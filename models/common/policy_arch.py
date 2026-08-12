import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from diffusers import FlowMatchEulerDiscreteScheduler
from models.common.transformers import DiffusionTransformerDecoderBlock, TimestepEmbedding, FinalLayer
from models.common import NoInitWrapper


class FlowMatchingPolicy(nn.Module):
    def __init__(self, output_dim, total_slots, hidden_dim=256,
                 attn_heads=4, horizon=8, num_layers=4,
                 num_inference_steps=5, state_dim=9, image_size=128, **kwargs):
        super().__init__()

        self.output_dim = output_dim
        self.horizon = horizon
        self.total_slots = total_slots
        self.hidden_dim = hidden_dim
        self.attn_heads = attn_heads
        self.num_layers = num_layers
        self.num_inference_steps = num_inference_steps
        self.state_dim = state_dim
        self.image_size = image_size
        # ResNet-18 downsamples by 32x, giving (image_size//32)^2 spatial patches
        self.spatial_patches = (image_size // 32) ** 2

        # Flow matching scheduler
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=200)

        # Use pretrained ResNet-18 for cameras (fully trainable)
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        resnet_sequential = nn.Sequential(*list(resnet.children())[:-2]).to(memory_format=torch.channels_last)
        self.resnet_features = NoInitWrapper(resnet_sequential)

        # ResNet-18 outputs 512 channels at the last conv layer
        self.resnet_to_hidden = nn.Linear(512, self.hidden_dim)
        self.resnet_norm = nn.LayerNorm(self.hidden_dim)

        # State projection
        self.state_projection = nn.Linear(self.state_dim, self.hidden_dim)
        self.state_norm = nn.LayerNorm(self.hidden_dim)

        # Calculate number of tokens:
        # - Visual tokens: spatial_patches per camera (overhead + gripper = 2 cameras)
        # - State token: 1 (single token)
        # - Total context: total_slots * spatial_patches * 2 + 1
        # - Action queries: horizon tokens (noisy actions)
        visual_tokens = self.total_slots * self.spatial_patches * 2
        context_len = visual_tokens + 1

        # Positional embeddings for context and action tokens (scaled for stable init)
        self.context_pos_embedding = nn.Parameter(torch.randn(1, context_len, self.hidden_dim) * 0.02)
        self.action_pos_embedding = nn.Parameter(torch.randn(1, self.horizon, self.hidden_dim) * 0.02)

        # Timestep embedding for diffusion
        self.timestep_embed = TimestepEmbedding(self.hidden_dim)

        # Action to hidden projection
        self.action_projection = nn.Linear(self.output_dim, self.hidden_dim)
        self.action_norm = nn.LayerNorm(self.hidden_dim)

        # Causal mask for horizon tokens: float additive mask, 0.0 where attention is
        # allowed (lower triangle + diagonal) and -inf where it is blocked (upper triangle).
        action_mask = torch.triu(torch.full((self.horizon, self.horizon), float('-inf')), diagonal=1)
        self.register_buffer('action_mask', action_mask)

        # DiT decoder blocks with cross-attention to context
        self.transformer_blocks = nn.ModuleList([
            DiffusionTransformerDecoderBlock(
                hidden_size=self.hidden_dim,
                num_heads=self.attn_heads,
                mlp_ratio=4
            )
            for _ in range(self.num_layers)
        ])

        self.final_layer = FinalLayer(self.hidden_dim, self.output_dim)

    def encode_context(self, observations):
        """Encode observations into context tokens (expensive, cache this during inference).

        Args:
            observations: Dict with keys:
                - observation.images.overhead_camera{i}: float32 tensor (B, 3, H, W) for i=1..total_slots
                - observation.images.gripper_camera{i}: float32 tensor (B, 3, H, W) for i=1..total_slots
                - observation.state: float32 tensor (B, state_dim)
        """
        B = observations['observation.state'].shape[0]
        H = W = self.image_size
        P = self.spatial_patches  # patches per camera

        # Collect all camera images from buffers
        overhead_cameras = []
        gripper_cameras = []

        for i in range(1, self.total_slots + 1):
            gripper_cam = observations[f'observation.images.gripper_camera{i}'].contiguous(memory_format=torch.channels_last)
            gripper_cameras.append(gripper_cam)

            overhead_cam = observations[f'observation.images.overhead_camera{i}'].contiguous(memory_format=torch.channels_last)
            overhead_cameras.append(overhead_cam)

        # Stack cameras: B x T x 3 x H x W, then flatten to (B*T) x 3 x H x W
        gripper_cameras = torch.stack(gripper_cameras, dim=1).reshape(B * self.total_slots, 3, H, W)
        overhead_cameras = torch.stack(overhead_cameras, dim=1).reshape(B * self.total_slots, 3, H, W)

        # Extract state
        state = observations['observation.state']  # B x state_dim

        # Process both cameras together through ResNet: (2*B*T) x 3 x H x W
        both_cameras = torch.cat([overhead_cameras, gripper_cameras], dim=0)
        all_feats = self.resnet_features(both_cameras)  # (2*B*T) x 512 x sqrt(P) x sqrt(P)

        # Convert to tokens: (2*B*T) x P x hidden_dim
        all_tokens = all_feats.flatten(2).permute(0, 2, 1)
        all_tokens = self.resnet_to_hidden(all_tokens)
        all_tokens = self.resnet_norm(all_tokens)

        # Reshape to separate overhead and gripper: B x T x (2*P) x hidden_dim
        all_tokens = all_tokens.view(2, B * self.total_slots, P, self.hidden_dim).permute(1, 0, 2, 3)
        all_tokens = all_tokens.reshape(B * self.total_slots, 2 * P, self.hidden_dim)
        camera_tokens = all_tokens.view(B, self.total_slots, 2 * P, self.hidden_dim)

        # Flatten camera tokens: B x (total_slots * P * 2) x hidden_dim
        camera_tokens = camera_tokens.flatten(1, 2)

        # Project state to a single token
        state_tokens = self.state_projection(state)  # B x hidden_dim
        state_tokens = self.state_norm(state_tokens)
        state_tokens = state_tokens.unsqueeze(1)  # B x 1 x hidden_dim

        # Concatenate: B x (total_slots * P * num_cameras + 1) x hidden_dim
        context_tokens = torch.cat([camera_tokens, state_tokens], dim=1)
        context_tokens = context_tokens + self.context_pos_embedding

        return context_tokens

    def denoise_step(self, context_tokens, noisy_actions, timesteps):
        """Single denoising step given cached context tokens."""
        # Project noisy actions to tokens
        # noisy_actions: B x H x output_dim
        action_tokens = self.action_projection(noisy_actions)  # B x H x hidden_dim
        action_tokens = self.action_norm(action_tokens)
        action_tokens = action_tokens + self.action_pos_embedding

        # Get timestep conditioning
        t_emb = self.timestep_embed(timesteps)  # B x hidden_dim

        # Apply DiT blocks with cross-attention
        # No causal mask on action tokens: all horizon tokens are noisy simultaneously
        # during denoising, so full bidirectional attention is correct here.
        for block in self.transformer_blocks:
            action_tokens = block(action_tokens, context_tokens, t_emb,
                                  x_mask=None, mem_mask=None)

        # Predict denoised actions
        predicted_actions = self.final_layer(action_tokens, t_emb)  # B x H x output_dim
        return predicted_actions

    def forward(self, observations, noisy_actions, timesteps):
        """Full forward pass (used during training)."""
        context_tokens = self.encode_context(observations)
        return self.denoise_step(context_tokens, noisy_actions, timesteps)

    def get_action(self, observations, deterministic=True):
        # to be used by evaluator - denoise from pure noise
        self.eval()  # Ensure model is in eval mode to disable dropout
        with torch.inference_mode():
            B = observations['observation.state'].shape[0]
            device = observations['observation.state'].device

            # Encode context once (expensive ResNet computation)
            context_tokens = self.encode_context(observations)

            # Start from pure noise
            actions = torch.randn(B, self.horizon, self.output_dim, device=device)

            # Set up scheduler for inference
            self.noise_scheduler.set_timesteps(self.num_inference_steps, device=device)

            # Denoise iteratively (reusing cached context)
            for t in self.noise_scheduler.timesteps:
                timesteps = t.expand(B).to(device=device)
                model_output = self.denoise_step(context_tokens, actions, timesteps)
                actions = self.noise_scheduler.step(model_output, t, actions).prev_sample

        return actions

    def compute_loss(self, observations, expert_actions):
        # to be used by proxy - flow matching training
        B, H, A = expert_actions.shape
        device = observations['observation.state'].device

        # Set up scheduler timesteps for training (needed by scale_noise)
        self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps, device=device)

        # Sample random indices into the timesteps
        idx = torch.randint(0, len(self.noise_scheduler.timesteps), (B,), device=device)
        timesteps = self.noise_scheduler.timesteps[idx]

        # Sample noise (starting point)
        noise = torch.randn_like(expert_actions)

        # Add noise using scheduler's scale_noise method
        # x_t = sigma * noise + (1 - sigma) * sample
        noisy_actions = self.noise_scheduler.scale_noise(expert_actions, timesteps, noise)

        # Predict the velocity (direction from noise to data)
        predicted_velocity = self.forward(observations, noisy_actions, timesteps)

        # Flow matching objective: predict velocity field (noise - data)
        target = noise - expert_actions

        loss = F.mse_loss(predicted_velocity, target, reduction='mean')

        return loss


class SingleStepTransformerPolicy(nn.Module):
    """Transformer policy that predicts a single action (horizon=1).

    Observations are dicts with keys:
        - observation.images.overhead_camera{i}: float32 tensor (B, 3, H, W) for i=1..total_slots
        - observation.images.gripper_camera{i}: float32 tensor (B, 3, H, W) for i=1..total_slots
        - observation.state: float32 tensor (B, state_dim)
    """

    def __init__(self, output_dim, total_slots, hidden_dim=256,
                 attn_heads=4, horizon=1,
                 num_layers=4, device="cuda", state_dim=9, image_size=128, **kwargs):
        super().__init__()

        self.output_dim = output_dim
        self.total_slots = total_slots
        self.hidden_dim = hidden_dim
        self.attn_heads = attn_heads
        self.num_layers = num_layers
        self.device = device
        self.state_dim = state_dim
        self.image_size = image_size
        # ResNet-18 downsamples by 32x, giving (image_size//32)^2 spatial patches
        self.spatial_patches = (image_size // 32) ** 2
        assert horizon == 1, "SingleStepTransformerPolicy must have horizon=1"

        # Use pretrained ResNet-18 for cameras (fully trainable)
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        resnet_sequential = nn.Sequential(*list(resnet.children())[:-2]).to(memory_format=torch.channels_last)
        self.resnet_features = NoInitWrapper(resnet_sequential)

        # ResNet-18 outputs 512 channels at the last conv layer
        self.resnet_to_hidden = nn.Linear(512, self.hidden_dim)
        self.resnet_norm = nn.LayerNorm(self.hidden_dim)

        # State projection
        self.state_projection = nn.Linear(self.state_dim, self.hidden_dim)
        self.state_norm = nn.LayerNorm(self.hidden_dim)

        # Positional embeddings for camera tokens: per-camera-slot and per-spatial-patch.
        # Camera slot captures temporal position + camera identity; spatial patch captures 2D location in feature map.
        n_cams = total_slots * 2  # overhead + gripper per buffer slot
        self.camera_slot_embedding = nn.Parameter(torch.randn(1, n_cams, self.hidden_dim) * 0.02)
        self.spatial_patch_embedding = nn.Parameter(torch.randn(1, self.spatial_patches, self.hidden_dim) * 0.02)

        # Positional embeddings for state token and action query
        self.pos_embedding = nn.Parameter(torch.randn(1, 2, self.hidden_dim) * 0.02)  # state + action query
        self.action_query = nn.Parameter(torch.randn(1, 1, self.hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.attn_heads,
            dim_feedforward=self.hidden_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.action_head = nn.Linear(self.hidden_dim, self.output_dim)

    def encode_context(self, observations):
        """Encode dict observations into context tokens.

        Collects all camera images (overhead + gripper across all buffer steps) and
        processes them through ResNet. Each camera token receives a camera-slot
        embedding (temporal position + camera identity) and a spatial-patch embedding
        (2D location in the ResNet feature map). Appends a single state token.

        Returns:
            context_tokens: (B, N_cams * spatial_patches + 1, hidden_dim)
        """
        B = observations['observation.state'].shape[0]
        H = W = self.image_size
        P = self.spatial_patches  # patches per camera

        # Collect all cameras as tensors (overhead + gripper per buffer slot)
        cameras = []
        for i in range(1, self.total_slots + 1):
            cameras.append(observations[f'observation.images.overhead_camera{i}'])
            cameras.append(observations[f'observation.images.gripper_camera{i}'])

        # Process all cameras through ResNet in one batch
        cam_tensors = torch.stack(cameras, dim=1)                                      # B x N_cams x 3 x H x W
        cam_tensors = cam_tensors.reshape(B * len(cameras), 3, H, W)                  # (B*N_cams) x 3 x H x W
        cam_tensors = cam_tensors.contiguous(memory_format=torch.channels_last)       # match ResNet weight layout

        feats = self.resnet_features(cam_tensors)                                      # (B*N_cams) x 512 x sqrt(P) x sqrt(P)
        tokens = feats.flatten(2).permute(0, 2, 1)                                    # (B*N_cams) x P x 512
        tokens = self.resnet_to_hidden(tokens)                                         # (B*N_cams) x P x hidden_dim
        tokens = self.resnet_norm(tokens)

        # Add camera-slot + spatial-patch positional embeddings
        n_cams = len(cameras)
        cam_tokens = tokens.reshape(B, n_cams, P, self.hidden_dim)                 # B x N_cams x P x hidden_dim
        cam_tokens = (cam_tokens
                      + self.camera_slot_embedding.unsqueeze(2)                    # 1 x N_cams x 1 x hidden_dim
                      + self.spatial_patch_embedding.unsqueeze(1))                 # 1 x 1 x P x hidden_dim
        camera_tokens_flat = cam_tokens.reshape(B, n_cams * P, self.hidden_dim)   # B x (N_cams*P) x hidden_dim

        # State — single token
        state = observations['observation.state']                                      # B x state_dim
        state_token = self.state_norm(self.state_projection(state)).unsqueeze(1)      # B x 1 x hidden_dim

        return torch.cat([camera_tokens_flat, state_token], dim=1)                    # B x (N_cams*P + 1) x hidden_dim

    def forward(self, observations):
        B = observations['observation.state'].shape[0]

        conditional_tokens = self.encode_context(observations)

        # Append action query; pos_embedding applied to last 2 tokens (state token + action query)
        action_query = self.action_query.expand(B, -1, -1)
        tokens = torch.cat([conditional_tokens, action_query], dim=1)
        # Apply pos_embedding only to the last 2 tokens (state token + action query)
        tokens[:, -2:, :] = tokens[:, -2:, :] + self.pos_embedding

        transformer_out = self.transformer(tokens)
        action_token = transformer_out[:, -1, :]          # B x hidden_dim
        action = self.action_head(action_token).unsqueeze(1)  # B x 1 x output_dim
        return action

    def get_action(self, observations, deterministic=True):
        # to be used by evaluator; returns B x 1 x output_dim
        self.eval()
        with torch.inference_mode():
            action = self.forward(observations)  # B x 1 x output_dim
        return action

    def compute_loss(self, observations, expert_actions):
        # expert_actions: B x 1 x output_dim
        predicted_actions = self.forward(observations)  # B x 1 x output_dim
        expert_actions = expert_actions.view(predicted_actions.shape)

        loss = F.smooth_l1_loss(predicted_actions, expert_actions, reduction='mean')
        return loss


if __name__ == "__main__":
    import time

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    B = 4
    total_slots = 3
    horizon = 8
    action_dim = 8
    num_runs = 100

    def create_dict_observations(batch_size, total_slots, device, image_size=128, state_dim=9):
        obs_dict = {}
        for i in range(1, total_slots + 1):
            obs_dict[f'observation.images.gripper_camera{i}'] = torch.rand(batch_size, 3, image_size, image_size, dtype=torch.float32, device=device)
            obs_dict[f'observation.images.overhead_camera{i}'] = torch.rand(batch_size, 3, image_size, image_size, dtype=torch.float32, device=device)
        obs_dict['observation.state'] = torch.randn(batch_size, state_dim, device=device)
        return obs_dict

    print("\n=== Testing with overhead + gripper cameras ===")
    observations_dict = create_dict_observations(B, total_slots, device=device)

    model = FlowMatchingPolicy(output_dim=action_dim, total_slots=total_slots, horizon=horizon,
                               hidden_dim=512, num_inference_steps=5)
    model.to(device)
    model.eval()
    print(f"FlowMatchingPolicy parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    expert_actions = torch.randn(B, horizon, action_dim, device=device)
    loss = model.compute_loss(observations_dict, expert_actions)
    print(f"Training loss: {loss.item():.4f}")

    with torch.inference_mode():
        _ = model.get_action(observations_dict)
        start = time.perf_counter()
        for _ in range(num_runs):
            actions = model.get_action(observations_dict)
        elapsed = (time.perf_counter() - start) / num_runs * 1000
    print(f"FlowMatchingPolicy output: actions {actions.shape}, inference: {elapsed:.2f} ms/batch")

    single_step_model = SingleStepTransformerPolicy(output_dim=action_dim, total_slots=total_slots,
                                                    hidden_dim=512, num_layers=4)
    single_step_model.to(device)
    single_step_model.eval()
    print(f"SingleStepTransformerPolicy parameters: {sum(p.numel() for p in single_step_model.parameters() if p.requires_grad)}")

    expert_actions_single = torch.randn(B, 1, action_dim, device=device)
    loss = single_step_model.compute_loss(observations_dict, expert_actions_single)
    print(f"Training loss: {loss.item():.4f}")

    with torch.inference_mode():
        _ = single_step_model.get_action(observations_dict)
        start = time.perf_counter()
        for _ in range(num_runs):
            action = single_step_model.get_action(observations_dict)
        elapsed = (time.perf_counter() - start) / num_runs * 1000
    print(f"SingleStepTransformerPolicy output: action {action.shape}, inference: {elapsed:.2f} ms/batch")
