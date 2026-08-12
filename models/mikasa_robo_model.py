from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.models as models
from models.common import NoInitWrapper
from models.common.policy_arch import FlowMatchingPolicy, SingleStepTransformerPolicy


class SelectorFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int=64,
                 hidden_dim=256, use_full_state=True):
        super().__init__(observation_space, features_dim)

        self.hidden_dim = hidden_dim
        self.use_full_state = use_full_state
        
        # Use pretrained ResNet-18 for cameras (fully trainable)
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        resnet_sequential = nn.Sequential(*list(resnet.children())[:-2]).to(memory_format=torch.channels_last)
        self.resnet_features = NoInitWrapper(resnet_sequential)
        
        # ResNet-18 outputs 512 channels at the last conv layer
        resnet_output_size = 512 * 4 * 4  # 8192 per camera
        
        # Calculate input size based on configuration (overhead + gripper cameras)
        num_cameras = 2
        state_dim = 25 if self.use_full_state else 9  # 25: tcp_pose(7) + qpos(9) + qvel(9), 9: qpos only
        
        # State projection layer
        self.state_projection = nn.Linear(state_dim, self.hidden_dim)
        self.state_norm = nn.LayerNorm(self.hidden_dim)
        
        # Projection head to combine features
        self.feature_head = nn.Sequential(
            nn.Linear(resnet_output_size * num_cameras + self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.features_dim),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        B = observations.shape[0]

        # Parse observation
        # Observation format: overhead_camera(49152) + gripper_camera(49152) + state(25) = 98329
        gripper_camera = observations[:, 49152:98304].reshape(B, 128, 128, 3).permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)  # B x 3 x 128 x 128
        
        # Extract state based on configuration
        if self.use_full_state:
            state = observations[:, 98304:]  # B x 25 (tcp_pose + qpos + qvel)
        else:
            state = observations[:, 98304+7:98304+7+9]  # B x 9 (only qpos)
        
        # Extract features from cameras
        # Process both cameras in parallel by concatenating along batch dimension
        overhead_camera = observations[:, :49152].reshape(B, 128, 128, 3).permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)  # B x 3 x 128 x 128
        # Concatenate both cameras and process together
        both_cameras = torch.cat([overhead_camera, gripper_camera], dim=0)  # (2*B) x 3 x 128 x 128
        all_feats = self.resnet_features(both_cameras).flatten(1)  # (2*B) x 8192
        # Reshape to (B, 16384) - concatenating overhead and gripper features
        camera_feats = all_feats.view(2, B, 8192).permute(1, 0, 2).reshape(B, 16384)

        # Project state to hidden_dim
        state_projected = self.state_projection(state)  # B x hidden_dim
        state_projected = self.state_norm(state_projected)
        
        # Concatenate all features
        fused = torch.cat([camera_feats, state_projected], dim=1)
        
        features = self.feature_head(fused)
        return features


class ProxyModel(nn.Module):
    def __init__(self, output_dim, total_slots, hidden_dim=256,
                 attn_heads=4, horizon=8,
                 use_full_state=True, num_layers=4, device="cuda"):
        super().__init__()

        self.output_dim = output_dim
        self.horizon = horizon
        self.total_slots = total_slots
        self.hidden_dim = hidden_dim
        self.attn_heads = attn_heads
        self.use_full_state = use_full_state
        self.num_layers = num_layers
        self.device = device
        
        # Use pretrained ResNet-18 for cameras (fully trainable)
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        resnet_sequential = nn.Sequential(*list(resnet.children())[:-2]).to(memory_format=torch.channels_last)
        self.resnet_features = NoInitWrapper(resnet_sequential)
        
        # ResNet-18 outputs 512 channels at 4x4 spatial resolution for 128x128 input
        self.resnet_to_hidden = nn.Linear(512, self.hidden_dim)
        self.resnet_norm = nn.LayerNorm(self.hidden_dim)
        
        # State projection based on configuration
        state_dim = 25 if self.use_full_state else 9  # 25: tcp_pose(7) + qpos(9) + qvel(9), 9: qpos only
        self.state_projection = nn.Linear(state_dim, self.hidden_dim)
        self.state_norm = nn.LayerNorm(self.hidden_dim)
        
        # Calculate number of tokens:
        # - Visual tokens: 16 per camera (4x4 spatial grid from ResNet), overhead + gripper
        # - State token: 1 per buffer element
        # - Total: total_slots * (16 * 2 + 1) + horizon queries
        tokens_per_buffer = 16 * 2 + 1
        num_tokens = self.total_slots * tokens_per_buffer + self.horizon
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, self.hidden_dim))
        self.horizon_queries = nn.Parameter(torch.randn(1, self.horizon, self.hidden_dim))
        
        # Create causal mask for horizon queries only
        context_len = num_tokens - self.horizon
        mask = torch.zeros(num_tokens, num_tokens, dtype=torch.bool)
        mask[context_len:, context_len:] = torch.triu(torch.ones(self.horizon, self.horizon, dtype=torch.bool), diagonal=1)
        self.register_buffer('causal_mask', mask)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.attn_heads,
            dim_feedforward=self.hidden_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        self.action_head = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, observations):
        B = observations.shape[0]
        observations = observations.view(B, self.total_slots, -1)
        
        # Parse observations for all buffer elements at once
        # Observation format: overhead_camera(49152) + gripper_camera(49152) + state(25) = 98329
        # Note: T in shape comments refers to total_slots
        observations_flat = observations.reshape(B * self.total_slots, -1)
        
        gripper_camera = observations_flat[:, 49152:98304].reshape(B * self.total_slots, 128, 128, 3).permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)  # (B*T) x 3 x 128 x 128
        
        # Extract state based on configuration
        if self.use_full_state:
            state = observations_flat[:, 98304:]  # (B*T) x 25 (tcp_pose + qpos + qvel)
        else:
            state = observations_flat[:, 98304+7:98304+7+9]  # (B*T) x 9 (only qpos)
        
        # Extract features from cameras (batch processing all buffer elements)
        # Process both cameras in parallel by concatenating along batch dimension
        overhead_camera = observations_flat[:, :49152].reshape(B * self.total_slots, 128, 128, 3).permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)  # (B*T) x 3 x 128 x 128
        # Concatenate both cameras and process together
        both_cameras = torch.cat([overhead_camera, gripper_camera], dim=0)  # (2*B*T) x 3 x 128 x 128
        all_feats = self.resnet_features(both_cameras)  # (2*B*T) x 512 x 4 x 4

        # Process all camera tokens together (no split needed)
        all_tokens = all_feats.flatten(2).permute(0, 2, 1)  # (2*B*T) x 16 x 512
        all_tokens = self.resnet_to_hidden(all_tokens)  # (2*B*T) x 16 x hidden_dim
        all_tokens = self.resnet_norm(all_tokens)

        # Reshape to separate overhead and gripper: first B*T are overhead, next B*T are gripper
        # Reshape to (2, B*T, 16, hidden_dim) then permute to (B*T, 2, 16, hidden_dim)
        all_tokens = all_tokens.view(2, B * self.total_slots, 16, self.hidden_dim).permute(1, 0, 2, 3)
        # Now reshape to (B*T, 32, hidden_dim) and then to (B, T, 32, hidden_dim)
        all_tokens = all_tokens.reshape(B * self.total_slots, 32, self.hidden_dim)
        camera_tokens = all_tokens.view(B, self.total_slots, 32, self.hidden_dim)

        # Project state
        state_tokens = self.state_projection(state)  # (B*T) x hidden_dim
        state_tokens = self.state_norm(state_tokens)
        # Reshape back to separate batch and time: B x T x 1 x hidden_dim
        state_tokens = state_tokens.view(B, self.total_slots, 1, self.hidden_dim)
        
        # Concatenate camera and state tokens: B x T x (16*2 + 1) x hidden_dim
        conditional_tokens = torch.cat([camera_tokens, state_tokens], dim=2)
        
        # Flatten buffer and token dimensions: B x (T * tokens_per_buffer) x hidden_dim
        conditional_tokens = conditional_tokens.flatten(1, 2)
        
        # Add horizon queries
        horizon_queries = self.horizon_queries.expand(B, -1, -1)
        tokens = torch.cat([conditional_tokens, horizon_queries], dim=1)
        tokens = tokens + self.pos_embedding[:, :tokens.size(1), :]
        
        transformer_out = self.transformer(tokens, mask=self.causal_mask)
        horizon_tokens = transformer_out[:, -self.horizon:, :]
        actions = self.action_head(horizon_tokens)  # B x H x output_dim (continuous actions)
        return actions

    def get_action(self, observations, deterministic=True):
        # to be used by evaluator
        self.eval()  # Ensure model is in eval mode to disable dropout
        with torch.inference_mode():
            actions = self.forward(observations)
        return actions
    
    def compute_loss(self, observations, expert_actions):
        # to be used by proxy 
        predicted_actions = self.forward(observations)
        B, H, A = predicted_actions.shape
        expert_actions = expert_actions.view(B, H, A)
        
        loss = F.l1_loss(
            predicted_actions,
            expert_actions,
            reduction='mean'
        )
        
        return loss
    
    def compute_loss_for_reward(self, observations, expert_actions):
        # to be used by vectorized selector env - returns per-sample losses
        self.eval()  # Ensure model is in eval mode
        with torch.inference_mode():
            predicted_actions = self.forward(observations)
            B, H, A = predicted_actions.shape
            expert_actions = expert_actions.view(B, H, A)
            
            loss = F.l1_loss(predicted_actions.cpu(), 
                                expert_actions, 
                                reduction='none').sum(dim=(1, 2))
        
        return loss


class MikasaRoboPolicy(nn.Module):
    _POLICY_TYPES = {
        "flow_matching_dit": FlowMatchingPolicy,
        "single_step_transformer": SingleStepTransformerPolicy,
    }
    STATE_DIM = 9    # qpos only
    IMAGE_SIZE = 128  # camera image height and width (square)

    def __init__(self, policy_type: str, state_grounded_actions: bool = False, *args, **kwargs):
        super().__init__()
        if policy_type not in self._POLICY_TYPES:
            raise ValueError(f"Unknown policy type '{policy_type}'. Choose from {list(self._POLICY_TYPES)}")
        self.state_grounded_actions = state_grounded_actions
        kwargs.setdefault('state_dim', self.STATE_DIM)
        kwargs.setdefault('image_size', self.IMAGE_SIZE)
        self.policy = self._POLICY_TYPES[policy_type](*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.policy.forward(*args, **kwargs)

    def get_action(self, observations, deterministic=True):
        actions = self.policy.get_action(observations, deterministic)
        if self.state_grounded_actions:
            state = observations['observation.state']  # B x 9
            actions = actions.clone()
            actions[:, :, :7] -= state[:, :7].unsqueeze(1)
        return actions

    def compute_loss(self, observations, expert_actions):
        if self.state_grounded_actions:
            state = observations['observation.state']  # B x 9
            expert_actions = expert_actions.clone()
            expert_actions[:, :, :7] += state[:, :7].unsqueeze(1)
        return self.policy.compute_loss(observations, expert_actions)


if __name__ == "__main__":
    import time

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    B = 4
    total_slots = 3
    horizon = 8
    action_dim = 8
    num_runs = 100

    # Create test data
    observations = torch.randn(B, total_slots * 98329).to(device)
    expert_actions = torch.randn(B, horizon, action_dim).to(device)
    observation = torch.randn(B, 98329).to(device)

    # Test ProxyModel
    proxy_model = ProxyModel(output_dim=action_dim, total_slots=total_slots, horizon=horizon,
                         hidden_dim=512, use_full_state=True)
    proxy_model.to(device)
    proxy_model.eval()  # Disable dropout for inference
    print(f"ProxyModel parameters: {sum(p.numel() for p in proxy_model.parameters() if p.requires_grad)}")
    
    with torch.inference_mode():
        # Warmup
        _ = proxy_model.get_action(observations)
        
        # Time inference
        start = time.perf_counter()
        for _ in range(num_runs):
            actions = proxy_model.get_action(observations)
        elapsed = (time.perf_counter() - start) / num_runs * 1000
        
        loss = proxy_model.compute_loss(observations, expert_actions)
        reward = proxy_model.compute_loss_for_reward(observations, expert_actions.cpu())
    
    print(f"ProxyModel output: actions {actions.shape}, loss: {loss.item():.4f}, reward: {reward}, inference: {elapsed:.2f} ms/batch")
    
    # Test SelectorFeatureExtractor
    selector_model = SelectorFeatureExtractor(None, features_dim=128, hidden_dim=256,
                                             use_full_state=True)
    selector_model.to(device)
    selector_model.eval()  # Disable dropout for inference
    print(f"SelectorFeatureExtractor parameters: {sum(p.numel() for p in selector_model.parameters() if p.requires_grad)}")
    
    with torch.inference_mode():
        # Warmup
        _ = selector_model(observation)
        
        # Time inference
        start = time.perf_counter()
        for _ in range(num_runs):
            features = selector_model(observation)
        elapsed = (time.perf_counter() - start) / num_runs * 1000
    
    print(f"SelectorFeatureExtractor output: features {features.shape}, inference: {elapsed:.2f} ms/batch")

    # Test ProxyModel with minimal config
    proxy_model_minimal = ProxyModel(output_dim=action_dim, total_slots=total_slots, horizon=horizon,
                                 hidden_dim=512, use_full_state=False)
    proxy_model_minimal.to(device)
    proxy_model_minimal.eval()  # Disable dropout for inference
    print(f"ProxyModel parameters: {sum(p.numel() for p in proxy_model_minimal.parameters() if p.requires_grad)}")
    
    with torch.inference_mode():
        # Warmup
        _ = proxy_model_minimal.get_action(observations)
        
        # Time inference
        start = time.perf_counter()
        for _ in range(num_runs):
            actions = proxy_model_minimal.get_action(observations)
        elapsed = (time.perf_counter() - start) / num_runs * 1000
        
        loss = proxy_model_minimal.compute_loss(observations, expert_actions)
        reward = proxy_model_minimal.compute_loss_for_reward(observations, expert_actions.cpu())
    
    print(f"ProxyModel output: actions {actions.shape}, loss: {loss.item():.4f}, reward: {reward}, inference: {elapsed:.2f} ms/batch")
    
    selector_model_minimal = SelectorFeatureExtractor(None, features_dim=128, hidden_dim=256,
                                                      use_full_state=False)
    selector_model_minimal.to(device)
    selector_model_minimal.eval()  # Disable dropout for inference
    print(f"SelectorFeatureExtractor parameters: {sum(p.numel() for p in selector_model_minimal.parameters() if p.requires_grad)}")
    
    with torch.inference_mode():
        # Warmup
        _ = selector_model_minimal(observation)
        
        # Time inference
        start = time.perf_counter()
        for _ in range(num_runs):
            features = selector_model_minimal(observation)
        elapsed = (time.perf_counter() - start) / num_runs * 1000
    
    print(f"SelectorFeatureExtractor output: features {features.shape}, inference: {elapsed:.2f} ms/batch")
    
