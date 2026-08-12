from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.models as models
from models.common import NoInitWrapper
from models.common.policy_arch import FlowMatchingPolicy, SingleStepTransformerPolicy
from problems.real_robot_problem.lerobot_utils import (
    CAMERA_NAME,
    IMAGE_SIZE,
    PIXELS_PER_CAMERA,
    STATE_DIM,
)


class SelectorFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int=64,
                 hidden_dim=256):
        super().__init__(observation_space, features_dim)

        self.hidden_dim = hidden_dim

        # Use pretrained ResNet-18 for cameras (fully trainable)
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        resnet_sequential = nn.Sequential(*list(resnet.children())[:-2]).to(memory_format=torch.channels_last)
        self.resnet_features = NoInitWrapper(resnet_sequential)

        # ResNet-18 outputs 512 channels at the last conv layer
        resnet_output_size = 512 * (IMAGE_SIZE // 32) ** 2  # 8192

        # State projection layer
        self.state_projection = nn.Linear(STATE_DIM, self.hidden_dim)
        self.state_norm = nn.LayerNorm(self.hidden_dim)

        # Projection head to combine features
        self.feature_head = nn.Sequential(
            nn.Linear(resnet_output_size + self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.features_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        B = observations.shape[0]

        # Observation format: wrist_camera(49152) + state(8) = 49160
        wrist_camera = observations[:, :PIXELS_PER_CAMERA].reshape(
            B, IMAGE_SIZE, IMAGE_SIZE, 3).permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
        state = observations[:, PIXELS_PER_CAMERA:]  # B x STATE_DIM

        camera_feats = self.resnet_features(wrist_camera).flatten(1)  # B x 8192

        state_projected = self.state_norm(self.state_projection(state))

        fused = torch.cat([camera_feats, state_projected], dim=1)
        return self.feature_head(fused)


class ProxyModel(nn.Module):
    def __init__(self, output_dim, total_slots, hidden_dim=256,
                 attn_heads=4, horizon=8,
                 num_layers=4, device="cuda"):
        super().__init__()

        self.output_dim = output_dim
        self.horizon = horizon
        self.total_slots = total_slots
        self.hidden_dim = hidden_dim
        self.attn_heads = attn_heads
        self.num_layers = num_layers
        self.device = device

        # Use pretrained ResNet-18 for cameras (fully trainable)
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        resnet_sequential = nn.Sequential(*list(resnet.children())[:-2]).to(memory_format=torch.channels_last)
        self.resnet_features = NoInitWrapper(resnet_sequential)

        # ResNet-18 outputs 512 channels at 4x4 spatial resolution for 128x128 input
        self.spatial_patches = (IMAGE_SIZE // 32) ** 2
        self.resnet_to_hidden = nn.Linear(512, self.hidden_dim)
        self.resnet_norm = nn.LayerNorm(self.hidden_dim)

        # State projection: joint1..joint7 + gripper
        self.state_projection = nn.Linear(STATE_DIM, self.hidden_dim)
        self.state_norm = nn.LayerNorm(self.hidden_dim)

        # Per slot: one token per spatial patch, plus a state token
        num_tokens = self.total_slots * (self.spatial_patches + 1) + self.horizon
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

        # T in shape comments refers to total_slots
        observations_flat = observations.reshape(B * self.total_slots, -1)

        wrist_camera = observations_flat[:, :PIXELS_PER_CAMERA].reshape(
            B * self.total_slots, IMAGE_SIZE, IMAGE_SIZE, 3).permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
        state = observations_flat[:, PIXELS_PER_CAMERA:]  # (B*T) x STATE_DIM

        feats = self.resnet_features(wrist_camera)  # (B*T) x 512 x 4 x 4

        tokens = feats.flatten(2).permute(0, 2, 1)  # (B*T) x P x 512
        tokens = self.resnet_norm(self.resnet_to_hidden(tokens))
        camera_tokens = tokens.view(B, self.total_slots, self.spatial_patches, self.hidden_dim)

        state_tokens = self.state_norm(self.state_projection(state))
        state_tokens = state_tokens.view(B, self.total_slots, 1, self.hidden_dim)

        conditional_tokens = torch.cat([camera_tokens, state_tokens], dim=2).flatten(1, 2)

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


class RealRobotPolicy(nn.Module):
    _POLICY_TYPES = {
        "flow_matching_dit": FlowMatchingPolicy,
        "single_step_transformer": SingleStepTransformerPolicy,
    }

    def __init__(self, policy_type: str, *args, **kwargs):
        super().__init__()
        if policy_type not in self._POLICY_TYPES:
            raise ValueError(f"Unknown policy type '{policy_type}'. Choose from {list(self._POLICY_TYPES)}")
        kwargs.setdefault('state_dim', STATE_DIM)
        kwargs.setdefault('image_size', IMAGE_SIZE)
        # policy_arch is shared with 2-camera mikasa, so it takes a sequence
        kwargs.setdefault('camera_names', (CAMERA_NAME,))
        self.policy = self._POLICY_TYPES[policy_type](*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.policy.forward(*args, **kwargs)

    def get_action(self, observations, deterministic=True):
        return self.policy.get_action(observations, deterministic)

    def compute_loss(self, observations, expert_actions):
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
    obs_dim = PIXELS_PER_CAMERA + STATE_DIM
    num_runs = 100

    # Create test data
    observations = torch.randn(B, total_slots * obs_dim).to(device)
    expert_actions = torch.randn(B, horizon, action_dim).to(device)
    observation = torch.randn(B, obs_dim).to(device)

    # Test ProxyModel
    proxy_model = ProxyModel(output_dim=action_dim, total_slots=total_slots, horizon=horizon,
                             hidden_dim=512)
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
    selector_model = SelectorFeatureExtractor(None, features_dim=128, hidden_dim=256)
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

    # Test RealRobotPolicy (single-step transformer)
    policy_model = RealRobotPolicy(
        policy_type="flow_matching_dit",
        output_dim=action_dim, total_slots=total_slots, hidden_dim=512, num_layers=4,
    )
    policy_model.to(device)
    policy_model.eval()
    print(f"RealRobotPolicy parameters: {sum(p.numel() for p in policy_model.parameters() if p.requires_grad)}")

    policy_obs = {}
    for i in range(1, total_slots + 1):
        policy_obs[f'observation.images.{CAMERA_NAME}{i}'] = torch.rand(
            B, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    policy_obs['observation.state'] = torch.randn(B, STATE_DIM, device=device)

    policy_horizon = getattr(policy_model.policy, 'horizon', 1)
    loss = policy_model.compute_loss(policy_obs, torch.randn(B, policy_horizon, action_dim, device=device))
    with torch.inference_mode():
        _ = policy_model.get_action(policy_obs)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_runs):
            action = policy_model.get_action(policy_obs)
        if device.type == 'cuda':
            torch.cuda.synchronize()  # queued kernels must finish before we read the clock
        elapsed = (time.perf_counter() - start) / num_runs * 1000

    print(f"RealRobotPolicy output: action {action.shape}, loss: {loss.item():.4f}, inference: {elapsed:.2f} ms/batch")
