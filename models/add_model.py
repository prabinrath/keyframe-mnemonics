from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
import torch.nn.functional as F
import torch.nn as nn
import torch


class SelectorFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 16):
        super().__init__(observation_space, features_dim)
        self.network = nn.Sequential(
            nn.Linear(3, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations.float()).squeeze()


class ProxyModel(nn.Module):
    def __init__(self, output_dim, total_slots, embd_dim=32, device="cpu"):
        super().__init__()
        self.device = device
        input_dim = total_slots * 3
        self.proj = nn.Sequential(
            nn.Linear(input_dim, embd_dim),
            nn.ReLU(),
            nn.Linear(embd_dim, embd_dim),
            nn.ReLU(),
            nn.Linear(embd_dim, embd_dim),
            nn.ReLU(),
            nn.Linear(embd_dim, output_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.proj(observations.float())

    def get_action(self, observations):
        self.eval()
        with torch.inference_mode():
            actions = self.forward(observations)
        return actions

    def compute_loss(self, observations, expert_actions):
        return F.mse_loss(self.forward(observations), expert_actions)

    def compute_loss_for_reward(self, observations, expert_actions):
        self.eval()
        actions = self.get_action(observations).reshape(expert_actions.shape)
        return F.mse_loss(actions.cpu(), expert_actions, reduction="none").mean(dim=-1)


class AddPolicy(ProxyModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
