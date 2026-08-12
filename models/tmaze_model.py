from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
import torch.nn.functional as F
import torch.nn as nn
import torch


class SelectorFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int=16):
        super().__init__(observation_space, features_dim)
        self.network = nn.Sequential(
            nn.Linear(2, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations).squeeze()
    

class ProxyModel(nn.Module):
    def __init__(self, output_dim, total_slots, embd_dim=16, device="cpu"):
        super().__init__()
        self.device = device
        self.register_buffer(
            "valid_actions",
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [-1.0, 0.0],
                    [0.0, -1.0],
                ],
                dtype=torch.float32,
            ),
        )
        self.proj = nn.Sequential(
            nn.Linear(total_slots * 2, embd_dim),
            nn.ReLU(),
            nn.Linear(embd_dim, embd_dim),
            nn.ReLU(),
            nn.Linear(embd_dim, embd_dim),
            nn.ReLU(),
            nn.Linear(embd_dim, output_dim)
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.proj(observations)          
        return x

    def decode_action(self, pred_actions: torch.Tensor) -> torch.Tensor:
        distances = (pred_actions.unsqueeze(1) - self.valid_actions.unsqueeze(0)).abs().mean(dim=-1)
        nearest_idx = distances.argmin(dim=-1)
        return self.valid_actions[nearest_idx]
    
    def get_action(self, observations):
        # to be used by evaluator
        self.eval()  # Ensure model is in eval mode to disable dropout
        with torch.inference_mode():
            pred_actions = self.forward(observations)
            actions = self.decode_action(pred_actions)
        return actions
    
    def compute_loss(self, observations, expert_actions):
        # to be used by proxy
        return F.mse_loss(self.forward(observations), expert_actions)
    
    def compute_loss_for_reward(self, observations, expert_actions):
        # to be used by vectorized selector env - returns per-sample losses
        self.eval()  # Ensure model is in eval mode
        with torch.inference_mode():
            actions = self.forward(observations).reshape(expert_actions.shape)
        return F.mse_loss(actions.cpu(), expert_actions, reduction='none').mean(dim=-1)
    

class TMazePolicy(ProxyModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
