"""
LTMB model architectures (vision + direction, no language conditioning).

Observations are flattened 7x7x3 minigrid images plus direction (148 dims).
"""
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.distributions import Categorical
import gymnasium as gym
import torch.nn as nn
import torch


class SelectorFeatureExtractor(BaseFeaturesExtractor):
    """CNN feature extractor for the PPO selector. Input: (148,) image + direction."""

    def __init__(self, observation_space: gym.Space, features_dim: int = 64,
                 hidden_dim: int = 128, in_ch: int = 3):
        super().__init__(observation_space, features_dim)
        self.hidden_dim = hidden_dim
        self.in_ch = in_ch

        self.stem = nn.Sequential(
            nn.Conv2d(self.in_ch, self.hidden_dim, 3, padding=0),
            nn.LayerNorm([self.hidden_dim, 5, 5]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=0),
            nn.LayerNorm([self.hidden_dim, 3, 3]),
            nn.ReLU(inplace=True),
        )

        self.dir_mlp = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.feature_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 10, features_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        B = observations.shape[0]
        image = observations[:, :147].reshape(B, 7, 7, 3).permute(0, 3, 1, 2) / 10.0
        direction = observations[:, 147:148].float()
        visual_feats = self.stem(image.float())
        image_tokens = visual_feats.flatten(1)
        dir_token = self.dir_mlp(direction)
        features = self.feature_head(torch.cat([dir_token, image_tokens], dim=1))
        return features.squeeze()


class ProxyModel(nn.Module):
    """
    Transformer-based action predictor for LTMB (vision + direction).

    Input:  (B, total_slots * 148)  -- flattened buffer of images + directions
    Output: (B, horizon, 7)         -- action logits per horizon step
    """

    def __init__(self, output_dim, total_slots, hidden_dim: int = 128,
                 in_ch: int = 3,
                 attn_heads: int = 4, entropy_coef: float = 0.01,
                 horizon: int = 8, device="cuda"):
        super().__init__()

        self.output_dim = output_dim 
        self.horizon = horizon
        self.total_slots = total_slots
        self.hidden_dim = hidden_dim
        self.in_ch = in_ch
        self.attn_heads = attn_heads
        self.entropy_coef = entropy_coef
        self.device = device

        self.stem = nn.Sequential(
            nn.Conv2d(self.in_ch, self.hidden_dim, 3, padding=0),
            nn.LayerNorm([self.hidden_dim, 5, 5]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=0),
            nn.LayerNorm([self.hidden_dim, 3, 3]),
            nn.ReLU(inplace=True),
        )

        self.dir_mlp = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.ReLU(inplace=True),
        )

        # total_slots image patches (9 tokens each) + total_slots direction tokens + horizon query tokens
        num_tokens = self.total_slots * 10 + self.horizon
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, self.hidden_dim))
        self.horizon_queries = nn.Parameter(torch.randn(1, self.horizon, self.hidden_dim))

        # causal mask: horizon queries attend causally to each other, freely to context
        context_len = num_tokens - self.horizon
        mask = torch.zeros(num_tokens, num_tokens, dtype=torch.bool)
        mask[context_len:, context_len:] = torch.triu(
            torch.ones(self.horizon, self.horizon, dtype=torch.bool), diagonal=1
        )
        self.register_buffer('causal_mask', mask)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.attn_heads,
            dim_feedforward=self.hidden_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.action_head = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, observations):
        B = observations.shape[0]
        observations = observations.view(B, self.total_slots, -1)
        images = observations[:, :, :147].reshape(
            B * self.total_slots, 7, 7, 3
        ).permute(0, 3, 1, 2) / 10.0
        directions = observations[:, :, 147:148].reshape(B * self.total_slots, 1).float()

        feats = self.stem(images.float())
        feats = feats.view(B, self.total_slots, self.hidden_dim, 3, 3)
        image_tokens = feats.flatten(3, 4).transpose(2, 3).reshape(
            B, self.total_slots * 9, self.hidden_dim
        )
        dir_tokens = self.dir_mlp(directions).view(B, self.total_slots, self.hidden_dim)
        tokens = torch.cat([dir_tokens, image_tokens], dim=1)

        horizon_queries = self.horizon_queries.expand(B, -1, -1)
        tokens = torch.cat([tokens, horizon_queries], dim=1)
        tokens = tokens + self.pos_embedding[:, :tokens.size(1), :]

        transformer_out = self.transformer(tokens, mask=self.causal_mask)
        horizon_tokens = transformer_out[:, -self.horizon:, :]
        logits = self.action_head(horizon_tokens)
        return logits

    def get_action(self, observations, deterministic=True):
        self.eval()
        with torch.inference_mode():
            logits = self.forward(observations)
            B, H, A = logits.shape
            distribution = Categorical(logits=logits.reshape(B * H, A))
        if deterministic:
            action = distribution.logits.argmax(dim=-1).view(B, H)
        else:
            action = distribution.sample().view(B, H)
        return action

    def compute_loss(self, observations, expert_actions):
        logits = self.forward(observations)
        B, H, A = logits.shape
        expert_actions = expert_actions[:, :H].contiguous().view(B, H).long()
        ce_loss = nn.functional.cross_entropy(
            logits.reshape(B * H, A),
            expert_actions.reshape(B * H),
            reduction='none'
        ).view(B, H).mean()

        distribution = Categorical(logits=logits.reshape(B * H, A))
        entropy = distribution.entropy().mean()
        return ce_loss - self.entropy_coef * entropy

    def compute_loss_for_reward(self, observations, expert_actions):
        self.eval()
        with torch.inference_mode():
            logits = self.forward(observations).cpu()
            B, H, A = logits.shape
            expert_actions = expert_actions[:, :H].contiguous().view(B, H).long()
            ce_loss = nn.functional.cross_entropy(
                logits.reshape(B * H, A),
                expert_actions.reshape(B * H),
                reduction='none'
            ).view(B, H).sum(dim=1)
        return ce_loss

class LtmbPolicy(ProxyModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

if __name__ == "__main__":
    B, total_slots, horizon = 10, 3, 8
    obs = torch.randint(0, 10, (B, total_slots * 148))
    actions = torch.randint(0, 7, (B, horizon))

    model = ProxyModel(output_dim=7, total_slots=total_slots, horizon=horizon)
    print(f"ProxyModel params: {sum(p.numel() for p in model.parameters()):,}")
    with torch.no_grad():
        out = model.get_action(obs)
        loss = model.compute_loss(obs, actions)
        reward = model.compute_loss_for_reward(obs, actions)
    print(f"actions: {out.shape}, loss: {loss:.4f}, reward: {reward.shape}")

    sel = SelectorFeatureExtractor(None, features_dim=64)
    single_obs = torch.randint(0, 10, (B, 148))
    print(f"SelectorFeatureExtractor params: {sum(p.numel() for p in sel.parameters()):,}")
    with torch.no_grad():
        feats = sel(single_obs)
    print(f"features: {feats.shape}")
