from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
import torch.nn.functional as F
import torch.nn as nn
import torch


class SelectorFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        num_symbols: int,
        noise_symbols: int,
        features_dim: int = 32,
    ):
        super().__init__(observation_space, features_dim * 2)
        max_val = num_symbols + noise_symbols + 2
        self.embed = nn.Embedding(max_val + 1, features_dim)
        self.network = nn.Sequential(
            nn.Linear(features_dim, features_dim * 2),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        embd_out = self.embed(observations.long())
        return self.network(embd_out).squeeze()


class ProxyModel(nn.Module):
    def __init__(
        self,
        output_dim,
        total_slots,
        num_symbols,
        noise_symbols,
        embd_dim=32,
        hidden_dim=256,
        device="cpu",
    ):
        super().__init__()
        self.device = device
        self.output_dim = output_dim
        self.total_slots = total_slots
        max_val = num_symbols + noise_symbols + 2
        self.embed = nn.Embedding(max_val + 1, embd_dim)
        self.embed.weight.requires_grad_(False)
        self.proj = nn.Sequential(
            nn.Linear(self.total_slots * embd_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embd_dim),
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations.long()
        x = self.embed(x)
        x = x.flatten(start_dim=1)
        x = self.proj(x)
        return x

    def get_action(self, observations, deterministic=True):
        self.eval()
        with torch.inference_mode():
            pred_emb = self.forward(observations)
            # decodes by nearest token embedding rather than applying output_dim-sized logits.
            token_embs = self.embed.weight
            distances = (pred_emb.unsqueeze(1) - token_embs.unsqueeze(0)).abs().mean(dim=-1)
            actions = distances.argmin(dim=-1)
        return actions

    def compute_loss(self, observations, expert_actions):
        pred_emb = self.forward(observations)
        targets = expert_actions.view(-1).long().to(pred_emb.device)
        target_emb = self.embed(targets)
        return F.l1_loss(pred_emb, target_emb)

    def compute_loss_for_reward(self, observations, expert_actions):
        self.eval()
        with torch.inference_mode():
            pred_emb = self.forward(observations).cpu()
            targets = expert_actions.view(-1).long().to(self.embed.weight.device)
            target_emb = self.embed(targets).cpu()
            loss = F.l1_loss(pred_emb, target_emb, reduction="none").mean(dim=-1)
        return loss


class ScatteredCopyPolicy(nn.Module):
    def __init__(
        self,
        output_dim,
        total_slots,
        num_symbols,
        noise_symbols,
        embd_dim=32,
        hidden_dim=256,
        attn_heads=4,
        num_layers=2,
        device="cpu",
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self.output_dim = output_dim
        self.total_slots = total_slots
        max_val = num_symbols + noise_symbols + 2

        self.embed = nn.Embedding(max_val + 1, embd_dim)
        self.embed.weight.requires_grad_(False)
        self.token_proj = nn.Linear(embd_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.total_slots + 1, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=attn_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(hidden_dim, embd_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations.long()
        x = self.embed(x)
        x = self.token_proj(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, :x.size(1)]
        x = self.transformer(x)
        return self.proj(x[:, 0])

    def get_action(self, observations, deterministic=True):
        self.eval()
        with torch.inference_mode():
            pred_emb = self.forward(observations)
            # decodes by nearest token embedding rather than applying output_dim-sized logits.
            token_embs = self.embed.weight
            distances = (pred_emb.unsqueeze(1) - token_embs.unsqueeze(0)).abs().mean(dim=-1)
            actions = distances.argmin(dim=-1)
        return actions

    def compute_loss(self, observations, expert_actions):
        pred_emb = self.forward(observations)
        targets = expert_actions.view(-1).long().to(pred_emb.device)
        target_emb = self.embed(targets)
        return F.l1_loss(pred_emb, target_emb)
