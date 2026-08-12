'''
Author: Prabin Kumar Rath (prath4@asu.edu)
'''

import torch
import torch.nn as nn
import math


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiffusionTransformerDecoderBlock(nn.Module):
    """
    Derived from https://github.com/facebookresearch/DiT
    A DiT decoder block with adaptive layer norm zero (adaLN-Zero) conditioning,
    built on top of nn.TransformerDecoderLayer.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4, mlp_drop=0.):
        super().__init__()
        self.layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * mlp_ratio,
            dropout=mlp_drop,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        # Replace affine norms — adaLN handles scale/shift conditioning
        self.layer.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.layer.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.layer.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )
        # Zero-initialize so gates start at 0 (identity at init)
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, mem, c, x_mask=None, mem_mask=None):
        shift_msa, scale_msa, gate_msa, \
        shift_mca, scale_mca, gate_mca, \
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(9, dim=1)

        x_msa = modulate(self.layer.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.layer.self_attn(x_msa, x_msa, x_msa, attn_mask=x_mask, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * self.layer.dropout1(attn_out)

        x_mca = modulate(self.layer.norm2(x), shift_mca, scale_mca)
        cross_out, _ = self.layer.multihead_attn(x_mca, mem, mem, attn_mask=mem_mask, need_weights=False)
        x = x + gate_mca.unsqueeze(1) * self.layer.dropout2(cross_out)

        x_mlp = modulate(self.layer.norm3(x), shift_mlp, scale_mlp)
        mlp_out = self.layer.linear2(self.layer.dropout(self.layer.activation(self.layer.linear1(x_mlp))))
        x = x + gate_mlp.unsqueeze(1) * self.layer.dropout3(mlp_out)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        # Zero-initialize the final linear so modulation starts neutral
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class TimestepEmbedding(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Derived from https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


if __name__ == '__main__':
    B, T, D, H = 2, 16, 256, 8
    t = torch.randint(0, 1000, (B,))
    x = torch.randn(B, T, D)
    mem = torch.randn(B, T, D)

    t_emb = TimestepEmbedding(D)
    dec_block = DiffusionTransformerDecoderBlock(D, H)
    final = FinalLayer(D, out_channels=D)

    dec_block.eval()

    with torch.no_grad():
        c = t_emb(t)                              # (B, D)

        dec_out = dec_block(x, mem, c)            # (B, T, D)
        assert dec_out.shape == (B, T, D), f"dec: {dec_out.shape}"

        out = final(dec_out, c)                   # (B, T, D)
        assert out.shape == (B, T, D), f"final: {out.shape}"

    print("All assertions passed.")
    print(f"  TimestepEmbedding : {c.shape}")
    print(f"  DecoderBlock out  : {dec_out.shape}")
    print(f"  FinalLayer out    : {out.shape}")
