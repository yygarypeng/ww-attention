import torch
import torch.nn as nn

class WBosonFourVectorLayer(nn.Module):
    """
    Compute W four-vectors from leptons and predicted neutrino 3-momenta.
    """
    def forward(self, lep0, lep1, nu_3mom):
        nu0_3, nu1_3 = nu_3mom[..., :3], nu_3mom[..., 3:]
        # neutrino energies as |p| for (approx) massless
        nu0_E = torch.sqrt(torch.clamp(torch.sum(nu0_3 ** 2, dim=-1, keepdim=True), min=1e-16))
        nu1_E = torch.sqrt(torch.clamp(torch.sum(nu1_3 ** 2, dim=-1, keepdim=True), min=1e-16))
        nu0_4 = torch.cat([nu0_3, nu0_E], dim=-1)
        nu1_4 = torch.cat([nu1_3, nu1_E], dim=-1)
        return torch.cat([lep0 + nu0_4, lep1 + nu1_4], dim=-1)
    
class Standardization(nn.Module):
    def __init__(self, mean, std, eps=1e-16):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        self.eps = eps

    def forward(self, x):
        return (x - self.mean) / (self.std + self.eps)

class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.3):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, key_padding_mask=None):
        # x: [Batch, Seq_Len, d_model]
        res = x
        x = self.norm1(x)
        # Self-Attention: Q, K, and V are all 'x'
        x, _ = self.mha(x, x, x, key_padding_mask=key_padding_mask)
        x = res + x
        
        x = x + self.ffn(self.norm2(x))
        return x
    
class CrossAttentionBlock(nn.Module):
    """
    The Detective (Query) scans the Witnesses (Context).
    Uses the Pre-Norm architecture for training stability.
    """
    def __init__(self, d_model, nhead, dropout=0.3):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model // 4),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, queries, context, key_padding_mask=None):
        # 1. Attention Phase
        res = queries
        q = self.norm_q(queries)
        kv = self.norm_kv(context)
        
        # Cross-attention: Query comes from 'queries', Key/Value from 'context'
        # attn_out shape: [Batch, 2, d_model]
        attn_out, _ = self.mha(query=q, key=kv, value=kv, key_padding_mask=key_padding_mask)
        x = res + attn_out
        
        # 2. Feed-Forward Phase
        x = x + self.ffn(x)
        return x