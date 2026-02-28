import torch
import torch.nn as nn
import pytorch_lightning as L

from layers import SelfAttentionBlock, CrossAttentionBlock, WBosonFourVectorLayer, Standardization
from losses import (
    huber_loss, neg_r2_loss, w_mass_mae_losses, w_mass_mmd_losses,
    higgs_mass_loss, nu_mass_loss, aux_mom_mmd_loss, alpha_loss
)


class WAttentionNet(nn.Module):
    def __init__(
            self, input_dim, 
            std_mean_train, std_scale_train, 
            d_model=128, nhead=8, num_self_attn=3, num_cross_attn=3
        ):
        super().__init__()
        self.d_model = d_model
        self.norm = Standardization(std_mean_train, std_scale_train)

        # 1. Embedders
        self.lep_embed = nn.Linear(4, d_model)
        self.jet_embed = nn.Linear(4, d_model)
        self.met_embed = nn.Linear(2, d_model)
        hl_input_dim = input_dim - 22
        self.hl_embed = nn.Linear(hl_input_dim, d_model)

        # 2. Dynamic query generator (10 -> d_model*2)
        self.query_generator = nn.Sequential(
            nn.LayerNorm(10), # lep0(4), lep1(4), met(2)
            nn.Linear(10, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model * 2),
        )

        # 3. Context refiner
        self.context_refiners = nn.ModuleList(
            [SelfAttentionBlock(d_model, nhead) for _ in range(num_self_attn)]
        )

        # 4. Cross-Attention
        self.investigations = nn.ModuleList(
            [CrossAttentionBlock(d_model, nhead) for _ in range(num_cross_attn)]
        )

        # 5. Output head
        self.to_nu_mom = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, 32),
            nn.SiLU(),
            nn.Linear(32, 3),
        )

        # 6. Physics decoder
        self.w_layer = WBosonFourVectorLayer()

    def forward(self, x):
        # Expected x: [lep0(4), lep1(4), jet0(4), jet1(4), jet2(4), met(2), hl(...)]
        x_std = self.norm(x)
        batch_size = x.shape[0]

        # Key mask for empty jets
        key_mask = torch.zeros((batch_size, 7), dtype=torch.bool, device=x.device)
        key_mask[:, 2] = (x[:, 8:12].abs().sum(dim=1) == 0)
        key_mask[:, 3] = (x[:, 12:16].abs().sum(dim=1) == 0)
        key_mask[:, 4] = (x[:, 16:20].abs().sum(dim=1) == 0)

        # Step 1: Embedding to get initial context
        l0 = self.lep_embed(x_std[:, 0:4])
        l1 = self.lep_embed(x_std[:, 4:8])
        j0 = self.jet_embed(x_std[:, 8:12])
        j1 = self.jet_embed(x_std[:, 12:16])
        j2 = self.jet_embed(x_std[:, 16:20])
        met = self.met_embed(x_std[:, 20:22])
        hl = self.hl_embed(x_std[:, 22:])
        
        # Combine into Context: [Batch, 7, d_model]
        context = torch.stack([l0, l1, j0, j1, j2, met, hl], dim=1)

        # Step 2: Self-Attention
        for refiner in self.context_refiners:
            context = refiner(context, key_padding_mask=key_mask)

        # Step 3: Dynamic queries
        q_input = torch.cat([x_std[:, 0:4], x_std[:, 4:8], x_std[:, 20:22]], dim=-1)
        # queries shape: [Batch, 2, d_model] ~ p(queries | leptons, random_guess)
        queries = self.query_generator(q_input).reshape(batch_size, 2, self.d_model)
        # queries = self.w_queries.expand(batch_size, -1, -1) # static

        # Step 4: Cross-Attention
        refined_w = queries
        for investigator in self.investigations:
            refined_w = investigator(refined_w, context, key_padding_mask=key_mask)

        # Step 5: Regression
        nu0_out = self.to_nu_mom(refined_w[:, 0, :])
        nu1_out = self.to_nu_mom(refined_w[:, 1, :])

        # Combine neutrino outputs and decode to W four-vectors
        nu_3mom = torch.cat([nu0_out, nu1_out], dim=-1) # Shape: [B, 6]
        swap_nu_3mom = torch.cat([nu1_out, nu0_out], dim=-1)
        
        # Get the deterministic physical 4-vectors
        w_4vecs = self.w_layer(x[:, 0:4], x[:, 4:8], nu_3mom)
        swap_w_4vecs = self.w_layer(x[:, 0:4], x[:, 4:8], swap_nu_3mom)

        return w_4vecs, swap_w_4vecs


class LightningWAttention(L.LightningModule):
    def __init__(
            self, input_dim, std_mean_train, std_scale_train,
            lr=1e-4, loss_weights=None,
            d_model=128, nhead=8, num_self_attn=6, num_cross_attn=6
        ):
        super().__init__()
        self.save_hyperparameters()
        self.model = WAttentionNet(
            input_dim, std_mean_train, std_scale_train,
            d_model=d_model, nhead=nhead,
            num_self_attn=num_self_attn, num_cross_attn=num_cross_attn
        )
        defaults = {
            "huber": 1.0,
            "higgs_mass": 0.0,
            "w0_mass_mae": 0.0, "w1_mass_mae": 0.0,
            "alpha": 0.0,
            # auxiliary monitoring losses:
            "neg_r2": 0.0, 
            "nu_mass": 0.0,
            "aux_mom_mmd0": 0.0, "aux_mom_mmd1": 0.0,
            "w_mass_mmd0": 0.0, "w_mass_mmd1": 0.0,
        }
        self.loss_weights = {**defaults, **(loss_weights or {})}
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def _compute_losses(self, x, y, y_pred):
        # Compute tuple-returning losses once to avoid redundant calls
        w_mass_mmd  = w_mass_mmd_losses(y, y_pred)
        aux_mom_mmd = aux_mom_mmd_loss(y, y_pred, self.current_epoch)
        w_mass_mae  = w_mass_mae_losses(y, y_pred)

        losses = {
            "huber":        huber_loss(y, y_pred),
            "higgs_mass":   higgs_mass_loss(y_pred),
            "w_mass_mmd0":  w_mass_mmd[0],
            "w_mass_mmd1":  w_mass_mmd[1],
            "alpha":        alpha_loss(x, y, y_pred),
            # auxiliary monitoring losses:
            "neg_r2":       neg_r2_loss(y, y_pred),
            "nu_mass":      nu_mass_loss(x, y_pred),
            "aux_mom_mmd0": aux_mom_mmd[0],
            "aux_mom_mmd1": aux_mom_mmd[1],
            "w0_mass_mae":  w_mass_mae[0],
            "w1_mass_mae":  w_mass_mae[1],
        }
        reduced = {k: v.mean() for k, v in losses.items()}
        cls_cri = (
            self.loss_weights["huber"]       * reduced["huber"] +
            self.loss_weights["higgs_mass"]  * reduced["higgs_mass"] +
            self.loss_weights["alpha"]       * reduced["alpha"]
        )
        return reduced, cls_cri

    def _shared_step(self, batch):
        x, y = batch
        y_pred, swap_y_pred = self(x)
        losses,      cls_cri      = self._compute_losses(x, y, y_pred)
        swap_losses, swap_cls_cri = self._compute_losses(x, y, swap_y_pred)

        use_swap = swap_cls_cri < cls_cri
        total = torch.where(
            use_swap,
            sum(self.loss_weights[k] * v for k, v in swap_losses.items()),
            sum(self.loss_weights[k] * v for k, v in losses.items()),
        )
        merged = {k: torch.where(use_swap, swap_losses[k], losses[k]) for k in losses}
        return merged, total

    def _log_losses(self, prefix, losses, total):
        self.log(f"{prefix}loss", total.detach(), prog_bar=True, on_step=False, on_epoch=True)
        for k, v in losses.items():
            self.log(f"{prefix}{k}_loss", v.detach(), prog_bar=False, on_step=False, on_epoch=True)

    def training_step(self, batch, batch_idx):
        losses, total = self._shared_step(batch)
        self._log_losses("", losses, total)
        return total

    def validation_step(self, batch, batch_idx):
        losses, total = self._shared_step(batch)
        self._log_losses("val_", losses, total)
        return total

    def test_step(self, batch, batch_idx):
        losses, total = self._shared_step(batch)
        self._log_losses("test_", losses, total)
        return total

    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=self.lr)
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=5e-4)
