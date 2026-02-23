import torch
import torch.nn as nn
import pytorch_lightning as L

from layers import SelfAttentionBlock, CrossAttentionBlock, WBosonFourVectorLayer, Standardization
from losses import (
    huber_loss, neg_r2_loss, w_mass_mae_losses, w_mass_mmd_losses,
    higgs_mass_loss, nu_mass_loss, aux_mom_mmd_loss, alpha_mmd_loss
)


class WBosonDetectiveNet(nn.Module):
    def __init__(self, input_dim, std_mean_train, std_scale_train, d_model=256, nhead=8):
        super().__init__()
        self.norm = Standardization(std_mean_train, std_scale_train)
        
        # 1. Specialized Embedders (Witnesses)
        self.lep_embed = nn.Linear(4, d_model)
        self.jet_embed = nn.Linear(4, d_model)
        self.met_embed = nn.Linear(2, d_model)
        hl_input_dim = input_dim - 22 # 2 leps (8) + 3 jets (12) + 1 met (2)
        self.hl_embed  = nn.Linear(hl_input_dim, d_model)

        # 2. The Detectives (Learned Latent Queries)
        # Two detectives: one for W0/nu0, one for W1/nu1 (1 for the latter expand to batch size in forward())
        self.w_queries = nn.Parameter(torch.randn(1, 2, d_model))
        
        # 3. The Context Refiner (Self-Attention)
        self.context_refiner = SelfAttentionBlock(d_model, nhead)
        self.refined_context_refiner = SelfAttentionBlock(d_model, nhead)

        # 4. The Cross-Attention Engine
        self.investigation = CrossAttentionBlock(d_model, nhead)
        self.refined_investigation = CrossAttentionBlock(d_model, nhead)
        self.rerefined_investigation = CrossAttentionBlock(d_model, nhead)
        
        # 5. Output Head: answer to 3D momentum (px, py, pz)
        self.to_nu_mom = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 3),
        )
        # 6. Physics decoder layer to get W four-vectors from leptons and predicted neutrino 3-momenta
        self.w_layer = WBosonFourVectorLayer()

    def forward(self, x):
        # Expected x: [lep0(4), lep1(4), jet0(4), jet1(4), jet2(4), met(2), hl(...)]
        x_std = self.norm(x)
        batch_size = x.shape[0]

        key_mask = torch.zeros((batch_size, 7), dtype=torch.bool, device=x.device)
        # Check Jet 0 (indices 8:12), Jet 1 (12:16), Jet 2 (16:20)
        # If the 4-vector is all zeros, set key_mask to True
        key_mask[:, 2] = (torch.abs(x[:, 8:12]).sum(dim=1) == 0)  # Jet 0 slot
        key_mask[:, 3] = (torch.abs(x[:, 12:16]).sum(dim=1) == 0) # Jet 1 slot
        key_mask[:, 4] = (torch.abs(x[:, 16:20]).sum(dim=1) == 0) # Jet 2 slot

        # --- Step 1: Embedding the Witnesses ---
        l0 = self.lep_embed(x_std[:, 0:4])
        l1 = self.lep_embed(x_std[:, 4:8])
        j0 = self.jet_embed(x_std[:, 8:12])
        j1 = self.jet_embed(x_std[:, 12:16])
        j2 = self.jet_embed(x_std[:, 16:20])
        met = self.met_embed(x_std[:, 20:22])
        hl = self.hl_embed(x_std[:, 22:])
        
        # Combine into Context: [Batch, 7, d_model]
        context = torch.stack([l0, l1, j0, j1, j2, met, hl], dim=1)

        # --- Step 2: Self-Attention (The Relation) ---
        context = self.context_refiner(context, key_padding_mask=key_mask)
        context = self.refined_context_refiner(context, key_padding_mask=key_mask)
        
        # --- Step 3: Prepare the Detectives ---
        # queries shape: [Batch, 2, d_model]
        queries = self.w_queries.expand(batch_size, -1, -1)

        # --- Step 4: Cross-Attention (The Interaction) ---
        # The 2 detectives scan the 7 witnesses
        refined_w = self.investigation(queries, context, key_padding_mask=key_mask)
        refined_w = self.refined_investigation(refined_w, context, key_padding_mask=key_mask)
        refined_w = self.rerefined_investigation(refined_w, context, key_padding_mask=key_mask)
        # --- Step 5: Final Regression ---
        nu0_3mom = self.to_nu_mom(refined_w[:, 0, :])
        nu1_3mom = self.to_nu_mom(refined_w[:, 1, :])

        # Combine back into the format for w_layer: [Batch, 6]
        nu_3mom = torch.cat([nu0_3mom, nu1_3mom], dim=-1)
        # Use raw (original) leptons for the final physics layer to preserve accuracy
        out = self.w_layer(x[:, 0:4], x[:, 4:8], nu_3mom)
        return out
    
class LightningWBoson(L.LightningModule):
    def __init__(self, input_dim, std_mean_train, std_scale_train, lr=1e-4, loss_weights=None):
        super().__init__()
        self.save_hyperparameters()
        self.model = WBosonDetectiveNet(input_dim, std_mean_train, std_scale_train) # give a base model structure for forward() 
        defaults = {
            "huber": 1.0, "nu_mass": 0.0, "higgs_mass": 0.0,
            "w0_mass_mae": 0.0, "w1_mass_mae": 0.0,
            "w_mass_mmd0": 0.0, "w_mass_mmd1": 0.0,
            "dinu_pt": 0.0, "neg_r2": 0.0, 
            "aux_mom_mmd0": 0.0, "aux_mom_mmd1": 0.0,
            "alpha_mmd": 0.0,
        }
        self.loss_weights = {**defaults, **(loss_weights or {})}
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def _compute_losses(self, x, y, y_pred):

        losses = {
            "huber": huber_loss(y, y_pred),
            "nu_mass": nu_mass_loss(x, y_pred),
            "higgs_mass": higgs_mass_loss(y_pred),
            "w0_mass_mae": w_mass_mae_losses(y, y_pred)[0],
            "w1_mass_mae": w_mass_mae_losses(y, y_pred)[1],
            "w_mass_mmd0": w_mass_mmd_losses(y, y_pred)[0],
            "w_mass_mmd1": w_mass_mmd_losses(y, y_pred)[1],
            "neg_r2": neg_r2_loss(y, y_pred),
            "aux_mom_mmd0": aux_mom_mmd_loss(y, y_pred, self.current_epoch)[0],
            "aux_mom_mmd1": aux_mom_mmd_loss(y, y_pred, self.current_epoch)[1],
            "alpha_mmd": alpha_mmd_loss(x, y, y_pred),
        }
        total = sum(self.loss_weights[k] * v for k, v in losses.items())
        return total.mean(), losses

    def _log_losses(self, prefix, losses, total):
        self.log(f"{prefix}loss", total.detach(), prog_bar=False, on_step=False, on_epoch=True)
        for k, v in losses.items():
            self.log(f"{prefix}{k}_loss", v.detach(), prog_bar=False, on_step=False, on_epoch=True)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        total, losses = self._compute_losses(x, y, y_pred)
        self._log_losses("", losses, total)
        return total

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        total, losses = self._compute_losses(x, y, y_pred)
        self._log_losses("val_", losses, total)
        return total

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        total, losses = self._compute_losses(x, y, y_pred)
        self._log_losses("test_", losses, total)
        return total

    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=self.lr)
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=5e-4)
