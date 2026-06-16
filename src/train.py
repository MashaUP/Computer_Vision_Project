import os
import time
from math import exp
import numpy as np
import torch
torch.set_float32_matmul_precision('high')
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchvision.ops import MLP
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from lejepa.multivariate import SlicingUnivariateTest
from lejepa.univariate import EppsPulley
from src.dataset import get_ssl_transform
from src.network import METERModel

# ==========================================
# Encoder output dimensions
# ==========================================
ENC_DIMS = {
    "xxs": 160,
    "xs":  192,
    "s":   320
}

# ==========================================
# Loss — exact copy from the professor (loss.py)
# With the addition of the KITTI branch (valid_mask) necessary
# for the sparse LiDAR GT
# ==========================================

def gaussian(window_size, sigma):
    gauss = torch.Tensor([
        exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, val_range, window_size=11, window=None, size_average=True, full=False):
    L = val_range
    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq  = mu1.pow(2)
    mu2_sq  = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12   = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k  = np.stack((edge_kx, edge_ky))
        edge_k  = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.edge_conv(x)
        out = out.contiguous().view(-1, 2, x.size(2), x.size(3))
        return out


class balanced_loss_function(nn.Module):
    """
    Exact copy of the professor's loss (loss.py).
    - NYU: Dense GT, no mask, val_range=1000 (centimeters)
    - KITTI: Sparse GT (LiDAR), valid_mask applied, val_range=80 (meters)
    lambda_1=0.5, lambda_2=100, lambda_3=100 as per paper Section IV-A.
    """
    def __init__(self, device, dataset_name="kitti"):
        super(balanced_loss_function, self).__init__()
        self.cos          = nn.CosineSimilarity(dim=1, eps=0)
        self.get_gradient = Sobel().to(device)
        self.device       = device
        self.dataset_name = str(dataset_name).lower().strip()
        self.lambda_1     = 0.5
        if "nyu" in self.dataset_name:
            self.lambda_2 = 100    # NYU: depth in cm
            self.lambda_3 = 100
        else:
            self.lambda_2 = 1      # KITTI: depth in m
            self.lambda_3 = 1

    def forward(self, output, depth):
        # Force float32 for numerical stability with mixed precision
        output = output.float()
        depth  = depth.float()

        # Clamp output to minimum positive values ONLY inside the loss.
        # Leaves the decoder free in the forward pass, but protects SSIM and Sobel
        # from division by zero in the early epochs (random decoder → output~0).
        eps = 1e-3
        output = torch.clamp(output, min=eps)

        with torch.no_grad():
            ones = torch.ones_like(depth)

        if "nyu" in self.dataset_name:
            # ---- NYU: 1:1 copy from the professor ----
            depth_grad  = self.get_gradient(depth)
            output_grad = self.get_gradient(output)

            depth_grad_dx  = depth_grad[:,  0, :, :].contiguous().view_as(depth)
            depth_grad_dy  = depth_grad[:,  1, :, :].contiguous().view_as(depth)
            output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(depth)
            output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(depth)

            depth_normal  = torch.cat((-depth_grad_dx,  -depth_grad_dy,  ones), 1)
            output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)

            loss_depth  = torch.abs(output - depth).mean()
            loss_dx     = torch.abs(output_grad_dx - depth_grad_dx).mean()
            loss_dy     = torch.abs(output_grad_dy - depth_grad_dy).mean()
            loss_normal = self.lambda_2 * torch.abs(
                1 - self.cos(output_normal, depth_normal)
            ).mean()
            loss_ssim   = (1 - ssim(output, depth, val_range=1000.0)) * self.lambda_3

        else:
            # ---- KITTI: valid_mask for sparse GT ----
            valid_mask    = (depth > 0.001).float()
            valid_pixels  = torch.clamp(valid_mask.sum(), min=1.0)
            # In empty pixels we replace the prediction with the GT (zero)
            # so Sobel and SSIM do not see noise in the sky
            output_masked = output * valid_mask + depth * (1 - valid_mask)

            depth_grad  = self.get_gradient(depth)
            output_grad = self.get_gradient(output_masked)

            depth_grad_dx  = depth_grad[:,  0, :, :].contiguous().view_as(depth)
            depth_grad_dy  = depth_grad[:,  1, :, :].contiguous().view_as(depth)
            output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(depth)
            output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(depth)

            depth_normal  = torch.cat((-depth_grad_dx,  -depth_grad_dy,  ones), 1)
            output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)

            # Compute valid_grad_mask (3x3 erosion of valid_mask) to filter out boundary artifacts from Sobel/normals
            valid_grad_mask = -F.max_pool2d(-valid_mask, kernel_size=3, stride=1, padding=1)
            valid_grad_pixels = torch.clamp(valid_grad_mask.sum(), min=1.0)

            loss_depth  = (torch.abs(output_masked - depth) * valid_mask).sum() / valid_pixels
            loss_dx     = (torch.abs(output_grad_dx - depth_grad_dx) * valid_grad_mask).sum() / valid_grad_pixels
            loss_dy     = (torch.abs(output_grad_dy - depth_grad_dy) * valid_grad_mask).sum() / valid_grad_pixels
            cos_sim     = self.cos(output_normal, depth_normal).unsqueeze(1)
            loss_normal = self.lambda_2 * (
                (torch.abs(1 - cos_sim) * valid_grad_mask).sum() / valid_grad_pixels
            )
            loss_ssim   = (
                1 - ssim(output_masked, depth, val_range=80.0)
            ) * self.lambda_3

        # Paper METER Eq. 1: L = L_depth + λ_1 * L_grad + λ_2 * L_norm + λ_3 * L_SSIM
        # We use the paper formula (* λ_1) on both datasets. The professor's code in
        # loss.py uses (/ λ_1), which inverts the intended weighting (×4 instead of ×0.5).
        # On KITTI we already use the paper formula; on NYU we now align it.
        loss_grad = (loss_dx + loss_dy) * self.lambda_1
        return loss_depth, loss_ssim, loss_normal, loss_grad


# ==========================================
# LeJEPA SSL Pre-training
# ==========================================
class LeJEPAMeter(L.LightningModule):
    """
    Self-supervised pre-training with LeJEPA.
    Uses the METERModel encoder and trains it with
    invariance loss + SIGReg (SlicingUnivariateTest).
    Multi-crop: 2 global views (256px) + 6 local views (128px).
    """
    def __init__(
        self,
        arch_type="xxs",
        proj_dim=64,
        num_slices=256,
        lamb=0.02,
        lr=5e-4,
        weight_decay=5e-2,
        warmup_epochs=10,
        dataset="kitti",
    ):
        super().__init__()
        self.arch_type = arch_type
        self.save_hyperparameters()

        print("\n[INFO] Initializing LeJEPA SSL model")
        self.encoder = METERModel(arch_type=arch_type).encoder
        self.encoder.requires_grad_(True)

        enc_dim = ENC_DIMS[arch_type]
        print(f"[INFO] Encoder dim: {enc_dim}")

        # Multi-crop augmentations
        self.global_aug = get_ssl_transform(global_view=True,  dataset=dataset)
        self.local_aug  = get_ssl_transform(global_view=False, dataset=dataset)

        # Projector MLP: enc_dim -> 256 -> 512 -> proj_dim
        self.projector = MLP(
            enc_dim,
            [256, 512, proj_dim],
            norm_layer=nn.LayerNorm
        )

        # SIGReg: Sliced Isotropic Gaussian Regularization
        self.sigreg = SlicingUnivariateTest(
            univariate_test=EppsPulley(t_max=5.0, n_points=21),
            num_slices=num_slices,
            reduction="mean"
        )
        print("[INFO] LeJEPA model ready")

    def forward(self, x):
        features, _ = self.encoder(x)
        emb  = features.mean(dim=[-2, -1])
        proj = self.projector(emb)
        return emb, proj

    def _lejepa_loss(self, proj):
        """
        proj: (num_views, batch, proj_dim)
        Invariance loss + SIGReg as per LeJEPA paper.
        """
        # Clamp + cast float32 before SIGReg for numerical stability
        proj_safe = torch.clamp(proj.float(), -100.0, 100.0)

        inv_loss    = (proj_safe.mean(0, keepdim=True) - proj_safe).square().mean()
        sigreg_loss = self.sigreg(proj_safe.flatten(0, 1).contiguous())

        loss = (1 - self.hparams.lamb) * inv_loss + self.hparams.lamb * sigreg_loss
        return loss, inv_loss, sigreg_loss

    def _shared_ssl_step(self, batch):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        B = x.shape[0]

        # Generate independent global views per image in the batch
        global_views = []
        for _ in range(2):
            global_views.append(torch.stack([self.global_aug(img) for img in x]))
        # Run forward pass on batched global views
        global_proj = torch.stack([
            self(g_view)[1] for g_view in global_views
        ])  # (2, B, proj_dim)

        # Generate independent local views per image in the batch
        local_views = []
        for _ in range(6):
            local_views.append(torch.stack([self.local_aug(img) for img in x]))
        # Run forward pass on batched local views
        local_proj = torch.stack([
            self(l_view)[1] for l_view in local_views
        ])  # (6, B, proj_dim)

        proj = torch.cat([global_proj, local_proj], dim=0)  # (8, B, proj_dim)
        return self._lejepa_loss(proj)

    def training_step(self, batch, batch_idx):
        loss, inv, sigreg = self._shared_ssl_step(batch)
        self.log_dict(
            {"ssl/loss": loss, "ssl/inv": inv, "ssl/sigreg": sigreg},
            prog_bar=True, on_step=True, on_epoch=True, sync_dist=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, inv, sigreg = self._shared_ssl_step(batch)
        if not torch.isfinite(loss):
            return None
        self.log_dict(
            {"val/loss": loss, "val/inv": inv, "val/sigreg": sigreg},
            prog_bar=True, on_step=False, on_epoch=True, sync_dist=True
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=self.hparams.warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs - self.hparams.warmup_epochs,
            eta_min=self.hparams.lr / 1000
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[self.hparams.warmup_epochs]
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


# ==========================================
# Supervised Fine-Tuning
# ==========================================
class METERFinetune(L.LightningModule):
    """
    Supervised fine-tuning of METER for depth estimation.

    Exact optimizer and scheduler from the paper (Section IV-A):
    - AdamW: beta1=0.9, beta2=0.999, wd=0.01, lr=0.001
    - StepLR: step_size=20, gamma=0.1
    - 60 epochs, batch_size=128 (from notebook)

    NYU: depth in centimeters (0-1000), metrics in meters (/ 100)
    KITTI: depth in meters (0-80), valid_mask for sparse GT
    """
    def __init__(
        self,
        arch_type="xxs",
        lr=1e-3,
        weight_decay=0.01,
        pretrained_ckpt=None,
        freeze_encoder=False,
        dataset_name="kitti"
    ):
        super().__init__()
        self.start_time      = None
        self.total_train_time = 0
        self.arch_type       = arch_type
        self.save_hyperparameters()

        print(f"\n[INFO] Initializing METERFinetune — dataset: {dataset_name}")
        self.model       = METERModel(arch_type=arch_type)
        self.loss_fn = balanced_loss_function(self.device, dataset_name=dataset_name)

        # Load SSL encoder weights if provided
        if pretrained_ckpt:
            self._load_ssl_encoder(pretrained_ckpt)

        if freeze_encoder:
            for p in self.model.encoder.parameters():
                p.requires_grad = False
            self.model.encoder.eval()
            print("[INFO] Encoder frozen")

        print("[INFO] METERFinetune ready")

    def _load_ssl_encoder(self, ckpt_path):
        """Load only the encoder weights from the LeJEPA SSL checkpoint."""
        if not os.path.exists(ckpt_path):
            print(f"[WARNING] Checkpoint not found: {ckpt_path}")
            return
        try:
            from src.utils import load_meter_pretrained_weights
            load_meter_pretrained_weights(self.model, ckpt_path, device=self.device)
            print(f"[INFO] SSL encoder loaded successfully from {ckpt_path}")
        except Exception as e:
            print(f"[ERROR] SSL loading failed: {e}")

    def on_fit_start(self):
        # Move the loss to the correct device (fix device mismatch with Lightning)
        self.loss_fn = self.loss_fn.to(self.device)

    def forward(self, x):
        out = self.model(x)
        out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=True)
        is_nyu = "nyu" in self.hparams.dataset_name.lower()
        max_depth = 1000.0 if is_nyu else 80.0

        if is_nyu:
            # 1.2x margin: scale = max_depth * 1.2 so the GT maximum (1000 cm)
            # maps to sigmoid output ≈ 0.83, well before the saturated region.
            # No clamp at top: that would zero the gradient on saturated pixels.
            # Predictions can briefly reach up to 1200 cm during training;
            # the L1 loss naturally pulls them back to the GT distribution.
            out = (max_depth * 1.2) * torch.sigmoid(out)
        else:
            # KITTI: keep original sigmoid (works well)
            out = max_depth * torch.sigmoid(out)
        return out
    
    def _compute_metrics(self, pred, y, dataset_name):
        """
        Computes RMSE, AbsRel, Delta1 according to the official rules of the paper.
        NYU: crop [16:H-16, 24:W-24], depth in meters (pred and GT divided by 100)
        KITTI: Garg/Eigen crop, depth in meters (already in meters)
        """
        _, _, H, W = y.shape
        crop_mask  = torch.zeros_like(y > 0)
        valid_depth = y > 0.001

        if "nyu" in dataset_name:
            # Convert cm -> m for metrics
            y    = y    / 100.0
            pred = pred / 100.0
            y    = torch.clamp(y,    min=1e-3, max=10.0)
            pred = torch.clamp(pred, min=1e-3, max=10.0)
            crop_mask[:, :, 16:H-16, 24:W-24] = True
        else:
            y    = torch.clamp(y,    min=1e-3, max=80.0)
            pred = torch.clamp(pred, min=1e-3, max=80.0)
            top, bottom = int(0.40810811 * H), int(0.99189189 * H)
            left, right = int(0.03594771 * W), int(0.96405229 * W)
            crop_mask[:, :, top:bottom, left:right] = True

        valid_mask = valid_depth & crop_mask
        if valid_mask.sum() == 0:
            return None, None, None

        p = pred[valid_mask].float()
        t = y[valid_mask].float()

        rmse    = torch.sqrt(torch.mean((p - t) ** 2))
        abs_rel = torch.mean(torch.abs(p - t) / t)
        delta1  = (torch.max(p / t, t / p) < 1.25).float().mean()
        return rmse, abs_rel, delta1

    def _shared_step(self, batch, stage):
        rgb, depth_gt = batch
        pred = self(rgb)
        l_depth, l_ssim, l_norm, l_grad = self.loss_fn(pred, depth_gt)
        loss = l_depth + l_ssim + l_norm + l_grad

        # Nan guard: if a component is nan, skip the batch
        # (can happen in early iterations with random weights)
        if not torch.isfinite(loss):
            return loss, pred

        self.log_dict(
            {
                f"{stage}/loss":   loss,
                f"{stage}/l1":     l_depth,
                f"{stage}/ssim":   l_ssim,
                f"{stage}/normal": l_norm,
                f"{stage}/grad":   l_grad,
            },
            prog_bar=(stage == "train"),
            on_step=True, on_epoch=True, sync_dist=True
        )
        return loss, pred

    def training_step(self, batch, batch_idx):
        loss, _ = self._shared_step(batch, "train")
        if not torch.isfinite(loss):
            return None
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred = self._shared_step(batch, "val")
        if not torch.isfinite(loss):
            # Log sentinel values to avoid blocking ModelCheckpoint
            self.log("val/rmse",    torch.tensor(999.0), prog_bar=True, on_epoch=True, sync_dist=True)
            self.log("val/abs_rel", torch.tensor(999.0), prog_bar=True, on_epoch=True, sync_dist=True)
            self.log("val/delta1",  torch.tensor(0.0),   prog_bar=True, on_epoch=True, sync_dist=True)
            self.log("val_rmse",    torch.tensor(999.0), on_epoch=True, sync_dist=True)
            self.log("val_abs_rel", torch.tensor(999.0), on_epoch=True, sync_dist=True)
            self.log("val_delta1",  torch.tensor(0.0),   on_epoch=True, sync_dist=True)
            return None
        _, y = batch
        dataset_name = str(self.hparams.dataset_name).lower().strip()
        rmse, abs_rel, delta1 = self._compute_metrics(pred, y, dataset_name)
        if rmse is not None:
            self.log("val/rmse",    rmse,    prog_bar=True, on_epoch=True, sync_dist=True)
            self.log("val/abs_rel", abs_rel, prog_bar=True, on_epoch=True, sync_dist=True)
            self.log("val/delta1",  delta1,  prog_bar=True, on_epoch=True, sync_dist=True)
            self.log("val_rmse",    rmse,    on_epoch=True, sync_dist=True)
            self.log("val_abs_rel", abs_rel, on_epoch=True, sync_dist=True)
            self.log("val_delta1",  delta1,  on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        _, pred = self._shared_step(batch, "test")
        _, y    = batch
        dataset_name = str(self.hparams.dataset_name).lower().strip()
        suffix = "nyu" if "nyu" in dataset_name else "kitti"
        rmse, abs_rel, delta1 = self._compute_metrics(pred, y, dataset_name)
        if rmse is not None:
            l1 = torch.mean(torch.abs(
                pred[y > 0.001].float() - y[y > 0.001].float()
            ))
            self.log(f"test/rmse_{suffix}",    rmse,    prog_bar=True, sync_dist=True)
            self.log(f"test/abs_rel_{suffix}",  abs_rel, prog_bar=True, sync_dist=True)
            self.log(f"test/delta1_{suffix}",   delta1,  prog_bar=True, sync_dist=True)
            self.log(f"test/l1_{suffix}",       l1,      prog_bar=True, sync_dist=True)
            return l1
        return torch.tensor(0.0, device=pred.device)

    def configure_optimizers(self):
        """
        AdamW + StepLR (paper Sec. IV-A).
        If a SSL checkpoint was loaded, encoder gets 10x smaller LR to preserve
        the pretrained geometric features (differential learning rate).
        """
        base_lr = self.hparams.lr
        encoder_lr_mult = 0.1 if self.hparams.pretrained_ckpt else 1.0

        enc_decay, enc_nodecay = [], []
        dec_decay, dec_nodecay = [], []

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            is_encoder = name.startswith("encoder.")
            no_decay = p.ndim <= 1 or name.endswith(".bias")
            if is_encoder and no_decay:
                enc_nodecay.append(p)
            elif is_encoder:
                enc_decay.append(p)
            elif no_decay:
                dec_nodecay.append(p)
            else:
                dec_decay.append(p)

        param_groups = [
            {"params": enc_decay,   "lr": base_lr * encoder_lr_mult, "weight_decay": self.hparams.weight_decay},
            {"params": enc_nodecay, "lr": base_lr * encoder_lr_mult, "weight_decay": 0.0},
            {"params": dec_decay,   "lr": base_lr,                   "weight_decay": self.hparams.weight_decay},
            {"params": dec_nodecay, "lr": base_lr,                   "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))
        print(f"[INFO] Optimizer — encoder LR mult: {encoder_lr_mult} (pretrained={bool(self.hparams.pretrained_ckpt)})")

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=20,
            gamma=0.1,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "strict": True,
            },
        }

    def on_train_start(self):
        self.start_time = time.time()

    def on_train_end(self):
        if self.start_time:
            self.total_train_time = time.time() - self.start_time
            print(f"[INFO] Training completed in {self.total_train_time:.2f}s")


# ==========================================
# Callbacks
# ==========================================
def get_callbacks(save_dir, monitor="val/rmse", dataset_name="nyu", arch_name="xxs"):
    dataset_clean = str(dataset_name).lower().strip()
    is_nyu = "nyu" in dataset_clean

    safe_monitor = monitor.replace('/', '_')
    checkpoint = ModelCheckpoint(
        dirpath=save_dir,
        filename=f"best-{{epoch:02d}}-{{{safe_monitor}:.4f}}-{dataset_name}-{arch_name}",
        monitor=monitor,
        mode="min",
        save_top_k=1,
        save_last=True
    )

    callbacks = [checkpoint, LearningRateMonitor(logging_interval="epoch")]

    if is_nyu:
        # NYU: train for full 60 epochs (paper Sec. IV-A) without early stop.
        # The StepLR drops at epoch 20 and 40 need both to be effective.
        print("[INFO] NYU: EarlyStopping disabled, training for full max_epochs.")
    else:
        # KITTI: keep early stopping with patience 15 (unchanged).
        callbacks.append(EarlyStopping(
            monitor=monitor, patience=15, mode="min", strict=False
        ))

    return callbacks


# ==========================================
# Test block
# ==========================================
if __name__ == "__main__":
    print("\n========== TRAIN TEST ==========")
    model = LeJEPAMeter(arch_type="xxs")
    dummy = torch.randn(2, 3, 256, 256)
    emb, proj = model(dummy)
    print(f"[INFO] Embedding shape: {emb.shape}")
    print(f"[INFO] Projection shape: {proj.shape}")
    print("\n========== DONE ==========")