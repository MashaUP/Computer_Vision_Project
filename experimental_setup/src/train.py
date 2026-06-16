import os
import time
import math
import torch
import numpy as np
import torch.nn as nn
import lightning as L
from torchvision.ops import MLP
import torch.nn.functional as F
from lejepa.multivariate import SlicingUnivariateTest
from lejepa.univariate import EppsPulley
from src.dataset import get_ssl_spatial_transform, get_ssl_color_transform
from src.network import METERModel
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import random
import itertools
import torchvision.transforms.functional as TF
import gc

# ==========================================
# Encoder output dimensions
# ==========================================
ENC_DIMS = {
    "xxs": 160,
    "xs": 192,
    "s": 320
}

# ==========================================
# Depth loss WORKING DO NOT DELETE
# ==========================================
# Return a placeholder composite loss with active L1 depth term only
#def balanced_loss_function(device):
#    l1_loss = nn.L1Loss()
#    def loss_fn(pred, target):
#       pred = pred.float()
#        target = target.float()
#        l_depth = l1_loss(pred, target)
#        zero = torch.tensor(
#            0.0,
#           device=device
#        )
#        return (
#            l_depth,
#            zero,
#            zero,
#            zero
#        )
#    return loss_fn
##############################################

# ==========================================
# Depth loss
# ==========================================
def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, val_range, window_size=11, window=None, size_average=True, full=False, mask=None):
    L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if mask is not None:
        mask_cropped = F.interpolate(mask.float(), size=ssim_map.shape[-2:], mode='nearest') > 0
        if mask_cropped.sum() > 0:
            ret = ssim_map[mask_cropped].mean()
        else:
            ret = ssim_map.mean()
    else:
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
        edge_k = np.stack((edge_kx, edge_ky))

        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.edge_conv(x)
        out = out.contiguous().view(-1, 2, x.size(2), x.size(3))

        return out


class balanced_loss_function(nn.Module):

    def __init__(self, device):
        super(balanced_loss_function, self).__init__()
        # Added eps=1e-6 (changed from eps=0) to prevent division-by-zero NaN in FP16
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.get_gradient = Sobel().to(device)
        self.device = device
        self.lambda_1 = 0.5
        self.lambda_2 = 1
        self.lambda_3 = 1
        self._ssim_window = None      # cached Gaussian kernel
        self._ssim_window_key = None  # (size, device) invalidation key

    def forward(self, output, depth):
        if output.dim() == 3: output = output.unsqueeze(1)
        if depth.dim() == 3: depth = depth.unsqueeze(1)
        
        # Cast to float32 to prevent NaN/Inf overflows during mixed precision training (AMP)
        output = output.float()
        depth = depth.float()

        # Handle hybrid 2-channel targets (dense in channel 0, sparse in channel 1)
        if depth.shape[1] == 2:
            depth_dense = depth[:, 0:1, :, :]
            depth_sparse = depth[:, 1:2, :, :]
        else:
            depth_dense = depth
            depth_sparse = depth

        with torch.no_grad():
            # B. depth_dense is GT — no grad needed through Sobel on depth
            depth_grad = self.get_gradient(depth_dense)
            # C. ones via expand: zero-copy broadcast, no allocation
            ones = torch.ones(1, 1, 1, 1, device=depth.device, dtype=torch.float32
                              ).expand(depth.size(0), 1, depth.size(2), depth.size(3))

        output_grad = self.get_gradient(output)

        depth_grad_dx  = depth_grad[:, 0, :, :].contiguous().view_as(depth_dense)
        depth_grad_dy  = depth_grad[:, 1, :, :].contiguous().view_as(depth_dense)
        output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(depth_dense)
        output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(depth_dense)

        depth_normal  = torch.cat((-depth_grad_dx, -depth_grad_dy, ones), 1)
        output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)

        is_kitti  = output.shape[-1] > 300
        max_range = 80.0 if is_kitti else 10.0

        # Define masks
        # 1. Sparse mask (for L1 absolute depth loss)
        valid_mask_sparse = (depth_sparse > 1e-3) & (depth_sparse < max_range)
        # 2. Dense mask (for structural geometric losses: SSIM, Normal, Grad)
        valid_mask_dense = (depth_dense > 1e-3) & (depth_dense < max_range)

        if is_kitti:
            # Crop boundary mask
            b, c, h, w = depth.shape
            top    = int(0.4081 * h)
            bottom = int(0.9918 * h)
            left   = int(0.0359 * w)
            right  = int(0.9640 * w)
            
            # Apply cropping to both masks
            valid_mask_sparse[:, :, :top]    = False
            valid_mask_sparse[:, :, bottom:] = False
            valid_mask_sparse[:, :, :, :left]  = False
            valid_mask_sparse[:, :, :, right:] = False

            valid_mask_dense[:, :, :top]    = False
            valid_mask_dense[:, :, bottom:] = False
            valid_mask_dense[:, :, :, :left]  = False
            valid_mask_dense[:, :, :, right:] = False
        else:
            # Crop boundary mask for NYU Depth V2
            b, c, h, w = depth.shape
            top    = 16
            bottom = h - 16
            left   = 24
            right  = w - 24
            
            # Apply cropping to both masks
            valid_mask_sparse[:, :, :top]    = False
            valid_mask_sparse[:, :, bottom:] = False
            valid_mask_sparse[:, :, :, :left]  = False
            valid_mask_sparse[:, :, :, right:] = False

            valid_mask_dense[:, :, :top]    = False
            valid_mask_dense[:, :, bottom:] = False
            valid_mask_dense[:, :, :, :left]  = False
            valid_mask_dense[:, :, :, right:] = False

        # Float masks — no gather, no GPU-CPU sync
        w_sparse = valid_mask_sparse.float()
        n_sparse = w_sparse.sum().clamp(min=1.0)
        
        w_dense    = valid_mask_dense.float()
        n_dense    = w_dense.sum().clamp(min=1.0)
        w_dense_sq = valid_mask_dense.squeeze(1).float()
        n_dense_sq = w_dense_sq.sum().clamp(min=1.0)

        # Erode the dense mask for gradients and normals (ignore transitions/cliffs at mask boundaries)
        kernel = torch.ones(1, 1, 3, 3, device=depth.device)
        w_grad_dense = (F.conv2d(w_dense, kernel, padding=1) == 9).float()
        n_grad_dense = w_grad_dense.sum().clamp(min=1.0)
        w_grad_dense_sq = w_grad_dense.squeeze(1)
        n_grad_dense_sq = w_grad_dense_sq.sum().clamp(min=1.0)

        # Erode the dense mask for SSIM (ignore ssim windows overlapping with invalid 0.0 regions)
        # Using an 11x11 kernel to match the ssim window size.
        ssim_kernel = torch.ones(1, 1, 11, 11, device=depth.device)
        w_ssim_dense = (F.conv2d(w_dense, ssim_kernel, padding=5) == 121)

        # Compute losses
        loss_depth  = (torch.abs(output - depth_sparse) * w_sparse).sum() / n_sparse
        loss_dx     = (torch.abs(output_grad_dx - depth_grad_dx) * w_grad_dense).sum() / n_grad_dense
        loss_dy     = (torch.abs(output_grad_dy - depth_grad_dy) * w_grad_dense).sum() / n_grad_dense
        cos_sim     = self.cos(output_normal, depth_normal)

        # Scale up scale-invariant structural losses for KITTI to compensate for 8x larger depth range
        # (KITTI range is 0-80m, meaning L1 depth error is 8x larger than NYU range of 0-10m).
        # Scaling up the SSIM and Normals losses by 8x restores the successful loss balance
        # of the NYU setup, forcing the model to produce sharp boundaries instead of blurry blobs.
        l2_weight = self.lambda_2 * 8.0 if is_kitti else self.lambda_2
        l3_weight = self.lambda_3 * 8.0 if is_kitti else self.lambda_3

        loss_normal = l2_weight * (torch.abs(1 - cos_sim) * w_grad_dense_sq).sum() / n_grad_dense_sq

        # G. Cache SSIM Gaussian window: recompute only when size or device changes
        win_size = min(11, output.shape[-2], output.shape[-1])
        win_key  = (win_size, output.device)
        if self._ssim_window_key != win_key:
            self._ssim_window     = create_window(win_size, channel=1).to(output.device)
            self._ssim_window_key = win_key
        loss_ssim = (1 - ssim(output, depth_dense, val_range=max_range, mask=w_ssim_dense,
                               window=self._ssim_window, window_size=win_size)) * l3_weight
        loss_grad = (loss_dx + loss_dy) * self.lambda_1

        return loss_depth, loss_ssim, loss_normal, loss_grad

# ==========================================
# LeJEPA SSL
# ==========================================
class LeJEPAMeter(L.LightningModule):
    """
    Self-supervised LeJEPA stage with Dense Token-Level loss.

    Motivation (V-JEPA 2.1, §2.2):
        Original LeJEPA mean-pools the encoder feature map before projecting,
        so the encoder has NO incentive to preserve local spatial structure —
        it can simply aggregate global information to minimise the loss.
        By supervising every spatial token individually, we force each patch
        position to encode its own local content (dense features), which is
        critical for downstream dense tasks such as depth estimation.

    Architecture:
        encoder(x) → [B, C, H, W]
            ↓ adaptive_avg_pool2d → [B, C, TOKEN_GRID, TOKEN_GRID]
            ↓ flatten spatial → [B, N, C]   N = TOKEN_GRID²
            ↓ shared MLP projector (per token) → [B, N, proj_dim]

    Loss (LeJEPA §5 + V-JEPA 2.1 §2.3.1):
        L = (1-λ) · L_token_inv  +  λ · L_sigreg

        L_token_inv : token-level invariance across views
                      for each spatial position n, align predictions
                      from all V views → mean over V, B, N, D
        L_sigreg    : SIGReg on the flat distribution of ALL tokens
                      (V·B·N samples) to enforce isotropic Gaussian
    """

    # Fixed spatial grid: all crops are pooled to TOKEN_GRID×TOKEN_GRID
    # before projecting, so global (8×8) and local (4×4) crops share the
    # same N and can be compared token-by-token across views.
    TOKEN_GRID = 4   # → N = 16 tokens per image

    def __init__(
        self,
        arch_type="xs",
        proj_dim=64,
        num_slices=256,
        lamb=0.02,
        alpha=0.5,
        lr=5e-4,
        weight_decay=5e-2,
        warmup_epochs=10
    ):
        super().__init__()
        self.arch_type = arch_type
        self.save_hyperparameters()
        print("\n[INFO] Initializing Dense LeJEPA SSL model with Same-Crop Multi-View")
        full_model = METERModel(
            arch_type=arch_type
        )
        self.encoder = full_model.encoder
        self.encoder.requires_grad_(True)
        enc_dim = ENC_DIMS[arch_type]
        print(f"[INFO] Encoder dim: {enc_dim}")
        print(f"[INFO] Dense token grid: {self.TOKEN_GRID}×{self.TOKEN_GRID} = {self.TOKEN_GRID**2} tokens/image")
        self.global_spatial = get_ssl_spatial_transform(global_view=True)
        self.global_color   = get_ssl_color_transform(global_view=True)
        self.local_spatial  = get_ssl_spatial_transform(global_view=False)
        self.local_color    = get_ssl_color_transform(global_view=False)
        # The projector is a shared MLP applied independently to each token.
        # torchvision MLP operates on the last dimension, so it naturally
        # handles [B, N, C] → [B, N, proj_dim] without any changes.
        self.projector = MLP(
            enc_dim,
            [256, 512, proj_dim],
            norm_layer=nn.LayerNorm
        )
        self.sigreg = SlicingUnivariateTest(
            univariate_test=EppsPulley(
                t_max=5.0,
                n_points=21
            ),
            num_slices=num_slices,
            reduction="mean"
        )
        self.cumulative_flops = 0.0
        print("[INFO] Dense LeJEPA model ready")

    def on_train_batch_start(self, batch, batch_idx):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif torch.backends.mps.is_available():
            torch.mps.synchronize()
        self.step_start_time = time.time()

    def forward(self, x):
        """
        Returns
        -------
        emb  : [B, C]        — global pooled embedding (kept for compatibility)
        proj : [B, N, D]     — dense per-token projection
                               N = TOKEN_GRID² spatial positions
                               D = proj_dim
        """
        features, _ = self.encoder(x)           # [B, C, H, W]

        # ── Dense path ────────────────────────────────────────────────────
        # Adaptive pool normalises spatial resolution across crop scales:
        #   global crops (256px) → encoder H=8,W=8 → pool to 4×4
        #   local  crops (128px) → encoder H=4,W=4 → pool is a no-op
        # Result: every crop yields exactly TOKEN_GRID² comparable tokens.
        pooled = F.adaptive_avg_pool2d(
            features, self.TOKEN_GRID
        )                                       # [B, C, G, G]  G=TOKEN_GRID
        tokens = pooled.flatten(2).transpose(
            1, 2
        )                                       # [B, N, C]  N=G²

        # Shared MLP projector: applied token-by-token (last dim = C)
        proj = self.projector(tokens)           # [B, N, proj_dim]

        # ── Global path (for logging / fine-tuning compat) ────────────────
        emb = features.mean(dim=[-2, -1])       # [B, C]

        return emb, proj

    def lejepa_loss(self, proj):
        """
        Dense Same-Crop Multi-View LeJEPA loss.

        Parameters
        ----------
        proj : [V, B, N, D]
            V  = number of views  (2 global + 6 local = 8)
            B  = batch size
            N  = number of spatial tokens per image  (TOKEN_GRID²)
            D  = projection dimension

        Returns
        -------
        loss, dense_inv, global_inv, sigreg  — all scalar tensors
        """
        V, B, N, D = proj.shape
        alpha = self.hparams.alpha

        # ── 1. Same-Crop Dense Invariance ────────────────────────────────────
        # We compare token-level predictions ONLY between color-distorted versions
        # of the exact same crop, avoiding the spatial collapse template shortcut.
        inv_g0 = (proj[0] - proj[1]).square().mean()  # global crop 0
        inv_l0 = (proj[2] - proj[3]).square().mean()  # local crop 0
        inv_l1 = (proj[4] - proj[5]).square().mean()  # local crop 1
        inv_l2 = (proj[6] - proj[7]).square().mean()  # local crop 2
        dense_inv_loss = (inv_g0 + inv_l0 + inv_l1 + inv_l2) / 4.0

        # ── 2. Global Cross-Crop Invariance ──────────────────────────────────
        # To align high-level semantic context across different crops of the same image,
        # we average/pool tokens spatially to 1x1 first, then align them across views.
        proj_global = proj.mean(dim=2)  # [8, B, D]
        mean_global = proj_global.mean(dim=0, keepdim=True)  # [1, B, D]
        global_inv_loss = (mean_global - proj_global).square().mean()

        # Combine dense and global invariance
        inv_loss = (1 - alpha) * dense_inv_loss + alpha * global_inv_loss

        # ── 3. Flat Spatial-Batch SIGReg ─────────────────────────────────────
        # To prevent spatial collapse and local content invariance within each image,
        # we flatten the batch and spatial dimensions. By treating B * N as the
        # sample dimension, each of the V views is independently tested for
        # normality without cross-view duplication issues.
        proj_flat = proj.reshape(V, B * N, D).contiguous()      # [8, B * N, D]
        sigreg_loss   = self.sigreg(proj_flat)                 # scalar

        loss = (
            (1 - self.hparams.lamb) * inv_loss
            + self.hparams.lamb * sigreg_loss
        )
        return loss, dense_inv_loss, global_inv_loss, sigreg_loss
    
    def training_step(self, batch, batch_idx):
        # Batch is a base tensor [B, C, H, W] already on device (MPS or CUDA)
        x = batch
        batch_size = x.shape[0]

        # 1 global crop -> 2 color views
        global_crop = self.global_spatial(x)  # [B, C, 256, 256]
        g_view0 = self.global_color(global_crop)
        g_view1 = self.global_color(global_crop)

        # 3 local crops -> 2 color views each
        local_crop0 = self.local_spatial(x)  # [B, C, 128, 128]
        l0_view0 = self.local_color(local_crop0)
        l0_view1 = self.local_color(local_crop0)

        local_crop1 = self.local_spatial(x)  # [B, C, 128, 128]
        l1_view0 = self.local_color(local_crop1)
        l1_view1 = self.local_color(local_crop1)

        local_crop2 = self.local_spatial(x)  # [B, C, 128, 128]
        l2_view0 = self.local_color(local_crop2)
        l2_view1 = self.local_color(local_crop2)

        # Concatenate for batch processing:
        #   global_input: [2*B, C, 256, 256]
        #   local_input:  [6*B, C, 128, 128]
        global_input = torch.cat([g_view0, g_view1], dim=0)
        local_input  = torch.cat([l0_view0, l0_view1, l1_view0, l1_view1, l2_view0, l2_view1], dim=0)

        # Forward pass → proj is now [·B, N, D]  (dense tokens)
        _, global_proj = self(global_input)   # [2·B, N, D]
        _, local_proj  = self(local_input)    # [6·B, N, D]

        # Reshape to [V, B, N, D] — split the batch-of-views axis
        N, D = global_proj.shape[1], global_proj.shape[2]
        global_proj = global_proj.reshape(2, batch_size, N, D)  # [2, B, N, D]
        local_proj  = local_proj.reshape(6, batch_size, N, D)   # [6, B, N, D]

        # Stack all 8 views: [8, B, N, D]
        proj = torch.cat([global_proj, local_proj], dim=0)
        loss, dense_inv, global_inv, sigreg = self.lejepa_loss(proj)

        # Profile step FLOPs dynamically on first run using native measure_flops
        if not hasattr(self, "_cached_step_flops"):
            from lightning.fabric.utilities.throughput import measure_flops
            self.print(f"[INFO] Profiling pre-training step FLOPs dynamically on {self.device}...")
            was_training = self.training
            self.eval()
            with torch.no_grad():
                dummy_batch = torch.randn(batch_size, 3, 256, 256, device=self.device)
                
                def profile_step():
                    global_crop = self.global_spatial(dummy_batch)
                    g_view0 = self.global_color(global_crop)
                    g_view1 = self.global_color(global_crop)
                    
                    local_crop0 = self.local_spatial(dummy_batch)
                    l0_view0 = self.local_color(local_crop0)
                    l0_view1 = self.local_color(local_crop0)
                    
                    local_crop1 = self.local_spatial(dummy_batch)
                    l1_view0 = self.local_color(local_crop1)
                    l1_view1 = self.local_color(local_crop1)
                    
                    local_crop2 = self.local_spatial(dummy_batch)
                    l2_view0 = self.local_color(local_crop2)
                    l2_view1 = self.local_color(local_crop2)
                    
                    global_input = torch.cat([g_view0, g_view1], dim=0)
                    local_input  = torch.cat([l0_view0, l0_view1, l1_view0, l1_view1, l2_view0, l2_view1], dim=0)
                    
                    _, gp = self(global_input)
                    _, lp = self(local_input)
                    
                    N, D = gp.shape[1], gp.shape[2]
                    gp = gp.reshape(2, batch_size, N, D)
                    lp = lp.reshape(6, batch_size, N, D)
                    proj = torch.cat([gp, lp], dim=0)
                    loss, _, _, _ = self.lejepa_loss(proj)
                    return loss
                
                try:
                    total_flops = measure_flops(self, profile_step, loss_fn=lambda loss: loss)
                except Exception as e:
                    fwd_flops = measure_flops(self, profile_step)
                    total_flops = int(fwd_flops * 3.8)
                
                self._cached_step_flops = total_flops
                self.flops_per_batch = total_flops
                self.print(f"[INFO] Profiled pre-training step FLOPs: {self._cached_step_flops:,}")
            if was_training:
                self.train()

        step_flops = self._cached_step_flops
        self.cumulative_flops += step_flops

        # Synchronize devices to get exact wall time for this step
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif torch.backends.mps.is_available():
            torch.mps.synchronize()
            
        step_duration = time.time() - self.step_start_time if getattr(self, "step_start_time", None) else 0.0
        
        # Compute throughput (samples/sec)
        throughput = batch_size / max(step_duration, 1e-6)
        
        # Memory
        mem_mb = 0.0
        if torch.cuda.is_available():
            mem_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        elif torch.backends.mps.is_available():
            try:
                mem_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
            except:
                pass

        self.log_dict(
            {
                "ssl/loss": loss,
                "ssl/dense_inv": dense_inv,
                "ssl/global_inv": global_inv,
                "ssl/sigreg": sigreg,
                "ssl/step_flops": float(step_flops),
                "ssl/cumulative_flops": float(self.cumulative_flops),
                "ssl/throughput_samples_per_sec": throughput,
                "ssl/step_duration_sec": step_duration,
                "ssl/memory_allocated_mb": mem_mb,
            },
            prog_bar=True, on_step=True, on_epoch=True,
            sync_dist=(self.trainer.world_size > 1),
        )
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Batch is a base tensor [B, C, H, W] already on device
        x = batch
        batch_size = x.shape[0]

        # 1 global crop -> 2 color views
        global_crop = self.global_spatial(x)  # [B, C, 256, 256]
        g_view0 = self.global_color(global_crop)
        g_view1 = self.global_color(global_crop)

        # 3 local crops -> 2 color views each
        local_crop0 = self.local_spatial(x)  # [B, C, 128, 128]
        l0_view0 = self.local_color(local_crop0)
        l0_view1 = self.local_color(local_crop0)

        local_crop1 = self.local_spatial(x)  # [B, C, 128, 128]
        l1_view0 = self.local_color(local_crop1)
        l1_view1 = self.local_color(local_crop1)

        local_crop2 = self.local_spatial(x)  # [B, C, 128, 128]
        l2_view0 = self.local_color(local_crop2)
        l2_view1 = self.local_color(local_crop2)

        # Concatenate for batch processing
        global_input = torch.cat([g_view0, g_view1], dim=0)
        local_input  = torch.cat([l0_view0, l0_view1, l1_view0, l1_view1, l2_view0, l2_view1], dim=0)

        # Forward pass → proj is now [·B, N, D]  (dense tokens)
        _, global_proj = self(global_input)   # [2·B, N, D]
        _, local_proj  = self(local_input)    # [6·B, N, D]

        # Reshape to [V, B, N, D]
        N, D = global_proj.shape[1], global_proj.shape[2]
        global_proj = global_proj.reshape(2, batch_size, N, D)
        local_proj  = local_proj.reshape(6, batch_size, N, D)

        # Compute SSL loss (invariance + SigReg)
        proj = torch.cat([global_proj, local_proj], dim=0)  # [8, B, N, D]
        loss, dense_inv, global_inv, sigreg = self.lejepa_loss(proj)

        # Log validation metrics
        self.log_dict(
            {
                "val/loss":       loss,
                "val/ssl_loss":   loss,
                "val/dense_inv":  dense_inv,
                "val/global_inv": global_inv,
                "val/sigreg":     sigreg,
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=(self.trainer.world_size > 1),
        )
        return loss

    def on_train_epoch_end(self):
        _clean_cache()

    def on_validation_epoch_end(self):
        _clean_cache()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            total_iters=self.hparams.warmup_epochs
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=(
                self.trainer.max_epochs
                - self.hparams.warmup_epochs
            ),
            eta_min=self.hparams.lr / 1000
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                warmup_scheduler,
                cosine_scheduler
            ],
            milestones=[
                self.hparams.warmup_epochs
            ]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }

# ==========================================
# Fine tuning
# ==========================================
def _clean_cache():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.synchronize()   # flush pending Metal command buffers first
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


# Precompute all 27 channel-swap policies once at import time
# (matches: indices = list(product([0,1,2], repeat=3)) in the original METER code)
_CHANNEL_SWAP_POLICIES = list(itertools.product([0, 1, 2], repeat=3))


def apply_augmentations(rgb, depth):
    """
    METER-faithful data augmentations — matches augmentation2D() in
    https://github.com/lorenzopapa5/METER/blob/main/augmentation.py

    Notes on faithfulness:
      - "flip" step in original (img[..., ::1, :, :]) is a no-op (stride=1 bug); omitted.
      - "mirror" step (img[..., ::-1, :]) reverses the width axis → horizontal flip.
      - "random_crop" crops (H,W) images to (192,256). Our pipeline already outputs
        target resolution, so range=0 → no-op; omitted for clarity.
      - Depth shift: original uses randint(-10, 10) in cm → *0.01 for metres.
    All random draws use pure-Python random to avoid GPU-CPU synchronisation.
    """
    # 1. Random horizontal flip ("mirror" in METER)
    if random.random() < 0.5:
        rgb   = torch.flip(rgb,   dims=[-1])
        depth = torch.flip(depth, dims=[-1])

    # 2. Random vertical flip: DEACTIVATED (0% probability) to preserve critical vertical perspective cues.

    # 3. Channel swap — 27 policies with repetition: product([0,1,2], repeat=3)
    if random.random() < 0.5:
        policy = _CHANNEL_SWAP_POLICIES[random.randint(0, 26)]
        idx    = torch.tensor(list(policy), device=rgb.device)
        rgb    = rgb[:, idx, :, :]

    # 4. Colour shifting strategy (Gamma + brightness + color jitter)
    if random.random() < 0.5:
        # gamma and brightness (clamp rgb to 1e-6 before power to prevent Inf/NaN in FP16 gradients)
        rgb = torch.clamp(rgb, 1e-6, 1.0) ** random.uniform(0.9, 1.1)
        rgb = rgb *  random.uniform(0.9, 1.1)
        # per-channel colour jitter
        rgb = rgb *  torch.empty(1, 3, 1, 1, device=rgb.device).uniform_(0.9, 1.1)
        rgb = torch.clamp(rgb, 0.0, 1.0)
        
    # 5. Depth shifting strategy: DEACTIVATED (0% probability) to preserve absolute metric scale and prevent regression-to-the-mean.

    return rgb, depth


class METERFinetune(L.LightningModule):
    def __init__(
        self,
        arch_type="xs",
        lr_enc=1e-5,
        lr_dec=1e-4,
        weight_decay=1e-2,
        pretrained_ckpt=None,
        freeze_encoder=True,
        use_baseline=False,
        dataset="nyu",
        warmup_epochs=5,
        use_step_lr=False
    ):
        super().__init__()
        self.start_time = None
        self.total_train_time = 0
        self.cumulative_flops = 0.0
        self.arch_type = arch_type
        self.save_hyperparameters()
        print(f"\n[INFO] Initializing fine-tuning model (Baseline={use_baseline}, Dataset={dataset})")
        self.model = METERModel(arch_type=arch_type, use_baseline=use_baseline)
        self.loss_fn = balanced_loss_function(self.device)
        if pretrained_ckpt:
            from src.utils import load_meter_pretrained_weights
            self.model = load_meter_pretrained_weights(self.model, pretrained_ckpt, self.device)
        print("[INFO] Fine-tuning model ready")
        if freeze_encoder:
            for p in self.model.encoder.parameters(): p.requires_grad = False
            self.model.encoder.eval()
            
    def load_pretrained_weights(self, ckpt_path, freeze_encoder=True):
        print(f"[INFO] Loading pretrained checkpoint: {ckpt_path}")
        if not os.path.exists(ckpt_path):
            print("[WARNING] Pretrained checkpoint not found")
            return
        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            clean_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace("model.encoder.", "").replace("encoder.", "")
                clean_state_dict[new_key] = value
            msg = self.model.encoder.load_state_dict(
                clean_state_dict,
                strict=False
            )
            print(
                f"[INFO] Missing keys: "
                f"{len(msg.missing_keys)}"
            )
            print(
                f"[INFO] Unexpected keys: "
                f"{len(msg.unexpected_keys)}"
            )
            if freeze_encoder:
                print("[INFO] Freezing encoder")
                for param in self.model.encoder.parameters():
                    param.requires_grad = False
                self.model.encoder.eval()
        except Exception as e:
            print(f"[ERROR] SSL loading failed: {e}")

    def forward(self, x):
        target_size = x.shape[-2:]
        
        # ImageNet normalization deactivated for all datasets (NYU and KITTI) to maintain
        # a consistent, unwarped [0, 1] RGB distribution and prevent feature mismatch.
        out = self.model(x)
        out = torch.nn.functional.relu(out)
        out = torch.nn.functional.interpolate(
            out,
            size=target_size,
            mode="bilinear",
            align_corners=True
        )
        if getattr(self.hparams, "use_baseline", False):
            is_kitti = (x.shape[-1] > 300)
            if is_kitti:
                out = out / 10.0
            else:
                out = out / 100.0
        return out
    
    def shared_step(self, batch, stage):
        rgb, depth_gt = batch
        pred = self(rgb)
        l_depth, l_ssim, l_norm, l_grad = self.loss_fn(
            pred,
            depth_gt
        )
        loss = (
            l_depth
            + l_ssim
            + l_norm
            + l_grad
        )
        _sync = self.trainer.world_size > 1
        self.log_dict(
            {
                f"{stage}/loss": loss,
                f"{stage}/l1": l_depth,
                f"{stage}/ssim": l_ssim,
                f"{stage}/normal": l_norm,
                f"{stage}/grad": l_grad
            },
            prog_bar=(stage == "train"),
            on_step=True,
            on_epoch=True,
            sync_dist=_sync,
        )
        return loss
    
    def on_train_batch_start(self, batch, batch_idx):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif torch.backends.mps.is_available():
            torch.mps.synchronize()
        self.step_start_time = time.time()

    def training_step(self, batch, batch_idx):
        rgb, depth = batch
        rgb, depth = apply_augmentations(rgb, depth)
        batch_size = rgb.shape[0]
        
        # Profile step FLOPs dynamically on first run using native measure_flops
        if not hasattr(self, "_cached_step_flops"):
            from lightning.fabric.utilities.throughput import measure_flops
            self.print(f"[INFO] Profiling fine-tuning step FLOPs dynamically on {self.device}...")
            was_training = self.training
            self.eval()
            with torch.no_grad():
                dummy_rgb = torch.randn(batch_size, 3, rgb.shape[-2], rgb.shape[-1], device=self.device)
                dummy_depth = torch.randn(batch_size, depth.shape[1], depth.shape[-2], depth.shape[-1], device=self.device)
                
                def profile_step():
                    pred = self(dummy_rgb)
                    l_depth, l_ssim, l_norm, l_grad = self.loss_fn(pred, dummy_depth)
                    loss = l_depth + l_ssim + l_norm + l_grad
                    return loss
                
                try:
                    total_flops = measure_flops(self, profile_step, loss_fn=lambda loss: loss)
                except Exception as e:
                    fwd_flops = measure_flops(self, profile_step)
                    total_flops = int(fwd_flops * 3.8)
                
                self._cached_step_flops = total_flops
                self.flops_per_batch = total_flops
                self.print(f"[INFO] Profiled fine-tuning step FLOPs: {self._cached_step_flops:,}")
            if was_training:
                self.train()

        loss = self.shared_step(
            (rgb, depth),
            "train"
        )
        
        # Synchronize devices to get exact wall time for this step
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif torch.backends.mps.is_available():
            torch.mps.synchronize()
            
        step_duration = time.time() - self.step_start_time if getattr(self, "step_start_time", None) else 0.0
        
        step_flops = self._cached_step_flops
        self.cumulative_flops += step_flops
        
        # Compute throughput (samples/sec)
        throughput = batch_size / max(step_duration, 1e-6)
        
        # Memory
        mem_mb = 0.0
        if torch.cuda.is_available():
            mem_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        elif torch.backends.mps.is_available():
            try:
                mem_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
            except:
                pass
                
        self.log_dict(
            {
                "train/step_flops": float(step_flops),
                "train/cumulative_flops": float(self.cumulative_flops),
                "train/throughput_samples_per_sec": throughput,
                "train/step_duration_sec": step_duration,
                "train/memory_allocated_mb": mem_mb,
            },
            prog_bar=True, on_step=True, on_epoch=True,
            sync_dist=(self.trainer.world_size > 1),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # Extract sparse target for unbiased physical validation metrics against LiDAR
        if y.shape[1] == 2:
            y_sparse = y[:, 1:2, :, :]
        else:
            y_sparse = y

        pred = self(x)
        if pred.dim() > y_sparse.dim():
            pred = pred.squeeze(1)
            
        is_kitti = (y_sparse.shape[-1] > 300)
        if is_kitti:
            b, c, h, w = y_sparse.shape
            top = int(0.4081 * h)
            bottom = int(0.9918 * h)
            left = int(0.0359 * w)
            right = int(0.9640 * w)
            y_crop = y_sparse[:, :, top:bottom, left:right]
            pred_crop = pred[:, :, top:bottom, left:right]
            valid_mask = (y_crop > 1e-3) & (y_crop < 80.0)
            if valid_mask.sum() == 0:
                dummy_loss = torch.tensor(80.0, device=pred.device)
                self.log("val/rmse", dummy_loss, prog_bar=True, on_epoch=True, sync_dist=(self.trainer.world_size > 1))
                self.log("val_rmse", dummy_loss, prog_bar=False, on_epoch=True, sync_dist=(self.trainer.world_size > 1))
                self.log("val/abs_rel", dummy_loss, prog_bar=True, on_epoch=True, sync_dist=(self.trainer.world_size > 1))
                self.log("val/delta1", torch.tensor(0.0, device=pred.device), prog_bar=True, on_epoch=True, sync_dist=(self.trainer.world_size > 1))
                self.log("val/l1", dummy_loss, prog_bar=False, on_epoch=True, sync_dist=(self.trainer.world_size > 1))
                return dummy_loss
            p = pred_crop[valid_mask].float()
            t = y_crop[valid_mask].float()
            p = torch.clamp(p, min=1e-3, max=80.0)
        else:
            b, c, h, w = y_sparse.shape
            top = 16
            bottom = h - 16
            left = 24
            right = w - 24
            y_crop = y_sparse[:, :, top:bottom, left:right]
            pred_crop = pred[:, :, top:bottom, left:right]
            valid_mask = (y_crop > 1e-3) & (y_crop < 10.0)
            if valid_mask.sum() == 0:
                dummy_loss = torch.tensor(10.0, device=pred.device)
                self.log("val/rmse", dummy_loss, prog_bar=True, on_epoch=True, sync_dist=(self.trainer.world_size > 1))
                self.log("val_rmse", dummy_loss, prog_bar=False, on_epoch=True, sync_dist=(self.trainer.world_size > 1))
                self.log("val/abs_rel", dummy_loss, prog_bar=True, on_epoch=True, sync_dist=(self.trainer.world_size > 1))
                self.log("val/delta1", torch.tensor(0.0, device=pred.device), prog_bar=True, on_epoch=True, sync_dist=(self.trainer.world_size > 1))
                self.log("val/l1", dummy_loss, prog_bar=False, on_epoch=True, sync_dist=(self.trainer.world_size > 1))
                return dummy_loss
            p = pred_crop[valid_mask].float()
            t = y_crop[valid_mask].float()
            p = torch.clamp(p, min=1e-3, max=10.0)
            
        rmse = torch.sqrt(torch.mean((p - t) ** 2))
        abs_rel = torch.mean(torch.abs(p - t) / t)
        ratio = torch.max(p / t, t / p)
        delta1 = (ratio < 1.25).float().mean()
        l1_loss = torch.mean(torch.abs(p - t))
        self.log("val/rmse", rmse, prog_bar=True, on_epoch=True, sync_dist=(self.trainer.world_size > 1))
        self.log("val_rmse", rmse, prog_bar=False, on_epoch=True, sync_dist=(self.trainer.world_size > 1))
        self.log("val/abs_rel", abs_rel, prog_bar=True, on_epoch=True, sync_dist=(self.trainer.world_size > 1))
        self.log("val/delta1", delta1, prog_bar=True, on_epoch=True, sync_dist=(self.trainer.world_size > 1))
        self.log("val/l1", l1_loss, prog_bar=False, on_epoch=True, sync_dist=(self.trainer.world_size > 1))
        return l1_loss

    def on_train_epoch_end(self):
        _clean_cache()

    def on_validation_epoch_end(self):
        _clean_cache()

    def configure_optimizers(self):
        trainable_params = []
        if any(
            p.requires_grad
            for p in self.model.encoder.parameters()
        ):
            trainable_params.append({
                "params": self.model.encoder.parameters(),
                "lr": self.hparams.lr_enc
            })
        trainable_params.append({
            "params": self.model.decoder.parameters(),
            "lr": self.hparams.lr_dec
        })
        optimizer = torch.optim.AdamW(
            trainable_params,
            weight_decay=self.hparams.weight_decay
        )
        if getattr(self.hparams, "use_step_lr", False) or getattr(self.hparams, "warmup_epochs", 5) == 0:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=20,
                gamma=0.1
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch"
                }
            }

        # 1. Warmup scheduler: linearly increases from 0.01x to 1x LR over `warmup_epochs` epochs
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.hparams.warmup_epochs
        )
        # 2. Cosine annealing for the remaining epochs
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs - self.hparams.warmup_epochs,
            eta_min=1e-6
        )
        # Combine the two schedulers
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.hparams.warmup_epochs]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }
    
    def on_train_start(self):
        self.start_time = time.time()

    def on_train_end(self):
        if self.start_time:
            self.total_train_time = (time.time() - self.start_time) 
            print(f"[INFO] Training completed in {self.total_train_time:.2f} seconds")
# ==========================================
# Lightning callbacks
# ==========================================
def get_callbacks(save_dir, monitor="val/loss", dataset_name="nyu", arch_name="xxs", monitor_mode="min"):
    safe_monitor_name = monitor.replace('/', '_')

    # Saves the best checkpoint according to the monitored metric
    best_checkpoint = ModelCheckpoint(
        dirpath=save_dir,
        filename="best-{epoch:02d}-{" + safe_monitor_name + ":.4f}-" + dataset_name + "-" + arch_name,
        monitor=monitor,
        mode=monitor_mode,
        save_top_k=3,
        save_last=False,  # last.ckpt handled by the dedicated callback below
    )

    # In Lightning >= 2.x, save_last=True only writes last.ckpt when a new
    # best is found. This dedicated callback writes last.ckpt unconditionally
    # after every epoch, enabling safe resumption from any interruption.
    last_checkpoint = ModelCheckpoint(
        dirpath=save_dir,
        filename="last",
        monitor=None,     # no condition — always save
        save_top_k=1,     # keep only the most recent
        every_n_epochs=1,
        save_last=False,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    early_stop = EarlyStopping(
        monitor=monitor,
        patience=60,
        mode=monitor_mode,
        strict=False,
    )
    
    from lightning.pytorch.callbacks import ThroughputMonitor
    def get_batch_size(batch):
        if isinstance(batch, (list, tuple)):
            return len(batch[0])
        return len(batch)
    throughput = ThroughputMonitor(batch_size_fn=get_batch_size)
    
    return [best_checkpoint, last_checkpoint, lr_monitor, early_stop, throughput]

# ==========================================
# Test block
# ==========================================
if __name__ == "__main__":
    print("\n========== TRAIN TEST ==========")
    model = LeJEPAMeter(arch_type="xs")

    # Test forward on a global-scale crop
    dummy_global = torch.randn(2, 3, 256, 256)
    emb, proj = model(dummy_global)
    print(f"[INFO] Global crop — emb:  {emb.shape}")   # [2, C]
    print(f"[INFO] Global crop — proj: {proj.shape}")  # [2, N, D]  N=TOKEN_GRID²

    # Test forward on a local-scale crop (half resolution)
    dummy_local = torch.randn(2, 3, 128, 128)
    emb_l, proj_l = model(dummy_local)
    print(f"[INFO] Local  crop — emb:  {emb_l.shape}")
    print(f"[INFO] Local  crop — proj: {proj_l.shape}")  # same N as global

    # Test loss (simulating 2 global + 6 local views)
    N, D = proj.shape[1], proj.shape[2]
    fake_proj = torch.cat([
        proj.unsqueeze(0).expand(2, -1, -1, -1),     # 2 global views [2,2,N,D]
        proj_l.unsqueeze(0).expand(6, -1, -1, -1),   # 6 local  views [6,2,N,D]
    ], dim=0)  # [8, 2, N, D]
    loss, dense_inv, global_inv, sig = model.lejepa_loss(fake_proj)
    print(f"[INFO] Loss={loss.item():.4f}  dense_inv={dense_inv.item():.4f}  global_inv={global_inv.item():.4f}  sigreg={sig.item():.4f}")
    print("\n========== DONE ==========")
