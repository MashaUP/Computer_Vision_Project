"""Phase-1 METER evaluation utilities (ported from ``src/evaluation.py`` of the exam version).

Metric computation with the phase-1 conventions, full test-set evaluation, encoder feature
extraction, PCA geometric probing, prediction plots and latency benchmarking.

The two phase-1 setups stored NYU depth in different units: centimetres in the classic setup,
metres in the dense setup. ``compute_metrics(..., nyu_unit="cm" | "m")`` covers both.
For the new comparison protocol use ``evaluation/metrics.py``.
"""

import os
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# =====================================================================
# 1. Metric computation  (paper Section IV-C, Eq. 9-11)
# =====================================================================

def _nyu_crop_mask(depth):
    """Eigen crop adapted to 192x256: 16-px border top/bottom, 24-px left/right."""
    H, W = depth.shape[-2:]
    mask = torch.zeros_like(depth, dtype=torch.bool)
    mask[:, :, 16:H - 16, 24:W - 24] = True
    return mask


def _kitti_crop_mask(depth):
    """Garg/Eigen crop, proportional to image size (paper Section IV-B)."""
    H, W = depth.shape[-2:]
    top    = int(0.40810811 * H)
    bottom = int(0.99189189 * H)
    left   = int(0.03594771 * W)
    right  = int(0.96405229 * W)
    mask = torch.zeros_like(depth, dtype=torch.bool)
    mask[:, :, top:bottom, left:right] = True
    return mask


def compute_metrics(pred, target, dataset_name, nyu_unit="cm"):
    """
    Compute RMSE / AbsRel / delta1 following the official benchmark protocol.

    NYU   : depth converted cm -> m before evaluation; clamp to [1e-3, 10] m;
            Eigen crop applied.
    KITTI : depth already in metres; clamp to [1e-3, 80] m;
            Garg/Eigen crop applied.

    Parameters
    ----------
    pred         : (B, 1, H, W) prediction tensor — same units as target
    target       : (B, 1, H, W) ground-truth depth tensor
    dataset_name : 'nyu' or 'kitti'

    Returns
    -------
    dict {'RMSE', 'AbsRel', 'Delta1'} (float), or None if no valid pixel
    survives crop + clamp + validity mask.
    """
    dataset_name = str(dataset_name).lower().strip()
    pred   = pred.float()
    target = target.float()

    if "nyu" in dataset_name:
        if nyu_unit == "cm":             # classic setup: cm -> m
            pred   = pred   / 100.0
            target = target / 100.0
        elif nyu_unit != "m":
            raise ValueError(f"nyu_unit must be 'cm' or 'm', got {nyu_unit!r}")
        pred   = torch.clamp(pred,   min=1e-3, max=10.0)
        target = torch.clamp(target, min=1e-3, max=10.0)
        crop   = _nyu_crop_mask(target)
    else:
        pred   = torch.clamp(pred,   min=1e-3, max=80.0)
        target = torch.clamp(target, min=1e-3, max=80.0)
        crop   = _kitti_crop_mask(target)

    valid = (target > 0.001) & crop
    if valid.sum() == 0:
        return None

    p = pred[valid]
    t = target[valid]

    rmse    = torch.sqrt(torch.mean((p - t) ** 2))
    abs_rel = torch.mean(torch.abs(p - t) / t)
    delta1  = (torch.max(p / t, t / p) < 1.25).float().mean()

    return {
        "RMSE":   rmse.item(),
        "AbsRel": abs_rel.item(),
        "Delta1": delta1.item(),
    }


# =====================================================================
# 2. Full test-set evaluation loop
# =====================================================================

def evaluate_depth_model(model, dataloader, device, dataset_name="nyu"):
    """
    Run the full test-set evaluation and return mean RMSE / AbsRel / delta1.

    Parameters
    ----------
    model        : METERFinetune / METERModel
    dataloader   : DataLoader yielding (rgb, depth) batches
    device       : torch.device or str
    dataset_name : 'nyu' or 'kitti' — controls crop and unit conversion
    """
    print(f"\n[INFO] Starting evaluation — dataset: {dataset_name}")
    model.eval()
    model.to(device)

    rmse_list, abs_rel_list, delta1_list = [], [], []

    with torch.inference_mode():
        for batch_idx, (rgb, depth) in enumerate(dataloader):
            rgb   = rgb.to(device)
            depth = depth.to(device)

            pred = model(rgb)
            if isinstance(pred, (list, tuple)):
                pred = pred[0]

            # Up-sample to GT resolution (mirrors METERFinetune.forward)
            pred = torch.nn.functional.interpolate(
                pred,
                size=depth.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )

            results = compute_metrics(pred, depth, dataset_name)
            if results is None:
                continue

            rmse_list.append(results["RMSE"])
            abs_rel_list.append(results["AbsRel"])
            delta1_list.append(results["Delta1"])

            if batch_idx % 10 == 0:
                print(f"[INFO] Batch {batch_idx:04d} — "
                      f"RMSE {results['RMSE']:.4f}  "
                      f"AbsRel {results['AbsRel']:.4f}  "
                      f"Delta1 {results['Delta1']:.4f}")

    if not rmse_list:
        print("[WARN] No valid samples during evaluation.")
        return {"RMSE": float("nan"), "AbsRel": float("nan"), "Delta1": float("nan")}

    final = {
        "RMSE":   float(np.mean(rmse_list)),
        "AbsRel": float(np.mean(abs_rel_list)),
        "Delta1": float(np.mean(delta1_list)),
    }
    print(f"\n[INFO] Evaluation completed ({len(rmse_list)} batches)")
    print(f"[INFO]   RMSE   : {final['RMSE']:.4f}")
    print(f"[INFO]   AbsRel : {final['AbsRel']:.4f}")
    print(f"[INFO]   Delta1 : {final['Delta1']:.4f}")
    return final


# =====================================================================
# 3. Feature extraction & PCA
# =====================================================================

def extract_features(model, image_tensor, device):
    """
    Extract the final encoder feature map.

    Works transparently with the three wrappers used in the project:
        METERModel     -> model.encoder
        METERFinetune  -> model.model.encoder
        LeJEPAMeter    -> model.encoder
    """
    model = model.to(device)
    model.eval()
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        encoder = getattr(model, "encoder", None)
        if encoder is None and hasattr(model, "model"):
            encoder = getattr(model.model, "encoder", None)

        if encoder is not None:
            features, _ = encoder(image_tensor)
        else:
            features = model(image_tensor)
            if isinstance(features, (list, tuple)):
                features = features[0]

    return features


def features_to_pca(features):
    """
    Project a (1, C, H, W) feature map to a (H, W, 3) RGB image via PCA.
    Min-max normalised for visualisation.
    """
    n_components = min(3, features.shape[1])
    fm = features.squeeze(0).permute(1, 2, 0).cpu().numpy()
    h, w, c = fm.shape
    flat    = fm.reshape(-1, c)

    pca     = PCA(n_components=n_components)
    reduced = pca.fit_transform(flat)

    if n_components < 3:
        canvas = np.zeros((h * w, 3))
        canvas[:, :n_components] = reduced
    else:
        canvas = reduced

    canvas = (canvas - canvas.min()) / (canvas.max() - canvas.min() + 1e-8)
    return canvas.reshape(h, w, 3)


def extract_embeddings(model, dataloader, device="cuda", max_batches=20):
    """
    Run the encoder over up to `max_batches` batches and return:
        X : (N, C) global-average-pooled encoder embeddings
        y : (N,)   per-image average GT depth (zero if no depth in batch)

    Robust to both (rgb,) pretrain-mode and (rgb, depth) finetune-mode batches.
    Used by `plot_pca_embeddings` for latent-space visualization.
    """
    model = model.to(device)
    model.eval()

    encoder = getattr(model, "encoder", None)
    if encoder is None and hasattr(model, "model"):
        encoder = getattr(model.model, "encoder", None)
    if encoder is None:
        raise RuntimeError("Cannot find an `encoder` attribute on the given model.")

    embeddings, labels = [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            if isinstance(batch, (list, tuple)):
                rgb   = batch[0].to(device)
                depth = batch[1] if len(batch) > 1 else None
            else:
                rgb, depth = batch.to(device), None

            features, _ = encoder(rgb)
            emb = features.mean(dim=[-2, -1])      # GAP over spatial dims
            embeddings.append(emb.cpu())

            if depth is not None:
                avg_d = depth.view(depth.shape[0], -1).float().mean(dim=1)
            else:
                avg_d = torch.zeros(rgb.shape[0])
            labels.append(avg_d.cpu())

    X = torch.cat(embeddings, dim=0).numpy()
    y = torch.cat(labels,     dim=0).numpy()
    return X, y


# =====================================================================
# 4. Geometric probing  (Project 7, Objective 2)
# =====================================================================

def geometric_probing_pca(model, device, image_tensor, save_path=None):
    """
    Visualise the PCA feature map for a single image.
    Implements the zero-shot geometric probing of Project 7.
    """
    print("\n[INFO] Running geometric probing (single image)")
    features = extract_features(model, image_tensor, device)
    print(f"[INFO] Feature shape: {features.shape}")

    pca_img = features_to_pca(features)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.imshow(pca_img)
    ax.set_title(f"Geometric Probing — PCA ({features.shape[1]} channels)")
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved: {save_path}")

    plt.show()


def geometric_probing_test_set(model, device, test_loader, num_samples=4, save_path=None):
    """
    RGB + PCA feature map side-by-side for a small batch from the test set.
    Shows whether the encoder separates depth discontinuities without supervision.
    """
    print(f"\n[INFO] Running test-set probing ({num_samples} samples)")
    model.eval()

    batch  = next(iter(test_loader))
    images = batch[0] if isinstance(batch, (list, tuple)) else batch
    images = images[:num_samples].to(device)

    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 4 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)
    fig.suptitle("Zero-shot Geometric Probing — Test Set", fontsize=15)

    with torch.no_grad():
        for i in range(num_samples):
            single   = images[i].unsqueeze(0)
            feat     = extract_features(model, single, device)
            pca_img  = features_to_pca(feat)
            orig_img = single.squeeze(0).permute(1, 2, 0).cpu().numpy()
            orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min() + 1e-8)

            axes[i, 0].imshow(orig_img)
            axes[i, 0].set_title("RGB Input")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(pca_img)
            axes[i, 1].set_title(f"PCA Features ({feat.shape[1]} ch)")
            axes[i, 1].axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved: {save_path}")

    plt.show()


def plot_pca_embeddings(model, dataloader, device="cuda",
                        save_path=None, max_batches=20,
                        dataset_name=None, variant_label=None):
    """
    2-D PCA scatter of global encoder embeddings, color-coded by average
    GT depth. Complements `geometric_probing_test_set` by showing global
    structure (one dot per image) instead of per-pixel feature maps.
    """
    print("\n[INFO] Extracting embeddings for PCA scatter...")
    X, y = extract_embeddings(model, dataloader, device=device, max_batches=max_batches)
    print(f"[INFO] Embeddings shape: {X.shape}")

    pca   = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                    c=y, cmap="viridis", s=24, alpha=0.85)
    plt.colorbar(sc, ax=ax, label="Average GT depth")

    # Build title dynamically based on what's provided
    parts = []
    if variant_label: parts.append(variant_label)
    if dataset_name:  parts.append(dataset_name.upper())
    suffix = f" — {' | '.join(parts)}" if parts else ""
    ax.set_title(f"PCA of encoder embeddings{suffix}\n(colored by avg depth)")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved: {save_path}")

    plt.show()


# =====================================================================
# 5. Depth-prediction visualization
# =====================================================================

def _denorm_imagenet(rgb_batch):
    """Reverse ImageNet normalization for display."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (rgb_batch.cpu() * std + mean).clamp(0, 1)


def show_depth_predictions(models_dict, dataloader, device="cuda",
                            num_samples=4, save_path=None, dataset_name="nyu"):
    """
    Display RGB + GT + predictions from one or more models, side-by-side.

    Parameters
    ----------
    models_dict  : dict {label: model} — one prediction column per entry
    dataloader   : test DataLoader yielding (rgb, depth)
    num_samples  : number of rows in the figure
    dataset_name : 'nyu' (cm -> m, vmax=10) or 'kitti' (vmax=80)
    """
    assert isinstance(models_dict, dict) and models_dict, \
        "models_dict must be a non-empty dict {label: model}"

    for m in models_dict.values():
        m.eval()

    if "nyu" in dataset_name.lower():
        scale, unit, vmax = 1 / 100.0, "m", 10.0
    else:
        scale, unit, vmax = 1.0, "m", 80.0

    # Pull enough samples from the loader to fill `num_samples` rows
    rgb_buf, gt_buf, count = [], [], 0
    with torch.no_grad():
        for rgb, depth in dataloader:
            rgb_buf.append(rgb)
            gt_buf.append(depth)
            count += rgb.shape[0]
            if count >= num_samples:
                break
    rgb = torch.cat(rgb_buf, dim=0)[:num_samples]
    gt  = torch.cat(gt_buf,  dim=0)[:num_samples]

    rgb_show = _denorm_imagenet(rgb)
    gt_show  = gt.cpu() * scale

    # Run each model
    preds = {}
    for label, model in models_dict.items():
        model = model.to(device)
        with torch.no_grad():
            p = model(rgb.to(device))
        if isinstance(p, (list, tuple)):
            p = p[0]
        p = torch.nn.functional.interpolate(
            p, size=gt.shape[-2:], mode="bilinear", align_corners=True
        )
        preds[label] = (p.cpu() * scale).clamp(0, vmax)

    n_cols = 2 + len(models_dict)
    fig, axes = plt.subplots(num_samples, n_cols,
                              figsize=(3.5 * n_cols, 3.0 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(num_samples):
        axes[i, 0].imshow(rgb_show[i].permute(1, 2, 0).numpy())
        axes[i, 0].set_title("RGB" if i == 0 else "")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(gt_show[i, 0].numpy(), cmap="plasma", vmin=0, vmax=vmax)
        axes[i, 1].set_title(f"GT ({unit})" if i == 0 else "")
        axes[i, 1].axis("off")

        for k, (label, pred) in enumerate(preds.items()):
            axes[i, 2 + k].imshow(pred[i, 0].numpy(),
                                  cmap="plasma", vmin=0, vmax=vmax)
            axes[i, 2 + k].set_title(f"{label} ({unit})" if i == 0 else "")
            axes[i, 2 + k].axis("off")

    plt.suptitle(f"Depth predictions — {dataset_name.upper()}", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved: {save_path}")

    plt.show()


# =====================================================================
# 6. Efficiency benchmarking  (Project 7, Objective 4)
# =====================================================================

def count_parameters(model):
    """Total trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark_model(model, input_shape, device="cuda", warmup=10, runs=50):
    """
    Measure params (M), latency (ms/image), and FPS for a single input.

    Parameters
    ----------
    model       : torch.nn.Module
    input_shape : (C, H, W) — e.g. (3, 192, 256) for NYU
    warmup      : warmup iterations excluded from timing
    runs        : number of timed forward passes

    Returns
    -------
    dict {'params_m', 'latency_ms', 'fps'}
    """
    model = model.to(device)
    model.eval()
    if not isinstance(device, torch.device):
        device = torch.device(device)

    dummy = torch.randn(1, *input_shape).to(device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
        sync()
        t0 = time.time()
        for _ in range(runs):
            _ = model(dummy)
        sync()
        elapsed = time.time() - t0

    params_m   = count_parameters(model) / 1e6
    latency_ms = (elapsed / runs) * 1000.0
    fps        = 1000.0 / latency_ms if latency_ms > 0 else float("inf")

    print(f"[INFO] Benchmark: {params_m:.3f}M params | "
          f"{latency_ms:.2f} ms/img | {fps:.1f} FPS")
    return {"params_m": params_m, "latency_ms": latency_ms, "fps": fps}


# =====================================================================
# 7. Pixel-weighted evaluation loop (all configurations)
# =====================================================================

def evaluate_pixel_weighted(model, dataloader, device, dataset_name):
    """
    Pixel-weighted aggregation (correct), unlike batch-mean in evaluate_depth_model.
    Accumulates squared errors / abs errors / delta1 hits over all valid pixels.
    """
    model.eval().to(device)

    # Re-implement pixel-level aggregation
    sq_err_sum   = 0.0
    abs_rel_sum  = 0.0
    delta1_sum   = 0.0
    n_pixels     = 0

    with torch.inference_mode():
        for rgb, depth in dataloader:
            rgb   = rgb.to(device)
            depth = depth.to(device)
            pred  = model(rgb)
            if isinstance(pred, (list, tuple)):
                pred = pred[0]
            pred = torch.nn.functional.interpolate(
                pred, size=depth.shape[-2:], mode="bilinear", align_corners=True
            )

            # Apply same clamp + crop + valid_mask as compute_metrics
            ds = dataset_name.lower()
            p = pred.float()
            t = depth.float()
            if "nyu" in ds:
                p, t = p / 100.0, t / 100.0
                p = torch.clamp(p, min=1e-3, max=10.0)
                t = torch.clamp(t, min=1e-3, max=10.0)
                _, _, H, W = t.shape
                crop = torch.zeros_like(t, dtype=torch.bool)
                crop[:, :, 16:H-16, 24:W-24] = True
            else:
                p = torch.clamp(p, min=1e-3, max=80.0)
                t = torch.clamp(t, min=1e-3, max=80.0)
                _, _, H, W = t.shape
                top, bot = int(0.40810811 * H), int(0.99189189 * H)
                lef, rig = int(0.03594771 * W), int(0.96405229 * W)
                crop = torch.zeros_like(t, dtype=torch.bool)
                crop[:, :, top:bot, lef:rig] = True

            valid = (t > 0.001) & crop
            if valid.sum() == 0:
                continue

            p_v = p[valid]
            t_v = t[valid]
            sq_err_sum  += float(((p_v - t_v) ** 2).sum())
            abs_rel_sum += float((torch.abs(p_v - t_v) / t_v).sum())
            ratio = torch.maximum(p_v / t_v, t_v / p_v)
            delta1_sum  += float((ratio < 1.25).sum())
            n_pixels    += int(valid.sum())

    if n_pixels == 0:
        return {"RMSE": float("nan"), "AbsRel": float("nan"), "Delta1": float("nan")}

    return {
        "RMSE"  : (sq_err_sum  / n_pixels) ** 0.5,
        "AbsRel": abs_rel_sum  / n_pixels,
        "Delta1": delta1_sum   / n_pixels,
    }
