"""Self-supervised photometric loss (Monodepth2 / FlexDepth / BoRe-Depth standard) for step 3 (option A)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


def rot_from_axisangle(vec: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle rotation ``(B, 1, 3)`` into ``(B, 4, 4)`` rotation matrix."""
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)
    ca, sa = torch.cos(angle), torch.sin(angle)
    c = 1.0 - ca
    x, y, z = axis[..., 0].unsqueeze(1), axis[..., 1].unsqueeze(1), axis[..., 2].unsqueeze(1)
    xs, ys, zs = x * sa, y * sa, z * sa
    xc, yc, zc = x * c, y * c, z * c
    xyc, yzc, zxc = x * yc, y * zc, z * xc

    rot = torch.zeros((vec.shape[0], 4, 4), device=vec.device, dtype=vec.dtype)
    rot[:, 0, 0] = (x * xc + ca).squeeze()
    rot[:, 0, 1] = (xyc - zs).squeeze()
    rot[:, 0, 2] = (zxc + ys).squeeze()
    rot[:, 1, 0] = (xyc + zs).squeeze()
    rot[:, 1, 1] = (y * yc + ca).squeeze()
    rot[:, 1, 2] = (yzc - xs).squeeze()
    rot[:, 2, 0] = (zxc - ys).squeeze()
    rot[:, 2, 1] = (yzc + xs).squeeze()
    rot[:, 2, 2] = (z * zc + ca).squeeze()
    rot[:, 3, 3] = 1.0
    return rot


def transformation_from_parameters(axisangle: torch.Tensor, translation: torch.Tensor, invert: bool = False) -> torch.Tensor:
    """Convert ``(axisangle, translation)`` of shape ``(B, 1, 3)`` into a ``(B, 4, 4)`` rigid transform."""
    r = rot_from_axisangle(axisangle.float())
    t_vec = translation.float().clone()
    if invert:
        r = r.transpose(1, 2)
        t_vec = -t_vec
    t = torch.eye(4, device=t_vec.device, dtype=t_vec.dtype).unsqueeze(0).repeat(t_vec.shape[0], 1, 1)
    t[:, :3, 3] = t_vec.contiguous().view(-1, 3)
    return torch.matmul(r, t) if invert else torch.matmul(t, r)


def warp_image(src: torch.Tensor, depth: torch.Tensor, T: torch.Tensor, K: torch.Tensor, inv_K: torch.Tensor) -> torch.Tensor:
    """Warp ``src`` ``(B, 3, H, W)`` into the target frame using ``depth`` ``(B, 1, H, W)`` and ``T`` ``(B, 4, 4)``."""
    b, _, h, w = depth.shape
    depth = depth.float()
    src = src.float()
    T = T.float()
    K = K.float()
    inv_K = inv_K.float()

    ys, xs = torch.meshgrid(
        torch.arange(h, device=depth.device, dtype=torch.float32),
        torch.arange(w, device=depth.device, dtype=torch.float32),
        indexing="ij",
    )
    pix = torch.stack([xs.reshape(-1), ys.reshape(-1), torch.ones(h * w, device=depth.device, dtype=torch.float32)], dim=0)
    pix = pix.unsqueeze(0).expand(b, -1, -1)  # (B, 3, H*W)

    cam_pts = depth.reshape(b, 1, -1) * torch.matmul(inv_K[:, :3, :3], pix)
    cam_pts_h = torch.cat([cam_pts, torch.ones((b, 1, h * w), device=depth.device, dtype=torch.float32)], dim=1)

    proj = torch.matmul(torch.matmul(K, T)[:, :3, :], cam_pts_h)
    xy = proj[:, :2, :] / proj[:, 2:3, :].clamp(min=1e-7)
    grid = xy.reshape(b, 2, h, w).permute(0, 2, 3, 1).contiguous()
    grid[..., 0] = (grid[..., 0] / max(w - 1, 1)) * 2.0 - 1.0
    grid[..., 1] = (grid[..., 1] / max(h - 1, 1)) * 2.0 - 1.0
    return F.grid_sample(src, grid, mode="bilinear", padding_mode="border", align_corners=True)


def ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Monodepth2 structural dissimilarity ``(1 - SSIM) / 2`` clamped to ``[0, 1]``."""
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    x = F.pad(x.float(), (1, 1, 1, 1), mode="reflect")
    y = F.pad(y.float(), (1, 1, 1, 1), mode="reflect")
    mu_x = F.avg_pool2d(x, 3, 1)
    mu_y = F.avg_pool2d(y, 3, 1)
    sigma_x = F.avg_pool2d(x * x, 3, 1) - mu_x ** 2
    sigma_y = F.avg_pool2d(y * y, 3, 1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, 3, 1) - mu_x * mu_y
    num = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    return torch.clamp((1.0 - num / den) / 2.0, 0.0, 1.0)


def photometric_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """0.85 * DSSIM + 0.15 * L1 per pixel ``(B, 1, H, W)``."""
    l1 = torch.abs(pred.float() - target.float()).mean(1, keepdim=True)
    dssim = ssim(pred, target).mean(1, keepdim=True)
    return 0.85 * dssim + 0.15 * l1


def smoothness_loss(disp: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
    """Edge-aware L1 smoothness on mean-normalized disparity."""
    disp = disp.float()
    img = img.float()
    norm_disp = disp / (disp.mean(dim=(2, 3), keepdim=True) + 1e-7)
    grad_disp_x = torch.abs(norm_disp[:, :, :, :-1] - norm_disp[:, :, :, 1:])
    grad_disp_y = torch.abs(norm_disp[:, :, :-1, :] - norm_disp[:, :, 1:, :])
    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)
    grad_disp_x = grad_disp_x * torch.exp(-grad_img_x)
    grad_disp_y = grad_disp_y * torch.exp(-grad_img_y)
    return grad_disp_x.mean() + grad_disp_y.mean()


class PoseNet(nn.Module):
    """Standard Monodepth2 ResNet-18 6-channel pose network."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        old_conv1 = backbone.conv1
        conv1 = nn.Conv2d(6, old_conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            conv1.weight.copy_(torch.cat([old_conv1.weight, old_conv1.weight], dim=1) / 2.0)
        backbone.conv1 = conv1
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layers = nn.Sequential(backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4)

        self.squeeze = nn.Conv2d(512, 256, 1)
        self.pose0 = nn.Conv2d(256, 256, 3, 1, 1)
        self.pose1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.pose2 = nn.Conv2d(256, 6, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([img_a, img_b], dim=1)
        x = (x - 0.45) / 0.225
        f = self.layers(self.stem(x))
        out = self.relu(self.squeeze(f))
        out = self.relu(self.pose0(out))
        out = self.relu(self.pose1(out))
        out = self.pose2(out).mean(dim=(2, 3))
        out = 0.01 * out.view(-1, 1, 6)
        return out[..., :3], out[..., 3:]


class PhotometricLoss(nn.Module):
    """Monodepth2 self-supervised photometric + auto-masking + edge-aware smoothness loss."""

    def __init__(self, smooth_weight: float = 1e-3, pose_pretrained: bool = True):
        super().__init__()
        self.pose_net = PoseNet(pretrained=pose_pretrained)
        self.smooth_weight = float(smooth_weight)

    def forward(
        self,
        depth: torch.Tensor,
        prev_aug: torch.Tensor,
        target_aug: torch.Tensor,
        next_aug: torch.Tensor,
        prev: torch.Tensor,
        target: torch.Tensor,
        nxt: torch.Tensor,
        K: torch.Tensor,
        inv_K: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        aa_prev, t_prev = self.pose_net(prev_aug, target_aug)
        aa_next, t_next = self.pose_net(target_aug, next_aug)
        T_prev = transformation_from_parameters(aa_prev, t_prev, invert=True)
        T_next = transformation_from_parameters(aa_next, t_next, invert=False)

        depth_f = depth.float()
        warped_prev = warp_image(prev, depth_f, T_prev, K, inv_K)
        warped_next = warp_image(nxt, depth_f, T_next, K, inv_K)

        reproj = torch.cat([
            photometric_error(warped_prev, target),
            photometric_error(warped_next, target),
        ], dim=1)
        identity = torch.cat([
            photometric_error(prev, target),
            photometric_error(nxt, target),
        ], dim=1)
        if self.training:
            identity = identity + torch.randn_like(identity) * 1e-5

        reproj_loss = torch.cat([identity, reproj], dim=1).min(dim=1).values.mean()
        smooth = smoothness_loss(1.0 / depth_f.clamp(min=1e-3), target)
        loss = reproj_loss + self.smooth_weight * smooth
        return {"loss": loss, "reproj": reproj_loss, "smooth": smooth}
