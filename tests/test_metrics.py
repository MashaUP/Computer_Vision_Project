import pytest
import torch

from evaluation.metrics import depth_metrics, eval_mask


def _gt(dataset):
    h, w = (480, 640) if dataset == "nyu" else (375, 1242)
    cap = 10.0 if dataset == "nyu" else 80.0
    return torch.rand(h, w) * (cap - 1.0) + 0.5


@pytest.mark.parametrize("dataset", ["nyu", "kitti"])
def test_perfect_prediction(dataset):
    gt = _gt(dataset)
    m = depth_metrics(gt.clone(), gt, dataset, alignment="none")
    assert m["abs_rel"] < 1e-6 and m["rmse"] < 1e-5 and m["a1"] == 1.0


@pytest.mark.parametrize("dataset", ["nyu", "kitti"])
def test_median_alignment_removes_scale(dataset):
    gt = _gt(dataset)
    m = depth_metrics(3.7 * gt, gt, dataset, alignment="median")
    assert m["abs_rel"] < 1e-5 and m["a1"] == 1.0


@pytest.mark.parametrize("dataset", ["nyu", "kitti"])
def test_scale_shift_alignment_removes_affine_inverse_depth(dataset):
    gt = _gt(dataset)
    inv_pred = 0.5 * (1.0 / gt) + 0.02            # affine-invariant inverse depth
    m = depth_metrics(1.0 / inv_pred, gt, dataset, alignment="scale_shift")
    assert m["abs_rel"] < 1e-4 and m["a1"] == 1.0


def test_crops_cover_the_expected_area():
    nyu = eval_mask(torch.full((480, 640), 5.0), "nyu")
    assert nyu.sum() == (471 - 45) * (601 - 41)
    kitti = eval_mask(torch.full((375, 1242), 20.0), "kitti")
    assert 0.5 * 375 * 1242 < kitti.sum() < 0.6 * 375 * 1242
