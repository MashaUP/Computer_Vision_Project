import pytest
import torch

from models.meter import METERModel


@pytest.mark.parametrize("arch,params_m", [("xxs", 0.71), ("xs", 1.45), ("s", 3.30)])
@pytest.mark.parametrize("skip", [False, True])
def test_meter_shapes_and_size(arch, params_m, skip):
    m = METERModel(arch, skip_from_transformer=skip, verbose=False).eval()
    assert abs(sum(p.numel() for p in m.parameters()) / 1e6 - params_m) < 0.01
    with torch.no_grad():
        assert m(torch.rand(2, 3, 192, 256)).shape == (2, 1, 48, 64)      # NYU fine-tuning size
        assert m(torch.rand(1, 3, 192, 636)).shape == (1, 1, 48, 160)     # KITTI fine-tuning size


def test_skip_option_changes_only_the_deepest_skip():
    m = METERModel("xxs", verbose=False).eval()
    x = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        out_ref, skips_ref = m.encoder(x)
        m.encoder.skip_from_transformer = True
        out_tr, skips_tr = m.encoder(x)
    assert torch.equal(out_ref, out_tr)
    assert all(torch.equal(a, b) for a, b in zip(skips_ref[:3], skips_tr[:3]))
    assert not torch.equal(skips_ref[3], skips_tr[3])
