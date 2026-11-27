"""CPU tests for leak-free KITTI split generation and ``data.datasets.check_list``."""
from pathlib import Path

import pytest

from data.datasets import check_list
from data.download.download_kitti import EIGEN_TEST_DRIVES, is_val_drive, make_splits


def test_eigen_test_drives_match_sota_reference():
    sota_splits = Path(__file__).resolve().parents[1] / "sota" / "flexdepth" / "splits"
    test_file = sota_splits / "eigen_benchmark" / "test_files.txt"
    if not test_file.is_file():
        pytest.skip("sota/flexdepth submodule not checked out")

    ref_test = {ln.split()[0].split("/")[1] for ln in test_file.read_text().splitlines() if ln.strip()}
    assert EIGEN_TEST_DRIVES == ref_test
    assert len(EIGEN_TEST_DRIVES) == 28

    for zhou in ("train_files.txt", "val_files.txt"):
        p = sota_splits / "eigen_zhou" / zhou
        if p.is_file():
            zhou_drives = {ln.split()[0].split("/")[1] for ln in p.read_text().splitlines() if ln.strip()}
            assert EIGEN_TEST_DRIVES.isdisjoint(zhou_drives)


def test_make_splits_excludes_test_and_separates_val(tmp_path):
    data = tmp_path / "data"
    test_drive = "2011_09_26_drive_0002_sync"
    non_test_drives = [f"2011_09_26_drive_{i:04d}_sync" for i in range(500, 520)]
    val_drive = next(d for d in non_test_drives if is_val_drive(d))
    train_drive = next(d for d in non_test_drives if not is_val_drive(d))

    for drive in (test_drive, val_drive, train_drive):
        img_dir = data / "kitti" / "rgb" / "2011_09_26" / drive / "image_02" / "data"
        dep_dir = data / "kitti" / "depth" / "data_depth_annotated" / "train" / drive / "proj_depth" / "groundtruth" / "image_02"
        img_dir.mkdir(parents=True)
        dep_dir.mkdir(parents=True)
        for idx in (0, 1, 2):
            (img_dir / f"{idx:010d}.png").write_bytes(b"")
            (dep_dir / f"{idx:010d}.png").write_bytes(b"")

    ref = tmp_path / "eigen_test_ref.txt"
    ref.write_text(f"2011_09_26/{test_drive} 1 l\n2011_09_26/{test_drive} 99 l\n")

    counts = make_splits(data, eigen_test_ref=ref)
    assert counts["kitti_test_drives.txt"] == 28
    assert counts["kitti_raw_frames.txt"] == 3
    assert counts["kitti_raw_triplets_train.txt"] == 1
    assert counts["kitti_raw_triplets_val.txt"] == 1
    assert counts["kitti_labeled_val.txt"] == 3
    assert counts["kitti_eigen_test.txt"] == 1

    splits = data / "splits"
    for name in ("kitti_raw_frames.txt", "kitti_raw_triplets_train.txt", "kitti_raw_triplets_val.txt", "kitti_labeled_val.txt"):
        txt = (splits / name).read_text()
        assert test_drive not in txt
    assert val_drive not in (splits / "kitti_raw_frames.txt").read_text()
    assert val_drive not in (splits / "kitti_raw_triplets_train.txt").read_text()


def test_check_list_kitti_rejects_test_and_val(tmp_path, monkeypatch):
    monkeypatch.setenv("LG_DATA_ROOT", str(tmp_path))
    splits = tmp_path / "splits"
    splits.mkdir()

    test_lst = tmp_path / "has_test.txt"
    test_lst.write_text("kitti/rgb/2011_09_26/2011_09_26_drive_0002_sync/image_02/data/0000000000.png\n")
    with pytest.raises(ValueError, match="2011_09_26_drive_0002_sync"):
        check_list(test_lst, "kitti", "pretrain")
    with pytest.raises(ValueError, match="2011_09_26_drive_0002_sync"):
        check_list(test_lst, "kitti", "val")

    (splits / "kitti_val_drives.txt").write_text("2011_09_26_drive_0028_sync\n")
    val_lst = tmp_path / "has_val.txt"
    val_lst.write_text("kitti/rgb/2011_09_26/2011_09_26_drive_0028_sync/image_02/data/0000000000.png\n")
    with pytest.raises(ValueError, match="2011_09_26_drive_0028_sync"):
        check_list(val_lst, "kitti", "train")
    check_list(val_lst, "kitti", "val")

    ok_lst = tmp_path / "ok.txt"
    ok_lst.write_text("kitti/rgb/2011_09_26/2011_09_26_drive_0022_sync/image_02/data/0000000000.png\n")
    check_list(ok_lst, "kitti", "pretrain")
    check_list(ok_lst, "kitti", "train")
