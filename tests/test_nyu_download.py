"""NYU preparation: resumable download (local HTTP server with range requests) and extraction of the
labelled frames from a synthetic file with the layout of nyu_depth_v2_labeled.mat."""
import http.server
import threading
from functools import partial

import h5py
import numpy as np
import pytest
from scipy.io import savemat

from data.download import nyu


class RangeHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send_head(self):
        path = self.translate_path(self.path)
        data = open(path, "rb").read()
        rng = self.headers.get("Range")
        start = int(rng.split("=")[1].split("-")[0]) if rng else 0
        self.send_response(206 if rng else 200)
        self.send_header("Content-Length", str(len(data) - start))
        self.end_headers()
        if self.command == "HEAD":
            self.send_header = None
            return None
        self.wfile.write(data[start:])
        return None


@pytest.fixture
def server(tmp_path):
    srv_dir = tmp_path / "srv"
    srv_dir.mkdir()
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), partial(RangeHandler, directory=str(srv_dir)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield srv_dir, f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_download_resumes_partial_file(server, tmp_path, monkeypatch):
    monkeypatch.delenv("http_proxy", raising=False)
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    srv_dir, url = server
    payload = np.random.default_rng(0).integers(0, 255, 3_000_000, dtype=np.uint8).tobytes()
    (srv_dir / "part.zip").write_bytes(payload)
    dest = tmp_path / "dl" / "part.zip"
    dest.parent.mkdir()
    (dest.parent / "part.zip.part").write_bytes(payload[:1_234_567])     # interrupted download
    nyu.download(f"{url}/part.zip", dest)
    assert dest.read_bytes() == payload
    nyu.download(f"{url}/part.zip", dest)                                 # already complete: no-op
    assert dest.read_bytes() == payload


def test_extract_labeled(tmp_path, monkeypatch):
    monkeypatch.setenv("LG_DATA_ROOT", str(tmp_path))
    dl = tmp_path / "nyu" / "downloads"
    dl.mkdir(parents=True)
    n = 5
    rng = np.random.default_rng(0)
    with h5py.File(dl / "nyu_depth_v2_labeled.mat", "w") as f:            # MATLAB v7.3 layout (transposed)
        f["images"] = rng.integers(0, 255, (n, 3, 640, 480), dtype=np.uint8)
        f["depths"] = rng.uniform(0.5, 9.5, (n, 640, 480)).astype(np.float32)
        refs = f.create_dataset("scenes", (1, n), dtype=h5py.ref_dtype)
        for i, name in enumerate(["kitchen_0002a", "kitchen_0001b", "office_0003", "bedroom_0009", "bedroom_0010"]):
            refs[0, i] = f.create_dataset(f"refs/s{i}", data=np.array([[ord(c)] for c in name], dtype=np.uint16)).ref
    savemat(dl / "splits.mat", {"trainNdxs": np.array([[1], [3], [4]]), "testNdxs": np.array([[2], [5]])})
    assert nyu.cmd_extract_labeled(None) == 1          # not the official 795/654 counts on synthetic data
    test_lines = (tmp_path / "splits" / "nyu_labeled_test.txt").read_text().split("\n")
    assert test_lines[0].split()[2] == "kitchen_0001b" and len([ln for ln in test_lines if ln]) == 2
    assert (tmp_path / "splits" / "nyu_test_scenes.txt").read_text().split() == ["bedroom_0010", "kitchen_0001"]
    from PIL import Image
    rgb = np.asarray(Image.open(tmp_path / "nyu/labeled/test/0001_rgb.png"))
    depth = np.asarray(Image.open(tmp_path / "nyu/labeled/test/0001_depth.png"))
    assert rgb.shape == (480, 640, 3) and depth.shape == (480, 640) and depth.dtype == np.uint16
    with h5py.File(dl / "nyu_depth_v2_labeled.mat", "r") as f:
        np.testing.assert_array_equal(rgb, np.asarray(f["images"][1]).transpose(2, 1, 0))
        np.testing.assert_allclose(depth / 1000.0, np.asarray(f["depths"][1]).T, atol=6e-4)


def _val_names():
    """One training scene id and one validation scene id (the split is a hash of the id)."""
    ids = [f"office_{n:04d}" for n in range(1, 200)]
    return next(i for i in ids if not nyu.is_val_scene(i)), next(i for i in ids if nyu.is_val_scene(i))


def test_extract_labeled_rejects_scene_in_train_and_test(tmp_path, monkeypatch):
    monkeypatch.setenv("LG_DATA_ROOT", str(tmp_path))
    dl = tmp_path / "nyu" / "downloads"
    dl.mkdir(parents=True)
    n = 2
    with h5py.File(dl / "nyu_depth_v2_labeled.mat", "w") as f:
        f["images"] = np.zeros((n, 3, 640, 480), dtype=np.uint8)
        f["depths"] = np.ones((n, 640, 480), dtype=np.float32)
        refs = f.create_dataset("scenes", (1, n), dtype=h5py.ref_dtype)
        for i, name in enumerate(["kitchen_0001a", "kitchen_0001b"]):          # same scene id on both sides
            refs[0, i] = f.create_dataset(f"refs/s{i}", data=np.array([[ord(c)] for c in name], dtype=np.uint16)).ref
    savemat(dl / "splits.mat", {"testNdxs": np.array([[2]])})
    with pytest.raises(SystemExit, match="kitchen_0001"):
        nyu.cmd_extract_labeled(None)


def test_val_scene_does_not_depend_on_other_scenes():
    ids = [f"bedroom_{n:04d}" for n in range(1, 465)]
    val = {i for i in ids if nyu.is_val_scene(i)}
    assert 5 <= len(val) <= 45
    assert {i for i in ids[:100] + ids[200:] if nyu.is_val_scene(i)} == val - set(ids[100:200])


def test_frames_excludes_test_and_val_scenes(tmp_path, monkeypatch):
    monkeypatch.setenv("LG_DATA_ROOT", str(tmp_path))
    keep, held = _val_names()
    for scene in ["kitchen_0001a", keep, held]:
        d = tmp_path / "nyu/raw" / scene
        d.mkdir(parents=True)
        for k in range(25):
            (d / f"r-{k:04d}.ppm").write_bytes(b"")
    (tmp_path / "splits").mkdir()
    (tmp_path / "splits/nyu_test_scenes.txt").write_text("kitchen_0001\n")
    import argparse
    nyu.cmd_frames(argparse.Namespace(stride=10))
    lines = (tmp_path / "splits/nyu_raw_frames_stride10.txt").read_text().split()
    _, _, val = nyu.raw_scenes()
    assert val == {held} and len(lines) == 3                                   # 2 non-test scenes, 1 held out for validation
    assert not any("kitchen" in ln or next(iter(val)) in ln for ln in lines)


def test_triplets_split_scenes_and_skip_gaps(tmp_path, monkeypatch):
    monkeypatch.setenv("LG_DATA_ROOT", str(tmp_path))
    keep, held = _val_names()
    for scene in ["kitchen_0001a", keep, held]:
        d = tmp_path / "nyu/raw" / scene
        d.mkdir(parents=True)
        times = [k * 0.04 for k in range(41)]                                   # 25 FPS
        if scene == keep:
            times = times[:20] + [t + 30.0 for t in times[20:]]                  # a sync jump after frame 19
        for k, t in enumerate(times):
            (d / f"r-{1000 + t:.6f}-{k:010d}.ppm").write_bytes(b"")
    (tmp_path / "splits").mkdir()
    (tmp_path / "splits/nyu_test_scenes.txt").write_text("kitchen_0001\n")
    import argparse
    nyu.cmd_triplets(argparse.Namespace(step=10, max_gap=1.0))
    val = (tmp_path / "splits/nyu_raw_val_scenes.txt").read_text().split()
    train = (tmp_path / "splits/nyu_raw_triplets_train.txt").read_text().split("\n")[:-1]
    valt = (tmp_path / "splits/nyu_raw_triplets_val.txt").read_text().split("\n")[:-1]
    assert val == [held]                                                        # test scene never validated
    rows = train + valt
    assert not any("kitchen" in r for r in rows) and all(len(r.split()) == 3 for r in rows)
    assert all((val[0] in r) for r in valt) and not any(val[0] in r for r in train)
    # held scene: targets 10, 20, 30 -> 3 triplets; kept scene: only the target whose sources stay on one side of the jump
    assert len(valt) == 3 and len(train) == 1
