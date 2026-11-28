# Download scripts

## NYU Depth V2 (complete): `nyu.py`
Run on the lab machine in a CPU container (no GPU booking needed); files go to
`$LG_DATA_ROOT/nyu` (`/mnt/hdd2/<user>/data/nyu`).

```bash
scripts/alcor/run.sh cpu nyu_sizes    python3 -m data.download.nyu sizes            # check every URL first
scripts/alcor/run.sh cpu nyu_download python3 -m data.download.nyu download         # ~431 GB, resumable
scripts/alcor/run.sh cpu nyu_labeled  python3 -m data.download.nyu extract-labeled  # 795 train / 654 test PNGs
scripts/alcor/run.sh cpu nyu_raw      python3 -m data.download.nyu extract-raw      # add --delete-zip to save space
scripts/alcor/run.sh cpu nyu_frames   python3 -m data.download.nyu frames --stride 10
scripts/alcor/run.sh cpu nyu_triplets python3 -m data.download.nyu triplets         # val scenes + train/val triplets
```

- `download`: labelled `.mat` (2.8 GB), official `splits.mat`, and the 37 raw parts listed on the
  NYU page. Interrupted downloads resume (HTTP range requests) and every file is checked against
  the size reported by the server: re-run the same command until it reports all files complete.
- `extract-labeled`: the 1449 labelled frames as PNGs (depth in millimetres, uint16) and the lists
  `splits/nyu_labeled_{train,val,test}.txt` and `splits/nyu_test_scenes.txt`. The official split is
  795 train / 654 test; about 10% of the train **scenes** (fixed seed, whole scenes so none is split
  between train and val) go to `val`, so train has fewer than 795 frames.
- `frames`: every k-th RGB frame of every raw scene that is **not** a test scene, for LeJEPA
  pre-training and the BatchNorm re-estimation (`splits/nyu_raw_frames_stride<k>.txt`).
- `triplets`: ~5% of the non-test raw scenes held out for validation (hash of the scene id,
  `splits/nyu_raw_val_scenes.txt`) and the frame triplets for the photometric loss and the pseudo-depth
  (`splits/nyu_raw_triplets_{train,val}.txt`; truncated raw frames are skipped, `ppm_ok`).

All lists are written to `<LG_DATA_ROOT>/splits/`, not to the repository (`data/splits/README.md`).

## KITTI: `download_kitti.py`
Raw `image_02` + annotated depth + calibration, resumable; files go to `$LG_DATA_ROOT/kitti`
(`rgb/<date>/<drive>/image_02/data/*.png`, `depth/data_depth_annotated/{train,val}/`). Already downloaded on the lab.

```bash
scripts/alcor/run.sh cpu kitti_splits python3 -m data.download.download_kitti --make_splits   # CPU, seconds
```
`--make_splits` writes the seven `kitti_*` lists to `<LG_DATA_ROOT>/splits/` (it overwrites existing ones):
the 28 Eigen test drives are excluded entirely, ~10% of the other drives go to validation (hash of the drive name),
and `kitti_eigen_test.txt` holds the 652 benchmark frames with annotated ground truth. Run on 28/11: 109 train /
14 val / 28 test drives, 27,154 train triplets.
