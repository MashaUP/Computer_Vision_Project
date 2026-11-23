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
```

- `download`: labelled `.mat` (2.8 GB), official `splits.mat`, and the 37 raw parts listed on the
  NYU page. Interrupted downloads resume (HTTP range requests) and every file is checked against
  the size reported by the server: re-run the same command until it reports all files complete.
- `extract-labeled`: the 1449 labelled frames as PNGs (depth in millimetres, uint16) and the lists
  `splits/nyu_labeled_{train,test}.txt` (official split: 795 / 654) and `splits/nyu_test_scenes.txt`.
- `frames`: every k-th RGB frame of every raw scene that is **not** a test scene, for LeJEPA
  pre-training and the BatchNorm re-estimation (`splits/nyu_raw_frames_stride<k>.txt`).

## KITTI: `download_kitti.py`
Phase-1 script (raw `image_02` + annotated depth + calibration), to be adapted next.
