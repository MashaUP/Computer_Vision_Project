# Download scripts (phase 1, to be replaced)

These are the phase-1 scripts, moved here unchanged (`check_dataset.py` was `test_dataset.py`).
They will be replaced in the data step of the new protocol:

- `download_nyu.py` shuffles frames and renames them into pseudo-scenes, losing scene and order: the new
  script keeps scenes, splits validation by scene, and extracts only the StructDepth frames from NYU raw
  with HTTP range requests (the NYU server supports them).
- `download_kitti.py` (raw `image_02` + annotated depth + calibration) is the starting point for KITTI.

Datasets go to `LG_DATA_ROOT` (`/mnt/hdd2/<user>/data` on the lab machine), never into the repository.
