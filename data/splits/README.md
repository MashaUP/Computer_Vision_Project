# Split lists

The lists are **not** in this folder: the data scripts write them to the data root,
`<LG_DATA_ROOT>/splits/` (`/mnt/hdd2/<user>/data/splits/` on the lab, `/work/data/splits/` in the container),
and every `--list splits/...` / `--test-list splits/...` argument is resolved against the data root.

| Lists | Written by | Size (28/11) |
|---|---|---|
| `nyu_labeled_{train,val,test}.txt`, `nyu_test_scenes.txt` | `python -m data.download.nyu extract-labeled` | test 654 |
| `nyu_raw_frames_stride<k>.txt` | `python -m data.download.nyu frames --stride <k>` | |
| `nyu_raw_val_scenes.txt`, `nyu_raw_triplets_{train,val}.txt` | `python -m data.download.nyu triplets` | train 24,863 |
| `kitti_*` (7 files) | `python -m data.download.download_kitti --make_splits` | train 27,154, test 652 |

Test scenes and validation scenes / drives never enter LeJEPA pre-training or training
(`data.datasets.check_list` rejects lists that touch them). Counts and the leak checks: `docs/protocol.md`.
