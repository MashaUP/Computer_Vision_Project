# Phase-1 results (course project, METER + LeJEPA)

Final numbers of the exam version (commit `52aa283`, tag `v1.0-course-project`), kept as they were produced.

- `classic/{xxs,xs,s}/`: root setup (global-pooled LeJEPA, frame-level splits, NYU depth in cm).
- `dense/xxs/`: experimental setup (4x4 token LeJEPA, scene-level splits, NYU depth in m).

Each folder has `json/` (test metrics), `csv/` (comparison tables) and `logs/` (Lightning `hparams.yaml`
and `metrics.csv` per run). Figures and checkpoints are in the tagged commit.

The two setups used different splits and units, so their numbers are not directly comparable with each
other or with the new protocol.
