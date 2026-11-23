"""Third-party code, never modified.

* ``flexdepth/``, ``zipdepth/``, ``bore_depth/``: baselines as git submodules at pinned commits.
* ``meter_reference/``: the three course-provided METER files (architecture, loss, augmentation),
  locked and kept byte-identical; the working METER code is ``models/meter.py``.
* ``loader.py``: imports the baselines in isolation (see its docstring).
"""
