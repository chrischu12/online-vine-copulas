##%

from pathlib import Path

# `BASE_PATH` is the output root: every intermediate result, figure, table and
# runtime report is written below it, and the input `GSPGroup_NetD_final.RData`
# is read from it. The simulation study imports it from here as well, so this is
# the single place where the location is decided.
#
# Set it in `local_path.py` next to this file -- copy `local_path_example.py`.
# That file is git-ignored, so each machine keeps its own path and no tracked
# file has to be edited. Without it, results land in `forecasting_study/` in the
# user's home directory.

try:
    from .local_path import BASE_PATH  # noqa: F401  (git-ignored, per machine)
except ImportError:
    BASE_PATH = Path.home() / "forecasting_study"

BASE_PATH = Path(BASE_PATH)
