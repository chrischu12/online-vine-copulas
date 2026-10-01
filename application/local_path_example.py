"""Template for a machine-local output root.

Copy this file to `local_path.py` (which is git-ignored) and set the path to the
folder where the results of this package should live. `GSPGroup_NetD_final.RData`
goes into that same folder. Nothing else needs to be edited, and your path never
enters the repository.
"""

from pathlib import Path

BASE_PATH = Path(r"C:\path\to\forecasting_study")
