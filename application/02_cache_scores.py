"""Lightweight score cache for the table-only plotting path.

Each `result_*.pkl` is around 650 MB, because it carries the per-step
coefficient, family and prediction stores that only the figures need. The
tables need just `df_scores`, `means` and `all_update_times`, a few megabytes in
total. This script copies those fields into `scores_*.pkl` next to the originals,
so that `TABLES_ONLY=1 python 03_plots.py` runs in seconds instead of minutes.

Run after the copula scripts, and again whenever they are re-run:

    python 02_cache_scores.py
"""

import pickle
from pathlib import Path

from online_copula_experiments.application.const_and_helper.constants import (
    APP_DIR,
    SCHEMES,
)

# Fields the table sections use. Everything else is dropped. The three timing
# fields are what the computation-times table needs: the schemes that re-fit
# monthly record fit_times/update_times, the fully online one records a single
# all_update_times whose first element is the initial batch fit.
KEEP = ("setting_name", "setting", "mode", "means", "df_scores",
        "fit_times", "update_times", "all_update_times")

MB = 1024**2


def cache_folder(folder: Path) -> int:
    written = 0
    for src in sorted(folder.glob("result_*.pkl")):
        dst = folder / src.name.replace("result_", "scores_", 1)
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            print(f"  {dst.name:52s} up to date")
            continue
        with open(src, "rb") as fh:
            res = pickle.load(fh)
        slim = {k: res[k] for k in KEEP if k in res}
        with open(dst, "wb") as fh:
            pickle.dump(slim, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(
            f"  {dst.name:52s} {src.stat().st_size / MB:7.1f} MB"
            f" -> {dst.stat().st_size / MB:6.2f} MB"
        )
        del res, slim
        written += 1
    return written


if __name__ == "__main__":
    total = 0
    for scheme in SCHEMES:
        folder = APP_DIR / scheme
        if not folder.is_dir() or not any(folder.glob("result_*.pkl")):
            continue
        print(f"{folder}:")
        total += cache_folder(folder)
    print(f"cached {total} result file(s)")
