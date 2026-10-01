"""Check what the interpreter actually loaded, before running the study.

A file on disk is not proof of what Python imports: a stale .pyc, a
non-editable install in site-packages, or a second copy earlier on sys.path
all leave the source tree looking correct while the imported module is not.
Everything below is therefore read off the loaded objects with
``inspect.getsource``, not off the checkout.

    python verify_install.py

Exits 0 if every check passes, 1 otherwise.
"""

import importlib.metadata
import inspect
import subprocess
import sys
from pathlib import Path

OK, BAD = "  ok   ", "  FAIL "
failures = []


def check(label, condition, detail=""):
    print((OK if condition else BAD) + label + (f"  {detail}" if detail else ""))
    if not condition:
        failures.append(label)


def submodule_sha(path):
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip() or "?"
    except Exception:
        return "?"


print("=" * 66)
print("Environment")
print("=" * 66)
print(f"  python      {sys.version.split()[0]}")
print(f"  executable  {sys.executable}")

import numpy as np
import pandas as pd

print(f"  numpy       {np.__version__}")
print(f"  pandas      {pd.__version__}")
try:
    cow = pd.options.mode.copy_on_write
except Exception:
    cow = "n/a (pandas 3: always on)"
print(f"  copy_on_write {cow}")

print()
print("=" * 66)
print("Packages: where they were imported from")
print("=" * 66)

import ondil
import vinecopulas

root = Path(__file__).resolve().parent
for name, mod in (("ondil", ondil), ("vinecopulas", vinecopulas)):
    path = Path(mod.__file__).resolve()
    try:
        version = importlib.metadata.version(name)
    except Exception:
        version = "?"
    inside = root in path.parents
    print(f"  {name:<12} {version:<10} {path}")
    check(f"{name} imported from this checkout (editable install)", inside)

print(f"  ondil submodule HEAD       {submodule_sha(root / 'ondil')}")
print(f"  VineCopulas submodule HEAD {submodule_sha(root / 'VineCopulas')}")

print()
print("=" * 66)
print("Source-level checks on the loaded objects")
print("=" * 66)

from vinecopulas.vinecopula import fit_vinecopstructure

src = inspect.getsource(fit_vinecopstructure)
check("fit_vinecopstructure sets the used flag with .loc",
      'order.loc[inde, "used"]' in src)
check("fit_vinecopstructure has no chained assignment",
      'order["used"][inde] =' not in src)

from ondil.estimators import MultivariateOnlineDistributionalRegressionPath as Est

est_src = inspect.getsource(Est)
check("estimator uses the observed-information weight",
      "weights = -(" in est_src)
check("estimator falls back to the expected information",
      "_expected_information" in est_src)

print()
print("=" * 66)
print("Behavioural check: one pair-copula per vine edge")
print("=" * 66)

from scipy import stats as st

A = np.array([
    [1, 2, 3, 4, 4],
    [2, 3, 4, 3, np.nan],
    [3, 4, 2, np.nan, np.nan],
    [4, 1, np.nan, np.nan, np.nan],
    [0, np.nan, np.nan, np.nan, np.nan],
])
RHOS, N = [0.90, 0.50, 0.80, 0.30], 2000

rng = np.random.default_rng(20260914)
z = [rng.standard_normal(N)]
for r in RHOS:
    z.append(r * z[-1] + np.sqrt(1 - r**2) * rng.standard_normal(N))
u = st.norm.cdf(np.column_stack(z))

import io as _io
import contextlib

with contextlib.redirect_stdout(_io.StringIO()):
    _, B, _, _, _ = fit_vinecopstructure(
        u1=u, copsi=[1], a=A, X_df=pd.DataFrame({"intercept": np.ones(N)}),
        online=0, printing=False, method_tree_1="ols", method_tree_2plus="ols",
        equation_tree_1={0: {0: "intercept"}},
        equation_tree_2plus={0: {0: "intercept"}},
        scale_inputs_tree_1=False, scale_inputs_tree_2plus=False,
        fit_intercept=True,
    )


def scalar(v):
    while isinstance(v, dict):
        v = v[sorted(v)[0]]
    return float(np.asarray(v, dtype=float).ravel()[0])


tree1 = [scalar(B[0, k]) for k in range(4)]
print("  fitted tree-1 parameters: " + ", ".join(f"{v:.6f}" for v in tree1))
check("the four first-tree edges are distinct",
      len({round(v, 9) for v in tree1}) == 4,
      "identical values mean the tree collapsed onto its first edge")

print()
print("=" * 66)
if failures:
    print(f"{len(failures)} CHECK(S) FAILED — do not run the study:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("All checks passed. The environment is ready.")
sys.exit(0)
