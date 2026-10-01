"""Per-step runtime, peak RAM and parallelization strategy for the pipelines.

Used by the `00_main.py` drivers: `run_and_measure` replaces a plain
`subprocess.run` and samples the child's whole process tree while it works, so
running the pipeline the usual way produces the report as a side effect.

Peak RAM is the maximum summed RSS over the step's process tree. The `loky`
backend spawns fresh interpreters instead of forking, so the workers hold
independent copies of the data and the sum is not double-counting shared pages.
The figure therefore scales with the number of workers, which is why the report
records the declared parallelization strategy next to it, together with the
process count actually observed.
"""

import json
import platform
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

try:
    import psutil
except ImportError:  # pragma: no cover - reported once, measurement degrades
    psutil = None

MB = 1024**2
GB = 1024**3
SAMPLE_INTERVAL = 0.5


# --------------------------------------------------------------------------
# static scan: what parallelization does a step script declare?
# --------------------------------------------------------------------------

_PARALLEL = re.compile(r"Parallel\s*\(([^)]*)\)", re.S)
_NJOBS = re.compile(r"n_jobs\s*=\s*([^,)]+)")
_BACKEND = re.compile(r"backend\s*=\s*[\"']([^\"']+)[\"']")


def parallel_strategy(script: Path) -> str:
    """Read the joblib configuration out of a step script.

    Returns a human-readable string such as ``loky, n_jobs=min(50, cpu_count)``
    or ``sequential`` when the script never calls ``Parallel``.
    """
    try:
        src = Path(script).read_text(encoding="utf-8")
    except OSError:
        return "unknown"

    # constants like `N_JOBS = min(50, os.cpu_count() or 1)`
    consts = dict(
        re.findall(r"^\s*(N_JOBS[A-Z_]*)\s*=\s*(.+?)\s*(?:#.*)?$", src, re.M)
    )

    found = []
    for call in _PARALLEL.findall(src):
        backend = _BACKEND.search(call)
        n_jobs = _NJOBS.search(call)
        if n_jobs is None:
            continue
        expr = n_jobs.group(1).strip()
        expr = consts.get(expr, expr)
        expr = expr.replace("os.cpu_count() or 1", "cpu_count")
        expr = expr.replace("os.cpu_count()", "cpu_count")
        found.append(f"{backend.group(1) if backend else 'default'}, n_jobs={expr}")

    if not found:
        return "sequential"
    return "; ".join(dict.fromkeys(found))  # de-duplicate, keep order


# --------------------------------------------------------------------------
# dynamic measurement
# --------------------------------------------------------------------------


def _sample(proc, record, interval):
    """Poll the child's process tree until it exits, recording the maxima."""
    try:
        root = psutil.Process(proc.pid)
    except psutil.NoSuchProcess:
        return
    while proc.poll() is None:
        procs = [root]
        try:
            procs += root.children(recursive=True)
        except psutil.NoSuchProcess:
            break
        rss = 0
        alive = 0
        for p in procs:
            try:
                rss += p.memory_info().rss
                alive += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        record["peak_ram_mb"] = max(record["peak_ram_mb"], rss / MB)
        record["max_processes"] = max(record["max_processes"], alive)
        time.sleep(interval)


def _remove_quietly(path: Path, attempts: int = 6, delay: float = 0.5) -> None:
    """Delete a file, tolerating a Windows lock held by lingering workers.

    The step's stderr handle is inherited by its `loky` workers, which can
    outlive the parent by a moment, so the log is occasionally still locked
    when the step returns. Retry briefly and then give up: a leftover log is
    harmless, but an exception here would abort the whole pipeline.
    """
    for attempt in range(attempts):
        try:
            path.unlink(missing_ok=True)
            return
        except OSError:
            if attempt == attempts - 1:
                return
            time.sleep(delay)


def run_and_measure(file: str, folder: Path, env: dict, args=None) -> dict:
    """Run one pipeline step as a subprocess and measure it.

    Drop-in for `subprocess.run([sys.executable, path], env=env)`; returns a
    record with wall time, peak RAM, observed process count and the declared
    parallelization strategy.
    """
    args = list(args or [])
    path = Path(folder) / file
    label = f"{file} {' '.join(args)}".strip()

    record = {
        "step": label,
        "strategy": parallel_strategy(path),
        "peak_ram_mb": 0.0,
        "max_processes": 0,
    }

    # stdout still streams to the console. stderr goes to a file so that a
    # traceback is not lost when the console does not capture the child's
    # output (as in Spyder on Windows); it is echoed below if the step fails.
    err_path = Path(folder) / f".stderr_{Path(file).stem}.log"
    try:
        err = open(err_path, "w", encoding="utf-8", errors="replace")
    except OSError:
        # Same lock as above, seen from the other side: a previous run's log is
        # still held open. Write next to it rather than refuse to start.
        err_path = err_path.with_suffix(f".{int(time.time())}.log")
        err = open(err_path, "w", encoding="utf-8", errors="replace")

    t0 = time.perf_counter()
    with err:
        proc = subprocess.Popen([sys.executable, str(path), *args], env=env, stderr=err)

        watcher = None
        if psutil is not None:
            watcher = threading.Thread(
                target=_sample, args=(proc, record, SAMPLE_INTERVAL), daemon=True
            )
            watcher.start()

        proc.wait()
        if watcher is not None:
            watcher.join(timeout=2 * SAMPLE_INTERVAL)

    record["seconds"] = round(time.perf_counter() - t0, 1)
    record["peak_ram_mb"] = round(record["peak_ram_mb"], 1)
    record["returncode"] = proc.returncode

    if proc.returncode != 0:
        print(f"File {label} did not run successfully (exit {proc.returncode}).")
        try:
            stderr = err_path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            stderr = ""
        if stderr:
            tail = stderr.splitlines()[-40:]
            print("--- stderr (last 40 lines) " + "-" * 40)
            print("\n".join(tail))
            print("-" * 67)
            print(f"full stderr: {err_path}")
        record["error"] = stderr.splitlines()[-1] if stderr else "no stderr captured"
    else:
        _remove_quietly(err_path)
        print(
            f"Successfully ran {label}. "
            f"[{fmt_time(record['seconds'])}, peak {record['peak_ram_mb'] / 1024:.2f} GB, "
            f"{record['max_processes']} processes]"
        )
    return record


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def machine_info() -> dict:
    info = {
        "host": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor(),
        "python": sys.version.split()[0],
        "started": datetime.now().isoformat(timespec="seconds"),
    }
    if psutil is not None:
        vm = psutil.virtual_memory()
        info.update(
            cores_physical=psutil.cpu_count(logical=False),
            cores_logical=psutil.cpu_count(logical=True),
            ram_total_gb=round(vm.total / GB, 1),
            ram_available_gb_at_start=round(vm.available / GB, 1),
        )
    return info


def fmt_time(seconds) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.2f} h"


def render(report: dict) -> tuple:
    """Render a report dict as (README section, full report file).

    The section is the machine and the step table; the report file adds the
    notes, so that two spliced blocks in one README do not repeat them.
    """
    m = report["machine"]
    records = report["steps"]
    cores = (
        f"{m.get('cores_physical', '?')} physical / {m.get('cores_logical', '?')} logical cores"
    )
    lines = [
        f"- Machine: {m['host']}, {m['os']}",
        f"- CPU: {m['cpu']} ({cores})",
        f"- RAM: {m.get('ram_total_gb', '?')} GB total, "
        f"{m.get('ram_available_gb_at_start', '?')} GB free at start",
        f"- Python {m['python']}, run started {m['started']}",
        "",
        "| Step | Wall time | Peak RAM | Parallelization | Processes seen |",
        "|---|---:|---:|---|---:|",
    ]
    for r in records:
        lines.append(
            f"| `{r['step']}` | {fmt_time(r['seconds'])} | "
            f"{r['peak_ram_mb'] / 1024:.2f} GB | {r['strategy']} | "
            f"{r.get('max_processes', '')} |"
        )
    lines.append(f"| **Total** | **{fmt_time(report['total_seconds'])}** | | | |")
    section = "\n".join(lines) + "\n"

    lines += [
        "",
        "Wall time and peak RAM are measured; the parallelization column is read "
        "from the step scripts, and the last column is the process count actually "
        "observed. `cpu_count` resolves to the cores of the machine "
        "above, so a step requesting `min(50, cpu_count)` runs with fewer workers "
        "-- and proportionally less memory -- on a smaller machine.",
        "",
        "Peak RAM is the maximum summed resident set size over the step's process "
        "tree. `loky` spawns fresh interpreters rather than forking, so workers "
        "hold independent copies of the data and the sum does not double-count "
        "shared pages.",
    ]
    markdown = f"# Runtime report — {report['pipeline']}\n\n" + "\n".join(lines) + "\n"
    return section, markdown


def readme_markers(name: str) -> tuple:
    """Start and end marker that delimit a pipeline's block in the README."""
    return (
        f"<!-- runtime-report:{name}:start -->",
        f"<!-- runtime-report:{name}:end -->",
    )


def update_readme(readme: Path, name: str, section: str) -> bool:
    """Replace the `name` block of `readme` with `section`.

    The README keeps the measured numbers between HTML comment markers, so the
    resource requirements are refreshed by re-running the pipeline rather than
    copied by hand. A README without the markers is left untouched: the report
    files are written either way.
    """
    readme = Path(readme)
    start, end = readme_markers(name)
    try:
        text = readme.read_text(encoding="utf-8")
    except OSError:
        print(f"  README not found at {readme}; skipped.")
        return False

    i, j = text.find(start), text.find(end)
    if i == -1 or j == -1 or j < i:
        print(f"  No {start} ... {end} block in {readme}; skipped.")
        return False

    new = text[: i + len(start)] + "\n" + section + text[j:]
    if new != text:
        readme.write_text(new, encoding="utf-8")
    return True


def write_report(records, out_dirs, name: str, machine=None, readme=None) -> list:
    """Write `runtime_report_<name>.md` and `.json` into each of `out_dirs`.

    Accepts a single path or an iterable of them, so the report can sit both
    next to the code and in the results folder that is handed on with the
    figures and tables. With `readme` given, the same table is spliced into that
    file's marker block, so the README states the resource requirements of the
    most recent run without anybody editing it. Returns the markdown paths.
    """
    if isinstance(out_dirs, (str, Path)):
        out_dirs = [out_dirs]
    out_dirs = [Path(d) for d in out_dirs]

    machine = machine or machine_info()
    total = round(sum(r["seconds"] for r in records), 1)

    report = {
        "pipeline": name,
        "total_seconds": total,
        "machine": machine,
        "steps": records,
    }

    section, markdown = render(report)

    written = []
    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"runtime_report_{name}.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        md = out_dir / f"runtime_report_{name}.md"
        md.write_text(markdown, encoding="utf-8")
        written.append(md)

    if readme is not None:
        update_readme(readme, name, section)
    return written


# --------------------------------------------------------------------------
# CLI: adopt a report produced elsewhere
# --------------------------------------------------------------------------
# A run on a compute server writes its report into that machine's BASE_PATH.
# Point this at the resulting .json to copy it into reports/ and refresh the
# README, so the numbers in the package are the numbers of the real run:
#
#   python -m online_copula_experiments.runtime_monitor <runtime_report_*.json>


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        print("usage: python -m online_copula_experiments.runtime_monitor "
              "<runtime_report_*.json> [...]")
        return 2

    pkg_dir = Path(__file__).resolve().parent
    for arg in argv:
        report = json.loads(Path(arg).read_text(encoding="utf-8"))
        name = report["pipeline"]
        section, markdown = render(report)

        out_dir = pkg_dir / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"runtime_report_{name}.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        (out_dir / f"runtime_report_{name}.md").write_text(markdown, encoding="utf-8")
        update_readme(pkg_dir / "README.md", name, section)
        print(f"Adopted {name} report from {arg}: "
              f"reports/ updated, README refreshed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
