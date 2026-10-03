"""
Shared helpers for the diarization benchmark harness: paths, run hygiene,
resource limits, memory readings, percentiles and reference-file loaders.

Everything here is benchmark-side instrumentation. Nothing in the service is
changed by importing it.
"""

# pylint: disable=import-outside-toplevel,broad-exception-caught

import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000

BENCH_DIR = Path(__file__).resolve().parent
ROOT = BENCH_DIR.parents[1]  # transcription_service/
DATA_DIR = BENCH_DIR / "data"
RESULTS_DIR = BENCH_DIR / "results"
CONFIGS_DIR = BENCH_DIR / "configs"
BASELINES_DIR = BENCH_DIR / "baselines"

# Load above this fraction of the CPUs the benchmark may use is flagged.
LOAD_HIGH_FRACTION = 0.75


# ---------------------------------------------------------------- generic ---


def rel_path(path) -> str:
    """
    Path as recorded in reports: relative to transcription_service/ when it
    lives there, otherwise `<outside>/<basename>`, so a committed baseline
    never carries a machine's absolute paths.
    """
    if path is None:
        return None
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return f"<outside>/{resolved.name}"


def git_rev(path: Path = ROOT) -> str:
    """Short git revision of the checkout, or 'unknown'."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return os.environ.get("SCRIBEAR_BUILD_COMMIT") or "unknown"


def git_dirty(path: Path = ROOT) -> bool | None:
    """True when the checkout has uncommitted changes, None if unknown."""
    try:
        out = subprocess.check_output(
            ["git", "-C", str(path), "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except (subprocess.CalledProcessError, OSError):
        return None


def percentile(values, q: float) -> float | None:
    """
    Nearest-rank percentile, the definition node-server's LatencyWindow and
    the monitoring sidecar use, so a p95 here means the same thing as the
    p95 on the fleet dashboard.
    """
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    rank = int(np.ceil(q * len(ordered)))
    idx = min(max(rank - 1, 0), len(ordered) - 1)
    return ordered[idx]


def summarize(values, digits: int = 3) -> dict:
    """count / mean / p50 / p95 / max of a sample, None-safe."""
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    arr = [float(v) for v in values]
    return {
        "count": len(arr),
        "mean": round(float(np.mean(arr)), digits),
        "p50": round(percentile(arr, 0.5), digits),
        "p95": round(percentile(arr, 0.95), digits),
        "max": round(max(arr), digits),
    }


def now_iso() -> str:
    """Timestamp with offset, as the existing reports use."""
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def write_json(path: Path, payload: dict) -> None:
    """Writes a pretty JSON report, creating parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def ensure_hf_token_env() -> str:
    """
    Makes sure the pyannote context's token variable is set, accepting the
    HF_TOKEN spelling the HuggingFace CLI uses. Returns the variable name.
    """
    var = "HUGGINGFACE_ACCESS_TOKEN"
    if not os.environ.get(var) and os.environ.get("HF_TOKEN"):
        os.environ[var] = os.environ["HF_TOKEN"]
    return var


# ------------------------------------------------------ resource limits ---


def _read_first(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return None


def resource_limits() -> dict:
    """
    The CPU and memory the benchmark is allowed to use, recorded in every
    report so numbers from different boxes are never compared blind.

    Inside a container the cgroup v2 files are authoritative; BENCH_CPU_LIMIT
    and BENCH_MEM_LIMIT_GB (set by docker/run_in_docker.sh) are recorded as
    the requested limits. Natively there is no limit and the machine's totals
    are reported instead.
    """
    runner = os.environ.get("SCRIBEAR_BENCH_RUNNER", "native")
    limits: dict = {
        "runner": runner,
        "requested_cpus": _float_env("BENCH_CPU_LIMIT"),
        "requested_memory_gb": _float_env("BENCH_MEM_LIMIT_GB"),
        "cgroup_cpus": None,
        "cgroup_memory_gb": None,
        "host_cpus": os.cpu_count(),
        "host_memory_gb": round(total_memory_mb() / 1024, 2),
    }
    cpu_max = _read_first("/sys/fs/cgroup/cpu.max")
    if cpu_max and cpu_max.split()[0] != "max":
        quota, period = cpu_max.split()[:2]
        limits["cgroup_cpus"] = round(float(quota) / float(period), 2)
    mem_max = _read_first("/sys/fs/cgroup/memory.max")
    if mem_max and mem_max != "max":
        limits["cgroup_memory_gb"] = round(int(mem_max) / 1024**3, 2)
    limits["effective_cpus"] = (
        limits["cgroup_cpus"] or limits["requested_cpus"] or limits["host_cpus"]
    )
    limits["effective_memory_gb"] = (
        limits["cgroup_memory_gb"]
        or limits["requested_memory_gb"]
        or limits["host_memory_gb"]
    )
    return limits


def _float_env(name: str) -> float | None:
    value = os.environ.get(name)
    try:
        return float(value) if value else None
    except ValueError:
        return None


def baseline_key(limits: dict | None = None) -> str:
    """
    Name of the committed baseline a report should be gated against:
    `linux-cpu-<cpus>c<mem>g` inside the container, `native-<os>-<arch>`
    otherwise.
    """
    limits = limits or resource_limits()
    if limits["runner"] == "docker":
        cpus = limits["effective_cpus"]
        mem = limits["effective_memory_gb"]
        cpus_txt = str(int(cpus)) if float(cpus).is_integer() else str(cpus)
        mem_txt = str(int(round(mem)))
        return f"linux-cpu-{cpus_txt}c{mem_txt}g"
    return f"native-{platform.system().lower()}-{platform.machine()}"


# ------------------------------------------------------------- hygiene ---


def total_memory_mb() -> float:
    """Physical memory of the machine (or VM) in MB."""
    if sys.platform == "darwin":
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"])
            return int(out) / 1024**2
        except (subprocess.CalledProcessError, OSError, ValueError):
            return 0.0
    meminfo = _read_first("/proc/meminfo") or ""
    for line in meminfo.splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) / 1024
    return 0.0


def _darwin_memory() -> tuple[float, float]:
    """(free_mb, swap_used_mb) on macOS from vm_stat and sysctl."""
    free_mb = 0.0
    swap_used_mb = 0.0
    try:
        out = subprocess.check_output(["vm_stat"], text=True)
        page_size = 4096
        counts: dict[str, int] = {}
        for line in out.splitlines():
            if "page size of" in line:
                page_size = int(line.split("page size of")[1].split()[0])
            elif ":" in line:
                key, value = line.split(":", 1)
                value = value.strip().rstrip(".")
                if value.isdigit():
                    counts[key.strip()] = int(value)
        # "free" + "inactive" + "speculative" is what the OS can hand out
        # without swapping; "free" alone is near zero on any busy Mac.
        free_pages = (
            counts.get("Pages free", 0)
            + counts.get("Pages inactive", 0)
            + counts.get("Pages speculative", 0)
        )
        free_mb = free_pages * page_size / 1024**2
    except (subprocess.CalledProcessError, OSError):
        pass
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "vm.swapusage"], text=True
        )
        # "total = 18432.00M  used = 16922.56M  free = 1509.44M  (encrypted)"
        for part in out.replace("  ", " ").split("M"):
            if "used =" in part:
                swap_used_mb = float(part.split("used =")[1].strip())
    except (subprocess.CalledProcessError, OSError, ValueError):
        pass
    return free_mb, swap_used_mb


def _linux_memory() -> tuple[float, float]:
    """(free_mb, swap_used_mb) on Linux from /proc/meminfo (+ cgroup)."""
    values: dict[str, int] = {}
    for line in (_read_first("/proc/meminfo") or "").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1].isdigit():
            values[parts[0].rstrip(":")] = int(parts[1])
    free_mb = values.get("MemAvailable", values.get("MemFree", 0)) / 1024
    swap_used_mb = (
        values.get("SwapTotal", 0) - values.get("SwapFree", 0)
    ) / 1024
    # Inside a cgroup the container's own swap usage is the one that matters.
    cg_swap = _read_first("/sys/fs/cgroup/memory.swap.current")
    if cg_swap and cg_swap.isdigit():
        swap_used_mb = max(swap_used_mb, int(cg_swap) / 1024**2)
    cg_max = _read_first("/sys/fs/cgroup/memory.max")
    cg_cur = _read_first("/sys/fs/cgroup/memory.current")
    if cg_max and cg_max != "max" and cg_cur and cg_cur.isdigit():
        free_mb = min(free_mb, (int(cg_max) - int(cg_cur)) / 1024**2)
    return free_mb, swap_used_mb


def system_state() -> dict:
    """Free memory, swap in use and load averages right now."""
    if sys.platform == "darwin":
        free_mb, swap_used_mb = _darwin_memory()
    else:
        free_mb, swap_used_mb = _linux_memory()
    try:
        load1, load5, load15 = os.getloadavg()
    except OSError:
        load1 = load5 = load15 = None
    return {
        "taken_at": now_iso(),
        "free_memory_mb": round(free_mb),
        "swap_used_mb": round(swap_used_mb),
        "load_1m": None if load1 is None else round(load1, 2),
        "load_5m": None if load5 is None else round(load5, 2),
        "load_15m": None if load15 is None else round(load15, 2),
    }


def hygiene_check(limits: dict | None = None) -> dict:
    """
    Records the machine state before a run and flags conditions that are
    known to pollute timings: swap in use (the audit's 17 GB of swap produced
    single ticks of 25 to 39 s that never reproduce in isolation) and a load
    average above three quarters of the CPUs the run may use.
    """
    limits = limits or resource_limits()
    state = system_state()
    warnings: list[str] = []
    if state["swap_used_mb"] and state["swap_used_mb"] > 0:
        warnings.append(
            f"swap in use: {state['swap_used_mb']} MB; timings may carry "
            "paging outliers"
        )
    cpus = float(limits.get("effective_cpus") or os.cpu_count() or 1)
    if (
        state["load_1m"] is not None
        and state["load_1m"] > LOAD_HIGH_FRACTION * cpus
    ):
        warnings.append(
            f"load average {state['load_1m']} exceeds {LOAD_HIGH_FRACTION:.0%} "
            f"of {cpus:g} CPUs; other processes are competing for the cores"
        )
    free_needed_mb = 3072
    if state["free_memory_mb"] < free_needed_mb:
        warnings.append(
            f"only {state['free_memory_mb']} MB free; a diarized worker needs "
            f"about 2 GB and the benchmark process another 1 GB"
        )
    for line in warnings:
        print(f"HYGIENE WARNING: {line}", file=sys.stderr, flush=True)
    return {"state": state, "warnings": warnings, "clean": not warnings}


# -------------------------------------------------------------- memory ---


def peak_rss_mb() -> float:
    """Peak resident set size of this process."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux kilobytes
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024


def current_rss_mb(pid: int | None = None) -> float:
    """Current resident set size of a process (default: this one)."""
    pid = pid or os.getpid()
    if sys.platform != "darwin":
        status = _read_first(f"/proc/{pid}/status") or ""
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
        return 0.0
    try:
        out = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(pid)], text=True
        )
        return int(out.strip() or 0) / 1024
    except (subprocess.CalledProcessError, OSError, ValueError):
        return 0.0


def child_pids(pid: int) -> list[int]:
    """Direct children of a process."""
    if sys.platform != "darwin":
        children: list[int] = []
        try:
            for entry in os.listdir("/proc"):
                if not entry.isdigit():
                    continue
                stat = _read_first(f"/proc/{entry}/stat")
                if not stat:
                    continue
                # pid (comm) state ppid ... - comm may contain spaces
                after = stat[stat.rfind(")") + 2 :].split()
                if len(after) > 1 and int(after[1]) == pid:
                    children.append(int(entry))
        except OSError:
            pass
        return children
    try:
        out = subprocess.check_output(["pgrep", "-P", str(pid)], text=True)
        return [int(p) for p in out.split()]
    except (subprocess.CalledProcessError, OSError, ValueError):
        return []


def process_tree_rss_mb(pid: int) -> dict:
    """RSS of a process and all of its descendants, in MB."""
    total = 0.0
    per_pid: dict[int, float] = {}
    stack = [pid]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        rss = current_rss_mb(current)
        per_pid[current] = round(rss, 1)
        total += rss
        stack.extend(child_pids(current))
    return {"total_mb": round(total, 1), "per_pid": per_pid}


# --------------------------------------------------------- environment ---


def environment_info(device: str = "cpu") -> dict:
    """Software and hardware facts recorded in every report."""
    info: dict = {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "device": device,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "resource_limits": resource_limits(),
    }
    info["baseline_key"] = baseline_key(info["resource_limits"])
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_threads"] = torch.get_num_threads()
    except Exception:
        info["torch"] = None
    try:
        import pyannote.audio

        info["pyannote_audio"] = pyannote.audio.__version__
    except Exception:
        info["pyannote_audio"] = None
    try:
        import faster_whisper

        info["faster_whisper"] = faster_whisper.__version__
    except Exception:
        info["faster_whisper"] = None
    return info


# ---------------------------------------------------------- references ---


def load_audio(path: Path) -> np.ndarray:
    """16 kHz mono float32 samples."""
    samples, rate = sf.read(str(path), dtype="float32")
    if rate != SAMPLE_RATE:
        raise SystemExit(f"{path}: expected {SAMPLE_RATE} Hz, got {rate} Hz")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return np.ascontiguousarray(samples, dtype=np.float32)


def load_rttm(path: Path):
    """Reference RTTM as a pyannote Annotation."""
    from pyannote.core import Annotation, Segment

    annotation = Annotation(uri=path.stem)
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        start, duration, speaker = float(parts[3]), float(parts[4]), parts[7]
        annotation[Segment(start, start + duration)] = speaker
    return annotation


def load_uem(path: Path, fallback_end: float):
    """Scoring region as a pyannote Timeline (whole file when no UEM)."""
    from pyannote.core import Segment, Timeline

    timeline = Timeline(uri=path.stem)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 4:
                timeline.add(Segment(float(parts[2]), float(parts[3])))
    if len(timeline) == 0:
        timeline.add(Segment(0.0, fallback_end))
    return timeline


def rttm_turns(path: Path) -> list[tuple[float, float, str]]:
    """Plain (start, end, speaker) tuples from an RTTM, sorted by start."""
    turns = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 8 and parts[0] == "SPEAKER":
            start, duration = float(parts[3]), float(parts[4])
            turns.append((start, start + duration, parts[7]))
    return sorted(turns)


def write_rttm(path: Path, turns, uri: str) -> None:
    """Writes (start, end, speaker) tuples as RTTM lines."""
    lines = [
        f"SPEAKER {uri} 1 {start:.3f} {end - start:.3f} <NA> <NA> {speaker} <NA> <NA>"
        for start, end, speaker in turns
        if end > start
    ]
    path.write_text(
        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
    )


def write_uem(path: Path, start: float, end: float, uri: str) -> None:
    """Writes a single-region UEM."""
    path.write_text(f"{uri} 1 {start:.3f} {end:.3f}\n", encoding="utf-8")
