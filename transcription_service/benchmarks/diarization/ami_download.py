"""
Shared AMI download / crop helpers for the hard-case and soak preparers.

Sources (documented in hard_cases.json and docs/speaker_diarization.md):
  audio      - AMI corpus mirror, single distant microphone Array1-01
               (AMI Meeting Corpus, CC BY 4.0)
  references - pyannote/AMI-diarization-setup, only_words RTTMs + UEMs
Nothing downloaded here is ever committed; everything lands in the
gitignored benchmarks/diarization/data/ folder.
"""

# pylint: disable=missing-function-docstring

import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

AMI_MIRROR = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
SETUP = "https://raw.githubusercontent.com/pyannote/AMI-diarization-setup/main"


def download(url: str, dest: Path, retries: int = 3) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            print(f"downloading {url}", file=sys.stderr, flush=True)
            with (
                urllib.request.urlopen(url, timeout=120) as response,
                open(tmp, "wb") as out,
            ):
                shutil.copyfileobj(response, out)
            tmp.replace(dest)
            return
        except (OSError, urllib.error.URLError) as error:
            if attempt == retries:
                raise SystemExit(
                    f"failed to download {url}: {error}"
                ) from error
            time.sleep(2 * attempt)


def fetch_meeting(data_dir: Path, meeting: str) -> tuple[Path, Path, Path]:
    """Full Array1-01 WAV, RTTM and UEM of one AMI meeting."""
    wav = data_dir / f"{meeting}.Array1-01.wav"
    rttm = data_dir / f"{meeting}.rttm"
    uem = data_dir / f"{meeting}.uem"
    download(f"{AMI_MIRROR}/{meeting}/audio/{meeting}.Array1-01.wav", wav)
    download(f"{SETUP}/only_words/rttms/test/{meeting}.rttm", rttm)
    download(f"{SETUP}/uems/test/{meeting}.uem", uem)
    return wav, rttm, uem


def crop_wav(source: Path, dest: Path, start: float, duration: float) -> None:
    """16 kHz mono PCM16 crop via ffmpeg (same command the AMI prep uses)."""
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required")
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(source),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            str(dest),
        ],
        check=True,
    )


def crop_turns(turns, start: float, end: float):
    """Reference turns intersecting [start, end), clipped and shifted."""
    out = []
    for s, e, speaker in turns:
        s2, e2 = max(s, start), min(e, end)
        if e2 > s2:
            out.append((s2 - start, e2 - start, speaker))
    return out


def read_uem(path: Path):
    regions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 4:
            regions.append((float(parts[2]), float(parts[3])))
    return regions


def crop_regions(regions, start: float, end: float):
    out = []
    for s, e in regions:
        s2, e2 = max(s, start), min(e, end)
        if e2 > s2:
            out.append((s2 - start, e2 - start))
    return out or [(0.0, end - start)]
