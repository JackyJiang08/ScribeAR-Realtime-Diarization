"""
The evaluation sets of the diarization benchmark.

- `dev`: the three AMI meetings of the Phase 2 baselines (fast loop).
- `standard`: every AMI test-set meeting (pyannote/AMI-diarization-setup
  `lists/test.meetings.txt`: 16 meetings, single distant microphone
  Array1-01, first 10 minutes, only_words references) plus the VoxConverse
  test subset of voxconverse_subset.json (speaker counts 1 to 8).
- `ami`: the 16 AMI meetings alone.

Every set is a list of WAV paths under data/; references sit beside each
WAV with the same stem (.rttm, .uem). Audio is never committed.
"""

import json
from pathlib import Path

from bench_common import BENCH_DIR, DATA_DIR

SUFFIX = "_10min"
DEV_MEETINGS = ["ES2004a", "IS1009a", "TS3003a"]
AMI_TEST_MEETINGS = [
    "IS1009a",
    "IS1009b",
    "IS1009c",
    "IS1009d",
    "ES2004a",
    "ES2004b",
    "ES2004c",
    "ES2004d",
    "TS3003a",
    "TS3003b",
    "TS3003c",
    "TS3003d",
    "EN2002a",
    "EN2002b",
    "EN2002c",
    "EN2002d",
]
SETS = ("dev", "ami", "standard")


def voxconverse_ids() -> list[str]:
    """
    Ids of the VoxConverse subset manifest, in manifest order
    """
    manifest = BENCH_DIR / "voxconverse_subset.json"
    if not manifest.exists():
        return []
    return [
        f["id"]
        for f in json.loads(manifest.read_text(encoding="utf-8"))["files"]
    ]


def set_wavs(name: str, data_dir: Path | None = None) -> list[Path]:
    """
    WAV paths of a named set (`dev`, `ami`, `standard`). Missing files are
    returned too so the caller can report what to prepare
    """
    ami = data_dir or DATA_DIR / "ami"
    vox = DATA_DIR / "voxconverse"
    if name == "dev":
        return [ami / f"{m}{SUFFIX}.wav" for m in DEV_MEETINGS]
    if name == "ami":
        return [ami / f"{m}{SUFFIX}.wav" for m in AMI_TEST_MEETINGS]
    if name == "standard":
        return [ami / f"{m}{SUFFIX}.wav" for m in AMI_TEST_MEETINGS] + [
            vox / f"{i}{SUFFIX}.wav" for i in voxconverse_ids()
        ]
    raise ValueError(f"unknown set {name!r}; one of {SETS}")


def missing_in(wavs: list[Path]) -> list[Path]:
    """
    The WAVs of a set that are not prepared yet
    """
    return [w for w in wavs if not w.exists()]
