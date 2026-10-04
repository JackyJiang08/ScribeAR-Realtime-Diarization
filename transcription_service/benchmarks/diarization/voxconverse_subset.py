"""
VoxConverse test subset for the diarization benchmark: meetings "in the
wild" (YouTube debates, news panels) with speaker counts from one to eight,
to complement the AMI meetings (always four people in a room).

Source: VoxConverse v0.3 (Chung et al., Interspeech 2020), annotations from
https://github.com/joonson/voxconverse (test/), audio from the test-set
archive https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_test_wav.zip
(4.3 GB). Both are released for research under CC BY 4.0; the copyright of
the recordings stays with the original owners. Nothing downloaded here is
ever committed: everything lands in the gitignored data/ folder.

Selection (`--select`, deterministic): for every speaker count from 1 to 8,
the longest test file of at least 8 minutes (ties by name). Each is cropped
to its first 10 minutes like the AMI meetings; the reference speaker count
inside the crop can be lower than the whole file's.

Download (`--fetch`): only the selected members are read from the archive,
through HTTP range requests (the server supports them), so about 170 MB is
transferred instead of 4.3 GB. Falls back to a plain full download with
`--full-zip <path>` when range requests are unavailable.

Usage (from transcription_service/):
    uv run python benchmarks/diarization/voxconverse_subset.py --select
    uv run python benchmarks/diarization/voxconverse_subset.py --fetch
"""

# pylint: disable=missing-function-docstring

import argparse
import io
import json
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ami_download import crop_turns  # noqa: E402
from bench_common import (  # noqa: E402
    BENCH_DIR,
    DATA_DIR,
    rttm_turns,
    write_rttm,
    write_uem,
)

ANNOTATIONS_REPO = "https://github.com/joonson/voxconverse.git"
TEST_ZIP = (
    "https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/"
    "voxconverse_test_wav.zip"
)
MANIFEST = BENCH_DIR / "voxconverse_subset.json"
VOX_DIR = DATA_DIR / "voxconverse"
ANNOTATIONS = VOX_DIR / "annotations"
MIN_DURATION_SEC = 480.0
MINUTES = 10


class HttpRangeFile(io.RawIOBase):
    """
    Read-only, seekable file over an HTTP resource that honours Range
    requests, with a read-ahead block so zipfile's small reads do not
    each become a request
    """

    def __init__(self, url: str, block: int = 8 << 20):
        super().__init__()
        self.url = url
        self.block = block
        self.pos = 0
        self._buf = b""
        self._buf_start = 0
        with urllib.request.urlopen(
            urllib.request.Request(url, method="HEAD"), timeout=60
        ) as response:
            if response.headers.get("Accept-Ranges", "none") == "none":
                raise OSError("server does not accept range requests")
            self.size = int(response.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def _fetch(self, start: int, end: int) -> bytes:
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={start}-{end - 1}"}
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.read()

    def read(self, size=-1):
        if size is None or size < 0:
            size = self.size - self.pos
        end = min(self.size, self.pos + size)
        if end <= self.pos:
            return b""
        buf_end = self._buf_start + len(self._buf)
        if not (self._buf_start <= self.pos and end <= buf_end):
            fetch_end = min(self.size, max(end, self.pos + self.block))
            self._buf = self._fetch(self.pos, fetch_end)
            self._buf_start = self.pos
        chunk = self._buf[self.pos - self._buf_start : end - self._buf_start]
        self.pos += len(chunk)
        return chunk

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def ensure_annotations() -> Path:
    if not (ANNOTATIONS / "test").is_dir():
        ANNOTATIONS.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "-q",
                ANNOTATIONS_REPO,
                str(ANNOTATIONS),
            ],
            check=True,
        )
    return ANNOTATIONS / "test"


def survey(test_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(test_dir.glob("*.rttm")):
        turns = rttm_turns(path)
        rows.append(
            {
                "id": path.stem,
                "speakers": len({t[2] for t in turns}),
                "duration_sec": round(
                    max((t[1] for t in turns), default=0.0), 1
                ),
                "speech_sec": round(sum(t[1] - t[0] for t in turns), 1),
            }
        )
    return rows


def select(rows: list[dict]) -> list[dict]:
    chosen = []
    for count in range(1, 9):
        candidates = [
            r
            for r in rows
            if r["speakers"] == count and r["duration_sec"] >= MIN_DURATION_SEC
        ]
        candidates.sort(key=lambda r: (-r["duration_sec"], r["id"]))
        if candidates:
            chosen.append(candidates[0])
    return chosen


def write_manifest(chosen: list[dict]) -> None:
    manifest = {
        "_comment": [
            "VoxConverse test subset for the diarization standard set.",
            "Audio is never committed: voxconverse_subset.py --fetch reads",
            "only these members from the test archive (CC BY 4.0, research",
            "use; copyright stays with the video owners) and crops each to",
            "its first 10 minutes with the reference RTTM from",
            "https://github.com/joonson/voxconverse (test/).",
            "Selection: for every speaker count 1..8 the longest test file",
            "of at least 8 minutes (ties by name); 'speakers' is the whole",
            "file's count, the first 10 minutes can hold fewer.",
        ],
        "sources": {
            "audio": TEST_ZIP,
            "annotations": ANNOTATIONS_REPO + " (test/*.rttm, v0.3)",
            "license": "CC BY 4.0 (research), https://www.robots.ox.ac.uk/~vgg/data/voxconverse/",
        },
        "crop_minutes": MINUTES,
        "files": chosen,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def crop(wav_in: Path, member_id: str, test_dir: Path) -> None:
    limit = MINUTES * 60.0
    wav_out = VOX_DIR / f"{member_id}_{MINUTES}min.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(wav_in),
            "-t",
            f"{limit:.3f}",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            str(wav_out),
        ],
        check=True,
    )
    turns = crop_turns(rttm_turns(test_dir / f"{member_id}.rttm"), 0.0, limit)
    stem = wav_out.stem
    write_rttm(VOX_DIR / f"{stem}.rttm", turns, stem)
    duration = min(limit, _wav_seconds(wav_out))
    write_uem(VOX_DIR / f"{stem}.uem", 0.0, duration, stem)
    print(
        f"{member_id}: {len(turns)} turns, "
        f"{len({t[2] for t in turns})} speakers in the first {MINUTES} min"
    )


def _wav_seconds(path: Path) -> float:
    import soundfile as sf

    info = sf.info(str(path))
    return info.frames / info.samplerate


def fetch(chosen: list[dict], full_zip: Path | None, force: bool) -> None:
    VOX_DIR.mkdir(parents=True, exist_ok=True)
    raw_dir = VOX_DIR / "raw"
    raw_dir.mkdir(exist_ok=True)
    test_dir = ensure_annotations()
    wanted = {r["id"] for r in chosen}
    missing = {
        i
        for i in wanted
        if force or not (VOX_DIR / f"{i}_{MINUTES}min.wav").exists()
    }
    if not missing:
        print("every selected file is prepared")
        return
    source = full_zip.open("rb") if full_zip else HttpRangeFile(TEST_ZIP)
    with zipfile.ZipFile(source) as archive:
        members = {
            Path(n).stem: n for n in archive.namelist() if n.endswith(".wav")
        }
        for member_id in sorted(missing):
            name = members.get(member_id)
            if name is None:
                raise SystemExit(f"{member_id}.wav not in the archive")
            raw = raw_dir / f"{member_id}.wav"
            if not raw.exists():
                print(
                    f"fetching {name} ({archive.getinfo(name).file_size >> 20} MB)",
                    flush=True,
                )
                with archive.open(name) as src, raw.open("wb") as dst:
                    while True:
                        block = src.read(4 << 20)
                        if not block:
                            break
                        dst.write(block)
            crop(raw, member_id, test_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--select",
        action="store_true",
        help="choose the subset and write the manifest",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="download and crop the manifest's files",
    )
    parser.add_argument(
        "--full-zip", default=None, help="already downloaded test archive"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.select:
        rows = survey(ensure_annotations())
        chosen = select(rows)
        write_manifest(chosen)
        for row in chosen:
            print(
                f"{row['id']}  speakers {row['speakers']}  {row['duration_sec']:.0f} s"
            )
        print(f"wrote {MANIFEST.relative_to(BENCH_DIR.parents[1])}")
    if args.fetch:
        if not MANIFEST.exists():
            raise SystemExit("run --select first")
        chosen = json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]
        fetch(
            chosen, Path(args.full_zip) if args.full_zip else None, args.force
        )
    if not (args.select or args.fetch):
        parser.print_help()


if __name__ == "__main__":
    main()
