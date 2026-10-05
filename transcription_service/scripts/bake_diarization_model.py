"""
Downloads the pyannote diarization model into a plain directory and verifies
the pipeline loads from it offline, for the diarization Docker images.

Runs at image build time with the HuggingFace token from a BuildKit secret
(a file), so the token never lands in a layer and the running container
needs neither the token nor the network. The directory holds `config.yaml`
and the `segmentation/`, `embedding/` and `plda/` folders the config refers
to by `$model`, which is how pyannote loads a local checkpoint.

Usage:
  python scripts/bake_diarization_model.py \
      --model pyannote/speaker-diarization-community-1 \
      --dest /opt/scribear/models/pyannote/speaker-diarization-community-1 \
      --token-file /run/secrets/hf_token
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

REQUIRED = ("config.yaml", "segmentation", "embedding")


def main() -> int:
    """
    Downloads, verifies and reports; exit code 0 only when the pipeline loads
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HuggingFace model id")
    parser.add_argument("--dest", required=True, help="directory to bake into")
    parser.add_argument(
        "--token-file",
        default=None,
        help="file holding the HuggingFace token (a BuildKit secret); the "
        "HUGGINGFACE_ACCESS_TOKEN / HF_TOKEN environment variables are used "
        "when absent",
    )
    args = parser.parse_args()

    token = None
    if args.token_file:
        token = Path(args.token_file).read_text(encoding="utf-8").strip()
    token = (
        token
        or os.environ.get("HUGGINGFACE_ACCESS_TOKEN")
        or os.environ.get("HF_TOKEN")
    )
    if not token:
        print(
            "a HuggingFace token is required to download the gated model: "
            "pass --token-file (the BuildKit secret) or set "
            "HUGGINGFACE_ACCESS_TOKEN; accept the model terms at "
            "https://huggingface.co/pyannote/speaker-diarization-community-1 "
            "first",
            file=sys.stderr,
        )
        return 2

    # pylint: disable=import-outside-toplevel
    from huggingface_hub import snapshot_download

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    snapshot_download(repo_id=args.model, local_dir=str(dest), token=token)
    # huggingface_hub keeps download bookkeeping next to the files; the
    # image does not need it and it would be the one place a repo id and
    # ETags linger.
    shutil.rmtree(dest / ".cache", ignore_errors=True)
    downloaded = time.perf_counter() - started

    missing = [name for name in REQUIRED if not (dest / name).exists()]
    if missing:
        print(f"download incomplete, missing {missing} in {dest}", file=sys.stderr)
        return 1

    # Load exactly the way the service will: directory only, no token, and
    # with the hub forced offline so a dependency on the network shows here.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.pop("HUGGINGFACE_ACCESS_TOKEN", None)
    os.environ.pop("HF_TOKEN", None)
    from pyannote.audio import Pipeline

    started = time.perf_counter()
    pipeline = Pipeline.from_pretrained(str(dest))
    if pipeline is None:
        print(f"Pipeline.from_pretrained({dest}) returned nothing", file=sys.stderr)
        return 1
    loaded = time.perf_counter() - started

    size_mb = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file()) / 1e6
    print(
        f"baked {args.model} into {dest}: {size_mb:.0f} MB, downloaded in "
        f"{downloaded:.1f}s, loads offline in {loaded:.1f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
