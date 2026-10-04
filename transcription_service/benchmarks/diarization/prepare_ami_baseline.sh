#!/usr/bin/env bash
# Download the AMI meetings used by the diarization benchmark and crop them
# (audio + reference RTTM/UEM) to the first N minutes. The default list is
# the whole AMI test set of pyannote/AMI-diarization-setup (16 meetings);
# the three Phase 2 meetings ES2004a IS1009a TS3003a are the dev subset.
#
# Source data:
#   audio      - AMI corpus mirror, single distant microphone Array1-01
#   references - pyannote/AMI-diarization-setup, "only_words" RTTMs + UEMs
#
# Output lands in benchmarks/diarization/data/ami (gitignored).
#
# Usage: benchmarks/diarization/prepare_ami_baseline.sh [minutes] [meeting ...]

set -euo pipefail

MINUTES="${1:-10}"
shift || true
MEETINGS=("$@")
if [ ${#MEETINGS[@]} -eq 0 ]; then
  MEETINGS=(IS1009a IS1009b IS1009c IS1009d ES2004a ES2004b ES2004c ES2004d
            TS3003a TS3003b TS3003c TS3003d EN2002a EN2002b EN2002c EN2002d)
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="$HERE/data/ami"
AMI_MIRROR="https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
SETUP="https://raw.githubusercontent.com/pyannote/AMI-diarization-setup/main"
SECONDS_LIMIT=$((MINUTES * 60))
SUFFIX="_${MINUTES}min"

command -v ffmpeg >/dev/null || { echo "ffmpeg is required" >&2; exit 1; }
mkdir -p "$DATA"

for meeting in "${MEETINGS[@]}"; do
  wav="$DATA/$meeting.Array1-01.wav"
  [ -s "$wav" ] || curl -sSL --retry 3 -f -o "$wav" \
    "$AMI_MIRROR/$meeting/audio/$meeting.Array1-01.wav"
  [ -s "$DATA/$meeting.rttm" ] || curl -sSL --retry 3 -f -o "$DATA/$meeting.rttm" \
    "$SETUP/only_words/rttms/test/$meeting.rttm"
  [ -s "$DATA/$meeting.uem" ] || curl -sSL --retry 3 -f -o "$DATA/$meeting.uem" \
    "$SETUP/uems/test/$meeting.uem"

  ffmpeg -v error -y -i "$wav" -t "$SECONDS_LIMIT" -ac 1 -ar 16000 \
    -acodec pcm_s16le "$DATA/$meeting$SUFFIX.wav"

  # Keep reference turns that start inside the crop, trimming any that
  # run past its end
  awk -v T="$SECONDS_LIMIT" '$4 < T {
      d = $5; if ($4 + d > T) d = T - $4;
      printf "%s %s %s %.3f %.3f %s %s %s %s %s\n", $1,$2,$3,$4,d,$6,$7,$8,$9,$10
    }' "$DATA/$meeting.rttm" > "$DATA/$meeting$SUFFIX.rttm"
  awk -v T="$SECONDS_LIMIT" '$3 < T {
      e = $4; if (e > T) e = T;
      printf "%s %s %.3f %.3f\n", $1,$2,$3,e
    }' "$DATA/$meeting.uem" > "$DATA/$meeting$SUFFIX.uem"

  echo "$meeting: $(wc -l < "$DATA/$meeting$SUFFIX.rttm" | tr -d ' ') reference turns in first $MINUTES min"
done

echo "Prepared ${#MEETINGS[@]} meetings in $DATA (suffix $SUFFIX)"
