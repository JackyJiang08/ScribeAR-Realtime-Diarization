/**
 * A run of consecutive words attributed to the same speaker. `speaker` is the
 * provider's stable session label (e.g. `spk_0`) or `null` when the words have
 * no speaker attribution (diarization disabled or inconclusive).
 */
export interface SpeakerRun {
  speaker: string | null;
  text: string;
}

/**
 * Word-level input for building speaker runs: token array plus the optional
 * speaker array aligned with it (as delivered by the transcript stream).
 */
export interface SpeakerRunInput {
  text: string[];
  speakers?: (string | null)[] | null;
}

/**
 * Groups word tokens into {@link SpeakerRun}s, merging consecutive words with
 * the same speaker. Words whose speaker entry is missing or `null` form
 * `null`-speaker runs. A `speakers` array shorter than `text` treats the
 * uncovered tail as unattributed rather than throwing.
 */
export const wordsToSpeakerRuns = (
  text: string[],
  speakers?: (string | null)[] | null,
): SpeakerRun[] => {
  const runs: SpeakerRun[] = [];
  for (let i = 0; i < text.length; i += 1) {
    const word = text[i] ?? '';
    const speaker = speakers?.[i] ?? null;
    const lastRun = runs[runs.length - 1];
    if (lastRun?.speaker === speaker) {
      lastRun.text += word;
    } else {
      runs.push({ speaker, text: word });
    }
  }
  return runs;
};

/**
 * Builds speaker runs spanning multiple sequences, merging runs across
 * sequence boundaries when the speaker stays the same. Used when committing
 * an active section into a paragraph.
 */
export const sequencesToSpeakerRuns = (
  sequences: SpeakerRunInput[],
): SpeakerRun[] => {
  const text: string[] = [];
  const speakers: (string | null)[] = [];
  for (const sequence of sequences) {
    for (let i = 0; i < sequence.text.length; i += 1) {
      text.push(sequence.text[i] ?? '');
      speakers.push(sequence.speakers?.[i] ?? null);
    }
  }
  return wordsToSpeakerRuns(text, speakers);
};

/**
 * Human-readable display name for a speaker label. Provider labels `spk_N`
 * become `Speaker N+1`; any other label is shown as-is. Shared by the caption
 * display, the translated-caption panel and the transcript export so the
 * same voice reads the same everywhere.
 */
export const formatSpeakerName = (speaker: string): string => {
  const match = /^spk_(\d+)$/.exec(speaker);
  if (match !== null) {
    return `Speaker ${(Number(match[1]) + 1).toString()}`;
  }
  return speaker;
};

/**
 * The speaker a sequence is attributed to for turn-taking decisions: the
 * label most of its words carry, or `null` when none of them has one yet.
 * Ties go to the label seen first.
 */
export const dominantSpeaker = (
  speakers?: (string | null)[] | null,
): string | null => {
  if (!speakers) return null;
  const counts = new Map<string, number>();
  for (const speaker of speakers) {
    if (speaker === null) continue;
    counts.set(speaker, (counts.get(speaker) ?? 0) + 1);
  }
  let best: string | null = null;
  let bestCount = 0;
  for (const [speaker, count] of counts) {
    if (count > bestCount) {
      best = speaker;
      bestCount = count;
    }
  }
  return best;
};

/**
 * The last attributed speaker across a list of sequences, or `null` when no
 * word in them carries a label.
 */
export const lastAttributedSpeaker = (
  sequences: SpeakerRunInput[],
): string | null => {
  for (let s = sequences.length - 1; s >= 0; s -= 1) {
    const speakers = sequences[s]?.speakers ?? [];
    for (let i = speakers.length - 1; i >= 0; i -= 1) {
      const speaker = speakers[i];
      if (speaker !== null && speaker !== undefined) return speaker;
    }
  }
  return null;
};

/**
 * Plain-text rendering of speaker runs for a transcript file: every change of
 * speaker starts a new line prefixed with the speaker's display name, words
 * with no speaker continue the current line (or stand alone when nothing is
 * attributed, which is what a diarization-off transcript looks like).
 */
export const speakerRunsToText = (runs: SpeakerRun[]): string => {
  const lines: string[] = [];
  let current = '';
  for (const run of runs) {
    if (run.speaker === null) {
      current += run.text;
      continue;
    }
    if (current.trim() !== '') lines.push(current.trim());
    current = `${formatSpeakerName(run.speaker)}: ${run.text.trim()}`;
  }
  if (current.trim() !== '') lines.push(current.trim());
  return lines.join('\n');
};
