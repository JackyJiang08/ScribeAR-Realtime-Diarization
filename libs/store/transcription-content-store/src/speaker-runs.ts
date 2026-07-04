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
