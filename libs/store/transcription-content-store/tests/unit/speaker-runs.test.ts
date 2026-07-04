import { describe, expect, it } from 'vitest';

import {
  sequencesToSpeakerRuns,
  wordsToSpeakerRuns,
} from '#src/speaker-runs.js';

describe('wordsToSpeakerRuns', () => {
  it('returns an empty array for empty input', () => {
    expect(wordsToSpeakerRuns([])).toEqual([]);
  });

  it('groups all words into one null run when speakers are missing', () => {
    expect(wordsToSpeakerRuns([' Hello', ' world'])).toEqual([
      { speaker: null, text: ' Hello world' },
    ]);
  });

  it('groups all words into one null run when speakers is null', () => {
    expect(wordsToSpeakerRuns([' Hello', ' world'], null)).toEqual([
      { speaker: null, text: ' Hello world' },
    ]);
  });

  it('merges consecutive words with the same speaker', () => {
    expect(
      wordsToSpeakerRuns(
        [' Hi', ' there', ' Hey'],
        ['spk_0', 'spk_0', 'spk_1'],
      ),
    ).toEqual([
      { speaker: 'spk_0', text: ' Hi there' },
      { speaker: 'spk_1', text: ' Hey' },
    ]);
  });

  it('keeps null-speaker words as separate runs between speakers', () => {
    expect(
      wordsToSpeakerRuns([' a', ' b', ' c'], ['spk_0', null, 'spk_0']),
    ).toEqual([
      { speaker: 'spk_0', text: ' a' },
      { speaker: null, text: ' b' },
      { speaker: 'spk_0', text: ' c' },
    ]);
  });

  it('treats a shorter speakers array as unattributed tail', () => {
    expect(wordsToSpeakerRuns([' a', ' b'], ['spk_0'])).toEqual([
      { speaker: 'spk_0', text: ' a' },
      { speaker: null, text: ' b' },
    ]);
  });
});

describe('sequencesToSpeakerRuns', () => {
  it('merges runs across sequence boundaries for the same speaker', () => {
    expect(
      sequencesToSpeakerRuns([
        { text: [' one'], speakers: ['spk_0'] },
        { text: [' two'], speakers: ['spk_0'] },
        { text: [' three'], speakers: ['spk_1'] },
      ]),
    ).toEqual([
      { speaker: 'spk_0', text: ' one two' },
      { speaker: 'spk_1', text: ' three' },
    ]);
  });

  it('handles sequences without speaker data', () => {
    expect(
      sequencesToSpeakerRuns([
        { text: [' plain'] },
        { text: [' text'], speakers: null },
      ]),
    ).toEqual([{ speaker: null, text: ' plain text' }]);
  });
});
