import { describe, expect, it } from 'vitest';

import {
  dominantSpeaker,
  formatSpeakerName,
  lastAttributedSpeaker,
  sequencesToSpeakerRuns,
  speakerRunsToText,
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

describe('formatSpeakerName', () => {
  it('turns provider labels into one-based speaker names', () => {
    expect(formatSpeakerName('spk_0')).toBe('Speaker 1');
    expect(formatSpeakerName('spk_11')).toBe('Speaker 12');
  });

  it('passes other labels through', () => {
    expect(formatSpeakerName('Alice')).toBe('Alice');
  });
});

describe('dominantSpeaker', () => {
  it('is the label most words carry, ties to the first seen', () => {
    expect(dominantSpeaker(['spk_0', 'spk_1', 'spk_1'])).toBe('spk_1');
    expect(dominantSpeaker(['spk_0', null, 'spk_1'])).toBe('spk_0');
  });

  it('is null without any label', () => {
    expect(dominantSpeaker(null)).toBeNull();
    expect(dominantSpeaker([null, null])).toBeNull();
    expect(dominantSpeaker(undefined)).toBeNull();
  });
});

describe('lastAttributedSpeaker', () => {
  it('finds the last labelled word across sequences', () => {
    expect(
      lastAttributedSpeaker([
        { text: [' a'], speakers: ['spk_0'] },
        { text: [' b', ' c'], speakers: ['spk_1', null] },
        { text: [' d'] },
      ]),
    ).toBe('spk_1');
    expect(lastAttributedSpeaker([{ text: [' d'] }])).toBeNull();
  });
});

describe('speakerRunsToText', () => {
  it('writes one speaker turn per line with display names', () => {
    expect(
      speakerRunsToText([
        { speaker: 'spk_0', text: ' Hello there.' },
        { speaker: null, text: ' Yes.' },
        { speaker: 'spk_1', text: ' Hi.' },
      ]),
    ).toBe('Speaker 1: Hello there. Yes.\nSpeaker 2: Hi.');
  });

  it('is plain text when nothing is attributed', () => {
    expect(speakerRunsToText([{ speaker: null, text: ' Hello ' }])).toBe(
      'Hello',
    );
  });
});
