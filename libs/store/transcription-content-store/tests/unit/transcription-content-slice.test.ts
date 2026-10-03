import { describe, expect, it } from 'vitest';

import {
  type TranscriptionContentSlice,
  applySpeakersUpdate,
  commitParagraphBreak,
  handleTranscript,
  isAwaitingSpeakers,
  transcriptionContentReducer,
} from '#src/transcription-content-slice.js';

const emptyState = (): TranscriptionContentSlice =>
  transcriptionContentReducer(undefined, { type: 'init' });

describe('handleTranscript', () => {
  it('stores speaker data on finalized sequences', () => {
    const state = transcriptionContentReducer(
      emptyState(),
      handleTranscript({
        final: { text: [' Hello'], speakers: ['spk_0'] },
        inProgress: null,
      }),
    );

    expect(state.activeSection.sequences).toHaveLength(1);
    expect(state.activeSection.sequences[0]?.speakers).toEqual(['spk_0']);
  });

  it('accepts sequences without speaker data', () => {
    const state = transcriptionContentReducer(
      emptyState(),
      handleTranscript({
        final: { text: [' Hello'] },
        inProgress: { text: [' wor'] },
      }),
    );

    expect(state.activeSection.sequences[0]?.speakers).toBeUndefined();
    expect(state.inProgressTranscription?.text).toEqual([' wor']);
  });
});

describe('commitParagraphBreak', () => {
  it('builds speaker runs for the committed section', () => {
    let state = emptyState();
    state = transcriptionContentReducer(
      state,
      handleTranscript({
        final: { text: [' Hi', ' there'], speakers: ['spk_0', 'spk_0'] },
        inProgress: null,
      }),
    );
    state = transcriptionContentReducer(
      state,
      handleTranscript({
        final: { text: [' Hey'], speakers: ['spk_1'] },
        inProgress: null,
      }),
    );
    state = transcriptionContentReducer(state, commitParagraphBreak());

    expect(state.commitedSections).toHaveLength(1);
    expect(state.commitedSections[0]?.text).toBe(' Hi there Hey');
    expect(state.commitedSections[0]?.runs).toEqual([
      { speaker: 'spk_0', text: ' Hi there' },
      { speaker: 'spk_1', text: ' Hey' },
    ]);
  });

  it('produces a single null run for sections without speakers', () => {
    let state = emptyState();
    state = transcriptionContentReducer(
      state,
      handleTranscript({
        final: { text: [' Plain', ' text'] },
        inProgress: null,
      }),
    );
    state = transcriptionContentReducer(state, commitParagraphBreak());

    expect(state.commitedSections[0]?.runs).toEqual([
      { speaker: null, text: ' Plain text' },
    ]);
  });
});

describe('applySpeakersUpdate', () => {
  it('fills the unlabelled words of the finalized sequence it names', () => {
    let state = transcriptionContentReducer(
      emptyState(),
      handleTranscript({
        final: {
          text: [' Hello', ' there'],
          speakers: [null, null],
          sequenceId: 's0',
        },
        inProgress: null,
      }),
    );
    expect(isAwaitingSpeakers(state.activeSection.sequences[0]!)).toBe(true);

    state = transcriptionContentReducer(
      state,
      applySpeakersUpdate({
        sequenceId: 's0',
        speakers: ['spk_1', 'spk_1'],
        settled: true,
      }),
    );

    expect(state.activeSection.sequences[0]?.speakers).toEqual([
      'spk_1',
      'spk_1',
    ]);
    expect(state.activeSection.sequences[0]?.speakersSettled).toBe(true);
    expect(state.finalizedTranscription[0]?.speakers).toEqual([
      'spk_1',
      'spk_1',
    ]);
    expect(isAwaitingSpeakers(state.activeSection.sequences[0]!)).toBe(false);
  });

  it('never changes a label already shown', () => {
    let state = transcriptionContentReducer(
      emptyState(),
      handleTranscript({
        final: {
          text: [' Hello', ' there'],
          speakers: ['spk_0', null],
          sequenceId: 's0',
        },
        inProgress: null,
      }),
    );

    state = transcriptionContentReducer(
      state,
      applySpeakersUpdate({
        sequenceId: 's0',
        speakers: ['spk_2', 'spk_2'],
        settled: false,
      }),
    );

    expect(state.activeSection.sequences[0]?.speakers).toEqual([
      'spk_0',
      'spk_2',
    ]);
    expect(isAwaitingSpeakers(state.activeSection.sequences[0]!)).toBe(false);
  });

  it('keeps a sequence pending while an unsettled update leaves nulls', () => {
    let state = transcriptionContentReducer(
      emptyState(),
      handleTranscript({
        final: { text: [' a', ' b'], speakers: [null, null], sequenceId: 's0' },
        inProgress: null,
      }),
    );

    state = transcriptionContentReducer(
      state,
      applySpeakersUpdate({
        sequenceId: 's0',
        speakers: ['spk_0', null],
        settled: false,
      }),
    );

    expect(isAwaitingSpeakers(state.activeSection.sequences[0]!)).toBe(true);
  });

  it('ignores an update for a sequence it does not hold', () => {
    const before = transcriptionContentReducer(
      emptyState(),
      handleTranscript({
        final: { text: [' a'], speakers: [null], sequenceId: 's0' },
        inProgress: null,
      }),
    );

    const after = transcriptionContentReducer(
      before,
      applySpeakersUpdate({
        sequenceId: 's9',
        speakers: ['spk_0'],
        settled: true,
      }),
    );

    expect(after.activeSection.sequences[0]?.speakers).toEqual([null]);
  });

  it('does not treat a sequence without a sequenceId as pending', () => {
    expect(isAwaitingSpeakers({ text: [' a'], speakers: [null] })).toBe(false);
    expect(isAwaitingSpeakers({ text: [' a'] })).toBe(false);
  });
});
