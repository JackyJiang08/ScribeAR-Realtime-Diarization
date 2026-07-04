import { describe, expect, it } from 'vitest';

import {
  type TranscriptionContentSlice,
  commitParagraphBreak,
  handleTranscript,
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
