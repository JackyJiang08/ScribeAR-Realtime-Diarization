import { configureStore } from '@reduxjs/toolkit';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  type ParagraphCommitOptions,
  createParagraphCommitMiddleware,
} from '#src/paragraph-commit-middleware.js';
import {
  appendFinalizedTranscription,
  clearTranscription,
  handleTranscript,
  transcriptionContentReducer,
} from '#src/transcription-content-slice.js';

function createTestStore(options: ParagraphCommitOptions = {}) {
  return configureStore({
    reducer: { transcriptionContent: transcriptionContentReducer },
    middleware: (getDefaultMiddleware) =>
      getDefaultMiddleware().concat(createParagraphCommitMiddleware(options)),
  });
}

const final = (text: string, speaker: string | null) => ({
  final: { text: [text], speakers: [speaker], sequenceId: text.trim() },
  inProgress: null,
});

describe('createParagraphCommitMiddleware', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('commits the active paragraph when the speaker changes, before the new turn', () => {
    const store = createTestStore({ idleMs: null, maxSequences: null });

    store.dispatch(handleTranscript(final(' Hello', 'spk_0')));
    store.dispatch(handleTranscript(final(' there', 'spk_0')));
    expect(store.getState().transcriptionContent.commitedSections).toEqual([]);

    store.dispatch(handleTranscript(final(' Hi', 'spk_1')));

    const { commitedSections, activeSection } =
      store.getState().transcriptionContent;
    expect(commitedSections).toHaveLength(1);
    expect(commitedSections[0]?.runs).toEqual([
      { speaker: 'spk_0', text: ' Hello there' },
    ]);
    // The new speaker's words open the next paragraph.
    expect(activeSection.sequences.map((s) => s.text.join(''))).toEqual([
      ' Hi',
    ]);
  });

  it('does not commit for unlabelled words or the same speaker continuing', () => {
    const store = createTestStore({ idleMs: null, maxSequences: null });

    store.dispatch(handleTranscript(final(' Hello', 'spk_0')));
    store.dispatch(handleTranscript(final(' still', null)));
    store.dispatch(handleTranscript(final(' me', 'spk_0')));
    store.dispatch(appendFinalizedTranscription({ text: [' again'] }));

    expect(store.getState().transcriptionContent.commitedSections).toEqual([]);
    expect(
      store.getState().transcriptionContent.activeSection.sequences,
    ).toHaveLength(4);
  });

  it('commits after a pause in finalized text', () => {
    const store = createTestStore({ idleMs: 1000, maxSequences: null });

    store.dispatch(handleTranscript(final(' Hello', null)));
    vi.advanceTimersByTime(900);
    expect(store.getState().transcriptionContent.commitedSections).toEqual([]);
    // More text restarts the pause.
    store.dispatch(handleTranscript(final(' world', null)));
    vi.advanceTimersByTime(900);
    expect(store.getState().transcriptionContent.commitedSections).toEqual([]);

    vi.advanceTimersByTime(100);

    const { commitedSections, activeSection } =
      store.getState().transcriptionContent;
    expect(commitedSections).toHaveLength(1);
    expect(commitedSections[0]?.text).toBe(' Hello world');
    expect(activeSection.sequences).toEqual([]);
  });

  it('commits once a paragraph holds enough sequences', () => {
    const store = createTestStore({ idleMs: null, maxSequences: 2 });

    store.dispatch(handleTranscript(final(' one', 'spk_0')));
    expect(store.getState().transcriptionContent.commitedSections).toEqual([]);
    store.dispatch(handleTranscript(final(' two', 'spk_0')));

    expect(store.getState().transcriptionContent.commitedSections).toHaveLength(
      1,
    );
    expect(
      store.getState().transcriptionContent.activeSection.sequences,
    ).toEqual([]);
  });

  it('forgets a pending pause when the transcript is cleared', () => {
    const store = createTestStore({ idleMs: 1000, maxSequences: null });

    store.dispatch(handleTranscript(final(' Hello', null)));
    store.dispatch(clearTranscription());
    store.dispatch(
      appendFinalizedTranscription({ text: [' fresh'], speakers: ['spk_2'] }),
    );
    vi.advanceTimersByTime(1000);

    // One commit, from the second paragraph's own pause; nothing stale.
    const { commitedSections } = store.getState().transcriptionContent;
    expect(commitedSections).toHaveLength(1);
    expect(commitedSections[0]?.text).toBe(' fresh');
  });
});
