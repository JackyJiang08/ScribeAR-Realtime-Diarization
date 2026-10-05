import { type Middleware } from '@reduxjs/toolkit';

import { dominantSpeaker, lastAttributedSpeaker } from './speaker-runs.js';
import {
  type WithTranscriptionContent,
  appendFinalizedTranscription,
  clearTranscription,
  commitParagraphBreak,
  handleTranscript,
} from './transcription-content-slice.js';

/**
 * Options for {@link createParagraphCommitMiddleware}.
 */
export interface ParagraphCommitOptions {
  // Commit the active paragraph when a finalized sequence arrives whose
  // speaker differs from the paragraph's last attributed speaker, so each
  // speaker turn becomes its own paragraph and the label is announced once,
  // at the turn's start. Default true.
  onSpeakerChange?: boolean;
  // Commit the active paragraph after this many milliseconds without a new
  // finalized sequence (a pause in speech). `null` disables. Default 8000.
  idleMs?: number | null;
  // Commit the active paragraph once it holds this many finalized sequences,
  // so an uninterrupted monologue still reaches the live region in pieces
  // rather than all at once at the first pause. `null` disables. Default 6.
  maxSequences?: number | null;
}

const DEFAULT_IDLE_MS = 8000;
const DEFAULT_MAX_SEQUENCES = 6;

/**
 * Redux middleware that commits the active paragraph on the boundaries a
 * networked caption session has: a change of speaker, a pause, or enough
 * finalized text.
 *
 * The caption display announces only *committed* paragraphs to assistive
 * technology (its `role="log"` live region holds the committed sections; the
 * active paragraph and the interim text are `aria-hidden`, because they are
 * rewritten several times a second). Upstream commits paragraphs only from
 * the standalone webapp's WebSpeech provider, so in the client and kiosk
 * webapps no caption and no speaker label ever reached a screen reader. This
 * middleware gives those apps the same boundaries, with the speaker change
 * as the natural one now that speakers exist: a committed paragraph starts
 * with its speaker's label, read once in order before the words.
 *
 * Only `handleTranscript` and `appendFinalizedTranscription` (the actions a
 * networked session dispatches for finalized text) are watched; the
 * standalone webapp keeps its provider-driven commits and does not install
 * this.
 */
export const createParagraphCommitMiddleware =
  (
    options: ParagraphCommitOptions = {},
  ): Middleware<object, WithTranscriptionContent> =>
  (store) => {
    const onSpeakerChange = options.onSpeakerChange ?? true;
    const idleMs =
      options.idleMs === undefined ? DEFAULT_IDLE_MS : options.idleMs;
    const maxSequences =
      options.maxSequences === undefined
        ? DEFAULT_MAX_SEQUENCES
        : options.maxSequences;

    let idleTimer: ReturnType<typeof setTimeout> | null = null;

    const clearIdleTimer = () => {
      if (idleTimer !== null) {
        clearTimeout(idleTimer);
        idleTimer = null;
      }
    };

    const activeSequences = () =>
      store.getState().transcriptionContent.activeSection.sequences;

    const commitIfAnything = () => {
      if (activeSequences().length > 0) {
        store.dispatch(commitParagraphBreak());
      }
    };

    const armIdleTimer = () => {
      clearIdleTimer();
      if (idleMs === null) return;
      idleTimer = setTimeout(() => {
        idleTimer = null;
        commitIfAnything();
      }, idleMs);
    };

    return (next) => (action) => {
      const incoming = handleTranscript.match(action)
        ? action.payload.final
        : appendFinalizedTranscription.match(action)
          ? action.payload
          : null;

      if (incoming !== null && onSpeakerChange) {
        // Decided before the reducer appends the new sequence: the turn
        // that ended belongs to the paragraph being closed, the one that
        // starts belongs to the next.
        const sequences = activeSequences();
        const current = lastAttributedSpeaker(sequences);
        const incomingSpeaker = dominantSpeaker(incoming.speakers);
        if (
          sequences.length > 0 &&
          current !== null &&
          incomingSpeaker !== null &&
          incomingSpeaker !== current
        ) {
          store.dispatch(commitParagraphBreak());
        }
      }

      const result = next(action);

      if (incoming !== null) {
        if (maxSequences !== null && activeSequences().length >= maxSequences) {
          store.dispatch(commitParagraphBreak());
        }
        armIdleTimer();
      }

      if (clearTranscription.match(action)) {
        clearIdleTimer();
      }

      return result;
    };
  };
