import { memo } from 'react';

import Box from '@mui/material/Box';

import {
  formatSpeakerName,
  getSpeakerColor,
} from '#src/utils/speaker-appearance.js';

/**
 * What a label slot shows while the speaker is not known. Reads as an
 * explicit "unattributed" state rather than as silence, and has the same
 * character count as `Speaker N`, so the slot does not change width when
 * the label arrives (tabular digits keep `?` and a digit the same width in
 * most fonts; a tenth speaker adds one character, the one case the slot
 * grows).
 */
export const PENDING_SPEAKER_LABEL = 'Speaker ?';

/**
 * Props for {@link SpeakerLabelSlot}.
 */
export interface SpeakerLabelSlotProps {
  // The speaker to show, `null` while unknown.
  speaker: string | null;
  // Whether a label may still arrive. When false and `speaker` is null, the
  // slot shows the pending text permanently: the provider never attributed
  // these words.
  pending: boolean;
  // When true the slot is kept but rendered empty: the speaker continues from
  // the previous line, so the label is not repeated. The width is kept so
  // text never moves if the previous line's label changes later.
  continuation: boolean;
  // Background the text is rendered on, for a readable label color.
  backgroundColor: string;
}

/**
 * A fixed-width inline slot at the start of a caption line that holds the
 * speaker label. It is rendered the moment the caption text is, with a
 * placeholder, and later filled in place: the slot never changes size and
 * nothing after it reflows when the label arrives.
 *
 * `aria-live="off"` on the slot stops a live region from announcing the
 * placeholder-to-label change as a text update; the label is still read in
 * order when a screen reader walks the line, and a newly added line is still
 * announced as an addition with whatever the slot held at that moment.
 */
export const SpeakerLabelSlot = memo(
  ({
    speaker,
    pending,
    continuation,
    backgroundColor,
  }: SpeakerLabelSlotProps) => {
    // Pending and settled-but-unattributed read the same: both are "no
    // speaker known"; they differ only in the data attribute, for tests and
    // styling hooks.
    const label =
      speaker !== null ? formatSpeakerName(speaker) : PENDING_SPEAKER_LABEL;
    const showText = speaker === null || !continuation;
    return (
      <Box
        component="span"
        aria-live="off"
        data-speaker-slot={speaker ?? (pending ? 'pending' : 'unattributed')}
        sx={{
          display: 'inline-block',
          // Wide enough for `Speaker 9:` in the caption font; a slot with a
          // shorter label keeps this width, so every line's text starts at
          // the same column and never moves when the label fills in.
          minWidth: '9ch',
          whiteSpace: 'nowrap',
          fontVariantNumeric: 'tabular-nums',
          fontWeight: 'bold',
          color:
            speaker !== null
              ? getSpeakerColor(speaker, backgroundColor)
              : 'inherit',
          opacity: speaker === null ? 0.7 : 1,
        }}
      >
        {showText ? `${label}: ` : ' '}
      </Box>
    );
  },
);
