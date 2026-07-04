import { Fragment, memo, useMemo } from 'react';

import Box from '@mui/material/Box';

import type { SpeakerRun } from '@scribear/transcription-content-store';

import {
  formatSpeakerName,
  getSpeakerColor,
} from '#src/utils/speaker-appearance.js';

/**
 * One renderable unit derived from a speaker run: its text plus whether a
 * speaker label (and preceding line break) should be shown before it.
 */
interface SpeakerRunRenderItem {
  key: number;
  needsLineBreak: boolean;
  labelSpeaker: string | null;
  text: string;
}

/**
 * Derives render items from speaker runs. A label is emitted when the
 * attributed speaker changes; unattributed (`null`) runs never interrupt the
 * current speaker. Pure so the component body performs no mutation.
 */
export const buildSpeakerRunRenderItems = (
  runs: SpeakerRun[],
  previousSpeaker: string | null,
): SpeakerRunRenderItem[] => {
  const items: SpeakerRunRenderItem[] = [];
  let currentSpeaker = previousSpeaker;
  let isFirstContent = previousSpeaker === null;
  for (const run of runs) {
    const labelSpeaker =
      run.speaker !== null && run.speaker !== currentSpeaker
        ? run.speaker
        : null;
    if (run.speaker !== null) {
      currentSpeaker = run.speaker;
    }
    items.push({
      // Runs are append-only within an immutable sequence, so positional
      // keys are stable.
      key: items.length,
      needsLineBreak: labelSpeaker !== null && !isFirstContent,
      labelSpeaker,
      text: run.text,
    });
    isFirstContent = false;
  }
  return items;
};

/**
 * Props for {@link SpeakerRunsText}.
 */
export interface SpeakerRunsTextProps {
  // Speaker runs to render, in reading order.
  runs: SpeakerRun[];
  // Last attributed speaker rendered before these runs, so a continuing
  // speaker does not repeat their label. `null` when nothing precedes.
  previousSpeaker: string | null;
  // Background color the text is rendered on, used to keep label colors readable.
  backgroundColor: string;
}

/**
 * Renders speaker runs as inline text. When the attributed speaker changes, a
 * colored `Speaker N:` label is inserted (on a new line unless it is the very
 * first content).
 *
 * Memoized so appending new sequences never re-renders existing ones.
 */
export const SpeakerRunsText = memo(
  ({ runs, previousSpeaker, backgroundColor }: SpeakerRunsTextProps) => {
    const items = useMemo(
      () => buildSpeakerRunRenderItems(runs, previousSpeaker),
      [runs, previousSpeaker],
    );

    return (
      <>
        {items.map((item) => (
          <Fragment key={item.key}>
            {item.needsLineBreak && <br />}
            {item.labelSpeaker !== null && (
              <Box
                component="span"
                sx={{
                  color: getSpeakerColor(item.labelSpeaker, backgroundColor),
                  fontWeight: 'bold',
                }}
              >
                {`${formatSpeakerName(item.labelSpeaker)}: `}
              </Box>
            )}
            <span>{item.text}</span>
          </Fragment>
        ))}
      </>
    );
  },
);
