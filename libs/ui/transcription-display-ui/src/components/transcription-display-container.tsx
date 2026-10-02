import { memo, useMemo } from 'react';

import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import { type SxProps, type Theme, useTheme } from '@mui/material/styles';

import {
  type ActiveSection,
  type TranscriptionSection,
  type TranscriptionSequence,
  wordsToSpeakerRuns,
} from '@scribear/transcription-content-store';

import { useTranscriptionDisplayHeight } from '#src/contexts/transcription-display-height-context.js';
import { useAutoScroll } from '#src/hooks/use-auto-scroll.js';
import { useContainerHeight } from '#src/hooks/use-container-height.js';

import { JumpToBottomButton } from './jump-to-bottom-button.js';
import { SpeakerRunsText } from './speaker-runs-text.js';

/**
 * Props for the internal {@link CommittedSections} component.
 */
interface CommittedSectionsProps {
  // The finalized transcription sections to render as static text.
  sections: TranscriptionSection[];
  // MUI sx styles applied to each section's Typography element.
  textStyle: SxProps<Theme>;
  // Background color transcription is rendered on, for readable speaker labels.
  backgroundColor: string;
}

// Memoized so active section transcription updates don't update the full committed history.
const CommittedSections = memo(
  ({ sections, textStyle, backgroundColor }: CommittedSectionsProps) => (
    <>
      {sections.map((section) => (
        <Typography key={section.id} color="transcriptionColor" sx={textStyle}>
          <SpeakerRunsText
            // Sections persisted before speaker support carry no runs.
            runs={section.runs ?? [{ speaker: null, text: section.text }]}
            previousSpeaker={null}
            backgroundColor={backgroundColor}
          />
        </Typography>
      ))}
    </>
  ),
);

/**
 * Props for the internal {@link ActiveSequenceText} component.
 */
interface ActiveSequenceTextProps {
  // The finalized sequence to render inside the active section.
  sequence: TranscriptionSequence;
  // Last attributed speaker before this sequence, for label suppression.
  previousSpeaker: string | null;
  // Background color transcription is rendered on, for readable speaker labels.
  backgroundColor: string;
}

// Memoized so appending sequences to the active section never re-renders
// existing ones; sequences are immutable once appended.
const ActiveSequenceText = memo(
  ({ sequence, previousSpeaker, backgroundColor }: ActiveSequenceTextProps) => {
    const runs = useMemo(
      () => wordsToSpeakerRuns(sequence.text, sequence.speakers),
      [sequence],
    );
    return (
      <SpeakerRunsText
        runs={runs}
        previousSpeaker={previousSpeaker}
        backgroundColor={backgroundColor}
      />
    );
  },
);

/**
 * Last attributed (non-null) speaker in a sequence, or `fallback` when the
 * sequence has no speaker attribution at all.
 */
const lastAttributedSpeaker = (
  sequence: TranscriptionSequence,
  fallback: string | null,
): string | null => {
  const speakers = sequence.speakers ?? [];
  for (let i = speakers.length - 1; i >= 0; i -= 1) {
    const speaker = speakers[i];
    if (speaker !== null && speaker !== undefined) return speaker;
  }
  return fallback;
};

/**
 * Pairs each active-section sequence with the speaker attributed just before
 * it, so each sequence renderer can suppress labels for continuing speakers.
 */
const buildActiveSequenceItems = (
  sequences: TranscriptionSequence[],
): { sequence: TranscriptionSequence; previousSpeaker: string | null }[] => {
  const items: {
    sequence: TranscriptionSequence;
    previousSpeaker: string | null;
  }[] = [];
  let previousSpeaker: string | null = null;
  for (const sequence of sequences) {
    items.push({ sequence, previousSpeaker });
    previousSpeaker = lastAttributedSpeaker(sequence, previousSpeaker);
  }
  return items;
};

/**
 * Bounded display preferences resolved against the current container height.
 */
interface BoundedDisplayPreferences {
  verticalPositionPx: number;
  numDisplayLines: number;
}

/**
 * Props for {@link TranscriptionDisplayContainer}.
 */
export interface TranscriptionDisplayContainerProps {
  // The finalized transcription sections rendered as static committed text.
  commitedSections: TranscriptionSection[];
  // The current in-progress transcription section rendered as live updating text.
  activeSection: ActiveSection;
  // Raw text for the currently streaming transcription chunk, appended after the active section sequences.
  inProgressTranscriptionText: string;
  // Word spacing applied to all transcription text, in em units.
  wordSpacingEm: number;
  // Font size applied to all transcription text, in pixels.
  fontSizePx: number;
  // Line height applied to all transcription text in pixels. Also used to calculate the display area height.
  lineHeightPx: number;
  // Returns the current display preferences (vertical position and line count) clamped to the container height.
  getBoundedDisplayPreferences: (
    containerHeightPx: number,
  ) => BoundedDisplayPreferences;
  // Fill the parent's height instead of the viewport's. Set when the container
  // shares the screen with something else (e.g. the translated caption panel),
  // so the two divide one viewport rather than each claiming all of it.
  fillParentHeight?: boolean;
  // Whether this region announces new text to assistive technology. Defaults to true.
  // Set false when another region on the page (e.g. translated captions) is the
  // one the reader has chosen to follow - two live regions carrying the same
  // speech announce it twice and make both unusable.
  announceUpdates?: boolean;
  // Return to following the speaker this many ms after the reader scrolls back,
  // if nothing scrolls and no sign of a reader arrives. `null` (the default)
  // leaves the view where they put it indefinitely. Set it on an unattended
  // display, where captions frozen for the rest of a session is a far worse
  // outcome than a reader losing their place once.
  idleReengageMs?: number | null;
}

/**
 * Renders the live transcription text with auto-scroll and user preference
 * styling. Reads container height from `TranscriptionDisplayHeightContext`.
 */
export const TranscriptionDisplayContainer = ({
  commitedSections,
  activeSection,
  inProgressTranscriptionText,
  wordSpacingEm,
  fontSizePx,
  lineHeightPx,
  getBoundedDisplayPreferences,
  fillParentHeight = false,
  announceUpdates = true,
  idleReengageMs = null,
}: TranscriptionDisplayContainerProps) => {
  const { containerHeightPx, setContainerHeightPx } =
    useTranscriptionDisplayHeight();
  const containerRef = useContainerHeight(setContainerHeightPx);
  const backgroundColor = useTheme().palette.background.default;

  const { verticalPositionPx, numDisplayLines } =
    getBoundedDisplayPreferences(containerHeightPx);
  const displayHeightPx = numDisplayLines * lineHeightPx;

  const { isAutoScrollEnabled, textContainerRef, handleScroll, jumpToBottom } =
    useAutoScroll(
      [
        commitedSections,
        activeSection,
        inProgressTranscriptionText,
        containerHeightPx,
        displayHeightPx,
      ],
      { lineHeightPx, label: 'transcription', idleReengageMs },
    );

  const textStyle = useMemo<SxProps<Theme>>(
    () => ({
      wordSpacing: `${wordSpacingEm.toString()}em`,
      fontSize: `${fontSizePx.toString()}px`,
      lineHeight: `${lineHeightPx.toString()}px`,
    }),
    [wordSpacingEm, fontSizePx, lineHeightPx],
  );

  const activeSequenceItems = useMemo(
    () => buildActiveSequenceItems(activeSection.sequences),
    [activeSection.sequences],
  );

  return (
    <Box
      sx={{ height: fillParentHeight ? '100%' : '100dvh', width: '100%', p: 2 }}
    >
      <Box ref={containerRef} sx={{ height: '100%' }}>
        <Stack direction="row">
          <Box
            ref={textContainerRef}
            onScroll={handleScroll}
            // Live-caption region for assistive technology. `role="log"` announces
            // only newly appended nodes (finalized/committed sections) and leaves
            // history in place; `polite` queues so it never interrupts the user;
            // `aria-relevant="additions text"` + `aria-atomic="false"` announce just
            // the new node, not the whole transcript. Interim text below is
            // `aria-hidden` so its word-by-word churn is never announced (it is
            // announced exactly once, later, when it becomes a committed section).
            // `tabIndex={0}` makes the region focusable so keyboard + AT users can
            // scroll back through history (arrow/PageUp/PageDown/Home/End) — the
            // scrollbar is visually hidden but the region stays keyboard-scrollable,
            // with a visible focus ring for SC 2.4.7.
            role="log"
            aria-live={announceUpdates ? 'polite' : 'off'}
            aria-relevant="additions text"
            aria-atomic="false"
            aria-label="Live transcription"
            tabIndex={0}
            sx={{
              marginTop: `${verticalPositionPx.toString()}px`,
              height: `${displayHeightPx.toString()}px`,
              width: '100%',
              overflowY: 'scroll',
              // Belt-and-braces, not the fix. Blink and Gecko reposition the
              // scroll offset to hold an "anchor" node still when content above
              // it changes size. The anchor should sit above everything that
              // mutates here, but a re-wrap from a late webfont or a width
              // change can still reach above it, and an append-only caption log
              // gains nothing from anchoring in any case. No-op in WebKit,
              // which does not implement it.
              overflowAnchor: 'none',
              // Keep overscroll inside this box: no chaining to the page, and a
              // damped rubber-band on iOS.
              overscrollBehavior: 'contain',
              '&::-webkit-scrollbar': {
                display: 'none',
              },
              msOverflowStyle: 'none',
              scrollbarWidth: 'none',
              '&:focus-visible': {
                outline: '2px solid',
                outlineColor: 'transcriptionColor',
                outlineOffset: '2px',
              },
            }}
          >
            <CommittedSections
              sections={commitedSections}
              textStyle={textStyle}
              backgroundColor={backgroundColor}
            />
            <Typography
              color="transcriptionColor"
              sx={textStyle}
              // Interim/in-progress results change many times per second; feeding
              // that churn to a live region makes speech stutter and a braille
              // display reflow continuously. Hide it from AT — sighted users still
              // see it live, and it is announced once when it moves to a committed
              // section above. (SC 4.1.3, 1.3.1)
              aria-hidden="true"
            >
              {/* Keyed memoized sequences so React only appends new nodes — never
                  mutates existing ones, keeping browser re-layout cost
                  proportional to each new chunk. */}
              {activeSequenceItems.map((item) => (
                <ActiveSequenceText
                  key={item.sequence.id}
                  sequence={item.sequence}
                  previousSpeaker={item.previousSpeaker}
                  backgroundColor={backgroundColor}
                />
              ))}
              <span>{inProgressTranscriptionText}</span>
            </Typography>
          </Box>
          <JumpToBottomButton
            visible={!isAutoScrollEnabled}
            onClick={jumpToBottom}
          />
        </Stack>
      </Box>
    </Box>
  );
};
