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
}: TranscriptionDisplayContainerProps) => {
  const { containerHeightPx, setContainerHeightPx } =
    useTranscriptionDisplayHeight();
  const containerRef = useContainerHeight(setContainerHeightPx);
  const backgroundColor = useTheme().palette.background.default;

  const { verticalPositionPx, numDisplayLines } =
    getBoundedDisplayPreferences(containerHeightPx);
  const displayHeightPx = numDisplayLines * lineHeightPx;

  const {
    isAutoScrollEnabled,
    setIsAutoScrollEnabled,
    textContainerRef,
    textBottomRef,
    handleScroll,
  } = useAutoScroll([
    commitedSections,
    activeSection,
    inProgressTranscriptionText,
    containerHeightPx,
    displayHeightPx,
  ]);

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
    <Box sx={{ height: '100dvh', width: '100%', p: 2 }}>
      <Box ref={containerRef} sx={{ height: '100%' }}>
        <Stack direction="row">
          <Box
            ref={textContainerRef}
            onScroll={handleScroll}
            sx={{
              marginTop: `${verticalPositionPx.toString()}px`,
              height: `${displayHeightPx.toString()}px`,
              width: '100%',
              overflowY: 'scroll',
              '&::-webkit-scrollbar': {
                display: 'none',
              },
              msOverflowStyle: 'none',
              scrollbarWidth: 'none',
            }}
          >
            <CommittedSections
              sections={commitedSections}
              textStyle={textStyle}
              backgroundColor={backgroundColor}
            />
            <Typography color="transcriptionColor" sx={textStyle}>
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
            <Box ref={textBottomRef} />
          </Box>
          <JumpToBottomButton
            visible={!isAutoScrollEnabled}
            onClick={() => {
              setIsAutoScrollEnabled(true);
            }}
          />
        </Stack>
      </Box>
    </Box>
  );
};
