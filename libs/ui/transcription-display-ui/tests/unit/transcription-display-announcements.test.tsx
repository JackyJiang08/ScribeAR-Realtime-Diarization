import { screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import type {
  ActiveSection,
  TranscriptionSection,
} from '@scribear/transcription-content-store';

import { TranscriptionDisplayContainer } from '#src/components/transcription-display-container.js';

import { axeViolations } from '../a11y.js';
import { renderWithProviders } from '../render.js';

function renderContainer(
  commitedSections: TranscriptionSection[],
  activeSection: ActiveSection = { id: 'active', sequences: [] },
) {
  return renderWithProviders(
    <TranscriptionDisplayContainer
      commitedSections={commitedSections}
      activeSection={activeSection}
      inProgressTranscriptionText=""
      wordSpacingEm={0}
      fontSizePx={32}
      lineHeightPx={40}
      getBoundedDisplayPreferences={() => ({
        verticalPositionPx: 0,
        numDisplayLines: 8,
      })}
    />,
  );
}

describe('speaker labels in the live region', () => {
  it('names each speaker once per turn across committed paragraphs', () => {
    renderContainer([
      {
        id: 's1',
        text: ' Hello everyone.',
        runs: [{ speaker: 'spk_0', text: ' Hello everyone.' }],
      },
      {
        // The same speaker continued after a pause commit: no second label.
        id: 's2',
        text: ' Today we start.',
        runs: [{ speaker: 'spk_0', text: ' Today we start.' }],
      },
      {
        id: 's3',
        text: ' Question?',
        runs: [{ speaker: 'spk_1', text: ' Question?' }],
      },
    ]);

    const log = screen.getByRole('log');
    const text = log.textContent;
    expect(text.match(/Speaker 1:/g)).toHaveLength(1);
    expect(text.match(/Speaker 2:/g)).toHaveLength(1);
    expect(text.indexOf('Speaker 1:')).toBeLessThan(text.indexOf('Hello'));
    expect(text.indexOf('Speaker 2:')).toBeLessThan(text.indexOf('Question'));
    expect(text).toContain('Today we start.');
  });

  it('does not re-label the active paragraph when its speaker continues', () => {
    const { container } = renderContainer(
      [
        {
          id: 's1',
          text: ' Hello.',
          runs: [{ speaker: 'spk_0', text: ' Hello.' }],
        },
      ],
      {
        id: 'active',
        sequences: [
          {
            id: 'seq1',
            text: [' Still', ' me.'],
            speakers: ['spk_0', 'spk_0'],
            sequenceId: 's7',
          },
        ],
      },
    );

    const hidden = container.querySelector('[aria-hidden="true"]');
    expect(hidden?.textContent).toContain('Still me.');
    // The label slot stays blank: a hanging indent, not a repeated name.
    expect(hidden?.textContent).not.toContain('Speaker 1:');
    expect(screen.getByRole('log').textContent).toContain('Speaker 1:');
  });

  it('has no axe violations with speakered paragraphs', async () => {
    const { container } = renderContainer([
      {
        id: 's1',
        text: ' Hello.',
        runs: [{ speaker: 'spk_0', text: ' Hello.' }],
      },
      {
        id: 's2',
        text: ' Hi.',
        runs: [{ speaker: 'spk_1', text: ' Hi.' }],
      },
    ]);
    expect(await axeViolations(container)).toEqual([]);
  });
});
