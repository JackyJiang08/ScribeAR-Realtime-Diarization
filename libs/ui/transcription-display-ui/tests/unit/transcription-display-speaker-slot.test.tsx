import { screen } from '@testing-library/react';
import { describe, expect } from 'vitest';

import type {
  ActiveSection,
  TranscriptionSequence,
} from '@scribear/transcription-content-store';

import { PENDING_SPEAKER_LABEL } from '#src/components/speaker-label-slot.js';
import { TranscriptionDisplayContainer } from '#src/components/transcription-display-container.js';

import { axeViolations } from '../a11y.js';
import { renderWithProviders, withProviders } from '../render.js';

const container = (sequences: TranscriptionSequence[]) => {
  const activeSection: ActiveSection = { id: 'active', sequences };
  return (
    <TranscriptionDisplayContainer
      commitedSections={[]}
      activeSection={activeSection}
      inProgressTranscriptionText=""
      wordSpacingEm={0}
      fontSizePx={32}
      lineHeightPx={40}
      getBoundedDisplayPreferences={() => ({
        verticalPositionPx: 0,
        numDisplayLines: 8,
      })}
    />
  );
};

function renderActive(sequences: TranscriptionSequence[]) {
  const rendered = renderWithProviders(container(sequences));
  return {
    ...rendered,
    rerenderActive: (next: TranscriptionSequence[]) => {
      rendered.rerender(withProviders(container(next)));
    },
  };
}

const slots = (container: HTMLElement) =>
  Array.from(container.querySelectorAll('[data-speaker-slot]'));

describe('speaker label slot', (it) => {
  it('shows a neutral placeholder for a caption whose label has not arrived', () => {
    const { container } = renderActive([
      {
        id: 'a',
        text: [' Hello', ' there'],
        speakers: [null, null],
        sequenceId: 's0',
      },
    ]);

    const [slot] = slots(container);
    expect(slot).toBeDefined();
    expect(slot?.textContent.trim()).toBe(`${PENDING_SPEAKER_LABEL}:`);
    expect(slot).toHaveAttribute('data-speaker-slot', 'pending');
    // The change from placeholder to label must never be announced as a text
    // update: the slot opts out of the surrounding live region.
    expect(slot).toHaveAttribute('aria-live', 'off');
    expect(screen.getByText(/Hello there/)).toBeInTheDocument();
  });

  it('fills the label into the same slot without changing the text nodes', () => {
    const pending: TranscriptionSequence = {
      id: 'a',
      text: [' Hello', ' there'],
      speakers: [null, null],
      sequenceId: 's0',
    };
    const { container, rerenderActive } = renderActive([pending]);
    const [slotBefore] = slots(container);
    const textBefore = screen.getByText(/Hello there/);

    rerenderActive([
      { ...pending, speakers: ['spk_1', 'spk_1'], speakersSettled: true },
    ]);

    const [slotAfter] = slots(container);
    // Same DOM node, new content: nothing was inserted before the text.
    expect(slotAfter).toBe(slotBefore);
    expect(slotAfter?.textContent.trim()).toBe('Speaker 2:');
    expect(screen.getByText(/Hello there/)).toBe(textBefore);
    // No line break was inserted between the slot and its text.
    expect(container.querySelectorAll('br')).toHaveLength(0);
  });

  it('empties the slot once the sequence is settled without a speaker', () => {
    const { container } = renderActive([
      {
        id: 'a',
        text: [' Hello'],
        speakers: [null],
        sequenceId: 's0',
        speakersSettled: true,
      },
    ]);

    const [slot] = slots(container);
    // The question mark never stays on screen: the slot keeps its width
    // (so nothing moves) but shows no label at all.
    expect(slot?.textContent.trim()).toBe('');
    expect(slot).toHaveAttribute('data-speaker-slot', 'unattributed');
    expect(screen.getByText(/Hello/)).toBeInTheDocument();
  });

  it('keeps the slot but blanks it when the speaker continues from the previous line', () => {
    const { container } = renderActive([
      { id: 'a', text: [' One.'], speakers: ['spk_0'], sequenceId: 's0' },
      { id: 'b', text: [' Two.'], speakers: ['spk_0'], sequenceId: 's1' },
      { id: 'c', text: [' Three.'], speakers: ['spk_1'], sequenceId: 's2' },
    ]);

    const [first, second, third] = slots(container);
    expect(first?.textContent.trim()).toBe('Speaker 1:');
    expect(second?.textContent.trim()).toBe('');
    expect(third?.textContent.trim()).toBe('Speaker 2:');
    // Every label-aware sequence after the first starts its own line.
    expect(container.querySelectorAll('br')).toHaveLength(2);
  });

  it('renders sequences without a sequenceId exactly as before', () => {
    const { container } = renderActive([
      { id: 'a', text: [' Plain', ' text'] },
      { id: 'b', text: [' more'], speakers: ['spk_0'] },
    ]);

    expect(slots(container)).toHaveLength(0);
    expect(screen.getByText(/Plain text/)).toBeInTheDocument();
    expect(screen.getByText('Speaker 1:')).toBeInTheDocument();
  });

  it('has no axe violations with pending and resolved labels', async () => {
    const { container } = renderActive([
      { id: 'a', text: [' One.'], speakers: ['spk_0'], sequenceId: 's0' },
      { id: 'b', text: [' Two.'], speakers: [null], sequenceId: 's1' },
    ]);
    expect(await axeViolations(container)).toEqual([]);
  });
});
