import { screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import {
  GAP_MARKER,
  type TranslatedSegment,
  TranslationStatus,
} from '@scribear/live-translation-store';

import { TranslatedCaptionsPanel } from '#src/components/translated-captions-panel.js';

import { axeViolations } from '../a11y.js';
import { renderWithProviders } from '../render.js';

const SEGMENTS: TranslatedSegment[] = [
  { id: 't1', text: 'Hola a todos.', kind: 'text' },
  { id: 't2', text: GAP_MARKER, kind: 'gap' },
  { id: 't3', text: 'Bienvenidos.', kind: 'text' },
];

function renderPanel(
  overrides: Partial<Parameters<typeof TranslatedCaptionsPanel>[0]> = {},
) {
  return renderWithProviders(
    <TranslatedCaptionsPanel
      segments={SEGMENTS}
      status={TranslationStatus.READY}
      targetLanguage="es"
      targetLanguageLabel="Spanish"
      downloadProgress={null}
      errorMessage={null}
      wordSpacingEm={0.25}
      fontSizePx={32}
      lineHeightPx={48}
      {...overrides}
    />,
  );
}

describe('TranslatedCaptionsPanel', () => {
  it('renders the translated captions', () => {
    renderPanel();

    const region = screen.getByRole('log');
    expect(region).toHaveTextContent('Hola a todos.');
    expect(region).toHaveTextContent('Bienvenidos.');
  });

  it('always states that the translation is machine produced', () => {
    renderPanel();

    expect(
      screen.getByText('In-browser translation - may contain errors'),
    ).toBeInTheDocument();
  });

  it('shows an ellipsis where captions were dropped to catch up', () => {
    renderPanel();

    expect(screen.getByRole('log')).toHaveTextContent(GAP_MARKER);
  });

  it('tags the caption region with the target language', () => {
    // Without `lang`, a screen reader reads Spanish captions with an English
    // voice, which is close to unintelligible.
    renderPanel();

    expect(screen.getByRole('log')).toHaveAttribute('lang', 'es');
  });

  it('reports a translation failure visibly rather than going quiet', () => {
    renderPanel({
      status: TranslationStatus.ERROR,
      errorMessage: 'No translations are available.',
    });

    expect(screen.getByRole('alert')).toHaveTextContent(
      'No translations are available.',
    );
  });

  it('shows download progress while the model is being fetched', () => {
    renderPanel({
      status: TranslationStatus.DOWNLOADING,
      downloadProgress: 0.42,
    });

    expect(screen.getByRole('status')).toHaveTextContent(
      /Downloading the Spanish language model/i,
    );
    expect(screen.getByRole('progressbar')).toHaveAttribute(
      'aria-valuenow',
      '42',
    );
  });

  it('shows an indeterminate bar before the browser reports progress', () => {
    renderPanel({
      status: TranslationStatus.DOWNLOADING,
      downloadProgress: null,
    });

    expect(screen.getByRole('progressbar')).not.toHaveAttribute(
      'aria-valuenow',
    );
  });

  it('holds its scroll position against the browser moving it', () => {
    // Blink and Gecko shift the offset to keep an "anchor" node still when
    // content above it resizes, which fights the auto-scroll pin; and
    // overscrolling the translation must not chain into the pane above it.
    renderPanel();

    expect(screen.getByRole('log')).toHaveStyle({
      overflowAnchor: 'none',
      overscrollBehavior: 'contain',
    });
  });

  it('keeps the jump-to-bottom control out of the way while it is following', () => {
    // Scrollback would be a trap without a way back; the control is rendered
    // from the start so its space is reserved, but stays hidden - from sighted
    // readers and assistive technology alike - until it is needed.
    renderPanel();

    expect(screen.getByRole('button', { hidden: true })).toHaveStyle({
      visibility: 'hidden',
    });
    expect(screen.queryByRole('button')).toBeNull();
  });

  it('has no automatically detectable accessibility violations', async () => {
    renderPanel();

    expect(await axeViolations()).toEqual([]);
  });
});

describe('TranslatedCaptionsPanel speaker labels', () => {
  const speakered: TranslatedSegment[] = [
    { id: 'a', text: 'Hola a todos.', kind: 'text', speaker: 'spk_0' },
    { id: 'b', text: 'Hoy empezamos.', kind: 'text', speaker: 'spk_0' },
    { id: 'c', text: GAP_MARKER, kind: 'gap' },
    { id: 'd', text: 'Una pregunta.', kind: 'text', speaker: 'spk_1' },
    { id: 'e', text: 'Claro.', kind: 'text', speaker: 'spk_0' },
  ];

  it('names each speaker once per turn, as the transcript does', () => {
    renderPanel({ segments: speakered });

    const text = screen.getByRole('log').textContent;
    expect(text.match(/Speaker 1:/g)).toHaveLength(2);
    expect(text.match(/Speaker 2:/g)).toHaveLength(1);
    expect(text.indexOf('Speaker 1:')).toBeLessThan(text.indexOf('Hola'));
    expect(text.indexOf('Speaker 2:')).toBeLessThan(text.indexOf('Una'));
    // Within one turn the second segment is not re-labelled.
    const firstTurn = text.slice(0, text.indexOf('Speaker 2:'));
    expect(firstTurn).toContain('Hoy empezamos.');
    expect(firstTurn.match(/Speaker 1:/g)).toHaveLength(1);
  });

  it('shows no labels when captions carry no speaker', () => {
    renderPanel();

    expect(screen.getByRole('log').textContent).not.toContain('Speaker');
  });

  it('has no axe violations with speaker labels', async () => {
    const { container } = renderPanel({ segments: speakered });
    expect(await axeViolations(container)).toEqual([]);
  });
});
