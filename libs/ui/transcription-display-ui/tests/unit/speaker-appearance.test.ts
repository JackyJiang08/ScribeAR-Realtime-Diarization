import { describe, expect, it } from 'vitest';

import {
  contrastRatio,
  formatSpeakerName,
  getSpeakerColor,
  parseHexColor,
  speakerPaletteIndex,
} from '#src/utils/speaker-appearance.js';

describe('parseHexColor', () => {
  it('parses 6-digit hex colors', () => {
    expect(parseHexColor('#ff8000')).toEqual({ r: 255, g: 128, b: 0 });
  });

  it('parses 3-digit hex colors', () => {
    expect(parseHexColor('#fff')).toEqual({ r: 255, g: 255, b: 255 });
  });

  it('returns null for non-hex colors', () => {
    expect(parseHexColor('rebeccapurple')).toBeNull();
    expect(parseHexColor('rgb(0, 0, 0)')).toBeNull();
  });
});

describe('speakerPaletteIndex', () => {
  it('maps provider labels directly to palette slots', () => {
    expect(speakerPaletteIndex('spk_0')).toBe(0);
    expect(speakerPaletteIndex('spk_3')).toBe(3);
    // Wraps around past the palette size
    expect(speakerPaletteIndex('spk_8')).toBe(0);
  });

  it('assigns custom labels a stable slot', () => {
    expect(speakerPaletteIndex('Alice')).toBe(speakerPaletteIndex('Alice'));
  });
});

describe('getSpeakerColor', () => {
  const white = { r: 255, g: 255, b: 255 };
  const black = { r: 0, g: 0, b: 0 };

  it('meets WCAG AA contrast on white backgrounds for many speakers', () => {
    for (let i = 0; i < 16; i += 1) {
      const color = parseHexColor(
        getSpeakerColor(`spk_${i.toString()}`, '#ffffff'),
      );
      expect(color).not.toBeNull();
      if (color !== null) {
        expect(contrastRatio(color, white)).toBeGreaterThanOrEqual(4.5);
      }
    }
  });

  it('meets WCAG AA contrast on black backgrounds for many speakers', () => {
    for (let i = 0; i < 16; i += 1) {
      const color = parseHexColor(
        getSpeakerColor(`spk_${i.toString()}`, '#000000'),
      );
      expect(color).not.toBeNull();
      if (color !== null) {
        expect(contrastRatio(color, black)).toBeGreaterThanOrEqual(4.5);
      }
    }
  });

  it('gives different speakers different colors on the same background', () => {
    expect(getSpeakerColor('spk_0', '#ffffff')).not.toBe(
      getSpeakerColor('spk_1', '#ffffff'),
    );
  });

  it('is deterministic for the same speaker and background', () => {
    expect(getSpeakerColor('spk_2', '#123456')).toBe(
      getSpeakerColor('spk_2', '#123456'),
    );
  });

  it('falls back to the base palette color on unparseable backgrounds', () => {
    expect(
      parseHexColor(getSpeakerColor('spk_0', 'not-a-color')),
    ).not.toBeNull();
  });
});

describe('formatSpeakerName', () => {
  it('turns provider labels into human-readable names', () => {
    expect(formatSpeakerName('spk_0')).toBe('Speaker 1');
    expect(formatSpeakerName('spk_11')).toBe('Speaker 12');
  });

  it('passes custom labels through unchanged', () => {
    expect(formatSpeakerName('Professor')).toBe('Professor');
  });
});
