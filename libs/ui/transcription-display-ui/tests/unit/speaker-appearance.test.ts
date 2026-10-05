import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
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

describe('getSpeakerColor on every background the apps ship', () => {
  // Every `backgroundColor` of libs/ui/theme-customization-ui's preset
  // themes, read from the source so a new preset is covered the day it is
  // added, plus the store default (`BASE_BACKGROUND_COLOR`, black).
  const presetSource = readFileSync(
    resolve(
      __dirname,
      '../../../theme-customization-ui/src/config/preset-themes.ts',
    ),
    'utf8',
  );
  const presetBackgrounds = Array.from(
    presetSource.matchAll(/backgroundColor:\s*'(#[0-9a-fA-F]{3,6})'/g),
    (match) => match[1] ?? '',
  );
  const backgrounds = Array.from(new Set(['#000000', ...presetBackgrounds]));

  it('finds every preset background', () => {
    // 23 distinct values at the time of writing; the regex must keep seeing
    // them or this suite silently tests nothing.
    expect(backgrounds.length).toBeGreaterThanOrEqual(20);
    expect(backgrounds).toContain('#5c5c5c');
    expect(backgrounds).toContain('#c2c2c2');
  });

  it.each(backgrounds)('reaches 4.5:1 for 16 speakers on %s', (background) => {
    const bg = parseHexColor(background);
    expect(bg).not.toBeNull();
    for (let i = 0; i < 16; i += 1) {
      const color = parseHexColor(
        getSpeakerColor(`spk_${i.toString()}`, background),
      );
      expect(color).not.toBeNull();
      if (bg && color) {
        expect(contrastRatio(color, bg)).toBeGreaterThanOrEqual(4.5);
      }
    }
  });

  it('reaches 4.5:1 on mid-luminance backgrounds the old cutoff failed', () => {
    for (const background of ['#808080', '#6e6e6e', '#5c5c5c', '#999999']) {
      const bg = parseHexColor(background);
      for (let i = 0; i < 8; i += 1) {
        const color = parseHexColor(
          getSpeakerColor(`spk_${i.toString()}`, background),
        );
        expect(bg && color && contrastRatio(color, bg)).toBeGreaterThanOrEqual(
          4.5,
        );
      }
    }
  });

  it('reaches 4.5:1 on any picker colour', () => {
    // A deterministic walk through the colour cube, the way a user with the
    // free picker might land anywhere.
    for (let r = 0; r < 256; r += 51) {
      for (let g = 0; g < 256; g += 51) {
        for (let b = 0; b < 256; b += 51) {
          const background = `#${[r, g, b]
            .map((v) => v.toString(16).padStart(2, '0'))
            .join('')}`;
          const bg = parseHexColor(background);
          const color = parseHexColor(getSpeakerColor('spk_3', background));
          expect(
            bg && color && contrastRatio(color, bg),
          ).toBeGreaterThanOrEqual(4.5);
        }
      }
    }
  });
});
