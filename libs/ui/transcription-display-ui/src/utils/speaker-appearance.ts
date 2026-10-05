/**
 * Utilities for rendering speaker labels: display names and colors that stay
 * readable against the user-configurable transcription background.
 */
import { formatSpeakerName } from '@scribear/transcription-content-store';

// The display name lives with the speaker runs in the content store, so the
// caption display, the translated-caption panel and the transcript export
// name a voice the same way; re-exported here for the existing callers.
export { formatSpeakerName };

/**
 * Colorblind-aware base palette (Okabe-Ito, minus black/white). Order matters:
 * speaker N takes the Nth entry, wrapping around when there are more speakers
 * than palette entries.
 */
const SPEAKER_BASE_PALETTE = [
  '#E69F00', // orange
  '#56B4E9', // sky blue
  '#009E73', // bluish green
  '#CC79A7', // reddish purple
  '#0072B2', // blue
  '#D55E00', // vermillion
  '#F0E442', // yellow
  '#999999', // grey
];

/** Minimum WCAG contrast ratio a speaker color must reach against the background. */
const MIN_CONTRAST_RATIO = 4.5;

/** Fraction moved toward white/black on each contrast-adjustment step. */
const ADJUST_STEP = 0.12;

/** Upper bound on adjustment iterations; 20 steps reaches near white/black. */
const MAX_ADJUST_STEPS = 20;

interface Rgb {
  r: number;
  g: number;
  b: number;
}

/**
 * Parses `#rgb` / `#rrggbb` CSS hex colors. Returns `null` for anything else
 * (named colors, rgb() strings) so callers can fall back gracefully.
 */
export const parseHexColor = (color: string): Rgb | null => {
  const hex = color.trim().replace(/^#/, '');
  if (/^[0-9a-fA-F]{3}$/.test(hex)) {
    const r = hex.charAt(0);
    const g = hex.charAt(1);
    const b = hex.charAt(2);
    return parseHexColor(`#${r}${r}${g}${g}${b}${b}`);
  }
  if (!/^[0-9a-fA-F]{6}$/.test(hex)) return null;
  return {
    r: parseInt(hex.slice(0, 2), 16),
    g: parseInt(hex.slice(2, 4), 16),
    b: parseInt(hex.slice(4, 6), 16),
  };
};

const toHex = ({ r, g, b }: Rgb): string => {
  const channel = (v: number) =>
    Math.round(Math.min(255, Math.max(0, v)))
      .toString(16)
      .padStart(2, '0');
  return `#${channel(r)}${channel(g)}${channel(b)}`;
};

/**
 * WCAG 2.x relative luminance of an sRGB color, in [0, 1].
 */
export const relativeLuminance = ({ r, g, b }: Rgb): number => {
  const linear = (channel: number) => {
    const c = channel / 255;
    return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  };
  return 0.2126 * linear(r) + 0.7152 * linear(g) + 0.0722 * linear(b);
};

/**
 * WCAG 2.x contrast ratio between two colors, in [1, 21].
 */
export const contrastRatio = (a: Rgb, b: Rgb): number => {
  const la = relativeLuminance(a);
  const lb = relativeLuminance(b);
  const [dark, light] = la < lb ? [la, lb] : [lb, la];
  return (light + 0.05) / (dark + 0.05);
};

const mixToward = (color: Rgb, target: Rgb, amount: number): Rgb => ({
  r: color.r + (target.r - color.r) * amount,
  g: color.g + (target.g - color.g) * amount,
  b: color.b + (target.b - color.b) * amount,
});

/**
 * Stable palette index for a speaker label. `spk_N` labels map directly to N;
 * any other label hashes deterministically so custom labels still get a
 * consistent color.
 */
export const speakerPaletteIndex = (speaker: string): number => {
  const match = /^spk_(\d+)$/.exec(speaker);
  if (match !== null) {
    return Number(match[1]) % SPEAKER_BASE_PALETTE.length;
  }
  let hash = 0;
  for (let i = 0; i < speaker.length; i += 1) {
    hash = (hash * 31 + speaker.charCodeAt(i)) | 0;
  }
  return Math.abs(hash) % SPEAKER_BASE_PALETTE.length;
};

const WHITE: Rgb = { r: 255, g: 255, b: 255 };
const BLACK: Rgb = { r: 0, g: 0, b: 0 };

/**
 * Returns a CSS color for the speaker that meets {@link MIN_CONTRAST_RATIO}
 * against `backgroundColor`. The base palette color is nudged toward white or
 * black until it is readable. The target is whichever of the two has the
 * higher contrast against the background, not the one a luminance cutoff
 * picks: on a mid-luminance background such as `#5c5c5c` or `#808080` white
 * tops out below 4.5:1 while black clears it, and the old cutoff sent every
 * speaker toward the failing side. Should the steps run out, the target
 * itself is returned, which always meets the ratio: for any background at
 * least one of white and black does. If the background cannot be parsed, the
 * base palette color is returned unchanged.
 */
export const getSpeakerColor = (
  speaker: string,
  backgroundColor: string,
): string => {
  const base = SPEAKER_BASE_PALETTE[speakerPaletteIndex(speaker)] ?? '#999999';
  const background = parseHexColor(backgroundColor);
  const baseRgb = parseHexColor(base);
  if (background === null || baseRgb === null) return base;

  const target: Rgb =
    contrastRatio(WHITE, background) >= contrastRatio(BLACK, background)
      ? WHITE
      : BLACK;

  let color = baseRgb;
  for (
    let step = 0;
    step < MAX_ADJUST_STEPS &&
    contrastRatio(color, background) < MIN_CONTRAST_RATIO;
    step += 1
  ) {
    color = mixToward(color, target, ADJUST_STEP);
  }
  if (contrastRatio(color, background) < MIN_CONTRAST_RATIO) {
    color = target;
  }
  return toHex(color);
};
