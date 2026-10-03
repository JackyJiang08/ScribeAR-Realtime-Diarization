import { type Static, Type } from 'typebox';

/**
 * One delivery of transcribed text from the upstream provider. `text` holds
 * the words as separate tokens so callers can render at word granularity.
 * `starts` / `ends` are aligned with `text` and carry seconds-from-stream-start
 * timestamps when the provider supplies them; `null` means the provider does
 * not emit per-token timing. `speakers` is aligned with `text` and carries
 * per-token speaker labels when the provider runs diarization; `null` (or a
 * null entry) means no speaker attribution is available (yet). `sequenceId`
 * is set on a finalized fragment when the provider runs diarization and may
 * still send labels for it later through a `speakersUpdate` message; a
 * client that ignores both still shows the caption text.
 */
export const TRANSCRIPT_FRAGMENT_SCHEMA = Type.Object(
  {
    text: Type.Array(Type.String()),
    starts: Type.Union([Type.Array(Type.Number()), Type.Null()]),
    ends: Type.Union([Type.Array(Type.Number()), Type.Null()]),
    speakers: Type.Optional(
      Type.Union([
        Type.Array(Type.Union([Type.String(), Type.Null()])),
        Type.Null(),
      ]),
    ),
    sequenceId: Type.Optional(Type.Union([Type.String(), Type.Null()])),
  },
  { $id: 'TranscriptFragment' },
);

/**
 * Late speaker labels for a finalized fragment that was already delivered,
 * named by the `sequenceId` it carried. `speakers` is aligned with that
 * fragment's `text`; a `null` entry is a word without a label yet, or - once
 * `settled` is true - a word that will never get one. A label already shown
 * for a word is never changed by a later update: updates only fill in nulls.
 */
export const SPEAKERS_UPDATE_SCHEMA = Type.Object(
  {
    sequenceId: Type.String(),
    speakers: Type.Array(Type.Union([Type.String(), Type.Null()])),
    settled: Type.Boolean(),
  },
  { $id: 'SpeakersUpdate' },
);

export type SpeakersUpdate = Static<typeof SPEAKERS_UPDATE_SCHEMA>;

export type TranscriptFragment = Static<typeof TRANSCRIPT_FRAGMENT_SCHEMA>;
