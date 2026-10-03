import { Type } from 'typebox';

import {
  type BaseRouteDefinition,
  type BaseWebSocketRouteSchema,
} from '@scribear/base-schema';

import { TranscriptionProviderConfigSchema } from '#src/provider-configs/index.js';

export enum TranscriptionStreamClientMessageType {
  AUTH = 'auth',
  CONFIG = 'config',
}

export enum TranscriptionStreamServerMessageType {
  TRANSCRIPT = 'transcript',
  /**
   * Late speaker labels for a finalized transcript sequence that was already
   * sent. Captions are never held back for diarization: a finalized sequence
   * arrives with a `sequence_id` and whatever labels are known, and this
   * message fills in the rest once the diarization job has covered its
   * audio. Sent only when the provider runs diarization.
   */
  SPEAKERS_UPDATE = 'speakers_update',
}

/**
 * One transcript sequence as the transcription service sends it. `speakers`
 * is aligned with `text` when diarization is on (`null` entries are words
 * without a label yet); `sequence_id` is set on finalized sequences when
 * diarization is on so a later `speakers_update` can name them. Both are
 * optional so a transcription service that predates them still validates.
 */
const TRANSCRIPT_SEQUENCE_SCHEMA = Type.Object({
  text: Type.Array(Type.String()),
  starts: Type.Union([Type.Array(Type.Number()), Type.Null()]),
  ends: Type.Union([Type.Array(Type.Number()), Type.Null()]),
  speakers: Type.Optional(
    Type.Union([
      Type.Array(Type.Union([Type.String(), Type.Null()])),
      Type.Null(),
    ]),
  ),
  sequence_id: Type.Optional(Type.Union([Type.String(), Type.Null()])),
});

const TRANSCRIPTION_STREAM_SCHEMA = {
  description: 'Accepts a connection from an client to a session',
  tags: [],
  params: Type.Object({
    providerKey: Type.String({ maxLength: 32 }),
  }),
  allowClientBinaryMessage: true,
  clientMessage: Type.Union([
    Type.Object({
      type: Type.Literal(TranscriptionStreamClientMessageType.AUTH),
      api_key: Type.String({ maxLength: 1024 }),
    }),
    Type.Object({
      type: Type.Literal(TranscriptionStreamClientMessageType.CONFIG),
      config: TranscriptionProviderConfigSchema,
      // Identifies which session/room this connection belongs to. Optional
      // so a transcription service that predates this field still validates.
      session_uid: Type.Optional(Type.Union([Type.String(), Type.Null()])),
      room_uid: Type.Optional(Type.Union([Type.String(), Type.Null()])),
    }),
  ]),
  allowServerBinaryMessage: false,
  serverMessage: Type.Union([
    Type.Object({
      type: Type.Literal(TranscriptionStreamServerMessageType.TRANSCRIPT),
      final: Type.Union([TRANSCRIPT_SEQUENCE_SCHEMA, Type.Null()]),
      in_progress: Type.Union([TRANSCRIPT_SEQUENCE_SCHEMA, Type.Null()]),
      // Ids of the source audio chunks that contributed to each transcript,
      // echoed back so the node server can correlate a transcript to the audio
      // frame it came from and measure latency. Optional so a transcription
      // service that predates this field still validates.
      final_chunk_ids: Type.Optional(
        Type.Union([Type.Array(Type.String()), Type.Null()]),
      ),
      in_progress_chunk_ids: Type.Optional(
        Type.Union([Type.Array(Type.String()), Type.Null()]),
      ),
    }),
    Type.Object({
      type: Type.Literal(TranscriptionStreamServerMessageType.SPEAKERS_UPDATE),
      // The `sequence_id` a finalized transcript sequence was sent with.
      sequence_id: Type.String(),
      // Aligned with that sequence's `text`. A `null` is a word without a
      // label yet, or - once `settled` - a word that will never get one. A
      // label already sent for a word is never changed by a later update.
      speakers: Type.Array(Type.Union([Type.String(), Type.Null()])),
      // True when no further update follows for this sequence.
      settled: Type.Boolean(),
    }),
  ]),
  closeCodes: {
    1000: { description: 'Normal closure' },
    1001: { description: 'Normal closure, going away' },
    1006: { description: 'Abnormal closure' },
    1007: {
      description: 'Invalid message format or configuration format received',
    },
    1008: { description: 'Authentication failure or timeout' },
    1011: { description: 'Internal server error' },
    1012: { description: 'Service Restart' },
    1013: { description: 'Try again later; refused, at capacity' },
  },
} satisfies BaseWebSocketRouteSchema;

const TRANSCRIPTION_STREAM_ROUTE: BaseRouteDefinition = {
  method: 'GET',
  websocket: true,
  url: '/transcription_stream/:providerKey',
};

export { TRANSCRIPTION_STREAM_SCHEMA, TRANSCRIPTION_STREAM_ROUTE };
