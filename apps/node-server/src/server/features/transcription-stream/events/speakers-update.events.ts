import { SPEAKERS_UPDATE_SCHEMA } from '@scribear/node-server-schema';

import type { ChannelDefinition } from '#src/server/shared/services/event-bus.service.js';

/**
 * Bus channel keyed by sessionUid for late speaker labels. The orchestrator
 * publishes one whenever the upstream provider labels a finalized transcript
 * fragment it had already delivered (`speakers_update` upstream); the
 * per-connection services forward it as a `speakersUpdate` server message.
 * The body is the server message minus its `type`, exactly as
 * {@link TranscriptChannel} carries the transcript body.
 */
export const SpeakersUpdateChannel: ChannelDefinition<
  typeof SPEAKERS_UPDATE_SCHEMA,
  [string]
> = {
  schema: SPEAKERS_UPDATE_SCHEMA,
  key: (sessionUid) => `speakers-update:${sessionUid}`,
};
