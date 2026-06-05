import type { VoiceRuntimeStatus } from './api'
import type { VoiceRuntimePhase } from './voice-ws'

const UI_VOICE_PHASES = new Set<VoiceRuntimePhase>([
  'idle',
  'connecting',
  'ready',
  'listening',
  'transcribing',
  'thinking',
  'synthesizing',
  'error',
])

export function resolveVoicePhaseFromRuntime(
  status: VoiceRuntimeStatus | null,
): VoiceRuntimePhase | null {
  if (!status) {
    return null
  }

  if (status.error) {
    return 'error'
  }

  const phase = status.phase?.toLowerCase()
  if (phase && UI_VOICE_PHASES.has(phase as VoiceRuntimePhase)) {
    return phase as VoiceRuntimePhase
  }

  // Runtime status reports backend readiness, not websocket connection progress.
  if (status.ready) {
    return 'ready'
  }

  return null
}

export function resolveVoiceOverlayPhase(
  clientPhase: VoiceRuntimePhase,
  runtimeStatus: VoiceRuntimeStatus | null,
): VoiceRuntimePhase {
  const runtimePhase = resolveVoicePhaseFromRuntime(runtimeStatus)
  if (clientPhase !== 'idle') {
    return clientPhase
  }
  return runtimePhase ?? 'idle'
}
