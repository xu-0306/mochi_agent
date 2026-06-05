const METER_PATTERN = [
  0.18, 0.32, 0.48, 0.68, 0.42, 0.78, 0.56, 0.28,
  0.44, 0.64, 0.36, 0.58, 0.82, 0.46, 0.62, 0.74,
  0.52, 0.88, 0.66, 0.4, 0.72, 0.54, 0.34, 0.5,
  0.7, 0.43, 0.24, 0.47, 0.31, 0.16, 0.38, 0.22,
]

export const VOICE_INPUT_SIGNAL_THRESHOLD = 0.015
export const VOICE_AUTO_STOP_SILENCE_MS = 900

export interface VoiceEndpointTracker {
  speechDetected: boolean
  silenceDurationMs: number
}

export interface VoiceEndpointUpdate {
  next: VoiceEndpointTracker
  speechStarted: boolean
  speechEnded: boolean
  shouldFinalize: boolean
}

export function calculateVoiceInputLevel(input: Float32Array): number {
  if (input.length === 0) {
    return 0
  }

  let sumSquares = 0
  for (let i = 0; i < input.length; i += 1) {
    const sample = input[i] ?? 0
    sumSquares += sample * sample
  }

  const rms = Math.sqrt(sumSquares / input.length)
  return Math.max(0, Math.min(1, rms * 6))
}

export function hasVoiceInputSignal(level: number): boolean {
  return level >= VOICE_INPUT_SIGNAL_THRESHOLD
}

export function advanceVoiceEndpointTracker(
  tracker: VoiceEndpointTracker,
  hasSignal: boolean,
  frameDurationMs: number,
  silenceTimeoutMs = VOICE_AUTO_STOP_SILENCE_MS,
): VoiceEndpointUpdate {
  if (hasSignal) {
    return {
      next: {
        speechDetected: true,
        silenceDurationMs: 0,
      },
      speechStarted: !tracker.speechDetected,
      speechEnded: false,
      shouldFinalize: false,
    }
  }

  if (!tracker.speechDetected) {
    return {
      next: tracker,
      speechStarted: false,
      speechEnded: false,
      shouldFinalize: false,
    }
  }

  const nextSilenceDurationMs = tracker.silenceDurationMs + Math.max(0, frameDurationMs)
  const shouldFinalize = nextSilenceDurationMs >= silenceTimeoutMs
  return {
    next: shouldFinalize
      ? { speechDetected: false, silenceDurationMs: 0 }
      : { speechDetected: true, silenceDurationMs: nextSilenceDurationMs },
    speechStarted: false,
    speechEnded: shouldFinalize,
    shouldFinalize,
  }
}

export function buildVoiceMeterAmplitudes(
  level: number,
  barCount = 32,
): number[] {
  const safeLevel = Math.max(0, Math.min(1, level))
  const floor = 0.08
  const gain = 0.18 + safeLevel * 0.82

  return Array.from({ length: barCount }, (_, index) => {
    const patternValue = METER_PATTERN[index % METER_PATTERN.length] ?? 0.25
    return Math.max(floor, Math.min(1, floor + patternValue * gain))
  })
}
