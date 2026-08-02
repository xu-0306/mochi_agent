/** Dependency-free adaptive-runtime SSE envelope boundary. */
export type AdaptiveRuntimeEnvelope = { type: string; event: unknown }
export function normalizeAdaptiveRuntimeEnvelope(value: unknown): AdaptiveRuntimeEnvelope | null {
  if (!value || typeof value !== 'object') return null
  const frame = value as Record<string, unknown>
  if (frame.type !== 'ordinary_chat_adaptive_runtime_event' && frame.type !== 'ordinary_chat_adaptive_runtime') return null
  return { type: String(frame.type), event: frame.event }
}

export async function* decodeSseJsonFrames(stream: ReadableStream<Uint8Array>): AsyncGenerator<unknown> {
  const reader = stream.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  const consume = function* (source: string): Generator<unknown> {
    for (const frame of source.replaceAll('\r\n', '\n').split('\n\n')) {
      const data = frame
        .split('\n')
        .filter((line) => line.startsWith('data:'))
        .map((line) => line.slice(5).trimStart())
        .join('\n')
      if (!data || data === '[DONE]') continue
      try {
        yield JSON.parse(data)
      } catch {
        // Malformed public frames are ignored; consumers resume from their
        // exact durable cursor rather than treating arbitrary text as state.
      }
    }
  }
  try {
    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      // Normalize after concatenation so a CR/LF delimiter split across two
      // network chunks is still recognized.
      buffer = buffer.replaceAll('\r\n', '\n')
      const pivot = buffer.lastIndexOf('\n\n')
      if (pivot < 0) continue
      const ready = buffer.slice(0, pivot)
      buffer = buffer.slice(pivot + 2)
      yield* consume(ready)
    }
    buffer += decoder.decode()
    if (buffer.trim()) yield* consume(buffer)
  } finally {
    reader.releaseLock()
  }
}
