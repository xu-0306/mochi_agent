import assert from 'node:assert/strict'
import { decodeSseJsonFrames, normalizeAdaptiveRuntimeEnvelope } from './ordinary-chat-runtime-stream.ts'
const encoder = new TextEncoder()
const chunks = [
  encoder.encode('event: ordinary_chat_adaptive_runtime\r\ndata: {"type":"ordinary_chat_adaptive_runtime_event",\r'),
  encoder.encode('\ndata: "event":{"event_id":"e"}}\r\n\r\ndata: {bad}\n\n'),
]
const stream = new ReadableStream<Uint8Array>({ start(controller) { for (const chunk of chunks) controller.enqueue(chunk); controller.close() } })
const frames = []; for await (const frame of decodeSseJsonFrames(stream)) frames.push(frame)
const envelope = normalizeAdaptiveRuntimeEnvelope(frames[0])
assert.equal((envelope?.event as { event_id: string }).event_id, 'e')
console.log('ordinary chat runtime stream tests passed')
