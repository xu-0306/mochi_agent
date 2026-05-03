'use client'

const DEFAULT_INPUT_SAMPLE_RATE_HZ = 16_000
const DEFAULT_OUTPUT_PCM_SAMPLE_RATE_HZ = 24_000
const PROCESSOR_BUFFER_SIZE = 4096

export type VoiceRuntimePhase =
  | 'idle'
  | 'connecting'
  | 'ready'
  | 'listening'
  | 'transcribing'
  | 'thinking'
  | 'synthesizing'
  | 'error'

export type VoiceStage = 'transcribing' | 'thinking' | 'synthesizing'

export interface VoiceTurnResult {
  turnId: number | null
  finalTranscription: string
  assistantText: string
}

export interface VoiceWsCallbacks {
  onPhaseChange?: (phase: VoiceRuntimePhase) => void
  onPartialTranscription?: (text: string) => void
  onFinalTranscription?: (text: string) => void
  onAssistantText?: (text: string) => void
  onTurnDone?: (result: VoiceTurnResult) => void
  onError?: (message: string, code?: string) => void
}

export interface VoiceWsClientOptions extends VoiceWsCallbacks {
  sessionId?: string
  idleTimeoutSeconds?: number
  inputSampleRateHz?: number
  outputPcmSampleRateHz?: number
}

interface VoiceServerMessageBase {
  type?: unknown
  turn_id?: unknown
  turnId?: unknown
}

interface VoiceTranscriptionMessage extends VoiceServerMessageBase {
  text?: unknown
  is_final?: unknown
  final?: unknown
}

interface VoiceStageMessage extends VoiceServerMessageBase {
  stage?: unknown
}

interface VoiceTextMessage extends VoiceServerMessageBase {
  text?: unknown
}

interface VoiceAudioChunkMessage extends VoiceServerMessageBase {
  data?: unknown
}

interface VoiceErrorMessage extends VoiceServerMessageBase {
  code?: unknown
  message?: unknown
}

interface VoiceTurnAccumulator {
  partialText: string
  finalText: string
  assistantText: string
  audioChunks: Uint8Array[]
}

export class VoiceWsClient {
  private readonly options: VoiceWsClientOptions
  private readonly turns = new Map<number | null, VoiceTurnAccumulator>()
  private readonly turnOrder: Array<number | null> = []

  private ws: WebSocket | null = null
  private stream: MediaStream | null = null
  private audioContext: AudioContext | null = null
  private sourceNode: MediaStreamAudioSourceNode | null = null
  private processorNode: ScriptProcessorNode | null = null
  private isRecording = false
  private phase: VoiceRuntimePhase = 'idle'
  private pendingPlay = Promise.resolve()

  constructor(options: VoiceWsClientOptions) {
    this.options = options
  }

  get currentPhase(): VoiceRuntimePhase {
    return this.phase
  }

  get recording(): boolean {
    return this.isRecording
  }

  async connect(): Promise<void> {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.setPhase('ready')
      return
    }

    await this.disconnect()
    this.setPhase('connecting')
    this.turns.clear()
    this.turnOrder.length = 0

    await new Promise<void>((resolve, reject) => {
      const ws = new WebSocket(this.buildSocketUrl())
      this.ws = ws

      const handleOpen = () => {
        cleanup()
        this.attachSocketHandlers(ws)
        this.setPhase('ready')
        resolve()
      }

      const handleError = () => {
        cleanup()
        reject(new Error('Unable to connect voice websocket.'))
      }

      const cleanup = () => {
        ws.removeEventListener('open', handleOpen)
        ws.removeEventListener('error', handleError)
      }

      ws.addEventListener('open', handleOpen)
      ws.addEventListener('error', handleError)
    })
  }

  async startRecording(): Promise<void> {
    if (this.isRecording) {
      return
    }
    await this.connect()

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        sampleRate: this.options.inputSampleRateHz ?? DEFAULT_INPUT_SAMPLE_RATE_HZ,
      },
      video: false,
    })
    this.stream = stream

    const audioContext = new AudioContext()
    this.audioContext = audioContext
    if (audioContext.state === 'suspended') {
      await audioContext.resume()
    }

    const sourceNode = audioContext.createMediaStreamSource(stream)
    this.sourceNode = sourceNode

    // Browser compatibility fallback: AudioWorklet is preferred, but not yet wired in this repo.
    const processorNode = audioContext.createScriptProcessor(PROCESSOR_BUFFER_SIZE, 1, 1)
    this.processorNode = processorNode

    processorNode.onaudioprocess = (event: AudioProcessingEvent) => {
      if (!this.isRecording) {
        return
      }
      const input = event.inputBuffer.getChannelData(0)
      const pcm16 = resampleFloat32ToPcm16(
        input,
        audioContext.sampleRate,
        this.options.inputSampleRateHz ?? DEFAULT_INPUT_SAMPLE_RATE_HZ,
      )
      if (pcm16.length === 0) {
        return
      }
      this.sendJson({
        type: 'audio_chunk',
        data: int16ToBase64(pcm16),
      })
    }

    sourceNode.connect(processorNode)
    processorNode.connect(audioContext.destination)

    this.isRecording = true
    this.setPhase('listening')
  }

  async stopRecording(): Promise<void> {
    if (!this.isRecording) {
      return
    }
    this.isRecording = false
    this.stopAudioCapture()
    this.sendJson({ type: 'vad_end' })
    this.setPhase('transcribing')
  }

  interrupt(): void {
    this.isRecording = false
    this.stopAudioCapture()
    this.sendJson({ type: 'interrupt' })
    this.setPhase('ready')
  }

  async disconnect(): Promise<void> {
    this.isRecording = false
    this.stopAudioCapture()

    if (this.ws) {
      const ws = this.ws
      this.ws = null
      try {
        ws.close()
      } catch {
        // noop
      }
    }
    this.setPhase('idle')
  }

  private attachSocketHandlers(ws: WebSocket): void {
    ws.onmessage = (event) => {
      if (typeof event.data !== 'string') {
        return
      }
      let payload: unknown
      try {
        payload = JSON.parse(event.data)
      } catch {
        return
      }
      this.handleServerMessage(payload)
    }

    ws.onerror = () => {
      this.raiseError('Voice websocket error.')
    }

    ws.onclose = () => {
      this.ws = null
      if (this.phase !== 'idle') {
        this.setPhase('idle')
      }
    }
  }

  private handleServerMessage(payload: unknown): void {
    if (!isRecord(payload)) {
      return
    }
    const msgType = typeof payload.type === 'string' ? payload.type : ''
    const turnId = this.extractTurnId(payload)

    if (msgType === 'transcription') {
      const msg = payload as unknown as VoiceTranscriptionMessage
      const text = typeof msg.text === 'string' ? msg.text : ''
      const isFinal = Boolean(msg.is_final ?? msg.final)
      const turn = this.getTurn(turnId)
      if (isFinal) {
        turn.finalText = text
        this.options.onFinalTranscription?.(text)
      } else {
        turn.partialText = text
        this.options.onPartialTranscription?.(text)
      }
      return
    }

    if (msgType === 'voice_stage') {
      const msg = payload as unknown as VoiceStageMessage
      const stage = typeof msg.stage === 'string' ? msg.stage : ''
      if (stage === 'transcribing' || stage === 'thinking' || stage === 'synthesizing') {
        this.setPhase(stage)
      }
      return
    }

    if (msgType === 'text') {
      const msg = payload as unknown as VoiceTextMessage
      const text = typeof msg.text === 'string' ? msg.text : ''
      const turn = this.getTurn(turnId)
      turn.assistantText = text
      this.options.onAssistantText?.(text)
      return
    }

    if (msgType === 'audio_chunk') {
      const msg = payload as unknown as VoiceAudioChunkMessage
      if (typeof msg.data === 'string') {
        const turn = this.getTurn(turnId)
        turn.audioChunks.push(base64ToBytes(msg.data))
      }
      return
    }

    if (msgType === 'done') {
      const msg = payload as unknown as VoiceServerMessageBase
      void this.handleTurnDone(this.extractTurnId(msg))
      return
    }

    if (msgType === 'interrupted') {
      this.turns.clear()
      this.turnOrder.length = 0
      this.setPhase('ready')
      return
    }

    if (msgType === 'error') {
      const msg = payload as unknown as VoiceErrorMessage
      this.raiseError(
        typeof msg.message === 'string' ? msg.message : 'Voice request failed.',
        typeof msg.code === 'string' ? msg.code : undefined,
      )
      return
    }
  }

  private async handleTurnDone(turnId: number | null): Promise<void> {
    const turn = this.turns.get(turnId)
    if (!turn) {
      this.setPhase('ready')
      return
    }

    const audio = joinChunks(turn.audioChunks)
    if (audio.length > 0) {
      this.pendingPlay = this.pendingPlay.then(() => this.playAudio(audio))
      await this.pendingPlay
    }

    this.options.onTurnDone?.({
      turnId,
      finalTranscription: turn.finalText,
      assistantText: turn.assistantText,
    })
    this.turns.delete(turnId)
    this.turnOrder.splice(this.turnOrder.indexOf(turnId), 1)

    if (!this.isRecording) {
      this.setPhase('ready')
    }
  }

  private async playAudio(audioBytes: Uint8Array): Promise<void> {
    const audioContext = this.audioContext ?? new AudioContext()
    this.audioContext = audioContext
    if (audioContext.state === 'suspended') {
      await audioContext.resume()
    }

    const decoded = await decodeAudioBufferWithFallback(
      audioContext,
      audioBytes,
      this.options.outputPcmSampleRateHz ?? DEFAULT_OUTPUT_PCM_SAMPLE_RATE_HZ,
    )
    if (!decoded) {
      return
    }
    const source = audioContext.createBufferSource()
    source.buffer = decoded
    source.connect(audioContext.destination)
    source.start(0)
    await waitForAudioSourceEnded(source)
  }

  private stopAudioCapture(): void {
    if (this.processorNode) {
      this.processorNode.onaudioprocess = null
      this.processorNode.disconnect()
      this.processorNode = null
    }
    if (this.sourceNode) {
      this.sourceNode.disconnect()
      this.sourceNode = null
    }
    if (this.stream) {
      for (const track of this.stream.getTracks()) {
        track.stop()
      }
      this.stream = null
    }
  }

  private sendJson(payload: Record<string, unknown>): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return
    }
    this.ws.send(JSON.stringify(payload))
  }

  private getTurn(turnId: number | null): VoiceTurnAccumulator {
    const existing = this.turns.get(turnId)
    if (existing) {
      return existing
    }
    const next: VoiceTurnAccumulator = {
      partialText: '',
      finalText: '',
      assistantText: '',
      audioChunks: [],
    }
    this.turns.set(turnId, next)
    this.turnOrder.push(turnId)
    return next
  }

  private extractTurnId(payload: VoiceServerMessageBase): number | null {
    const candidate = payload.turn_id ?? payload.turnId
    if (typeof candidate === 'number' && Number.isFinite(candidate)) {
      return candidate
    }
    if (typeof candidate === 'string' && candidate.trim().length > 0) {
      const parsed = Number.parseInt(candidate, 10)
      return Number.isFinite(parsed) ? parsed : null
    }
    return null
  }

  private setPhase(phase: VoiceRuntimePhase): void {
    if (this.phase === phase) {
      return
    }
    this.phase = phase
    this.options.onPhaseChange?.(phase)
  }

  private raiseError(message: string, code?: string): void {
    this.setPhase('error')
    this.options.onError?.(message, code)
  }

  private buildSocketUrl(): string {
    const origin = resolveVoiceOrigin()
    const protocol = origin.protocol === 'https:' ? 'wss:' : 'ws:'
    const query = new URLSearchParams()
    if (this.options.sessionId) {
      query.set('session_id', this.options.sessionId)
    }
    if (typeof this.options.idleTimeoutSeconds === 'number') {
      query.set('idle_timeout_seconds', String(this.options.idleTimeoutSeconds))
    }
    const suffix = query.size > 0 ? `?${query.toString()}` : ''
    return `${protocol}//${origin.host}/v1/voice${suffix}`
  }
}

function resolveVoiceOrigin(): URL {
  const configuredOrigin = process.env.NEXT_PUBLIC_MOCHI_API_BASE_URL?.trim()
  if (configuredOrigin) {
    return new URL(configuredOrigin)
  }

  const { hostname, port, protocol } = window.location
  if ((hostname === 'localhost' || hostname === '127.0.0.1') && port === '3000') {
    return new URL(`${protocol}//127.0.0.1:8000`)
  }

  return new URL(window.location.origin)
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function resampleFloat32ToPcm16(
  input: Float32Array,
  inputRate: number,
  outputRate: number,
): Int16Array {
  if (input.length === 0) {
    return new Int16Array(0)
  }

  if (inputRate === outputRate) {
    const direct = new Int16Array(input.length)
    for (let i = 0; i < input.length; i += 1) {
      direct[i] = floatToInt16(input[i])
    }
    return direct
  }

  const ratio = inputRate / outputRate
  const outputLength = Math.max(1, Math.floor(input.length / ratio))
  const result = new Int16Array(outputLength)
  for (let i = 0; i < outputLength; i += 1) {
    const index = i * ratio
    const left = Math.floor(index)
    const right = Math.min(left + 1, input.length - 1)
    const frac = index - left
    const sample = input[left] + (input[right] - input[left]) * frac
    result[i] = floatToInt16(sample)
  }
  return result
}

function floatToInt16(sample: number): number {
  const clamped = Math.max(-1, Math.min(1, sample))
  return clamped < 0 ? Math.round(clamped * 0x8000) : Math.round(clamped * 0x7fff)
}

function int16ToBase64(samples: Int16Array): string {
  const view = new Uint8Array(samples.buffer, samples.byteOffset, samples.byteLength)
  let binary = ''
  for (let i = 0; i < view.length; i += 1) {
    binary += String.fromCharCode(view[i])
  }
  return window.btoa(binary)
}

function base64ToBytes(base64: string): Uint8Array {
  const binary = window.atob(base64)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i)
  }
  return bytes
}

function joinChunks(chunks: Uint8Array[]): Uint8Array {
  const total = chunks.reduce((sum, chunk) => sum + chunk.byteLength, 0)
  const merged = new Uint8Array(total)
  let offset = 0
  for (const chunk of chunks) {
    merged.set(chunk, offset)
    offset += chunk.byteLength
  }
  return merged
}

async function decodeAudioBufferWithFallback(
  context: AudioContext,
  bytes: Uint8Array,
  pcmFallbackSampleRateHz: number,
): Promise<AudioBuffer | null> {
  const firstAttempt = await safeDecodeAudioData(context, bytes)
  if (firstAttempt) {
    return firstAttempt
  }

  const wavWrapped = wrapPcm16AsWav(bytes, pcmFallbackSampleRateHz, 1)
  return safeDecodeAudioData(context, wavWrapped)
}

async function safeDecodeAudioData(
  context: AudioContext,
  bytes: Uint8Array,
): Promise<AudioBuffer | null> {
  try {
    const copy = new Uint8Array(bytes)
    return await context.decodeAudioData(copy.buffer)
  } catch {
    return null
  }
}

function wrapPcm16AsWav(pcm: Uint8Array, sampleRate: number, channels: number): Uint8Array {
  const headerSize = 44
  const wav = new Uint8Array(headerSize + pcm.byteLength)
  const view = new DataView(wav.buffer)
  const blockAlign = channels * 2
  const byteRate = sampleRate * blockAlign

  writeAscii(view, 0, 'RIFF')
  view.setUint32(4, 36 + pcm.byteLength, true)
  writeAscii(view, 8, 'WAVE')
  writeAscii(view, 12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true)
  view.setUint16(22, channels, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, byteRate, true)
  view.setUint16(32, blockAlign, true)
  view.setUint16(34, 16, true)
  writeAscii(view, 36, 'data')
  view.setUint32(40, pcm.byteLength, true)
  wav.set(pcm, headerSize)
  return wav
}

function writeAscii(view: DataView, offset: number, value: string): void {
  for (let i = 0; i < value.length; i += 1) {
    view.setUint8(offset + i, value.charCodeAt(i))
  }
}

function waitForAudioSourceEnded(source: AudioBufferSourceNode): Promise<void> {
  return new Promise((resolve) => {
    source.onended = () => resolve()
  })
}
