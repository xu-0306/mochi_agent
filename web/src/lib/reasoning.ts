import type { ReasoningStep } from '@/lib/chat'

export interface InlineReasoningExtraction {
  content: string
  reasoning: string | null
}

export interface InlineReasoningStepOptions {
  index: number
  turnKey: string | null
  timestamp?: string
}

export interface InlineReasoningBuffer {
  visible: string
  reasoning: string
  pendingTag: string
  activeTag: string | null
  isInsideThink: boolean
  startedReasoningBlock: boolean
}

export interface InlineReasoningChunkResult {
  buffer: InlineReasoningBuffer
  contentDelta: string
  reasoningDelta: string | null
}

const CHANNEL_MARKER_RE = /(?:<\|channel\|?>|<channel\|>|<\|message\|?>|<message\|>)/gi
const HEADER_MARKER_RE = /(?:<\|start_header_id\|>|<\|end_header_id\|>|<\|im_start\|>|<\|im_end\|>|<\|eot_id\|>)/gi
const ROLE_SENTINEL_RE = /<[|｜]\s*(?:assistant|user|system|tool)\s*[|｜]>/gi
const CHANNEL_REASONING_PREFIX_RE =
  /^\s*(?:(?:<\|channel\|?>|<channel\|>|<\|message\|?>|<message\|>)\s*)+(?:(?:thought|analysis|reasoning)\s*)?(?:(?:<\|channel\|?>|<channel\|>|<\|message\|?>|<message\|>)\s*)*/i
const ROLE_PREFIX_RE =
  /^\s*(?:(?:<\|start_header_id\|>|<\|im_start\|>)\s*)+(?:assistant|user|system|tool)\s*(?:(?:<\|end_header_id\|>|<\|im_end\|>|<\|eot_id\|>)\s*)*/i
const ROLE_SENTINEL_PREFIX_RE = /^\s*<[|｜]\s*(?:assistant|user|system|tool)\s*[|｜]>\s*/i
const REASONING_TAGS = ['think', 'analysis', 'reasoning'] as const
const REASONING_OPEN_TAGS = REASONING_TAGS.map((tag) => `<${tag}>`)

function normalizeTagName(tag: string): string {
  return tag.trim().toLowerCase()
}

function findOpeningReasoningTag(source: string): { index: number; tag: string; token: string } | null {
  let bestMatch: { index: number; tag: string; token: string } | null = null
  for (const tag of REASONING_TAGS) {
    const token = `<${tag}>`
    const index = source.toLowerCase().indexOf(token)
    if (index === -1) {
      continue
    }
    if (bestMatch === null || index < bestMatch.index) {
      bestMatch = { index, tag, token }
    }
  }
  return bestMatch
}

function findReasoningClosingTag(source: string): { index: number; tag: string; token: string } | null {
  let bestMatch: { index: number; tag: string; token: string } | null = null
  for (const tag of REASONING_TAGS) {
    const token = `</${tag}>`
    const index = source.toLowerCase().indexOf(token)
    if (index === -1) {
      continue
    }
    if (bestMatch === null || index < bestMatch.index) {
      bestMatch = { index, tag, token }
    }
  }
  return bestMatch
}

function findPartialTagSuffix(source: string, candidates: ReadonlyArray<string>): string {
  const lowered = source.toLowerCase()
  for (let size = Math.min(source.length, Math.max(...candidates.map((candidate) => candidate.length - 1))); size > 0; size -= 1) {
    const suffix = lowered.slice(-size)
    if (candidates.some((candidate) => candidate.startsWith(suffix) && candidate !== suffix)) {
      return source.slice(-size)
    }
  }
  return ''
}

function normalizeTextBlock(value: string): string {
  return value
    .replace(/\r\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim()
}

export function stripReasoningArtifacts(value: string): string {
  return normalizeTextBlock(
    value
      .replace(CHANNEL_REASONING_PREFIX_RE, '')
      .replace(ROLE_PREFIX_RE, '')
      .replace(ROLE_SENTINEL_PREFIX_RE, '')
      .replace(CHANNEL_MARKER_RE, '')
      .replace(HEADER_MARKER_RE, '')
      .replace(ROLE_SENTINEL_RE, '')
  )
}

function consumeReasoningStream(
  buffer: InlineReasoningBuffer,
  chunk: string,
): InlineReasoningChunkResult {
  let source = `${buffer.pendingTag}${chunk.replace(/\r\n/g, '\n')}`
  let visible = buffer.visible
  let reasoning = buffer.reasoning
  let contentDelta = ''
  let reasoningDelta = ''
  let activeTag = buffer.activeTag
  let isInsideThink = buffer.isInsideThink
  let startedReasoningBlock = buffer.startedReasoningBlock

  while (source.length > 0) {
    if (activeTag) {
      const closeToken = `</${activeTag}>`
      const closeIndex = source.toLowerCase().indexOf(closeToken)
      if (closeIndex === -1) {
        const pendingTag = findPartialTagSuffix(source, [closeToken])
        const commit = pendingTag ? source.slice(0, -pendingTag.length) : source
        reasoning += commit
        reasoningDelta += commit
        return {
          buffer: {
            visible,
            reasoning,
            pendingTag,
            activeTag,
            isInsideThink: true,
            startedReasoningBlock,
          },
          contentDelta,
          reasoningDelta: normalizeTextBlock(reasoningDelta) || null,
        }
      }

      const nextReasoning = source.slice(0, closeIndex)
      reasoning += nextReasoning
      reasoningDelta += nextReasoning
      source = source.slice(closeIndex + closeToken.length)
      activeTag = null
      isInsideThink = false
      continue
    }

    const openMatch = findOpeningReasoningTag(source)
    if (openMatch === null) {
      const pendingTag = findPartialTagSuffix(source, REASONING_OPEN_TAGS)
      const commit = pendingTag ? source.slice(0, -pendingTag.length) : source
      visible += commit
      contentDelta += commit
      return {
        buffer: {
          visible,
          reasoning,
          pendingTag,
          activeTag: null,
          isInsideThink: false,
          startedReasoningBlock: buffer.startedReasoningBlock,
        },
        contentDelta,
        reasoningDelta: normalizeTextBlock(reasoningDelta) || null,
      }
    }

    const nextVisible = source.slice(0, openMatch.index)
    visible += nextVisible
    contentDelta += nextVisible
    source = source.slice(openMatch.index + openMatch.token.length)
    if (startedReasoningBlock || reasoning.length > 0) {
      reasoning += '\n\n'
      reasoningDelta += '\n\n'
    }
    startedReasoningBlock = true
    activeTag = normalizeTagName(openMatch.tag)
    isInsideThink = true
  }

  return {
    buffer: {
      visible,
      reasoning,
      pendingTag: '',
      activeTag,
      isInsideThink,
      startedReasoningBlock,
    },
    contentDelta,
    reasoningDelta: normalizeTextBlock(reasoningDelta) || null,
  }
}

export function extractInlineReasoning(content: string): InlineReasoningExtraction {
  const normalized = content.replace(/\r\n/g, '\n')
  const closingOnlyMatch = findReasoningClosingTag(normalized)
  if (!findOpeningReasoningTag(normalized) && closingOnlyMatch !== null) {
      const hidden = stripReasoningArtifacts(normalized.slice(0, closingOnlyMatch.index))
      const visible = stripReasoningArtifacts(
        normalized.slice(closingOnlyMatch.index + closingOnlyMatch.token.length)
      )
      return {
        content: visible,
        reasoning: hidden || null,
      }
  }

  const streamed = appendInlineReasoningChunk(createInlineReasoningBuffer(), normalized)
  const finalized = finalizeInlineReasoningBuffer(streamed.buffer)

  return {
    content: finalized.content,
    reasoning: finalized.reasoning,
  }
}

export function createInlineReasoningBuffer(): InlineReasoningBuffer {
  return {
    visible: '',
    reasoning: '',
    pendingTag: '',
    activeTag: null,
    isInsideThink: false,
    startedReasoningBlock: false,
  }
}

export function appendInlineReasoningChunk(
  buffer: InlineReasoningBuffer,
  chunk: string,
): InlineReasoningChunkResult {
  return consumeReasoningStream(buffer, chunk)
}

export function finalizeInlineReasoningBuffer(buffer: InlineReasoningBuffer): InlineReasoningExtraction {
  const pending = buffer.pendingTag.trim()
  const trailingVisible = !buffer.activeTag && pending ? pending : ''
  const trailingReasoning = buffer.activeTag && pending ? pending : ''
  return {
    content: stripReasoningArtifacts(`${buffer.visible}${trailingVisible}`),
    reasoning: stripReasoningArtifacts(`${buffer.reasoning}${trailingReasoning}`) || null,
  }
}

export function buildInlineReasoningStep(
  reasoning: string,
  options: InlineReasoningStepOptions,
): ReasoningStep {
  const { index, turnKey, timestamp } = options
  const stepTimestamp = timestamp ? new Date(timestamp) : new Date()
  return {
    id: ['reasoning-inline', turnKey ?? 'na', timestamp ?? 'now', String(index)].join('-'),
    type: 'thinking',
    content: stripReasoningArtifacts(reasoning),
    timestamp: Number.isNaN(stepTimestamp.getTime()) ? new Date() : stepTimestamp,
    status: 'success',
  }
}
