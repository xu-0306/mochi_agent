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
  isInsideThink: boolean
  startedReasoningBlock: boolean
}

export interface InlineReasoningChunkResult {
  buffer: InlineReasoningBuffer
  contentDelta: string
  reasoningDelta: string | null
}

function normalizeTextBlock(value: string): string {
  return value
    .replace(/\r\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim()
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
  let isInsideThink = buffer.isInsideThink
  let startedReasoningBlock = buffer.startedReasoningBlock

  while (source.length > 0) {
    if (isInsideThink) {
      const closeIndex = source.search(/<\/think>/i)
      if (closeIndex === -1) {
        const partialCloseMatch = source.match(/<\/th?i?n?k?$/i)
        const pendingTag = partialCloseMatch?.[0] ?? ''
        const commit = pendingTag ? source.slice(0, -pendingTag.length) : source
        reasoning += commit
        reasoningDelta += commit
        return {
          buffer: {
            visible,
            reasoning,
            pendingTag,
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
      source = source.slice(closeIndex + 8)
      isInsideThink = false
      continue
    }

    const openIndex = source.search(/<think>/i)
    if (openIndex === -1) {
      const partialOpenMatch = source.match(/<t?h?i?n?k?$/i)
      const pendingTag = partialOpenMatch?.[0] ?? ''
      const commit = pendingTag ? source.slice(0, -pendingTag.length) : source
      visible += commit
      contentDelta += commit
      return {
        buffer: {
          visible,
          reasoning,
          pendingTag,
          isInsideThink: false,
          startedReasoningBlock: buffer.startedReasoningBlock,
        },
        contentDelta,
        reasoningDelta: normalizeTextBlock(reasoningDelta) || null,
      }
    }

    const nextVisible = source.slice(0, openIndex)
    visible += nextVisible
    contentDelta += nextVisible
    source = source.slice(openIndex + 7)
    if (startedReasoningBlock || reasoning.length > 0) {
      reasoning += '\n\n'
      reasoningDelta += '\n\n'
    }
    startedReasoningBlock = true
    isInsideThink = true
  }

  return {
    buffer: {
      visible,
      reasoning,
      pendingTag: '',
      isInsideThink,
      startedReasoningBlock,
    },
    contentDelta,
    reasoningDelta: normalizeTextBlock(reasoningDelta) || null,
  }
}

export function extractInlineReasoning(content: string): InlineReasoningExtraction {
  const normalized = content.replace(/\r\n/g, '\n')
  const matches = [...normalized.matchAll(/<think>([\s\S]*?)<\/think>/gi)]
  if (matches.length === 0) {
    const closingOnlyIndex = normalized.search(/<\/think>/i)
    if (closingOnlyIndex !== -1) {
      const hidden = normalizeTextBlock(normalized.slice(0, closingOnlyIndex))
      const visible = normalizeTextBlock(normalized.slice(closingOnlyIndex + 8))
      return {
        content: visible,
        reasoning: hidden || null,
      }
    }

    return {
      content: normalized.trim(),
      reasoning: null,
    }
  }

  const reasoning = normalizeTextBlock(
    matches
      .map((match) => match[1] ?? '')
      .join('\n\n')
  )

  const visible = normalizeTextBlock(
    normalized.replace(/<think>[\s\S]*?<\/think>/gi, '\n')
  )

  return {
    content: visible,
    reasoning: reasoning || null,
  }
}

export function createInlineReasoningBuffer(): InlineReasoningBuffer {
  return {
    visible: '',
    reasoning: '',
    pendingTag: '',
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
  const trailingVisible = !buffer.isInsideThink && pending ? pending : ''
  const trailingReasoning = buffer.isInsideThink && pending ? pending : ''
  return {
    content: normalizeTextBlock(`${buffer.visible}${trailingVisible}`),
    reasoning: normalizeTextBlock(`${buffer.reasoning}${trailingReasoning}`) || null,
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
    content: reasoning,
    timestamp: Number.isNaN(stepTimestamp.getTime()) ? new Date() : stepTimestamp,
    status: 'success',
  }
}
