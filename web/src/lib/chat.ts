export type MessageType = 'user' | 'assistant' | 'system' | 'error'

export type MessageEventType =
  | 'thinking'
  | 'tool_call_request'
  | 'tool_call_result'
  | 'final_answer'
  | 'error'
  | 'text_chunk'

export type ReasoningStepType = 'thinking' | 'tool_call' | 'tool_result' | 'error'

export interface ReasoningStep {
  id: string
  type: ReasoningStepType
  content: string
  timestamp: Date
  toolName?: string
  toolArgs?: Record<string, unknown>
  toolResult?: unknown
  toolMeta?: Record<string, unknown>
  toolCallId?: string
  toolError?: string
  errorCode?: string
  status?: 'running' | 'success' | 'error'
}

export interface TokenStats {
  inputTokens: number
  outputTokens: number
  generationTimeMs: number
  finishReason?: string
}

export interface Message {
  id: string
  type: MessageType
  content: string
  timestamp: Date
  eventType?: MessageEventType
  turnKey?: string | null
  turnId?: string | null
  reasoningSteps?: ReasoningStep[]
  errorCode?: string
  isStreaming?: boolean
  tokenStats?: TokenStats
  reasoningBuffer?: {
    visible: string
    reasoning: string
    pendingTag: string
    isInsideThink: boolean
    startedReasoningBlock: boolean
  }
  inlineReasoningStepId?: string
}
