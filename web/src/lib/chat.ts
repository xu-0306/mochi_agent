export type MessageType = 'user' | 'assistant' | 'system' | 'error'

export type MessageEventType =
  | 'thinking'
  | 'status'
  | 'tool_call_request'
  | 'tool_call_result'
  | 'final_answer'
  | 'error'
  | 'text_chunk'

export type ReasoningStepType = 'thinking' | 'status' | 'tool_call' | 'tool_result' | 'error'

export interface ReasoningStep {
  id: string
  type: ReasoningStepType
  content: string
  timestamp: Date
  source?: 'model_summary' | 'runtime_progress' | string
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

export interface ChatAttachment {
  id?: string
  name: string
  path: string
  size?: number | null
  contentType?: string | null
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
  attachments?: ChatAttachment[]
  reasoningBuffer?: {
    visible: string
    reasoning: string
    pendingTag: string
    isInsideThink: boolean
    startedReasoningBlock: boolean
  }
  inlineReasoningStepId?: string
}
