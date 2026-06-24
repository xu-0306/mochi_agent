import type { WorkflowProgressCardView } from '@/components/workflow/types'
import type { DelegatedSubagentCardView } from '@/lib/subagent-tasks'

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

export interface ToolExposureDiagnostics {
  exposedTools: string[]
  workspaceBound?: boolean
  attachmentCount?: number
}

export interface ToolTransportDiagnostics {
  summaryApplied?: boolean
  overflowPersisted?: boolean
  referenceId?: string
  artifactPath?: string
  sourcePath?: string
}

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
  toolExposure?: ToolExposureDiagnostics
  transport?: ToolTransportDiagnostics
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
  source?: 'upload' | 'workspace_file' | 'workspace_selection' | 'image' | string | null
  lineStart?: number | null
  lineEnd?: number | null
  quote?: string | null
  note?: string | null
}

export type GoalCardKind = 'proposal' | 'revised_proposal' | 'started'

export type GoalCardExecutionMode = 'single_agent' | 'workflow'

export interface GoalCardView {
  kind: GoalCardKind
  label: string
  objective: string
  executionMode: GoalCardExecutionMode
  protocolId?: string | null
  models: string[]
  roleSummary?: string | null
  runtimeMode?: string | null
  riskNote?: string | null
  goalId?: string | null
  status?: string | null
  superseded?: boolean | null
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
    activeTag: string | null
    isInsideThink: boolean
    startedReasoningBlock: boolean
  }
  inlineReasoningStepId?: string
  goalCard?: GoalCardView
  workflowCard?: WorkflowProgressCardView
  workflowCompletion?: boolean
  subagentTaskCard?: DelegatedSubagentCardView
}
