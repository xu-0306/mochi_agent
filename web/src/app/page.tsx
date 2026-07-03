'use client'

import * as React from 'react'
import { useRouter } from 'next/navigation'
import {
  AlertCircle,
  FolderTree,
  ListTodo,
  Loader2,
  MoreHorizontal,
  Settings,
  SlidersHorizontal,
  Workflow,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  ChatInput,
  type ChatComposerSeed,
  type ChatInputModelOption,
} from '@/components/chat/ChatInput'
import {
  GoalDrawerContent,
  GoalHeaderChip,
  type GoalDrawerBlockerView,
  type GoalHeaderChipView,
} from '@/components/chat/GoalHeaderChip'
import { GoalFocusPanel } from '@/components/chat/GoalFocusPanel'
import { ExecutionTimeline } from '@/components/chat/ExecutionTimeline'
import { SubagentDrawer } from '@/components/chat/SubagentDrawer'
import { SubagentTimelineCard } from '@/components/chat/SubagentTimelineCard'
import { ChatMessage } from '@/components/chat/ChatMessage'
import { EmptyState } from '@/components/chat/EmptyState'
import { ExportDialog } from '@/components/chat/ExportDialog'
import { InferencePanel } from '@/components/chat/InferencePanel'
import { ScrollToBottom } from '@/components/chat/ScrollToBottom'
import { FloatingPanelShell } from '@/components/chat/FloatingPanelShell'
import { TaskPanel, type TaskPanelMode } from '@/components/chat/TaskPanel'
import { WorkflowPanel } from '@/components/chat/WorkflowPanel'
import { WorkspacePanel } from '@/components/chat/WorkspacePanel'
import { VoiceOverlay } from '@/components/voice/VoiceOverlay'
import { buildWorkflowProgressCardView } from '@/components/workflow/utils'
import * as api from '@/lib/api'
import type { ChatAttachment, GoalCardView, Message, ReasoningStep } from '@/lib/chat'
import {
  buildOptimisticConversationTurnMessages,
} from '@/lib/chat-send-contract'
import {
  resolveChatGoalWorkflowRouting,
  type ChatGoalWorkflowRoute,
} from '@/lib/chat-goal-routing'
import {
  resolveGoalContinuationDecision,
} from '@/lib/chat-goal-continuation'
import {
  layoutShowsEmptyState,
  layoutShowsExecutionHighlights,
  resolveGoalConversationBodyLayout,
} from '@/lib/goal-execution-surface'
import { buildGoalFocusCallout } from '@/lib/goal-focus-surface'
import {
  applyGoalProposalModelReadiness,
  buildGoalProposalProbeCandidates,
  type GoalProposalModelReadinessById,
} from '@/lib/goal-proposal-models'
import {
  buildGoalCardChromeCopy,
  buildGoalCardKindLabel,
  buildGoalCommandHelpMessage,
  buildGoalFollowUpMessage,
  buildGoalLifecycleMessage,
  buildGoalOpenWorkflowLabel,
  buildGoalPendingApprovalNotice,
  buildGoalReviewApprovalsLabel,
  buildGoalUiErrorMessage,
  buildLocalGoalProposalAssistantExplanation,
} from '@/lib/goal-proposal-copy'
import { resolvePreferredCurrentModelId } from '@/lib/current-model-selection'
import {
  mapAgentRunEventToTranscriptEvent,
  mergeSubagentSummaries,
  projectGoalSurfaceExecutionEvents,
  projectWorkflowRunEventToSessionEvent,
} from '@/lib/execution-transcript'
import {
  buildGoalProposalState as buildGoalProposalDraft,
  buildGoalSummaryFromGoal as buildGoalSummarySnapshot,
  buildSelectedModelsRolesPayload,
  detectGoalModelHints,
  mergeSelectedModelRoles,
  normalizeGoalExecutionMode,
  normalizeGoalExecutionTopology,
  normalizeGoalInteractionMode,
  normalizeSelectedModelRoles,
  type GoalSessionProposalLike,
  type GoalSessionSummaryLike,
} from '@/lib/goal-strategy-selection'
import { buildProjectedDisplayMessages } from '@/lib/chat-projections'
import { DELEGATE_SUBAGENT_TOOL_NAME } from '@/lib/subagent-tasks'
import {
  findRegeneratePrompt,
  isConversationEffectivelyEmpty,
  type FileChangeSummary,
} from '@/lib/chat-p2'
import { useI18n } from '@/lib/i18n'
import { mergeReasoningStep } from '@/lib/reasoning-steps'
import {
  getActivePreset,
  resolveEffectiveInferenceParams,
  useInferenceStore,
} from '@/lib/stores/inference-store'
import { useChatRuntimeStore } from '@/lib/stores/chat-runtime-store'
import { useProjectStore } from '@/lib/stores/project-store'
import { useSessionStore, type Session } from '@/lib/stores/session-store'
import { useTaskStore } from '@/lib/stores/task-store'
import { useUIStore } from '@/lib/stores/ui-store'
import { useWorkspaceStore } from '@/lib/stores/workspace-store'
import { cn } from '@/lib/utils'
import { resolveVoiceOverlayPhase, resolveVoicePhaseFromRuntime } from '@/lib/voice-phase'
import {
  VoiceWsClient,
  type VoiceCaptureDiagnostics,
  type VoiceRuntimePhase,
  type VoiceVadState,
  type VoiceTurnResult,
} from '@/lib/voice-ws'

const MODELS_UPDATED_EVENT = 'mochi:models-updated'
const DEFAULT_WORKFLOW_PROTOCOL: api.AgentRunProtocolId = 'teacher_student_distill'
const WORKFLOW_CARD_POLL_MS = 4000
const EXECUTION_TIMELINE_POLL_MS = 2000
const EXECUTION_TIMELINE_STREAM_RECOVER_MS = 1500
type WorkflowTemplate = 'standard' | 'research_debate'
type WorkflowRunPolicyPreset = 'short' | 'balanced' | 'long' | 'custom'
const WORKFLOW_PROTOCOL_OPTIONS: Array<{
  value: api.AgentRunProtocolId
  label: string
  description: string
}> = [
  {
    value: 'teacher_student_distill',
    label: 'Teacher / Student Distill',
    description: 'General multi-agent execution with teacher and student roles.',
  },
  {
    value: 'multi_agent_debate',
    label: 'Multi-Agent Debate',
    description: 'Parallel debate and judging workflow for harder decisions.',
  },
  {
    value: 'dr_zero_self_evolve',
    label: 'DR Zero Self-Evolve',
    description: 'Iterative proposal, solve, and verification loops.',
  },
  {
    value: 'controlled_subagent_execution',
    label: 'Controlled Execution',
    description: 'Subagents propose execution while the controller keeps runtime boundaries.',
  },
]

function parsePositiveInteger(value: unknown, fallback: number): number {
  const source = typeof value === 'string' ? value : String(value ?? '')
  const numeric = Number.parseInt(source, 10)
  if (!Number.isFinite(numeric) || numeric <= 0) {
    return fallback
  }
  return numeric
}

function sourceModeToEvidenceMode(mode: string): string {
  if (mode === 'local_only') {
    return 'rag'
  }
  if (mode === 'web_first') {
    return 'web'
  }
  return 'hybrid'
}

function normalizeEvidenceCollectionMode(value: unknown): string {
  if (value === 'local_only') {
    return 'rag'
  }
  if (value === 'web_only') {
    return 'web'
  }
  return typeof value === 'string' && value.trim().length > 0 ? value : 'hybrid'
}

function runPolicyPresetValues(preset: WorkflowRunPolicyPreset): Required<api.AgentRunRunPolicy> {
  if (preset === 'short') {
    return {
      max_wall_clock_sec: 600,
      heartbeat_timeout_sec: 45,
      checkpoint_interval_steps: 1,
      max_subagent_failures_per_role: 1,
      on_budget_exhausted: 'pause',
      on_subagent_disconnect: 'pause',
    }
  }
  if (preset === 'long') {
    return {
      max_wall_clock_sec: 5400,
      heartbeat_timeout_sec: 180,
      checkpoint_interval_steps: 2,
      max_subagent_failures_per_role: 3,
      on_budget_exhausted: 'finalize_partial',
      on_subagent_disconnect: 'retry_then_degrade',
    }
  }
  return {
    max_wall_clock_sec: 1800,
    heartbeat_timeout_sec: 90,
    checkpoint_interval_steps: 1,
    max_subagent_failures_per_role: 2,
    on_budget_exhausted: 'pause',
    on_subagent_disconnect: 'retry_then_degrade',
  }
}

interface ComposerEditState {
  messageId: string
  turnId: string | null
  seed: ChatComposerSeed
  resetKey: string
}

interface BackendChatResponse {
  session_id?: string
  sessionId?: string
  final_answer?: string
  content?: string
  model?: string
  events?: api.BackendChatEvent[]
}

interface ModelsResponse {
  configured_model?: string
  active_model?: Record<string, unknown> | null
  models?: Array<Record<string, unknown>>
  available_models?: Array<Record<string, unknown>>
}

interface ApiCompat {
  sendMessage?: (
    text: string,
    options?: {
      sessionId?: string
      projectId?: string | null
      model?: string
      toolMode?: 'disabled' | 'auto' | 'required'
      selectedSkillIds?: string[]
      attachments?: ChatAttachment[]
      systemPrompt?: string
      temperature?: number
      maxTokens?: number | null
      reserveOutputTokens?: number | null
      topP?: number
      minP?: number
      topK?: number
      frequencyPenalty?: number
      presencePenalty?: number
      repeatPenalty?: number
      reasoningEffort?: api.ReasoningEffort | null
    }
  ) => Promise<unknown>
  postChat?: (payload: {
    message: string
    session_id?: string
    sessionId?: string
    project_id?: string | null
    projectId?: string | null
    model?: string
    tool_mode?: 'disabled' | 'auto' | 'required'
    selected_skill_ids?: string[]
    selectedSkillIds?: string[]
    attachments?: ChatAttachment[]
    system_prompt?: string
    temperature?: number
    max_tokens?: number | null
    reserve_output_tokens?: number | null
    top_p?: number
    min_p?: number
    top_k?: number
    frequency_penalty?: number
    presence_penalty?: number
    repeat_penalty?: number
    reasoning_effort?: api.ReasoningEffort | null
  }) => Promise<unknown>
}

interface StreamChatChunk {
  event: Message | null
  rawEvent?: api.StreamChatEvent | null
  sessionId?: string
  trajectoryId?: string | null
  model?: string | null
  done?: boolean
}

type StreamSubagentEvent = api.ExecutionTranscriptEvent & {
  sessionId?: string | null
  agentRunId?: string | null
}

function createInitialMessages(t: (key: string) => string): Message[] {
  return [{
    id: 'system-ready',
    type: 'system',
    content: t('chat.system.ready'),
    timestamp: new Date(),
  }]
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function getString(value: unknown): string | null {
  return typeof value === 'string' && value.trim().length > 0 ? value : null
}

function getStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return []
  }
  return value.filter((item): item is string => typeof item === 'string' && item.trim().length > 0)
}

function deliveryStatusFromSubagentMessageType(type: string): string | null {
  const match = /^subagent_message_(queued|accepted|applied|deferred|cancelled|canceled|rejected)$/.exec(type)
  if (!match) {
    return null
  }
  return match[1] === 'canceled' ? 'cancelled' : match[1]
}

function normalizeStreamDeliveryFields(
  type: string,
  record: Record<string, unknown>,
  metadata: Record<string, unknown>
): {
  metadata: Record<string, unknown>
  messageId: string | null
  deliveryMode: string | null
  deliveryStatus: string | null
  deliveryReason: string | null
} {
  const normalizedMetadata = { ...metadata }
  const messageId =
    getString(record.message_id) ??
    getString(record.messageId) ??
    getString(normalizedMetadata.message_id) ??
    getString(normalizedMetadata.messageId) ??
    null
  const deliveryMode =
    getString(record.delivery_mode) ??
    getString(record.deliveryMode) ??
    getString(normalizedMetadata.delivery_mode) ??
    getString(normalizedMetadata.deliveryMode) ??
    null
  const deliveryStatus =
    getString(record.delivery_status) ??
    getString(record.deliveryStatus) ??
    getString(normalizedMetadata.delivery_status) ??
    getString(normalizedMetadata.deliveryStatus) ??
    deliveryStatusFromSubagentMessageType(type)
  const deliveryReason =
    getString(record.delivery_reason) ??
    getString(record.deliveryReason) ??
    getString(normalizedMetadata.delivery_reason) ??
    getString(normalizedMetadata.deliveryReason) ??
    null

  if (messageId && !getString(normalizedMetadata.message_id)) {
    normalizedMetadata.message_id = messageId
  }
  if (deliveryMode && !getString(normalizedMetadata.delivery_mode)) {
    normalizedMetadata.delivery_mode = deliveryMode
  }
  if (deliveryStatus && !getString(normalizedMetadata.delivery_status)) {
    normalizedMetadata.delivery_status = deliveryStatus
  }
  if (deliveryReason && !getString(normalizedMetadata.delivery_reason)) {
    normalizedMetadata.delivery_reason = deliveryReason
  }

  return {
    metadata: normalizedMetadata,
    messageId,
    deliveryMode,
    deliveryStatus,
    deliveryReason,
  }
}

function isSubagentTranscriptEventType(type: string | null | undefined): boolean {
  if (!type) {
    return false
  }
  return type.startsWith('subagent_') || type === 'runtime_blocked'
}

function normalizeStreamSubagentEvent(value: unknown): StreamSubagentEvent | null {
  const base = mapAgentRunEventToTranscriptEvent(
    typeof value === 'object' && value !== null ? (value as Record<string, unknown>) : {}
  )
  if (!base || !isSubagentTranscriptEventType(base.type)) {
    return null
  }

  const record = typeof value === 'object' && value !== null ? (value as Record<string, unknown>) : {}
  const metadata =
    typeof base.metadata === 'object' && base.metadata !== null ? base.metadata : {}
  const deliveryFields = normalizeStreamDeliveryFields(base.type, record, metadata as Record<string, unknown>)

  return {
    ...base,
    metadata: deliveryFields.metadata,
    status: base.status ?? deliveryFields.deliveryStatus ?? null,
    messageId: deliveryFields.messageId,
    deliveryMode: deliveryFields.deliveryMode,
    deliveryStatus: deliveryFields.deliveryStatus,
    deliveryReason: deliveryFields.deliveryReason,
    sessionId:
      getString(record.session_id) ??
      getString(record.sessionId) ??
      getString(deliveryFields.metadata.session_id) ??
      getString(deliveryFields.metadata.sessionId) ??
      null,
    agentRunId:
      getString(record.agent_run_id) ??
      getString(record.agentRunId) ??
      getString(record.run_id) ??
      getString(deliveryFields.metadata.agent_run_id) ??
      getString(deliveryFields.metadata.agentRunId) ??
      getString(deliveryFields.metadata.run_id) ??
      null,
  }
}

function getStreamSubagentDeliveryStatus(event: api.ExecutionTranscriptEvent): string | null {
  return (
    getString(event.deliveryStatus) ??
    getString(event.metadata.delivery_status) ??
    getString(event.metadata.deliveryStatus) ??
    deliveryStatusFromSubagentMessageType(event.type)
  )
}

function buildSubagentSummaryFromEvent(
  event: StreamSubagentEvent
): api.SubagentTranscriptSummary | null {
  const subagentId = event.subagentId ?? event.roleId ?? null
  if (!subagentId) {
    return null
  }

  const metadata = event.metadata
  const deliveryStatus = getStreamSubagentDeliveryStatus(event)
  const runtimeStatusForDelivery =
    getString((metadata as Record<string, unknown>).subagent_status) ??
    getString((metadata as Record<string, unknown>).subagentStatus) ??
    getString((metadata as Record<string, unknown>).runtime_status) ??
    getString((metadata as Record<string, unknown>).runtimeStatus) ??
    null
  if (deliveryStatus && !runtimeStatusForDelivery) {
    return null
  }
  const statusFromEvent =
    runtimeStatusForDelivery ??
    getString(event.status) ??
    getString((metadata as Record<string, unknown>).status) ??
    getString((metadata as Record<string, unknown>).state)
  const promptPreview =
    getString((metadata as Record<string, unknown>).prompt_preview) ??
    getString((metadata as Record<string, unknown>).promptPreview) ??
    event.summary ??
    null
  const outputPreview =
    getString((metadata as Record<string, unknown>).output_preview) ??
    getString((metadata as Record<string, unknown>).outputPreview) ??
    null

  let status = statusFromEvent ?? 'running'
  if (event.type === 'subagent_completed') {
    status = statusFromEvent ?? 'completed'
  } else if (event.type === 'runtime_blocked') {
    status = 'blocked'
  }

  return {
    subagentId,
    parentType: event.parentType ?? (event.agentRunId ? 'agent_run' : 'chat_turn'),
    parentId: event.parentId ?? event.agentRunId ?? event.sessionId ?? '',
    sessionId: event.sessionId ?? null,
    agentRunId: event.agentRunId ?? null,
    roleId:
      event.roleId ??
      getString((metadata as Record<string, unknown>).role_id) ??
      getString((metadata as Record<string, unknown>).roleId) ??
      null,
    title:
      event.title ??
      getString((metadata as Record<string, unknown>).title) ??
      null,
    modelId:
      event.modelId ??
      getString((metadata as Record<string, unknown>).model_id) ??
      getString((metadata as Record<string, unknown>).modelId) ??
      null,
    status,
    promptPreview,
    summary: event.content ?? event.summary ?? promptPreview,
    outputPreview,
    eventCount: 1,
    createdAt: event.createdAt ?? null,
    updatedAt: event.createdAt ?? null,
  }
}

function mergeTranscriptEvents(
  current: api.ExecutionTranscriptEvent[],
  next: api.ExecutionTranscriptEvent[]
): api.ExecutionTranscriptEvent[] {
  const merged = [...current]
  for (const event of next) {
    const duplicate = merged.some(
      (item) =>
        item.seq === event.seq &&
        item.type === event.type &&
        item.subagentId === event.subagentId &&
        item.createdAt === event.createdAt
    )
    if (!duplicate) {
      merged.push(event)
    }
  }
  return merged.sort((left, right) => {
    const leftSeq = left.seq ?? Number.MAX_SAFE_INTEGER
    const rightSeq = right.seq ?? Number.MAX_SAFE_INTEGER
    if (leftSeq !== rightSeq) {
      return leftSeq - rightSeq
    }
    const leftTime = Date.parse(left.createdAt ?? '') || 0
    const rightTime = Date.parse(right.createdAt ?? '') || 0
    return leftTime - rightTime
  })
}

function transcriptEventsForSubagent(
  events: api.ExecutionTranscriptEvent[],
  subagent: api.SubagentTranscriptSummary
): api.ExecutionTranscriptEvent[] {
  return events.filter((event) => {
    if (event.subagentId && event.subagentId === subagent.subagentId) {
      return true
    }
    if (event.roleId && subagent.roleId && event.roleId === subagent.roleId) {
      return true
    }
    return false
  })
}

function isGoalSurfaceSubagentVisible(
  subagent: api.SubagentTranscriptSummary,
  recentEvents: api.ExecutionTranscriptEvent[]
): boolean {
  const status = subagent.status.trim().toLowerCase()
  const hasIdentity = Boolean((subagent.title ?? '').trim() || (subagent.roleId ?? '').trim())
  const hasSummary = Boolean(
    (subagent.summary ?? '').trim() ||
    (subagent.promptPreview ?? '').trim() ||
    (subagent.outputPreview ?? '').trim()
  )

  if (recentEvents.length > 0) {
    return true
  }
  if (!hasIdentity) {
    return false
  }
  if (
    status === 'blocked' ||
    status === 'awaiting_approval' ||
    status === 'failed' ||
    status === 'cancelled' ||
    status === 'interrupted'
  ) {
    return true
  }
  if (status === 'completed' || status === 'running') {
    return hasSummary
  }
  return hasSummary
}

function buildDefaultWorkflowState(
  _reasoningEffort: api.ReasoningEffort | null
): api.SessionWorkflowState {
  return {
    enabled: false,
    bound_run_id: null,
    synced_run_event_count: 0,
    workspace_dir_override: null,
    config: {
      title: null,
      template: 'standard',
      protocol_id: DEFAULT_WORKFLOW_PROTOCOL,
      workspace_dir_override: null,
      reasoning_effort: null,
      selected_models_roles: {},
      run_policy_preset: 'balanced',
      run_policy: {},
      execution_policy: {},
      schedule: {},
      evidence: {},
      research: {},
    },
  }
}

function normalizeWorkflowState(
  value: api.SessionWorkflowState | null | undefined,
  reasoningEffort: api.ReasoningEffort | null
): api.SessionWorkflowState {
  const defaults = buildDefaultWorkflowState(reasoningEffort)
  const config = value?.config ?? {}
  return {
    enabled: value?.enabled ?? defaults.enabled,
    bound_run_id: value?.bound_run_id ?? defaults.bound_run_id,
    synced_run_event_count: value?.synced_run_event_count ?? defaults.synced_run_event_count,
    workspace_dir_override:
      value?.workspace_dir_override ?? config.workspace_dir_override ?? defaults.workspace_dir_override,
    config: {
      ...defaults.config,
      ...config,
      template: config.template === 'research_debate' ? 'research_debate' : 'standard',
      protocol_id: config.protocol_id ?? defaults.config?.protocol_id ?? DEFAULT_WORKFLOW_PROTOCOL,
      reasoning_effort: config.reasoning_effort ?? null,
      selected_models_roles: config.selected_models_roles ?? {},
      run_policy_preset:
        config.run_policy_preset === 'short' ||
        config.run_policy_preset === 'long' ||
        config.run_policy_preset === 'custom'
          ? config.run_policy_preset
          : 'balanced',
      run_policy: config.run_policy ?? {},
      execution_policy: config.execution_policy ?? {},
      schedule: normalizeWorkflowScheduleConfig(
        isRecord(config.schedule) ? config.schedule : {}
      ),
      evidence: config.evidence ?? {},
      research: isRecord(config.research) ? config.research : {},
    },
  }
}

function workflowScheduleEnabled(workflow: api.SessionWorkflowState): boolean {
  return Boolean(
    workflow.config?.schedule &&
      typeof workflow.config.schedule === 'object' &&
      workflow.config.schedule.enabled === true
  )
}

interface GoalSessionSummary extends GoalSessionSummaryLike {
  goal_id: string | null
  objective: string
  execution_mode: api.GoalExecutionMode
  interaction_mode: api.GoalInteractionMode
  execution_topology: api.GoalExecutionTopology
  strategy_id?: string | null
  selection_source?: string | null
  selection_reason?: string | null
  protocol_id: string | null
  bound_run_id: string | null
  protocol_selection: string | null
  selection_rationale: string | null
  models: string[]
  role_summary: string | null
  runtime_mode: string | null
  risk_note: string | null
  status: string | null
}

interface GoalSessionProposal extends GoalSessionProposalLike {
  proposal_id: string
  revision_index: number
  updated_at: string
  assistant_explanation: string | null
  assistant_explanation_source: string | null
}

interface GoalSessionState {
  active_goal_id: string | null
  active_goal_status: string | null
  execution_mode: api.GoalExecutionMode | null
  interaction_mode: api.GoalInteractionMode | null
  execution_topology: api.GoalExecutionTopology | null
  strategy_id?: string | null
  selection_source?: string | null
  selection_reason?: string | null
  bound_run_id: string | null
  protocol_selection: string | null
  selection_rationale: string | null
  default_route: 'chat' | 'goal' | 'workflow'
  last_goal_summary: GoalSessionSummary | null
  pending_proposal: GoalSessionProposal | null
}

interface GoalConversationAppendInput {
  sessionId: string
  attachments: ChatAttachment[]
  userContent: string
  assistantContent: string
  goalCard?: GoalCardView
  userMetadata?: Record<string, unknown>
  assistantMetadata?: Record<string, unknown>
}

interface SyncGoalWorkflowStateInput {
  sessionId: string
  baseWorkflow: api.SessionWorkflowState
  executionMode: api.GoalExecutionMode
  interactionMode?: api.GoalInteractionMode | null
  executionTopology?: api.GoalExecutionTopology | null
  goalStatus?: string | null
  runId?: string | null
}

interface GoalWorkflowRoutingHandlerInput {
  sessionId: string
  attachments: ChatAttachment[]
  selectedSkillIds: string[]
  route: Exclude<ChatGoalWorkflowRoute, { kind: 'direct_chat' }>
  requestText: string
  baseWorkflow: api.SessionWorkflowState
  baseGoalState: GoalSessionState
  workflowSessionProjectId: string | null
  effectiveWorkspaceDir: string | null
}

interface SendSessionContext {
  latestSessionState: {
    sessions: Session[]
    currentSessionDetail: api.SessionDetail | null
  }
  sessionAfterMaterialize: Session | undefined
  baseWorkflow: api.SessionWorkflowState
  baseGoalState: GoalSessionState
  workflowSessionProjectId: string | null
  effectiveWorkspaceDir: string | null
}

interface SendSessionScope {
  getSessionId: () => string
  getContext: () => SendSessionContext
  materializeIfNeeded: () => Promise<string>
}

interface DirectChatTurnInput {
  targetSessionId: string | null
  requestText: string
  attachments: ChatAttachment[]
  selectedSkillIds: string[]
  toolMode?: 'disabled' | 'auto' | 'required'
  normalizedWorkflow: api.SessionWorkflowState
  sessionScope: SendSessionScope
  systemPromptOverride?: string | null
}

interface ActiveGoalDirectTurnDecisionInput {
  sessionId: string
  requestText: string
  attachments: ChatAttachment[]
  activeGoalId: string
  baseWorkflow: api.SessionWorkflowState
  latestGoalSummary: GoalSessionSummary | GoalSessionProposal | null
  selectedSkillIds: string[]
  sessionScope: SendSessionScope
  systemPromptOverride?: string | null
}

function normalizeGoalSessionSummary(value: unknown): GoalSessionSummary | null {
  if (!isRecord(value)) {
    return null
  }

  const objective = getString(value.objective)
  const executionMode = normalizeGoalExecutionMode(value.execution_mode)
  if (!objective || !executionMode) {
    return null
  }

  return {
    goal_id: getString(value.goal_id) ?? null,
    objective,
    execution_mode: executionMode,
    interaction_mode: normalizeGoalInteractionMode(value.interaction_mode) ?? 'goal',
    execution_topology: normalizeGoalExecutionTopology(value.execution_topology) ?? 'single_agent',
    strategy_id: getString(value.strategy_id) ?? null,
    selection_source: getString(value.selection_source) ?? null,
    selection_reason:
      getString(value.selection_reason) ??
      getString(value.selection_rationale) ??
      null,
    protocol_id: getString(value.protocol_id) ?? null,
    bound_run_id: getString(value.bound_run_id) ?? null,
    protocol_selection:
      getString(value.protocol_selection) ??
      getString(value.strategy_id) ??
      getString(value.protocol_id) ??
      null,
    selection_rationale:
      getString(value.selection_rationale) ??
      getString(value.selection_reason) ??
      null,
    models: getStringArray(value.models),
    role_summary: getString(value.role_summary) ?? null,
    runtime_mode: getString(value.runtime_mode) ?? null,
    risk_note: getString(value.risk_note) ?? null,
    status: getString(value.status) ?? null,
  }
}

function normalizeGoalSessionProposal(value: unknown): GoalSessionProposal | null {
  if (!isRecord(value)) {
    return null
  }

  const summary = normalizeGoalSessionSummary(value)
  if (!summary) {
    return null
  }

  return {
    ...summary,
    proposal_id: getString(value.proposal_id) ?? getString(value.id) ?? `goal-proposal-${Date.now()}`,
    revision_index: Math.max(0, Number(getString(value.revision_index) ?? value.revision_index ?? 0) || 0),
    updated_at: getString(value.updated_at) ?? new Date().toISOString(),
    assistant_explanation: getString(value.assistant_explanation) ?? getString(value.assistantExplanation) ?? null,
    assistant_explanation_source:
      getString(value.assistant_explanation_source) ??
      getString(value.assistantExplanationSource) ??
      null,
  }
}

function normalizeGoalSessionState(value: api.SessionGoalState | null | undefined): GoalSessionState {
  if (!isRecord(value)) {
    return {
      active_goal_id: null,
      active_goal_status: null,
      execution_mode: null,
      interaction_mode: null,
      execution_topology: null,
      strategy_id: null,
      selection_source: null,
      selection_reason: null,
      bound_run_id: null,
      protocol_selection: null,
      selection_rationale: null,
      default_route: 'chat',
      last_goal_summary: null,
      pending_proposal: null,
    }
  }

  return {
    active_goal_id: getString(value.active_goal_id) ?? null,
    active_goal_status: getString(value.active_goal_status) ?? null,
    execution_mode:
      normalizeGoalExecutionMode(value.execution_mode) ??
      normalizeGoalSessionProposal(value.pending_proposal)?.execution_mode ??
      normalizeGoalSessionSummary(value.last_goal_summary)?.execution_mode ??
      null,
    interaction_mode:
      normalizeGoalInteractionMode(value.interaction_mode) ??
      normalizeGoalSessionProposal(value.pending_proposal)?.interaction_mode ??
      normalizeGoalSessionSummary(value.last_goal_summary)?.interaction_mode ??
      null,
    execution_topology:
      normalizeGoalExecutionTopology(value.execution_topology) ??
      normalizeGoalSessionProposal(value.pending_proposal)?.execution_topology ??
      normalizeGoalSessionSummary(value.last_goal_summary)?.execution_topology ??
      null,
    strategy_id:
      getString(value.strategy_id) ??
      normalizeGoalSessionProposal(value.pending_proposal)?.strategy_id ??
      normalizeGoalSessionSummary(value.last_goal_summary)?.strategy_id ??
      null,
    selection_source:
      getString(value.selection_source) ??
      normalizeGoalSessionProposal(value.pending_proposal)?.selection_source ??
      normalizeGoalSessionSummary(value.last_goal_summary)?.selection_source ??
      null,
    selection_reason:
      getString(value.selection_reason) ??
      normalizeGoalSessionProposal(value.pending_proposal)?.selection_reason ??
      normalizeGoalSessionSummary(value.last_goal_summary)?.selection_reason ??
      null,
    bound_run_id:
      getString(value.bound_run_id) ??
      normalizeGoalSessionProposal(value.pending_proposal)?.bound_run_id ??
      normalizeGoalSessionSummary(value.last_goal_summary)?.bound_run_id ??
      null,
    protocol_selection:
      getString(value.protocol_selection) ??
      getString(value.strategy_id) ??
      normalizeGoalSessionProposal(value.pending_proposal)?.protocol_selection ??
      normalizeGoalSessionSummary(value.last_goal_summary)?.protocol_selection ??
      null,
    selection_rationale:
      getString(value.selection_rationale) ??
      getString(value.selection_reason) ??
      normalizeGoalSessionProposal(value.pending_proposal)?.selection_rationale ??
      normalizeGoalSessionSummary(value.last_goal_summary)?.selection_rationale ??
      null,
    default_route:
      value.default_route === 'goal' || value.default_route === 'workflow'
        ? value.default_route
        : 'chat',
    last_goal_summary: normalizeGoalSessionSummary(value.last_goal_summary),
    pending_proposal: normalizeGoalSessionProposal(value.pending_proposal),
  }
}

function isGoalTerminalStatus(status: string | null | undefined): boolean {
  const normalized = (status ?? '').toLowerCase()
  return (
    normalized === 'completed' ||
    normalized === 'done' ||
    normalized === 'succeeded' ||
    normalized === 'cancelled' ||
    normalized === 'canceled' ||
    normalized === 'failed' ||
    normalized === 'error'
  )
}

function isGoalCompletedStatus(status: string | null | undefined): boolean {
  const normalized = (status ?? '').toLowerCase()
  return normalized === 'completed' || normalized === 'done' || normalized === 'succeeded'
}

function isGoalBlockedStatus(status: string | null | undefined): boolean {
  const normalized = (status ?? '').toLowerCase()
  return (
    normalized === 'blocked' ||
    normalized === 'paused' ||
    normalized === 'awaiting_approval' ||
    normalized === 'waiting_approval' ||
    normalized === 'awaiting_resources' ||
    normalized === 'stalled' ||
    normalized === 'partial'
  )
}

function isAgentRunTerminalStatus(status: string | null | undefined): boolean {
  const normalized = (status ?? '').toLowerCase()
  return (
    normalized === 'completed' ||
    normalized === 'done' ||
    normalized === 'succeeded' ||
    normalized === 'cancelled' ||
    normalized === 'canceled' ||
    normalized === 'failed' ||
    normalized === 'error' ||
    normalized === 'partial' ||
    normalized === 'awaiting_approval' ||
    normalized === 'awaiting_resources' ||
    normalized === 'stalled'
  )
}

function isSubagentTerminalStatus(status: string | null | undefined): boolean {
  const normalized = (status ?? '').toLowerCase()
  return (
    normalized === 'completed' ||
    normalized === 'done' ||
    normalized === 'succeeded' ||
    normalized === 'cancelled' ||
    normalized === 'canceled' ||
    normalized === 'failed' ||
    normalized === 'error'
  )
}

function isSubagentActiveStatus(status: string | null | undefined): boolean {
  const normalized = (status ?? '').toLowerCase()
  return [
    'active',
    'created',
    'in_progress',
    'pending',
    'processing',
    'queued',
    'resumed',
    'running',
    'started',
    'working',
  ].includes(normalized)
}

function getSubagentMessageDeliveryDefaults(
  status: string | null | undefined
): Pick<api.AgentRunSubagentMessageInput, 'deliveryMode' | 'interrupt'> {
  if (isSubagentActiveStatus(status)) {
    return {
      deliveryMode: 'inject_now',
      interrupt: true,
    }
  }
  if (isSubagentTerminalStatus(status)) {
    return {
      deliveryMode: 'resume_only',
      interrupt: false,
    }
  }
  return {}
}

function isLegacySubagentMessageDeliveryError(error: unknown): boolean {
  return error instanceof api.ApiError && (error.status === 400 || error.status === 422)
}

function uniqueStrings(values: Array<string | null | undefined>): string[] {
  return [...new Set(values.map((value) => (typeof value === 'string' ? value.trim() : '')).filter((value) => value.length > 0))]
}

function goalCardFromSummary(
  summary: GoalSessionSummary,
  kind: GoalCardView['kind'],
  overrides?: Partial<GoalCardView>
): GoalCardView {
  const copySource = overrides?.copySource ?? summary.objective
  const defaultLabel = buildGoalCardKindLabel(copySource, kind)

  return {
    kind,
    label: defaultLabel,
    objective: summary.objective,
    executionMode: summary.execution_mode,
    copySource,
    protocolId: summary.protocol_id,
    models: summary.models,
    roleSummary: summary.role_summary,
    runtimeMode: summary.runtime_mode,
    riskNote: summary.risk_note,
    goalId: summary.goal_id,
    status: summary.status,
    superseded: false,
    ...overrides,
  }
}

function resolveGoalWorkflowRouteUserContent(
  route: Exclude<ChatGoalWorkflowRoute, { kind: 'direct_chat' }>,
  requestText: string
): string {
  switch (route.kind) {
    case 'goal_help':
      return route.raw
    case 'goal_proposal':
      return route.content
    case 'workflow_proposal':
    case 'goal_revision':
    case 'goal_pending_follow_up':
      return requestText
    case 'goal_confirmation':
    case 'goal_lifecycle':
      return route.raw
  }
}

function getGoalAttemptRunId(goal: api.GoalSummary): string | null {
  const currentAttempt = goal.attempts.find((attempt) => attempt.attempt_id === goal.current_attempt_id)
  return currentAttempt?.agent_run_id ?? goal.attempts.at(-1)?.agent_run_id ?? null
}

type WorkflowScheduleType = 'interval' | 'once' | 'cron'

function defaultScheduleTimezone(): string {
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC'
  } catch {
    return 'UTC'
  }
}

function resolveWorkflowScheduleType(schedule: Record<string, unknown>): WorkflowScheduleType {
  if (typeof schedule.cron === 'string' && schedule.cron.trim()) {
    return 'cron'
  }
  if (typeof schedule.run_at === 'string' && schedule.run_at.trim()) {
    return 'once'
  }
  return 'interval'
}

function formatWorkflowScheduleRunAt(value: unknown): string {
  if (typeof value !== 'string' || !value.trim()) {
    return ''
  }
  const trimmed = value.trim()
  if (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$/.test(trimmed)) {
    return trimmed
  }
  const parsed = new Date(trimmed)
  if (Number.isNaN(parsed.getTime())) {
    return trimmed
  }
  const local = new Date(parsed.getTime() - parsed.getTimezoneOffset() * 60_000)
  return local.toISOString().slice(0, 16)
}

function normalizeWorkflowScheduleConfig(
  value: Record<string, unknown> | null | undefined
): Record<string, unknown> {
  if (!isRecord(value)) {
    return {}
  }
  const normalized: Record<string, unknown> = { ...value }
  const timezone = getString(normalized.timezone) ?? defaultScheduleTimezone()
  if (Object.keys(normalized).length === 0) {
    return normalized
  }
  const hasLegacyScheduleFields =
    getString(normalized.run_at) !== null ||
    getString(normalized.cron) !== null ||
    normalized.interval_seconds !== undefined
  normalized.enabled =
    normalized.enabled === false ? false : normalized.enabled === true || hasLegacyScheduleFields
  normalized.timezone = timezone

  const type = resolveWorkflowScheduleType(normalized)
  if (type === 'once') {
    const runAt = getString(normalized.run_at)
    if (runAt) {
      const parsed = new Date(runAt)
      normalized.run_at = Number.isNaN(parsed.getTime()) ? runAt : parsed.toISOString()
    }
    delete normalized.interval_seconds
    delete normalized.cron
    delete normalized.start_immediately
    return normalized
  }

  if (type === 'cron') {
    normalized.cron = getString(normalized.cron) ?? '0 9 * * 1'
    normalized.start_immediately = normalized.start_immediately !== false
    delete normalized.interval_seconds
    delete normalized.run_at
    return normalized
  }

  normalized.interval_seconds =
    typeof normalized.interval_seconds === 'number' && Number.isFinite(normalized.interval_seconds)
      ? normalized.interval_seconds
      : Number.parseInt(String(normalized.interval_seconds ?? ''), 10) || 3600
  normalized.start_immediately = normalized.start_immediately !== false
  delete normalized.cron
  delete normalized.run_at
  return normalized
}

function buildWorkflowScheduleConfig(
  schedule: Record<string, unknown>,
  type: WorkflowScheduleType,
  enabled: boolean
): Record<string, unknown> {
  const base = normalizeWorkflowScheduleConfig(schedule)
  const timezone = getString(base.timezone) ?? defaultScheduleTimezone()
  const autoPauseOnFailure = base.auto_pause_on_failure !== false
  const maxRuns = typeof base.max_runs === 'number' && Number.isFinite(base.max_runs)
    ? base.max_runs
    : null

  if (type === 'once') {
    return {
      enabled,
      run_at: getString(base.run_at) ?? '',
      timezone,
      max_runs: maxRuns ?? 1,
      auto_pause_on_failure: autoPauseOnFailure,
    }
  }

  if (type === 'cron') {
    return {
      enabled,
      cron: getString(base.cron) ?? '0 9 * * 1',
      timezone,
      start_immediately: base.start_immediately !== false,
      max_runs: maxRuns,
      auto_pause_on_failure: autoPauseOnFailure,
    }
  }

  return {
    enabled,
    interval_seconds:
      typeof base.interval_seconds === 'number' && Number.isFinite(base.interval_seconds)
        ? base.interval_seconds
        : 3600,
    timezone,
    start_immediately: base.start_immediately !== false,
    max_runs: maxRuns,
    auto_pause_on_failure: autoPauseOnFailure,
  }
}

function isWorkflowRunSettledForPolling(status: string | null | undefined): boolean {
  const normalized = (status ?? '').toLowerCase()
  return (
    normalized === 'succeeded' ||
    normalized === 'failed' ||
    normalized === 'cancelled' ||
    normalized === 'completed' ||
    normalized === 'done' ||
    normalized === 'error' ||
    normalized === 'partial' ||
    normalized === 'awaiting_resources' ||
    normalized === 'stalled'
  )
}

const INFERENCE_PARAM_KEY_MAP: Record<string, keyof ReturnType<typeof resolveEffectiveInferenceParams>> = {
  temperature: 'temperature',
  max_tokens: 'maxTokens',
  top_p: 'topP',
  min_p: 'minP',
  top_k: 'topK',
  frequency_penalty: 'frequencyPenalty',
  presence_penalty: 'presencePenalty',
  repeat_penalty: 'repeatPenalty',
}

function resolveModelId(model: Record<string, unknown> | null | undefined): string | null {
  if (!model) {
    return null
  }
  return (
    getString(model.id) ??
    getString(model.model_spec) ??
    getString(model.name) ??
    getString(model.model) ??
    getString(model.label)
  )
}

function normalizeResolvedModelId(
  model: Record<string, unknown> | null | undefined,
  modelId: string | null
): string | null {
  if (!modelId) {
    return null
  }

  const provider = getString(model?.provider)
  const backendType = getString(model?.backend_type)
  if ((provider === 'ollama' || backendType === 'ollama') && !modelId.startsWith('ollama:')) {
    return `ollama:${modelId}`
  }

  return modelId
}

function resolveModelOptionId(model: Record<string, unknown> | null | undefined): string | null {
  return normalizeResolvedModelId(model, resolveModelId(model))
}

function resolveModelLabel(model: Record<string, unknown>, modelId: string): string {
  return (
    getString(model.model) ??
    getString(model.name) ??
    getString(model.label) ??
    formatModelLabel(modelId)
  )
}

function formatModelLabel(modelId: string): string {
  const [provider, ...rest] = modelId.split(':')
  if (provider === 'ollama') {
    return rest.join(':') || modelId
  }
  if (
    provider === 'openai_compat' ||
    provider === 'openai_codex' ||
    provider === 'gemini' ||
    provider === 'anthropic' ||
    provider === 'vllm' ||
    provider === 'sglang' ||
    provider === 'tensorrt_llm'
  ) {
    return rest[rest.length - 1] ?? modelId
  }
  return modelId
}

function summarizeBaseUrl(baseUrl: string | null): string | null {
  if (!baseUrl) {
    return null
  }
  try {
    const url = new URL(baseUrl)
    return url.host || baseUrl
  } catch {
    return baseUrl
  }
}

function formatToolModeDetail(model: Record<string, unknown>): string | null {
  const metadata = isRecord(model.metadata) ? model.metadata : null
  const mode = getString(metadata?.tool_call_mode)
  const nativeStatus = getString(metadata?.native_tool_calling_status)
  if (mode === 'simulated_fallback') {
    if (nativeStatus === 'rejected_missing_parser') {
      return 'tools: simulated (native probe rejected by vLLM parser config)'
    }
    return 'tools: simulated fallback'
  }
  if (mode === 'native') {
    return 'tools: native'
  }
  return null
}

function formatModelSource(model: Record<string, unknown>): string | null {
  const provider = getString(model.provider)
  const backendType = getString(model.backend_type)
  const baseUrl = summarizeBaseUrl(getString(model.base_url))
  const toolMode = formatToolModeDetail(model)

  if (provider === 'openai_codex' || backendType === 'openai_codex') {
    return toolMode ? `OpenAI Codex 繚 ${toolMode}` : 'OpenAI Codex'
  }

  if (provider === 'openai_compat' || (backendType === 'openai_compat' && !provider)) {
    const source = baseUrl ?? 'OpenAI-Compatible'
    return toolMode ? `${source} 繚 ${toolMode}` : source
  }

  if (provider === 'gemini') {
    const source = baseUrl ? `Gemini @ ${baseUrl}` : 'Gemini'
    return toolMode ? `${source} | ${toolMode}` : source
  }

  if (provider === 'anthropic') {
    const source = baseUrl ? `Anthropic @ ${baseUrl}` : 'Anthropic'
    return toolMode ? `${source} | ${toolMode}` : source
  }

  if (provider === 'vllm') {
    const source = baseUrl ? `vLLM @ ${baseUrl}` : 'vLLM'
    return toolMode ? `${source} | ${toolMode}` : source
  }

  if (provider === 'sglang') {
    const source = baseUrl ? `SGLang @ ${baseUrl}` : 'SGLang'
    return toolMode ? `${source} | ${toolMode}` : source
  }

  if (provider === 'tensorrt_llm') {
    const source = baseUrl ? `TensorRT-LLM @ ${baseUrl}` : 'TensorRT-LLM'
    return toolMode ? `${source} | ${toolMode}` : source
  }

  if (provider === 'ollama' || backendType === 'ollama') {
    return toolMode ? `Ollama 繚 ${toolMode}` : 'Ollama'
  }

  if (provider === 'local') {
    const source = backendType ? `Local ${backendType}` : 'Local'
    return toolMode ? `${source} 繚 ${toolMode}` : source
  }

  const source = provider ?? backendType ?? baseUrl
  return toolMode && source ? `${source} 繚 ${toolMode}` : source
}

function displaySessionTitle(title: string | undefined, fallback: string): string {
  if (!title || title === '\u65b0\u5c0d\u8a71' || title === 'New chat') {
    return fallback
  }
  return title
}

function deriveModelOptions(payload: ModelsResponse): ChatInputModelOption[] {
  const candidates: ChatInputModelOption[] = []
  const seen = new Set<string>()

  const pushModel = (
    modelId: string | null,
    status: ChatInputModelOption['status'],
    label?: string | null,
    detail?: string | null
  ) => {
    if (!modelId || seen.has(modelId)) {
      return
    }
    seen.add(modelId)
    candidates.push({
      id: modelId,
      label: label ?? formatModelLabel(modelId),
      detail: detail ?? null,
      status,
    })
  }

  if (Array.isArray(payload.available_models)) {
    for (const entry of payload.available_models) {
      if (!isRecord(entry)) {
        continue
      }
      const modelId = resolveModelOptionId(entry)
      pushModel(
        modelId,
        'configured',
        modelId ? resolveModelLabel(entry, modelId) : null,
        formatModelSource(entry)
      )
    }
  }

  if (candidates.length === 0 && Array.isArray(payload.models)) {
    for (const entry of payload.models) {
      if (!isRecord(entry)) {
        continue
      }
      const modelId = resolveModelOptionId(entry)
      pushModel(
        modelId,
        'configured',
        modelId ? resolveModelLabel(entry, modelId) : null,
        formatModelSource(entry)
      )
    }
  }

  if (candidates.length === 0) {
    pushModel(resolveModelOptionId(payload.active_model ?? undefined), 'connected')
    pushModel(getString(payload.configured_model), 'configured')
  }

  if (isRecord(payload.active_model)) {
    const activeId = resolveModelOptionId(payload.active_model)
    const activeDetail = formatModelSource(payload.active_model)
    const activeLabel = activeId ? resolveModelLabel(payload.active_model, activeId) : null
    if (activeId) {
      const index = candidates.findIndex((candidate) => candidate.id === activeId)
      if (index >= 0) {
        candidates[index] = {
          ...candidates[index],
          label: activeLabel ?? candidates[index].label,
          detail: activeDetail ?? candidates[index].detail,
          status: 'connected',
        }
      } else {
        pushModel(activeId, 'connected', activeLabel, activeDetail)
      }
    }
  }

  return candidates
}

async function requestChat(
  text: string,
  sessionId: string | undefined,
  projectId: string | null | undefined,
  model: string | null,
  selectedSkillIds: string[],
  attachments: ChatAttachment[],
  toolMode: 'disabled' | 'auto' | 'required' | undefined,
  inference: {
    systemPrompt: string
    temperature: number
    maxTokens: number | null
    reserveOutputTokens: number | null
    topP: number
    minP: number
    topK: number
    frequencyPenalty: number
    presencePenalty: number
    repeatPenalty: number
    reasoningEffort: api.ReasoningEffort | null
  }
): Promise<BackendChatResponse> {
  const client = api as ApiCompat

  if (typeof client.postChat === 'function') {
    const response = await client.postChat({
      message: text,
      session_id: sessionId,
      project_id: projectId,
      sessionId,
      projectId,
      model: model ?? undefined,
      tool_mode: toolMode,
      selected_skill_ids: selectedSkillIds,
      selectedSkillIds,
      attachments,
      system_prompt: inference.systemPrompt,
      temperature: inference.temperature,
      max_tokens: inference.maxTokens,
      reserve_output_tokens: inference.reserveOutputTokens,
      top_p: inference.topP,
      min_p: inference.minP,
      top_k: inference.topK,
      frequency_penalty: inference.frequencyPenalty,
      presence_penalty: inference.presencePenalty,
      repeat_penalty: inference.repeatPenalty,
      reasoning_effort: inference.reasoningEffort,
    })
    return response as BackendChatResponse
  }

  if (typeof client.sendMessage === 'function') {
    const response = await client.sendMessage(text, {
      sessionId,
      projectId: projectId ?? undefined,
      model: model ?? undefined,
      toolMode,
      selectedSkillIds,
      attachments,
      systemPrompt: inference.systemPrompt,
      temperature: inference.temperature,
      maxTokens: inference.maxTokens,
      reserveOutputTokens: inference.reserveOutputTokens,
      topP: inference.topP,
      minP: inference.minP,
      topK: inference.topK,
      frequencyPenalty: inference.frequencyPenalty,
      presencePenalty: inference.presencePenalty,
      repeatPenalty: inference.repeatPenalty,
      reasoningEffort: inference.reasoningEffort,
    })
    return response as BackendChatResponse
  }

  throw new Error('Chat API client is unavailable.')
}

function isStreamUnavailable(error: unknown): boolean {
  return error instanceof api.ApiError && (error.status === 404 || error.status === 405)
}

function isVoiceStatusUnavailable(error: unknown): boolean {
  return error instanceof api.ApiError && (error.status === 404 || error.status === 405)
}

function isLocalModelId(modelId: string | null): boolean {
  if (!modelId) {
    return false
  }
  return modelId.startsWith('/') || /^[A-Za-z]:[\\/]/.test(modelId)
}

function applyStreamChunk(
  prev: Message[],
  chunk: StreamChatChunk,
  fallbackTurnKey: string
): Message[] {
  if (chunk.done) {
    return prev.map((message) =>
      message.turnKey === fallbackTurnKey ? { ...message, isStreaming: false } : message
    )
  }

  if (!chunk.event) {
    return prev
  }

  const nextMessage = chunk.event.turnKey
    ? chunk.event
    : {
        ...chunk.event,
        turnKey: fallbackTurnKey,
      }

  const targetIndex = prev.findIndex((message) => message.turnKey === nextMessage.turnKey)
  if (targetIndex === -1) {
    return [...prev, nextMessage]
  }

  const current = prev[targetIndex]
  const mergedReasoning = nextMessage.reasoningSteps?.reduce(
    (steps: ReasoningStep[], step: ReasoningStep) => mergeReasoningStep(steps, step),
    current.reasoningSteps ?? []
  ) ?? current.reasoningSteps

  const merged: Message = {
    ...current,
    ...nextMessage,
    id: current.id,
    turnKey: nextMessage.turnKey ?? fallbackTurnKey,
    content: nextMessage.content || current.content,
    reasoningSteps: mergedReasoning,
    isStreaming: nextMessage.eventType === 'final_answer'
      ? false
      : nextMessage.isStreaming ?? current.isStreaming ?? true,
    reasoningBuffer: nextMessage.reasoningBuffer ?? current.reasoningBuffer,
  }

  return prev.map((message, index) => (index === targetIndex ? merged : message))
}

function matchesTaskRuntimeContext(
  task: api.TaskSummary,
  currentSessionId: string | null | undefined,
  projectId: string | null | undefined
): boolean {
  if (currentSessionId) {
    return task.session_id === currentSessionId
  }
  if (projectId) {
    return task.project_id === projectId
  }
  return false
}

function isActiveTaskStatus(status: string): boolean {
  return ['queued', 'running', 'resumed', 'awaiting_approval'].includes(status)
}

function isFailedTaskStatus(status: string): boolean {
  return ['failed', 'error', 'cancelled'].includes(status)
}

function formatRuntimeBadgeCount(count: number): string {
  return count > 9 ? '9+' : String(count)
}

function HeaderRuntimeIndicator({
  tone,
  count,
  pulse = false,
}: {
  tone: 'info' | 'warning' | 'error'
  count?: number | null
  pulse?: boolean
}) {
  const palette =
    tone === 'error'
      ? 'bg-rose-500 text-white'
      : tone === 'warning'
        ? 'bg-amber-400 text-slate-950'
        : 'bg-primary-500 text-white'

  if (typeof count === 'number' && count > 0) {
    return (
      <span
        aria-hidden="true"
        className={cn(
          'pointer-events-none absolute -right-1 -top-0.5 inline-flex min-h-[1.125rem] min-w-[1.125rem] items-center justify-center rounded-full border border-canvas/80 px-1 text-[9px] font-semibold leading-none shadow-[0_0_0_1px_rgba(9,10,16,0.35)] opacity-95',
          palette,
          pulse ? 'animate-pulse' : null
        )}
      >
        {formatRuntimeBadgeCount(count)}
      </span>
    )
  }

  return (
    <span
      aria-hidden="true"
      className={cn(
        'pointer-events-none absolute right-1 top-1 inline-flex h-2 w-2 rounded-full border border-canvas/80 shadow-[0_0_0_1px_rgba(9,10,16,0.35)] opacity-90',
        palette,
        pulse ? 'animate-pulse' : null
      )}
    />
  )
}

export default function ChatPage() {
  const router = useRouter()
  const { t } = useI18n()
  const [modelOptions, setModelOptions] = React.useState<ChatInputModelOption[]>([])
  const [currentModel, setCurrentModel] = React.useState<string | null>(null)
  const [currentModelLoaded, setCurrentModelLoaded] = React.useState<boolean | null>(null)
  const [activeModelInfo, setActiveModelInfo] = React.useState<Record<string, unknown> | null>(null)
  const [activeLocalRuntimeStatus, setActiveLocalRuntimeStatus] = React.useState<api.LocalActiveModelRuntimeStatus | null>(null)
  const [isUnloadingCurrentModel, setIsUnloadingCurrentModel] = React.useState(false)
  const [modelSwitchError, setModelSwitchError] = React.useState<string | null>(null)
  const [settings, setSettings] = React.useState<api.Settings | null>(null)
  const [mobileInferenceOpen, setMobileInferenceOpen] = React.useState(false)
  const [taskPanelOpen, setTaskPanelOpen] = React.useState(false)
  const [taskPanelMode, setTaskPanelMode] = React.useState<TaskPanelMode>('default')
  const [taskPanelFocusedTaskId, setTaskPanelFocusedTaskId] = React.useState<string | null>(null)
  const [taskPanelFocusedApprovalIds, setTaskPanelFocusedApprovalIds] = React.useState<string[] | null>(null)
  const [goalDrawerOpen, setGoalDrawerOpen] = React.useState(false)
  const [goalDrawerBusyAction, setGoalDrawerBusyAction] = React.useState<'status' | 'pause' | 'resume' | 'stop' | null>(null)
  const [goalDrawerHealth, setGoalDrawerHealth] = React.useState<api.GoalHealthSummary | null>(null)
  const [goalDrawerHealthLoading, setGoalDrawerHealthLoading] = React.useState(false)
  const [goalDrawerHealthError, setGoalDrawerHealthError] = React.useState<string | null>(null)
  const [goalDrawerApprovals, setGoalDrawerApprovals] = React.useState<api.ApprovalSummary[]>([])
  const [goalDrawerApprovalsLoading, setGoalDrawerApprovalsLoading] = React.useState(false)
  const [goalDrawerApprovalError, setGoalDrawerApprovalError] = React.useState<string | null>(null)
  const [goalDrawerResolvingApprovalKey, setGoalDrawerResolvingApprovalKey] = React.useState<string | null>(null)
  const [executionTimelineEvents, setExecutionTimelineEvents] = React.useState<api.ExecutionTranscriptEvent[]>([])
  const [executionTimelineError, setExecutionTimelineError] = React.useState<string | null>(null)
  const [sessionSubagents, setSessionSubagents] = React.useState<api.SubagentTranscriptSummary[]>([])
  const [agentRunSubagents, setAgentRunSubagents] = React.useState<api.SubagentTranscriptSummary[]>([])
  const [activeSubagentId, setActiveSubagentId] = React.useState<string | null>(null)
  const [expandedSubagentIds, setExpandedSubagentIds] = React.useState<Set<string>>(() => new Set())
  const [activeSubagentDetail, setActiveSubagentDetail] = React.useState<api.SubagentTranscriptDetail | null>(null)
  const [subagentDrawerOpen, setSubagentDrawerOpen] = React.useState(false)
  const [subagentDrawerLoading, setSubagentDrawerLoading] = React.useState(false)
  const [subagentDrawerError, setSubagentDrawerError] = React.useState<string | null>(null)
  const [subagentGuidance, setSubagentGuidance] = React.useState('')
  const [subagentGuidanceSending, setSubagentGuidanceSending] = React.useState(false)
  const [subagentBusyAction, setSubagentBusyAction] = React.useState<'cancel' | 'resume' | null>(null)
  const lastTranscriptSeqRef = React.useRef<number | null>(null)
  const [workflowPanelOpen, setWorkflowPanelOpen] = React.useState(false)
  const [workspaceMobileOpen, setWorkspaceMobileOpen] = React.useState(false)
  const [queuedWorkspaceAttachments, setQueuedWorkspaceAttachments] = React.useState<ChatAttachment[]>([])
  const [queuedWorkspaceAttachmentsKey, setQueuedWorkspaceAttachmentsKey] = React.useState<string | undefined>(undefined)
  const [selectedPresetName, setSelectedPresetName] = React.useState('default')
  const [savingPreset, setSavingPreset] = React.useState(false)
  const [workflowBusy, setWorkflowBusy] = React.useState(false)
  const [workflowError, setWorkflowError] = React.useState<string | null>(null)
  const [workflowSaveState, setWorkflowSaveState] = React.useState<'idle' | 'saving' | 'saved' | 'error'>('idle')
  const [workflowLastSavedAt, setWorkflowLastSavedAt] = React.useState<string | null>(null)
  const [workflowLastSaveScope, setWorkflowLastSaveScope] = React.useState<'persisted' | 'draft' | null>(null)
  const [workflowDraftBySessionId, setWorkflowDraftBySessionId] = React.useState<
    Record<string, api.SessionWorkflowState>
  >({})
  const [workflowCardRun, setWorkflowCardRun] = React.useState<api.AgentRunDetail | null>(null)
  const [goalStrategyRegistry, setGoalStrategyRegistry] =
    React.useState<api.GoalStrategyRegistryResponse | null>(null)
  const [editState, setEditState] = React.useState<ComposerEditState | null>(null)
  const [voiceOpen, setVoiceOpen] = React.useState(false)
  const [voicePhase, setVoicePhase] = React.useState<VoiceRuntimePhase>('idle')
  const [voiceRecording, setVoiceRecording] = React.useState(false)
  const [voicePartialTranscription, setVoicePartialTranscription] = React.useState('')
  const [voiceFinalTranscription, setVoiceFinalTranscription] = React.useState('')
  const [voiceAssistantText, setVoiceAssistantText] = React.useState('')
  const [voiceInputLevel, setVoiceInputLevel] = React.useState(0)
  const [voiceVadState, setVoiceVadState] = React.useState<VoiceVadState | null>(null)
  const [voiceCaptureDiagnostics, setVoiceCaptureDiagnostics] = React.useState<VoiceCaptureDiagnostics | null>(null)
  const [voiceCaptureWarning, setVoiceCaptureWarning] = React.useState<string | null>(null)
  const [voiceErrorMessage, setVoiceErrorMessage] = React.useState<string | null>(null)
  const [voiceRuntimeStatus, setVoiceRuntimeStatus] = React.useState<api.VoiceRuntimeStatus | null>(null)
  const [, setVoiceRuntimeLoading] = React.useState(false)
  const [exportOpen, setExportOpen] = React.useState(false)
  const [showScrollToBottom, setShowScrollToBottom] = React.useState(false)
  const scrollRef = React.useRef<HTMLDivElement>(null)
  const shouldAutoScrollRef = React.useRef(true)
  const voiceClientRef = React.useRef<VoiceWsClient | null>(null)
  const voiceSessionIdRef = React.useRef<string | null>(null)
  const goalProposalModelReadinessRef = React.useRef<GoalProposalModelReadinessById>({})
  const goalProposalModelProbeInFlightRef = React.useRef<Set<string>>(new Set())
  const activeChatRunRef = React.useRef<{
    sessionId: string | null
    turnId: string | null
    chatRunId: string | null
  }>({
    sessionId: null,
    turnId: null,
    chatRunId: null,
  })
  const pendingChatCancelRef = React.useRef<api.ChatCancelResponse | null>(null)

  const {
    sessions,
    currentSessionId,
    currentSessionDetail,
    isLoadingDetail,
    createDraftSession,
    materializeDraftSession,
    moveSessionToProject,
    selectSession,
    updateLastMessage,
  } = useSessionStore()
  const activeProjectId = useProjectStore((state) => state.activeProjectId)
  const projects = useProjectStore((state) => state.projects)
  const runtimeTasks = useTaskStore((state) => state.tasks)
  const runtimeApprovals = useTaskStore((state) => state.approvals)
  const workspacePanelOpen = useUIStore((state) => state.workspacePanelOpen)
  const setWorkspacePanelOpen = useUIStore((state) => state.setWorkspacePanelOpen)
  const {
    panelOpen,
    setPanelOpen,
    sessionOverridesById,
    setSessionOverride,
    replaceSessionOverride,
    resetSessionOverride,
  } = useInferenceStore()
  const messagesBySessionId = useChatRuntimeStore((state) => state.messagesBySessionId)
  const streamingSessionId = useChatRuntimeStore((state) => state.streamingSessionId)
  const setSessionMessages = useChatRuntimeStore((state) => state.setSessionMessages)
  const updateSessionMessages = useChatRuntimeStore((state) => state.updateSessionMessages)
  const hydrateSessionMessages = useChatRuntimeStore((state) => state.hydrateSessionMessages)
  const moveSessionMessages = useChatRuntimeStore((state) => state.moveSessionMessages)
  const startStreaming = useChatRuntimeStore((state) => state.startStreaming)
  const finishStreaming = useChatRuntimeStore((state) => state.finishStreaming)
  const abortStreaming = useChatRuntimeStore((state) => state.abortStreaming)
  const setWorkspaceContext = useWorkspaceStore((state) => state.setContext)
  const loadWorkspaceTree = useWorkspaceStore((state) => state.loadTree)
  const loadWorkspaceChanges = useWorkspaceStore((state) => state.loadChanges)
  const currentSession = sessions.find((session) => session.id === currentSessionId)
  const effectiveProjectId = currentSession?.projectId ?? activeProjectId
  const hasActiveStream = streamingSessionId !== null
  const isStreaming = hasActiveStream
  const currentSessionMessages = currentSessionId ? messagesBySessionId[currentSessionId] : undefined
  const activeAgentSettings = settings?.agent
  const activePreset = getActivePreset(activeAgentSettings)
  const activeModelMetadata = isRecord(activeModelInfo?.metadata) ? activeModelInfo.metadata : null
  const supportedReasoningEfforts = React.useMemo(
    () =>
      getStringArray(activeModelMetadata?.supported_reasoning_efforts).filter(
        (value): value is api.ReasoningEffort =>
          value === 'none' ||
          value === 'minimal' ||
          value === 'low' ||
          value === 'medium' ||
          value === 'high' ||
          value === 'xhigh'
      ),
    [activeModelMetadata]
  )
  const supportedInferenceParameters = React.useMemo(
    () => getStringArray(activeModelMetadata?.supported_inference_parameters),
    [activeModelMetadata]
  )
  const supportsReasoningEffort = supportedReasoningEfforts.length > 0
  const disabledInferenceKeys = React.useMemo(
    () =>
      Object.entries(INFERENCE_PARAM_KEY_MAP)
        .filter(([key]) => supportedInferenceParameters.length > 0 && !supportedInferenceParameters.includes(key))
        .map(([, value]) => value),
    [supportedInferenceParameters]
  )
  const disabledReason = React.useMemo(() => {
    if (disabledInferenceKeys.length === 0) {
      return null
    }
    return getString(activeModelMetadata?.inference_policy_message) ?? 'This model ignores some chat inference controls.'
  }, [activeModelMetadata, disabledInferenceKeys.length])
  const sessionOverride = currentSessionId ? sessionOverridesById[currentSessionId] : undefined
  const effectiveInference = React.useMemo(
    () => resolveEffectiveInferenceParams(sessionOverride, activeAgentSettings),
    [activeAgentSettings, sessionOverride]
  )
  const persistedWorkflowState = React.useMemo(
    () =>
      normalizeWorkflowState(
        currentSessionDetail?.workflow ?? currentSession?.workflow ?? null,
        effectiveInference.reasoningEffort
      ),
    [currentSession?.workflow, currentSessionDetail?.workflow, effectiveInference.reasoningEffort]
  )
  const workflowState = React.useMemo(() => {
    if (!currentSessionId) {
      return normalizeWorkflowState(null, effectiveInference.reasoningEffort)
    }
    return normalizeWorkflowState(
      workflowDraftBySessionId[currentSessionId] ?? persistedWorkflowState,
      effectiveInference.reasoningEffort
    )
  }, [currentSessionId, effectiveInference.reasoningEffort, persistedWorkflowState, workflowDraftBySessionId])
  const currentSessionGoalState = React.useMemo(
    () => normalizeGoalSessionState(currentSessionDetail?.goal ?? currentSession?.goal ?? null),
    [currentSession?.goal, currentSessionDetail?.goal]
  )
  const workflowEnabled = Boolean(workflowState.enabled)
  const workflowBoundRunId = workflowState.bound_run_id ?? null
  const goalBoundRunId =
    currentSessionGoalState.bound_run_id ??
    currentSessionGoalState.last_goal_summary?.bound_run_id ??
    null
  const sessionBoundRunId = goalBoundRunId ?? workflowBoundRunId
  const workflowConfig = workflowState.config ?? {}
  const workflowTemplate: WorkflowTemplate =
    workflowConfig.template === 'research_debate' ? 'research_debate' : 'standard'
  const workflowProject = projects.find((project) => project.id === effectiveProjectId) ?? null
  const workflowProtocolId = workflowConfig.protocol_id ?? DEFAULT_WORKFLOW_PROTOCOL
  const workflowReasoningEffort =
    workflowConfig.reasoning_effort ?? null
  const workflowRunPolicyPreset: WorkflowRunPolicyPreset =
    workflowConfig.run_policy_preset === 'short' ||
    workflowConfig.run_policy_preset === 'long' ||
    workflowConfig.run_policy_preset === 'custom'
      ? workflowConfig.run_policy_preset
      : 'balanced'
  const workflowRunPolicy = React.useMemo(
    () => (workflowConfig.run_policy ?? {}) as api.AgentRunRunPolicy,
    [workflowConfig.run_policy]
  )
  const allVisibleSubagents = React.useMemo(
    () => mergeSubagentSummaries(sessionSubagents, agentRunSubagents),
    [agentRunSubagents, sessionSubagents]
  )
  const goalSurfaceTimelineEvents = React.useMemo(
    () => projectGoalSurfaceExecutionEvents(executionTimelineEvents),
    [executionTimelineEvents]
  )
  const goalSurfaceTimelineEventsById = React.useMemo(() => {
    const grouped = new Map<string, api.ExecutionTranscriptEvent[]>()
    for (const subagent of allVisibleSubagents) {
      grouped.set(subagent.subagentId, transcriptEventsForSubagent(goalSurfaceTimelineEvents, subagent))
    }
    return grouped
  }, [allVisibleSubagents, goalSurfaceTimelineEvents])
  const goalSurfaceSubagents = React.useMemo(
    () =>
      allVisibleSubagents.filter((subagent) =>
        isGoalSurfaceSubagentVisible(
          subagent,
          goalSurfaceTimelineEventsById.get(subagent.subagentId) ?? []
        )
      ),
    [allVisibleSubagents, goalSurfaceTimelineEventsById]
  )
  const subagentTimelineEventsById = React.useMemo(() => {
    const grouped = new Map<string, api.ExecutionTranscriptEvent[]>()
    for (const subagent of goalSurfaceSubagents) {
      grouped.set(
        subagent.subagentId,
        goalSurfaceTimelineEventsById.get(subagent.subagentId) ?? []
      )
    }
    return grouped
  }, [goalSurfaceSubagents, goalSurfaceTimelineEventsById])
  const workflowExecutionPolicy = React.useMemo(
    () => (workflowConfig.execution_policy ?? {}) as Record<string, unknown>,
    [workflowConfig.execution_policy]
  )
  const workflowEvidenceConfig = React.useMemo(
    () => (workflowConfig.evidence ?? {}) as Record<string, unknown>,
    [workflowConfig.evidence]
  )

  React.useEffect(() => {
    setExpandedSubagentIds((current) => {
      if (current.size === 0) {
        return current
      }
      const validIds = new Set(goalSurfaceSubagents.map((item) => item.subagentId))
      const next = new Set([...current].filter((id) => validIds.has(id)))
      return next.size === current.size ? current : next
    })
  }, [goalSurfaceSubagents])
  const workflowResearchConfig = React.useMemo(
    () => (workflowConfig.research ?? {}) as Record<string, unknown>,
    [workflowConfig.research]
  )
  const shouldSuppressWorkflowUiForGoal =
    currentSessionGoalState.interaction_mode === 'goal' &&
    currentSessionGoalState.execution_topology === 'single_agent' &&
    (currentSessionGoalState.active_goal_id !== null || currentSessionGoalState.pending_proposal !== null)
  const workflowScheduleConfig = React.useMemo(
    () => normalizeWorkflowScheduleConfig((workflowConfig.schedule ?? {}) as Record<string, unknown>),
    [workflowConfig.schedule]
  )
  const workflowScheduleType = React.useMemo(
    () => resolveWorkflowScheduleType(workflowScheduleConfig),
    [workflowScheduleConfig]
  )
  const workflowHasUnsavedChanges = React.useMemo(() => {
    if (!currentSessionId) {
      return false
    }
    return JSON.stringify(workflowState) !== JSON.stringify(persistedWorkflowState)
  }, [currentSessionId, persistedWorkflowState, workflowState])
  const workflowLastSavedLabel = React.useMemo(() => {
    if (!workflowLastSavedAt) {
      return null
    }
    const timestamp = Date.parse(workflowLastSavedAt)
    if (Number.isNaN(timestamp)) {
      return workflowLastSavedAt
    }
    return new Date(timestamp).toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    })
  }, [workflowLastSavedAt])
  React.useEffect(() => {
    let cancelled = false

    api.fetchGoalStrategies()
      .then((registry) => {
        if (!cancelled) {
          setGoalStrategyRegistry(registry)
        }
      })
      .catch(() => {
        if (!cancelled) {
          setGoalStrategyRegistry(null)
        }
      })

    return () => {
      cancelled = true
    }
  }, [])
  const sessionSecurityOverride = React.useMemo(
    () => currentSessionDetail?.security_override ?? currentSession?.securityOverride ?? null,
    [currentSession?.securityOverride, currentSessionDetail?.security_override]
  )
  const effectiveAutonomyMode = sessionSecurityOverride?.autonomy_mode ?? settings?.security?.autonomy_mode ?? 'trusted_workspace'
  const hasSessionAutonomyOverride = Boolean(sessionSecurityOverride?.autonomy_mode)
  const autonomyModeSourceLabel = hasSessionAutonomyOverride ? 'Session override' : 'Workspace default'
  const autonomyModeSourceDescription = hasSessionAutonomyOverride
    ? 'This chat is overriding the workspace safety default.'
    : 'This chat is using the workspace safety default.'
  const contextualRuntimeTasks = React.useMemo(
    () =>
      runtimeTasks.filter((task) =>
        matchesTaskRuntimeContext(task, currentSessionId, effectiveProjectId)
      ),
    [currentSessionId, effectiveProjectId, runtimeTasks]
  )
  const contextualRuntimeTaskIds = React.useMemo(
    () => new Set(contextualRuntimeTasks.map((task) => task.task_id)),
    [contextualRuntimeTasks]
  )
  const contextualPendingApprovals = React.useMemo(
    () =>
      runtimeApprovals.filter(
        (approval) =>
          approval.status === 'pending' &&
          typeof approval.task_id === 'string' &&
          contextualRuntimeTaskIds.has(approval.task_id)
      ),
    [contextualRuntimeTaskIds, runtimeApprovals]
  )
  const pendingApprovalCount = React.useMemo(() => {
    if (contextualRuntimeTaskIds.size === 0) {
      return 0
    }
    return contextualPendingApprovals.length
  }, [contextualPendingApprovals, contextualRuntimeTaskIds])
  const pendingGoalApprovalIds = React.useMemo(
    () =>
      Array.from(
        new Set(
          contextualPendingApprovals
            .map((approval) => approval.approval_id.trim())
            .filter((approvalId) => approvalId.length > 0)
        )
      ),
    [contextualPendingApprovals]
  )
  const goalHealthPendingApprovalCount = React.useMemo(() => {
    if (!goalDrawerHealth) {
      return 0
    }
    const pendingCount = goalDrawerHealth.approval_state?.pending_count
    if (typeof pendingCount === 'number' && Number.isFinite(pendingCount) && pendingCount > 0) {
      return pendingCount
    }
    return getStringArray(goalDrawerHealth.approval_state?.approval_ids).length
  }, [goalDrawerHealth])
  const goalSurfacePendingApprovalCount = React.useMemo(
    () => Math.max(pendingApprovalCount, goalHealthPendingApprovalCount),
    [goalHealthPendingApprovalCount, pendingApprovalCount]
  )
  const activeTaskCount = React.useMemo(
    () => contextualRuntimeTasks.filter((task) => isActiveTaskStatus(task.status)).length,
    [contextualRuntimeTasks]
  )
  const failedTaskCount = React.useMemo(
    () => contextualRuntimeTasks.filter((task) => isFailedTaskStatus(task.status)).length,
    [contextualRuntimeTasks]
  )
  const taskShortcutTitle = React.useMemo(() => {
    if (pendingApprovalCount > 0) {
      return `Tasks (${pendingApprovalCount} approval${pendingApprovalCount > 1 ? 's' : ''} waiting)`
    }
    if (failedTaskCount > 0) {
      return `Tasks (${failedTaskCount} issue${failedTaskCount > 1 ? 's' : ''})`
    }
    if (activeTaskCount > 0) {
      return `Tasks (${activeTaskCount} active)`
    }
    return 'Tasks'
  }, [activeTaskCount, failedTaskCount, pendingApprovalCount])
  const workflowUiSuppressedInHeader =
    currentSessionGoalState.interaction_mode === 'goal' &&
    currentSessionGoalState.execution_topology === 'single_agent'
  const workflowShortcutLabel = workflowUiSuppressedInHeader
    ? 'Workflow desk'
    : t('sidebar.workflows')
  const workflowShortcutTitle = React.useMemo(() => {
    const baseLabel = workflowShortcutLabel
    if (workflowError) {
      return `${baseLabel} (attention needed)`
    }
    if (workflowBoundRunId) {
      return `${baseLabel} (run active)`
    }
    if (workflowEnabled) {
      return `${baseLabel} (enabled)`
    }
    if (workflowUiSuppressedInHeader) {
      return `${baseLabel} (advanced overrides)`
    }
    return baseLabel
  }, [workflowBoundRunId, workflowEnabled, workflowError, workflowShortcutLabel, workflowUiSuppressedInHeader])
  const uploadTargetDir =
    projects.find((project) => project.id === effectiveProjectId)?.workspaceDir ??
    getString(settings?.paths?.workspace_dir) ??
    undefined

  React.useEffect(() => {
    setWorkspaceContext({
      sessionId: currentSessionId,
      projectId: effectiveProjectId,
    })
    void loadWorkspaceTree()
    void loadWorkspaceChanges()
  }, [
    currentSessionId,
    effectiveProjectId,
    loadWorkspaceChanges,
    loadWorkspaceTree,
    setWorkspaceContext,
  ])

  React.useEffect(() => {
    void useTaskStore.getState().load()
  }, [currentSessionId, effectiveProjectId])

  const effectiveWorkflowWorkspace =
    workflowState.workspace_dir_override ||
    workflowConfig.workspace_dir_override ||
    workflowProject?.workspaceDir ||
    uploadTargetDir ||
    ''
  const messages = React.useMemo<Message[]>(() => {
    if (currentSessionMessages && currentSessionMessages.length > 0) {
      return currentSessionMessages
    }

    if (!currentSessionId) {
      return createInitialMessages(t)
    }

    if (currentSessionDetail?.id === currentSessionId) {
      const replayMessages = api.buildMessagesFromSessionEvents(currentSessionDetail.events)
      return replayMessages.length > 0 ? replayMessages : createInitialMessages(t)
    }

    if (isLoadingDetail) {
      return [
        {
          id: `loading-${currentSessionId}`,
          type: 'system',
          content: t('chat.loadingSession'),
          timestamp: new Date(),
        },
      ]
    }

    return createInitialMessages(t)
  }, [currentSessionDetail, currentSessionId, currentSessionMessages, isLoadingDetail, t])
  const resolveGoalSurfaceCopySource = React.useCallback((
    goalId: string | null,
    objective: string | null | undefined
  ): string => {
    const normalizedObjective = (objective ?? '').trim()
    for (let index = messages.length - 1; index >= 0; index -= 1) {
      const goalCard = messages[index]?.goalCard
      if (!goalCard?.copySource) {
        continue
      }
      if (goalId && goalCard.goalId === goalId) {
        return goalCard.copySource
      }
      if (!goalId && normalizedObjective && goalCard.objective === normalizedObjective) {
        return goalCard.copySource
      }
    }
    return normalizedObjective
  }, [messages])
  const delegatedSubagentToolResultCount = React.useMemo(
    () =>
      messages.reduce((count, message) => (
        count + (message.reasoningSteps ?? []).filter(
          (step) => step.type === 'tool_result' && step.toolName === DELEGATE_SUBAGENT_TOOL_NAME
        ).length
      ), 0),
    [messages]
  )
  const projectedWorkflowRun = shouldSuppressWorkflowUiForGoal ? null : workflowCardRun
  const workflowProgressCard = React.useMemo(
    () => buildWorkflowProgressCardView(projectedWorkflowRun),
    [projectedWorkflowRun]
  )
  const displayMessages = React.useMemo<Message[]>(() => {
    return buildProjectedDisplayMessages({
      messages,
      runtimeTasks: contextualRuntimeTasks,
      workflowProgressCard,
      workflowRun: projectedWorkflowRun,
    })
  }, [contextualRuntimeTasks, messages, projectedWorkflowRun, workflowProgressCard])

  React.useEffect(() => {
    if (delegatedSubagentToolResultCount > 0) {
      void useTaskStore.getState().load()
    }
  }, [delegatedSubagentToolResultCount])

  React.useEffect(() => {
    if (activeTaskCount === 0 && pendingApprovalCount === 0) {
      return
    }
    const intervalId = window.setInterval(() => {
      void useTaskStore.getState().load()
    }, 8000)
    return () => window.clearInterval(intervalId)
  }, [activeTaskCount, pendingApprovalCount])

  React.useEffect(() => {
    const presetNames = activeAgentSettings?.presets.map((preset) => preset.name) ?? []
    const fallbackPreset =
      activeAgentSettings?.active_preset ??
      activePreset?.name ??
      presetNames[0] ??
      'default'

    setSelectedPresetName((current) => (
      presetNames.includes(current) ? current : fallbackPreset
    ))
  }, [activeAgentSettings, activePreset, currentSessionId])

  React.useEffect(() => {
    if (!currentSessionId) {
      setEditState(null)
      return
    }

    setWorkflowDraftBySessionId((current) => {
      const next = normalizeWorkflowState(
        current[currentSessionId] ?? persistedWorkflowState,
        effectiveInference.reasoningEffort
      )
      const existing = current[currentSessionId]
      if (JSON.stringify(existing ?? null) === JSON.stringify(next)) {
        return current
      }
      return {
        ...current,
        [currentSessionId]: next,
      }
    })
  }, [currentSessionId, effectiveInference.reasoningEffort, persistedWorkflowState])

  const scrollToBottom = React.useCallback(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [])

  React.useEffect(() => {
    if (shouldAutoScrollRef.current) {
      scrollToBottom()
    }
  }, [displayMessages, scrollToBottom])

  React.useEffect(() => {
    const element = scrollRef.current
    if (!element) {
      return
    }

    const handleScroll = () => {
      const distanceFromBottom = element.scrollHeight - element.scrollTop - element.clientHeight
      shouldAutoScrollRef.current = distanceFromBottom <= 200
      setShowScrollToBottom(distanceFromBottom > 200)
    }

    handleScroll()
    element.addEventListener('scroll', handleScroll)
    return () => element.removeEventListener('scroll', handleScroll)
  }, [])

  React.useEffect(() => {
    let cancelled = false
    const loadSettings = async () => {
      try {
        const nextSettings = await api.fetchSettings()
        if (cancelled) {
          return
        }
        setSettings(nextSettings)
      } catch {
        // keep previous settings on transient failures
      }
    }

    const handleSettingsUpdated = () => {
      void loadSettings()
    }

    void loadSettings()
    window.addEventListener('mochi:settings-updated', handleSettingsUpdated)
    return () => {
      cancelled = true
      window.removeEventListener('mochi:settings-updated', handleSettingsUpdated)
    }
  }, [])

  React.useEffect(() => {
    if (!currentSessionId || currentSessionDetail?.id !== currentSessionId) {
      return
    }
    const replayMessages = api.buildMessagesFromSessionEvents(currentSessionDetail.events)
    if (replayMessages.length === 0) {
      return
    }

    const runtimeMessages = currentSessionMessages ?? []
    const countMessagesMissingTurnId = (messages: Message[]) =>
      messages.reduce(
        (count, message) => (
          (message.type === 'user' || message.type === 'assistant') && !message.turnId
            ? count + 1
            : count
        ),
        0
      )
    const runtimeMissingTurnIds = countMessagesMissingTurnId(runtimeMessages)
    const replayMissingTurnIds = countMessagesMissingTurnId(replayMessages)
    const canImproveCanonicalTurnIds =
      runtimeMessages.length > 0 &&
      !hasActiveStream &&
      runtimeMissingTurnIds > 0 &&
      replayMissingTurnIds < runtimeMissingTurnIds

    if (canImproveCanonicalTurnIds) {
      setSessionMessages(currentSessionId, replayMessages)
      return
    }

    hydrateSessionMessages(currentSessionId, replayMessages)
  }, [
    currentSessionDetail,
    currentSessionId,
    currentSessionMessages,
    hasActiveStream,
    hydrateSessionMessages,
    setSessionMessages,
  ])

  const resolveMessagesForSession = React.useCallback((sessionId: string): Message[] => {
    const runtimeMessages = useChatRuntimeStore.getState().messagesBySessionId[sessionId]
    if (runtimeMessages && runtimeMessages.length > 0) {
      return runtimeMessages
    }

    const detail = useSessionStore.getState().currentSessionDetail
    if (detail?.id === sessionId) {
      const replayMessages = api.buildMessagesFromSessionEvents(detail.events)
      if (replayMessages.length > 0) {
        return replayMessages
      }
    }

    return createInitialMessages(t)
  }, [t])

  const moveWorkflowDraftState = React.useCallback((fromSessionId: string, toSessionId: string) => {
    if (fromSessionId === toSessionId) {
      return
    }

    setWorkflowDraftBySessionId((current) => {
      const existing = current[fromSessionId]
      if (existing === undefined) {
        return current
      }

      const next = { ...current }
      if (next[toSessionId] === undefined) {
        next[toSessionId] = existing
      }
      delete next[fromSessionId]
      return next
    })
  }, [])

  const resolveSendSessionContext = React.useCallback((
    resolvedSessionId: string,
    initialSessionId: string,
    targetSession: Session | undefined
  ): SendSessionContext => {
    const latestSessionState = useSessionStore.getState()
    const resolvedSession = latestSessionState.sessions.find((session) => session.id === resolvedSessionId)
    const resolvedSessionDetail =
      latestSessionState.currentSessionDetail?.id === resolvedSessionId
        ? latestSessionState.currentSessionDetail
        : null
    const baseWorkflow = normalizeWorkflowState(
      workflowDraftBySessionId[resolvedSessionId] ??
        (resolvedSessionId !== initialSessionId ? workflowDraftBySessionId[initialSessionId] : undefined) ??
        resolvedSession?.workflow ??
        resolvedSessionDetail?.workflow ??
        targetSession?.workflow,
      effectiveInference.reasoningEffort
    )
    const workflowSessionProjectId = resolvedSession?.projectId ?? activeProjectId ?? null

    return {
      latestSessionState,
      sessionAfterMaterialize: resolvedSession,
      baseWorkflow,
      baseGoalState: normalizeGoalSessionState(
        resolvedSession?.goal ??
          resolvedSessionDetail?.goal ??
          targetSession?.goal
      ),
      workflowSessionProjectId,
      effectiveWorkspaceDir:
        baseWorkflow.workspace_dir_override ||
        baseWorkflow.config?.workspace_dir_override ||
        projects.find((project) => project.id === workflowSessionProjectId)?.workspaceDir ||
        uploadTargetDir ||
        null,
    }
  }, [
    activeProjectId,
    effectiveInference.reasoningEffort,
    projects,
    uploadTargetDir,
    workflowDraftBySessionId,
  ])

  const createSendSessionScope = React.useCallback((
    initialSessionId: string,
    targetSession: Session | undefined
  ): SendSessionScope => {
    let sessionId = initialSessionId
    let sessionContext = resolveSendSessionContext(sessionId, initialSessionId, targetSession)

    const materializeIfNeeded = async () => {
      const latestSessionState = useSessionStore.getState()
      const currentSession = latestSessionState.sessions.find((session) => session.id === sessionId)
      const needsMaterialize = Boolean(currentSession?.isDraft) || sessionId.startsWith('draft-')
      if (!needsMaterialize) {
        sessionContext = resolveSendSessionContext(sessionId, initialSessionId, targetSession)
        return sessionId
      }

      const nextSessionId = await materializeDraftSession(sessionId)
      if (nextSessionId !== sessionId) {
        moveSessionMessages(sessionId, nextSessionId)
        moveWorkflowDraftState(sessionId, nextSessionId)
      }
      sessionId = nextSessionId
      sessionContext = resolveSendSessionContext(sessionId, initialSessionId, targetSession)
      return sessionId
    }

    return {
      getSessionId: () => sessionId,
      getContext: () => sessionContext,
      materializeIfNeeded,
    }
  }, [
    materializeDraftSession,
    moveSessionMessages,
    moveWorkflowDraftState,
    resolveSendSessionContext,
  ])

  const appendOptimisticConversationTurn = React.useCallback((
    sessionId: string,
    userContent: string,
    attachments: ChatAttachment[],
    turnKey: string,
    assistantPlaceholderContent = ''
  ) => {
    const next = buildOptimisticConversationTurnMessages({
      existingMessages: resolveMessagesForSession(sessionId),
      userContent,
      attachments,
      turnKey,
      assistantPlaceholderContent,
    })
    setSessionMessages(sessionId, next.messages)
    updateLastMessage(sessionId, next.lastMessageSummary)
  }, [
    resolveMessagesForSession,
    setSessionMessages,
    updateLastMessage,
  ])

  const upsertSessionDetail = React.useCallback(
    (detail: api.SessionDetail) => {
      useSessionStore.setState((state) => ({
        sessions: state.sessions.map((session) =>
          session.id === detail.id
            ? {
                ...session,
                title: detail.title || session.title,
                lastMessageAt: new Date(detail.updatedAt),
                messageCount: detail.eventCount,
                projectId: detail.projectId,
                workflow: detail.workflow,
                goal: detail.goal,
                securityOverride: detail.security_override,
                isDraft: false,
              }
            : session
        ),
        currentSessionDetail:
          state.currentSessionDetail?.id === detail.id || state.currentSessionId === detail.id
            ? detail
            : state.currentSessionDetail,
      }))
    },
    []
  )

  const persistSessionSecurityOverride = React.useCallback(
    async (
      sessionId: string,
      autonomyMode: api.SessionSecurityOverride['autonomy_mode']
    ) => {
      const targetSession = useSessionStore.getState().sessions.find((session) => session.id === sessionId)
      if (targetSession?.isDraft || sessionId.startsWith('draft-')) {
        useSessionStore.setState((state) => ({
          sessions: state.sessions.map((session) =>
            session.id === sessionId
              ? {
                  ...session,
                  securityOverride: { autonomy_mode: autonomyMode },
                }
              : session
          ),
          currentSessionDetail:
            state.currentSessionDetail?.id === sessionId
              ? {
                  ...state.currentSessionDetail,
                  security_override: { autonomy_mode: autonomyMode },
                }
              : state.currentSessionDetail,
        }))
        return null
      }

      const detail = await api.updateSessionSecurityOverride(sessionId, {
        autonomy_mode: autonomyMode,
      })
      upsertSessionDetail(detail)
      return detail
    },
    [upsertSessionDetail]
  )

  const persistSessionGoalState = React.useCallback(
    async (sessionId: string, nextGoalState: GoalSessionState) => {
      const goalPayload: api.SessionGoalState = {
        active_goal_id: nextGoalState.active_goal_id,
        active_goal_status: nextGoalState.active_goal_status,
        execution_mode: nextGoalState.execution_mode,
        interaction_mode: nextGoalState.interaction_mode,
        execution_topology: nextGoalState.execution_topology,
        strategy_id: nextGoalState.strategy_id ?? null,
        selection_source: nextGoalState.selection_source ?? null,
        selection_reason: nextGoalState.selection_reason ?? null,
        bound_run_id: nextGoalState.bound_run_id,
        protocol_selection: nextGoalState.protocol_selection,
        selection_rationale: nextGoalState.selection_rationale,
        default_route: nextGoalState.default_route,
        last_goal_summary: nextGoalState.last_goal_summary,
        pending_proposal: nextGoalState.pending_proposal,
      }

      const targetSession = useSessionStore.getState().sessions.find((session) => session.id === sessionId)
      if (targetSession?.isDraft || sessionId.startsWith('draft-')) {
        useSessionStore.setState((state) => ({
          sessions: state.sessions.map((session) =>
            session.id === sessionId
              ? {
                  ...session,
                  goal: goalPayload,
                }
              : session
          ),
          currentSessionDetail:
            state.currentSessionDetail?.id === sessionId
              ? {
                  ...state.currentSessionDetail,
                  goal: goalPayload,
                }
              : state.currentSessionDetail,
        }))
        return null
      }

      const detail = await api.updateSessionGoalState(sessionId, goalPayload)
      upsertSessionDetail(detail)
      return detail
    },
    [upsertSessionDetail]
  )

  const persistWorkflowState = React.useCallback(
    async (sessionId: string, nextWorkflow: api.SessionWorkflowState) => {
      const normalized = normalizeWorkflowState(nextWorkflow, effectiveInference.reasoningEffort)
      setWorkflowDraftBySessionId((current) => ({
        ...current,
        [sessionId]: normalized,
      }))

      const targetSession = useSessionStore.getState().sessions.find((session) => session.id === sessionId)
      if (targetSession?.isDraft || sessionId.startsWith('draft-')) {
        useSessionStore.setState((state) => ({
          sessions: state.sessions.map((session) =>
            session.id === sessionId
              ? {
                  ...session,
                  workflow: normalized,
                }
              : session
          ),
          currentSessionDetail:
            state.currentSessionDetail?.id === sessionId
              ? {
                  ...state.currentSessionDetail,
                  workflow: normalized,
                }
              : state.currentSessionDetail,
        }))
        return null
      }

      const detail = await api.updateSessionWorkflowState(sessionId, normalized)
      upsertSessionDetail(detail)
      setWorkflowDraftBySessionId((current) => ({
        ...current,
        [sessionId]: normalizeWorkflowState(detail.workflow, effectiveInference.reasoningEffort),
      }))
      return detail
    },
    [effectiveInference.reasoningEffort, upsertSessionDetail]
  )

  const syncWorkflowRunEventsToSession = React.useCallback(
    async (
      sessionId: string,
      runDetail: api.AgentRunDetail,
      baseWorkflowState: api.SessionWorkflowState
    ) => {
      const normalizedWorkflow = normalizeWorkflowState(baseWorkflowState, effectiveInference.reasoningEffort)
      const syncedCount = normalizedWorkflow.synced_run_event_count ?? 0
      const events = Array.isArray(runDetail.events) ? runDetail.events : []
      const nextEvents = events.slice(Math.max(0, syncedCount))

      if (nextEvents.length === 0) {
        const unchanged = normalizeWorkflowState(
          {
            ...normalizedWorkflow,
            bound_run_id: runDetail.run_id,
            synced_run_event_count: events.length,
          },
          effectiveInference.reasoningEffort
        )
        await persistWorkflowState(sessionId, unchanged)
        return unchanged
      }

      const mappedEvents = nextEvents
        .map((event, eventIndex) =>
          projectWorkflowRunEventToSessionEvent(event, {
            runId: runDetail.run_id,
            turnId: `${runDetail.run_id}:${syncedCount + eventIndex}`,
          })
        )
        .filter((event) => {
          if (event === null) {
            return false
          }
          if (event.type === 'message') {
            return Boolean(event.content) || (Array.isArray(event.attachments) && event.attachments.length > 0)
          }
          return true
        }) as Record<string, unknown>[]

      if (mappedEvents.length === 0) {
        const unchanged = normalizeWorkflowState(
          {
            ...normalizedWorkflow,
            bound_run_id: runDetail.run_id,
            synced_run_event_count: events.length,
          },
          effectiveInference.reasoningEffort
        )
        await persistWorkflowState(sessionId, unchanged)
        return unchanged
      }

      const detail = await api.appendSessionEvents(sessionId, mappedEvents)
      upsertSessionDetail(detail)

      const nextWorkflow = normalizeWorkflowState(
        {
          ...normalizedWorkflow,
          bound_run_id: runDetail.run_id,
          synced_run_event_count: events.length,
        },
        effectiveInference.reasoningEffort
      )
      await persistWorkflowState(sessionId, nextWorkflow)
      return nextWorkflow
    },
    [effectiveInference.reasoningEffort, persistWorkflowState, upsertSessionDetail]
  )

  React.useEffect(() => {
    if (!currentSessionId || !sessionBoundRunId) {
      setWorkflowCardRun(null)
      return
    }

    let cancelled = false
    let timeoutId: number | null = null
    setWorkflowCardRun((current) => (current?.run_id === sessionBoundRunId ? current : null))

    const scheduleNext = (run: api.AgentRunDetail | null) => {
      if (cancelled || isWorkflowRunSettledForPolling(run?.status)) {
        return
      }
      timeoutId = window.setTimeout(() => {
        void loadWorkflowRun()
      }, WORKFLOW_CARD_POLL_MS)
    }

    const loadWorkflowRun = async () => {
      try {
        const run = await api.fetchAgentRun(sessionBoundRunId)
        if (cancelled) {
          return
        }
        setWorkflowCardRun(run)
        try {
          await syncWorkflowRunEventsToSession(currentSessionId, run, workflowState)
        } catch {
          // The card is the primary UX; session event sync can recover on the next poll.
        }
        scheduleNext(run)
      } catch {
        scheduleNext(null)
      }
    }

    void loadWorkflowRun()

    return () => {
      cancelled = true
      if (timeoutId !== null) {
        window.clearTimeout(timeoutId)
      }
    }
  }, [currentSessionId, sessionBoundRunId, syncWorkflowRunEventsToSession, workflowState])

  React.useEffect(() => {
    if (!currentSessionId) {
      setSessionSubagents([])
      return
    }

    setSessionSubagents([])
    let cancelled = false
    const loadSessionSubagents = async () => {
      try {
        const subagents = await api.fetchSessionSubagents(currentSessionId)
        if (cancelled) {
          return
        }
        setSessionSubagents(subagents)
      } catch {
        if (!cancelled) {
          setSessionSubagents([])
        }
      }
    }

    void loadSessionSubagents()
    return () => {
      cancelled = true
    }
  }, [currentSessionId])

  React.useEffect(() => {
    if (!sessionBoundRunId) {
      setExecutionTimelineEvents((current) =>
        current.filter((event) => event.parentType !== 'agent_run' && event.parentType !== 'goal')
      )
      setExecutionTimelineError(null)
      setAgentRunSubagents([])
      setActiveSubagentId((current) => {
        if (!current) {
          return null
        }
        const stillExists = sessionSubagents.some((item) => item.subagentId === current)
        return stillExists ? current : null
      })
      setActiveSubagentDetail((current) => (current?.agentRunId ? null : current))
      setSubagentDrawerError(null)
      setSubagentDrawerLoading(false)
      lastTranscriptSeqRef.current = null
      return
    }

    let cancelled = false
    let timeoutId: number | null = null
    let streamAbortController: AbortController | null = null
    let shouldUsePollingFallback = false

    const clearScheduledPoll = () => {
      if (timeoutId !== null) {
        window.clearTimeout(timeoutId)
        timeoutId = null
      }
    }

    const scheduleNext = (
      runStatus?: string | null,
      delayMs: number = EXECUTION_TIMELINE_POLL_MS
    ) => {
      clearScheduledPoll()
      if (cancelled || isAgentRunTerminalStatus(runStatus ?? workflowCardRun?.status ?? null)) {
        return
      }
      timeoutId = window.setTimeout(() => {
        timeoutId = null
        void pollTranscript(true)
      }, delayMs)
    }

    const applyFallbackEvents = (runDetail: api.AgentRunDetail) => {
      const fallbackEvents = runDetail.events
        .map((event) => mapAgentRunEventToTranscriptEvent(event))
        .filter((event): event is api.ExecutionTranscriptEvent => event !== null)
      setExecutionTimelineEvents((current) => {
        const preservedSessionEvents = current.filter((event) => event.parentType === 'chat_turn')
        return mergeTranscriptEvents(preservedSessionEvents, fallbackEvents)
      })
      lastTranscriptSeqRef.current = fallbackEvents.reduce(
        (max, event) => Math.max(max, event.seq ?? max),
        0
      )
    }

    const pollTranscript = async (
      scheduleFollowUp = shouldUsePollingFallback
    ): Promise<string | null> => {
      try {
        const [runDetail, subagents] = await Promise.all([
          api.fetchAgentRun(sessionBoundRunId),
          api.fetchAgentRunSubagents(sessionBoundRunId).catch(() => []),
        ])
        if (cancelled) {
          return null
        }

        setWorkflowCardRun(runDetail)
        setAgentRunSubagents((current) => mergeSubagentSummaries(current, subagents))

        try {
          const events = await api.fetchAgentRunEvents(sessionBoundRunId, {
            afterSeq: lastTranscriptSeqRef.current,
          })
          if (cancelled) {
            return null
          }

          if (events.length > 0) {
            setExecutionTimelineEvents((current) => mergeTranscriptEvents(current, events))
            lastTranscriptSeqRef.current = events.reduce(
              (max, event) => Math.max(max, event.seq ?? max),
              lastTranscriptSeqRef.current ?? 0
            )
          } else if (lastTranscriptSeqRef.current === null) {
            applyFallbackEvents(runDetail)
          }
          setExecutionTimelineError(null)
        } catch (error) {
          if (cancelled) {
            return null
          }
          applyFallbackEvents(runDetail)
          setExecutionTimelineError(
            error instanceof Error ? error.message : 'Transcript endpoint unavailable.'
          )
        }

        if (scheduleFollowUp) {
          scheduleNext(runDetail.status)
        }
        return runDetail.status
      } catch (error) {
        if (cancelled) {
          return null
        }
        setExecutionTimelineError(
          error instanceof Error ? error.message : 'Failed to load execution timeline.'
        )
        if (scheduleFollowUp) {
          scheduleNext(null)
        }
        return null
      }
    }

    const streamTranscript = async () => {
      streamAbortController = new AbortController()
      try {
        for await (const event of api.streamAgentRunEvents(sessionBoundRunId, {
          afterSeq: lastTranscriptSeqRef.current,
          signal: streamAbortController.signal,
        })) {
          if (cancelled) {
            return
          }
          setExecutionTimelineEvents((current) => mergeTranscriptEvents(current, [event]))
          lastTranscriptSeqRef.current = Math.max(lastTranscriptSeqRef.current ?? 0, event.seq ?? 0)
          setExecutionTimelineError(null)
        }
      } catch (error) {
        if (cancelled) {
          return
        }
        shouldUsePollingFallback = true
        setExecutionTimelineError(
          error instanceof Error ? error.message : 'Transcript stream unavailable.'
        )
        scheduleNext(null, EXECUTION_TIMELINE_STREAM_RECOVER_MS)
        return
      } finally {
        streamAbortController = null
      }

      if (cancelled || shouldUsePollingFallback) {
        return
      }

      shouldUsePollingFallback = true
      scheduleNext(null, EXECUTION_TIMELINE_STREAM_RECOVER_MS)
    }

    void pollTranscript(false).then((runStatus) => {
      if (cancelled || shouldUsePollingFallback || isAgentRunTerminalStatus(runStatus)) {
        return
      }
      void streamTranscript()
    })

    return () => {
      cancelled = true
      streamAbortController?.abort()
      clearScheduledPoll()
    }
  }, [sessionBoundRunId, sessionSubagents, workflowCardRun?.status])

  React.useEffect(() => {
    if (!currentSessionId) {
      return
    }
    if (currentSessionDetail?.id !== currentSessionId) {
      return
    }

    const restoredEvents = currentSessionDetail.events
      .map((event) => normalizeStreamSubagentEvent(event))
      .filter((event): event is StreamSubagentEvent => event !== null)
      .filter((event) => (event.sessionId ?? currentSessionId) === currentSessionId)

    setExecutionTimelineEvents((current) => {
      const preservedNonChatTurn = current.filter((event) => event.parentType !== 'chat_turn')
      return restoredEvents.length > 0
        ? mergeTranscriptEvents(preservedNonChatTurn, restoredEvents)
        : preservedNonChatTurn
    })
    const restoredSummaries = restoredEvents
      .map((event) => buildSubagentSummaryFromEvent(event))
      .filter((event): event is api.SubagentTranscriptSummary => event !== null)
    if (restoredSummaries.length > 0) {
      setSessionSubagents((current) => mergeSubagentSummaries(current, restoredSummaries))
    }
  }, [currentSessionDetail, currentSessionId])

  const hydrateSessionFromDetail = React.useCallback(
    (detail: api.SessionDetail) => {
      const replayMessages = api.buildMessagesFromSessionEvents(detail.events)
      setSessionMessages(detail.id, replayMessages.length > 0 ? replayMessages : createInitialMessages(t))
      const lastRetainedMessage = [...replayMessages]
        .reverse()
        .find((message) => message.type === 'user' || message.type === 'assistant')
      if (lastRetainedMessage) {
        updateLastMessage(detail.id, lastRetainedMessage.content)
      }
    },
    [setSessionMessages, t, updateLastMessage]
  )

  const syncSessionFromServer = React.useCallback(async (sessionId: string) => {
    try {
      const detail = await api.fetchSession(sessionId)
      hydrateSessionFromDetail(detail)
      upsertSessionDetail(detail)
    } catch {
      // Keep the optimistic transcript if canonical session refresh fails.
    }
  }, [hydrateSessionFromDetail, upsertSessionDetail])

  const appendVoiceMessages = React.useCallback(
    (result: VoiceTurnResult) => {
      const transcript = result.finalTranscription.trim()
      const assistantText = result.assistantText.trim()
      if (!transcript && !assistantText) {
        return
      }

      const sessionId = voiceSessionIdRef.current ?? currentSessionId ?? createDraftSession(activeProjectId)
      const newMessages: Message[] = []
      if (transcript) {
        newMessages.push({
          id: `voice-user-${Date.now()}`,
          type: 'user',
          content: transcript,
          timestamp: new Date(),
        })
      }
      if (assistantText) {
        newMessages.push({
          id: `voice-assistant-${Date.now()}`,
          type: 'assistant',
          eventType: 'final_answer',
          content: assistantText,
          timestamp: new Date(),
        })
      }

      if (newMessages.length > 0) {
        setSessionMessages(sessionId, [...resolveMessagesForSession(sessionId), ...newMessages])
      }
      if (assistantText) {
        updateLastMessage(sessionId, assistantText)
      } else if (transcript) {
        updateLastMessage(sessionId, transcript)
      }
      void selectSession(sessionId)
      setVoiceFinalTranscription(transcript)
      setVoiceAssistantText(assistantText)
    },
    [activeProjectId, createDraftSession, currentSessionId, resolveMessagesForSession, selectSession, setSessionMessages, updateLastMessage]
  )

  const ensureVoiceClient = React.useCallback((sessionId: string): VoiceWsClient => {
    if (voiceClientRef.current && voiceSessionIdRef.current === sessionId) {
      return voiceClientRef.current
    }

    if (voiceClientRef.current) {
      void voiceClientRef.current.disconnect()
      voiceClientRef.current = null
    }

    voiceSessionIdRef.current = sessionId
    const client = new VoiceWsClient({
      sessionId,
      onPhaseChange: (phase) => {
        setVoicePhase(phase)
        if (phase !== 'error') {
          setVoiceErrorMessage(null)
        }
      },
      onRecordingChange: (recording) => {
        setVoiceRecording(recording)
      },
      onPartialTranscription: (text) => {
        setVoicePartialTranscription(text)
      },
      onFinalTranscription: (text) => {
        setVoiceFinalTranscription(text)
        setVoicePartialTranscription('')
      },
      onAssistantText: (text) => {
        setVoiceAssistantText(text)
      },
      onTurnDone: (result) => {
        appendVoiceMessages(result)
      },
      onCaptureDiagnostics: (diagnostics) => {
        setVoiceCaptureDiagnostics(diagnostics)
        setVoiceInputLevel(diagnostics.inputLevel)
      },
      onVadState: (state) => {
        setVoiceVadState(state)
        if (state === 'speech_started') {
          setVoiceCaptureWarning(null)
        }
      },
      onError: (message, code) => {
        setVoiceErrorMessage(code ? `${message} (${code})` : message)
        setVoiceCaptureWarning(null)
      },
    })
    voiceClientRef.current = client
    return client
  }, [appendVoiceMessages])

  React.useEffect(() => {
    if (
      !voiceRecording ||
      voicePhase !== 'listening' ||
      !voiceCaptureDiagnostics?.capturing ||
      voiceCaptureDiagnostics.hasInputSignal
    ) {
      setVoiceCaptureWarning(null)
      return
    }

    const timeoutId = window.setTimeout(() => {
      setVoiceCaptureWarning(t('chat.voice.noInputDetected'))
    }, 2500)

    return () => window.clearTimeout(timeoutId)
  }, [
    t,
    voiceCaptureDiagnostics?.capturing,
    voiceCaptureDiagnostics?.hasInputSignal,
    voicePhase,
    voiceRecording,
  ])

  const refreshVoiceRuntimeStatus = React.useCallback(async (): Promise<api.VoiceRuntimeStatus | null> => {
    setVoiceRuntimeLoading(true)
    try {
      const status = await api.fetchVoiceStatus()
      setVoiceRuntimeStatus(status)
      return status
    } catch (error) {
      if (isVoiceStatusUnavailable(error)) {
        setVoiceRuntimeStatus(null)
      } else {
        const detail = error instanceof Error ? error.message : 'Voice runtime status unavailable.'
        setVoiceRuntimeStatus({
          type: 'voice_runtime_status',
          phase: 'error',
          enabled: null,
          loaded: null,
          ready: false,
          error: detail,
          configured: {},
          sessionDiagnostics: {},
          raw: {},
        })
      }
      return null
    } finally {
      setVoiceRuntimeLoading(false)
    }
  }, [])

  React.useEffect(() => {
    return () => {
      const client = voiceClientRef.current
      voiceClientRef.current = null
      if (client) {
        void client.disconnect()
      }
    }
  }, [])

  const recordGoalProposalModelReadiness = React.useCallback((
    readinessEntries: GoalProposalModelReadinessById,
    selectedModelId?: string | null
  ) => {
    const entries = Object.entries(readinessEntries)
    if (entries.length === 0) {
      return
    }

    const mergedReadiness = {
      ...goalProposalModelReadinessRef.current,
      ...readinessEntries,
    }
    goalProposalModelReadinessRef.current = mergedReadiness
    setModelOptions((prev) => applyGoalProposalModelReadiness(
      prev,
      selectedModelId ?? currentModel,
      mergedReadiness
    ))
  }, [currentModel])

  const clearFailedGoalProposalModelReadiness = React.useCallback(() => {
    const nextEntries = Object.fromEntries(
      Object.entries(goalProposalModelReadinessRef.current).filter(([, value]) => value !== 'failed')
    ) as GoalProposalModelReadinessById
    goalProposalModelReadinessRef.current = nextEntries
    setModelOptions((prev) => applyGoalProposalModelReadiness(prev, currentModel, nextEntries))
  }, [currentModel])

  const probeGoalProposalModels = React.useCallback(async (
    modelIds: string[],
    selectedModelId?: string | null
  ) => {
    const candidates = uniqueStrings(modelIds).filter((modelId) => {
      const readiness = goalProposalModelReadinessRef.current[modelId]
      return (
        !goalProposalModelProbeInFlightRef.current.has(modelId) &&
        readiness !== 'ready' &&
        readiness !== 'failed'
      )
    })

    if (candidates.length === 0) {
      return goalProposalModelReadinessRef.current
    }

    for (const modelId of candidates) {
      goalProposalModelProbeInFlightRef.current.add(modelId)
    }

    try {
      const results = await Promise.allSettled(
        candidates.map(async (modelId) => {
          await api.testModelConnection({ modelId })
          return modelId
        })
      )
      const readinessEntries: GoalProposalModelReadinessById = {}
      results.forEach((result, index) => {
        readinessEntries[candidates[index]] = result.status === 'fulfilled' ? 'ready' : 'failed'
      })
      recordGoalProposalModelReadiness(readinessEntries, selectedModelId)
      return goalProposalModelReadinessRef.current
    } finally {
      for (const modelId of candidates) {
        goalProposalModelProbeInFlightRef.current.delete(modelId)
      }
    }
  }, [recordGoalProposalModelReadiness])

  const loadModels = React.useCallback(async (signal?: AbortSignal) => {
    const [modelsResponse, localRuntimeResult] = await Promise.all([
      fetch('/v1/models', {
        cache: 'no-store',
        signal,
      }),
      api.fetchActiveLocalModelRuntimeStatus().catch(() => null),
    ])
    if (!modelsResponse.ok) {
      throw new Error(`GET /v1/models failed: ${modelsResponse.status}`)
    }

    const payload = (await modelsResponse.json()) as ModelsResponse
    const nextConfiguredOptions = deriveModelOptions(payload)
    const activeModel = resolvePreferredCurrentModelId(
      {
        configuredModel: getString(payload.configured_model),
        activeModelId: resolveModelOptionId(payload.active_model ?? undefined),
      },
      nextConfiguredOptions
    )
    const nextActiveModelInfo = isRecord(payload.active_model) ? payload.active_model : null
    const activeModelMetadata = isRecord(payload.active_model?.metadata) ? payload.active_model?.metadata : null
    const loaded =
      activeModelMetadata && typeof activeModelMetadata.loaded === 'boolean'
        ? activeModelMetadata.loaded
        : null
    if (activeModel) {
      goalProposalModelReadinessRef.current = {
        ...goalProposalModelReadinessRef.current,
        [activeModel]: 'ready',
      }
    }
    const nextOptions = applyGoalProposalModelReadiness(
      nextConfiguredOptions,
      activeModel,
      goalProposalModelReadinessRef.current
    )

    setModelOptions(nextOptions)
    setCurrentModel(activeModel)
    setCurrentModelLoaded(loaded)
    setActiveModelInfo(nextActiveModelInfo)
    setActiveLocalRuntimeStatus(localRuntimeResult)
  }, [])

  React.useEffect(() => {
    let cancelled = false
    const controller = new AbortController()

    const refreshModels = async () => {
      try {
        await loadModels(controller.signal)
      } catch (error) {
        if (cancelled || (error instanceof DOMException && error.name === 'AbortError')) {
          return
        }
        setModelOptions((prev) => prev)
      }
    }

    const handleModelsUpdated = () => {
      clearFailedGoalProposalModelReadiness()
      void refreshModels()
    }
    const handleFocus = () => {
      void refreshModels()
    }
    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        void refreshModels()
      }
    }

    void refreshModels()
    window.addEventListener(MODELS_UPDATED_EVENT, handleModelsUpdated)
    window.addEventListener('focus', handleFocus)
    document.addEventListener('visibilitychange', handleVisibilityChange)

    return () => {
      cancelled = true
      controller.abort()
      window.removeEventListener(MODELS_UPDATED_EVENT, handleModelsUpdated)
      window.removeEventListener('focus', handleFocus)
      document.removeEventListener('visibilitychange', handleVisibilityChange)
    }
  }, [clearFailedGoalProposalModelReadiness, loadModels])

  const handleSwitchModel = React.useCallback(async (modelId: string) => {
    setModelSwitchError(null)

    try {
      const response = await fetch('/v1/models/switch', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ model: modelId }),
      })

      if (!response.ok) {
        throw new Error(`POST /v1/models/switch failed: ${response.status}`)
      }

      const nextModelPayload = (await response.json().catch(() => null)) as Record<string, unknown> | null
      const nextSettings = await api.fetchSettings()
      const nextModel = modelId

      setCurrentModel(nextModel)
      setActiveModelInfo(isRecord(nextModelPayload?.active_model) ? nextModelPayload?.active_model : null)
      setSettings(nextSettings)
      recordGoalProposalModelReadiness({ [nextModel]: 'ready' }, nextModel)
      setModelOptions((prev) => {
        if (prev.some((option) => option.id === nextModel)) {
          return prev.map((option) =>
            option.id === nextModel ? { ...option, status: 'connected' } : option
          )
        }
        return [
          ...prev,
          {
            id: nextModel,
            label: formatModelLabel(nextModel),
            status: 'connected',
          },
        ]
      })
      window.dispatchEvent(new Event('mochi:settings-updated'))
    } catch (error) {
      const detail = error instanceof Error ? error.message : t('chat.modelSwitchFailed')
      setModelSwitchError(`${t('chat.modelSwitchFailed')}: ${detail}`)
    }
  }, [recordGoalProposalModelReadiness, t])

  const handleUnloadCurrentModel = React.useCallback(async () => {
    setIsUnloadingCurrentModel(true)
    setModelSwitchError(null)
    try {
      const result = await api.unloadActiveLocalModelRuntime()
      setActiveLocalRuntimeStatus(result.activeRuntime)
      setCurrentModelLoaded(result.activeRuntime.loaded)
      window.dispatchEvent(new Event(MODELS_UPDATED_EVENT))
    } catch (error) {
      const detail = error instanceof Error ? error.message : 'Failed to unload current local model.'
      setModelSwitchError(`Failed to unload current local model: ${detail}`)
    } finally {
      setIsUnloadingCurrentModel(false)
    }
  }, [])

  const buildGoalProposalState = React.useCallback(
    (
      objective: string,
      executionMode: api.GoalExecutionMode,
      options?: {
        previous?: GoalSessionProposal | null
        revisionText?: string | null
        modelCandidates?: ChatInputModelOption[]
        currentModelId?: string | null
      }
    ): GoalSessionProposal =>
      buildGoalProposalDraft({
        objective,
        executionMode,
        previous: options?.previous ?? null,
        revisionText: options?.revisionText ?? null,
        modelCandidates: options?.modelCandidates ?? modelOptions,
        currentModelId: options?.currentModelId ?? currentModel,
        workflowSelectedModelRoles: workflowConfig.selected_models_roles,
        workflowTemplate,
        workflowProtocolId,
        workflowScheduleEnabled: workflowScheduleEnabled(workflowState),
        workflowScheduleType,
        effectiveAutonomyMode,
        goalStrategyRegistry,
        defaultWorkflowProtocol: DEFAULT_WORKFLOW_PROTOCOL,
      }),
    [
      currentModel,
      effectiveAutonomyMode,
      goalStrategyRegistry,
      modelOptions,
      workflowConfig.selected_models_roles,
      workflowProtocolId,
      workflowScheduleType,
      workflowState,
      workflowTemplate,
    ]
  )

  const buildGoalSummaryFromGoal = React.useCallback(
    (goal: api.GoalSummary, fallback?: GoalSessionSummary | GoalSessionProposal | null): GoalSessionSummary =>
      buildGoalSummarySnapshot(goal, fallback),
    []
  )

  const generateGoalProposalAssistantExplanation = React.useCallback(
    async (
      userMessage: string,
      proposal: GoalSessionProposal
    ): Promise<{ explanation: string; source: string }> => {
      try {
        const result = await api.generateGoalProposalAssistantCopy({
          message: userMessage.trim() || proposal.objective,
          proposalObjective: proposal.objective,
          executionMode: proposal.execution_mode,
          protocolSelection: proposal.protocol_selection,
          roleSummary: proposal.role_summary,
          runtimeMode: proposal.runtime_mode,
          revisionIndex: proposal.revision_index,
        })
        if (result.explanation.trim().length > 0) {
          return result
        }
      } catch {
        // Fall back locally so proposal creation never blocks on copy generation.
      }

      return {
        explanation: buildLocalGoalProposalAssistantExplanation(userMessage, proposal),
        source: 'fallback',
      }
    },
    []
  )

  const persistGoalConversation = React.useCallback(
    async ({
      sessionId,
      attachments,
      userContent,
      assistantContent,
      goalCard,
      userMetadata,
      assistantMetadata,
    }: GoalConversationAppendInput) => {
      const detail = await api.appendSessionEvents(sessionId, [
        {
          type: 'message',
          role: 'user',
          content: userContent,
          attachments,
          timestamp: new Date().toISOString(),
          metadata: userMetadata ?? {},
        },
        {
          type: 'message',
          role: 'assistant',
          content: assistantContent,
          timestamp: new Date().toISOString(),
          metadata: assistantMetadata ?? {},
          ...(goalCard ? { goal_card: goalCard } : {}),
        },
      ])
      upsertSessionDetail(detail)
      hydrateSessionFromDetail(detail)
    },
    [hydrateSessionFromDetail, upsertSessionDetail]
  )

  const syncWorkflowStateForGoal = React.useCallback(
    async ({
      sessionId,
      baseWorkflow,
      executionMode,
      interactionMode = null,
      executionTopology = null,
      goalStatus = null,
      runId = null,
    }: SyncGoalWorkflowStateInput) => {
      const goalTerminal = isGoalTerminalStatus(goalStatus)
      const nextBoundRunId = !goalTerminal ? runId : null
      const workflowEnabled =
        Boolean(nextBoundRunId) &&
        (interactionMode === 'workflow' || executionTopology === 'multi_agent' || executionMode === 'workflow') &&
        !goalTerminal &&
        goalStatus !== 'paused'

      await persistWorkflowState(
        sessionId,
        normalizeWorkflowState(
          {
            ...baseWorkflow,
            enabled: workflowEnabled,
            bound_run_id: nextBoundRunId,
            synced_run_event_count:
              nextBoundRunId && nextBoundRunId === baseWorkflow.bound_run_id
                ? baseWorkflow.synced_run_event_count ?? 0
                : 0,
          },
          effectiveInference.reasoningEffort
        )
      )
    },
    [effectiveInference.reasoningEffort, persistWorkflowState]
  )

  const handleGoalWorkflowRouting = React.useCallback(
    async ({
      sessionId,
      attachments,
      selectedSkillIds,
      route,
      requestText,
      baseWorkflow,
      baseGoalState,
      workflowSessionProjectId,
      effectiveWorkspaceDir,
    }: GoalWorkflowRoutingHandlerInput): Promise<boolean> => {
      const pendingProposal = baseGoalState.pending_proposal
      const latestGoalSummary = baseGoalState.last_goal_summary
      const activeGoalId = baseGoalState.active_goal_id
      const proposalRequested =
        route.kind === 'goal_proposal' ||
        route.kind === 'workflow_proposal'
      const proposalRevisionRequested = route.kind === 'goal_revision'
      const confirmationRequested = route.kind === 'goal_confirmation'

      if (route.kind === 'goal_help') {
        if (pendingProposal) {
          const pendingCard = goalCardFromSummary(
            pendingProposal,
            pendingProposal.revision_index > 0 ? 'revised_proposal' : 'proposal',
            {
              copySource: route.raw,
            }
          )
          await persistGoalConversation({
            sessionId,
            attachments,
            userContent: route.raw,
            assistantContent: pendingProposal.assistant_explanation ?? '',
            goalCard: pendingCard,
          })
          return true
        }

        if (latestGoalSummary) {
          const cardCopy = buildGoalCardChromeCopy(route.raw)
          const summaryCard = goalCardFromSummary(latestGoalSummary, 'started', {
            label: activeGoalId ? cardCopy.goalSummaryLabel : cardCopy.mostRecentGoalLabel,
            goalId: latestGoalSummary.goal_id,
            status: baseGoalState.active_goal_status ?? latestGoalSummary.status,
            copySource: route.raw,
          })
          await persistGoalConversation({
            sessionId,
            attachments,
            userContent: route.raw,
            assistantContent: activeGoalId
              ? buildGoalLifecycleMessage(route.raw, 'goal_manage_hint')
              : buildGoalLifecycleMessage(route.raw, 'no_active_goal'),
            goalCard: summaryCard,
          })
          return true
        }

        await persistGoalConversation({
          sessionId,
          attachments,
          userContent: route.raw,
          assistantContent: buildGoalCommandHelpMessage(route.raw),
        })
        return true
      }

      if (
        proposalRequested &&
        activeGoalId &&
        !isGoalTerminalStatus(baseGoalState.active_goal_status)
      ) {
        await persistGoalConversation({
          sessionId,
          attachments,
          userContent:
            route.kind === 'goal_proposal'
              ? route.content
              : requestText,
          assistantContent: buildGoalFollowUpMessage(
            route.kind === 'goal_proposal'
              ? route.content
              : requestText,
            'active_goal_exists'
          ),
        })
        return true
      }

      if (proposalRequested || proposalRevisionRequested) {
        const revisionSourceText = proposalRevisionRequested
          ? requestText
          : route.kind === 'goal_proposal'
            ? route.content
            : requestText
        const explicitExecutionMode =
          route.kind === 'workflow_proposal'
            ? 'workflow'
            : route.kind === 'goal_proposal'
              ? 'single_agent'
              : pendingProposal?.execution_mode ?? 'single_agent'
        const proposalObjective = proposalRevisionRequested
          ? pendingProposal?.objective ?? requestText
          : route.kind === 'goal_proposal'
            ? route.content
            : requestText
        const modelHintSource = revisionSourceText || proposalObjective
        const candidateModelOptions = applyGoalProposalModelReadiness(
          modelOptions,
          currentModel,
          goalProposalModelReadinessRef.current
        )
        const explicitModelHints = detectGoalModelHints(
          modelHintSource,
          candidateModelOptions,
          currentModel
        )
        const probeCandidates = buildGoalProposalProbeCandidates(
          candidateModelOptions,
          currentModel,
          explicitExecutionMode,
          explicitModelHints
        )
        if (probeCandidates.length > 0) {
          await probeGoalProposalModels(probeCandidates, currentModel)
        }
        const probedModelOptions = applyGoalProposalModelReadiness(
          modelOptions,
          currentModel,
          goalProposalModelReadinessRef.current
        )
        const nextProposal = buildGoalProposalState(proposalObjective, explicitExecutionMode, {
          previous: pendingProposal,
          revisionText: pendingProposal ? revisionSourceText : null,
          modelCandidates: probedModelOptions,
          currentModelId: currentModel,
        })
        const proposalConversationUserContent =
          proposalRevisionRequested || pendingProposal
            ? revisionSourceText
            : route.kind === 'workflow_proposal'
              ? requestText
              : route.kind === 'goal_proposal'
                ? route.content
                : requestText
        const assistantExplanation = await generateGoalProposalAssistantExplanation(
          proposalConversationUserContent || nextProposal.objective,
          nextProposal
        )
        const pendingProposalSummary: GoalSessionProposal = {
          ...nextProposal,
          assistant_explanation: assistantExplanation.explanation,
          assistant_explanation_source: assistantExplanation.source,
        }
        await persistSessionGoalState(sessionId, {
          active_goal_id: null,
          active_goal_status: null,
          execution_mode: pendingProposalSummary.execution_mode,
          interaction_mode: pendingProposalSummary.interaction_mode,
          execution_topology: pendingProposalSummary.execution_topology,
          bound_run_id: pendingProposalSummary.bound_run_id,
          protocol_selection: pendingProposalSummary.protocol_selection,
          selection_rationale: pendingProposalSummary.selection_rationale,
          default_route:
            pendingProposalSummary.execution_mode === 'workflow' ? 'workflow' : 'goal',
          last_goal_summary: latestGoalSummary,
          pending_proposal: pendingProposalSummary,
        })
        await persistGoalConversation({
          sessionId,
          attachments,
          userContent: proposalConversationUserContent,
          assistantContent: pendingProposalSummary.assistant_explanation ?? '',
          goalCard: goalCardFromSummary(
            pendingProposalSummary,
            pendingProposalSummary.revision_index > 0 ? 'revised_proposal' : 'proposal',
            {
              copySource: proposalConversationUserContent,
            }
          ),
        })
        return true
      }

      if (confirmationRequested && pendingProposal) {
        const goalAgentModelId = pendingProposal.models[0] ?? currentModel ?? modelOptions[0]?.id ?? null
        const selectedGoalModelRoles: Record<string, string> = goalAgentModelId
          ? { agent: goalAgentModelId }
          : {}
        const createdGoal = await api.createGoal({
          objective: pendingProposal.objective,
          execution_mode: pendingProposal.execution_mode,
          interaction_mode: pendingProposal.interaction_mode,
          execution_topology: pendingProposal.execution_topology,
          protocol_id: pendingProposal.protocol_id,
          bound_run_id: pendingProposal.bound_run_id,
          protocol_selection: pendingProposal.protocol_selection,
          selection_rationale: pendingProposal.selection_rationale,
          topic: pendingProposal.objective,
          projectId: workflowSessionProjectId,
          workspaceDir: effectiveWorkspaceDir,
          selected_models_roles:
            Object.keys(selectedGoalModelRoles).length > 0
              ? buildSelectedModelsRolesPayload(selectedGoalModelRoles)
              : {},
          summary: {
            operator_message: pendingProposal.objective,
            selected_skill_ids: selectedSkillIds,
            source_session_id: sessionId,
          },
          metadata: {
            channel: 'chat_goal',
            source_session_id: sessionId,
            pending_proposal_id: pendingProposal.proposal_id,
          },
        })
        const startedGoal = await api.startGoal(createdGoal.goal_id)
        const startedSummary = buildGoalSummaryFromGoal(startedGoal, pendingProposal)
        const startedRunId = getGoalAttemptRunId(startedGoal)

        await syncWorkflowStateForGoal({
          sessionId,
          baseWorkflow,
          executionMode: startedGoal.execution_mode,
          interactionMode: startedGoal.interaction_mode,
          executionTopology: startedGoal.execution_topology,
          goalStatus: startedGoal.status,
          runId: startedRunId,
        })
        await persistSessionGoalState(sessionId, {
          active_goal_id: startedGoal.goal_id,
          active_goal_status: startedGoal.status,
          execution_mode: startedGoal.execution_mode,
          interaction_mode: startedGoal.interaction_mode,
          execution_topology: startedGoal.execution_topology,
          bound_run_id: startedSummary.bound_run_id,
          protocol_selection: startedSummary.protocol_selection,
          selection_rationale: startedSummary.selection_rationale,
          default_route: startedGoal.execution_mode === 'workflow' ? 'workflow' : 'goal',
          last_goal_summary: startedSummary,
          pending_proposal: null,
        })
        await persistGoalConversation({
          sessionId,
          attachments,
          userContent: route.raw,
          assistantContent: buildGoalLifecycleMessage(
            pendingProposal.objective || route.raw,
            'goal_started'
          ),
          goalCard: goalCardFromSummary(startedSummary, 'started', {
            goalId: startedGoal.goal_id,
            status: startedGoal.status,
            copySource: route.raw,
          }),
        })
        return true
      }

      if (!activeGoalId) {
        if (route.kind === 'goal_lifecycle' && route.action === 'status' && pendingProposal) {
          await persistGoalConversation({
            sessionId,
            attachments,
            userContent: route.raw,
            assistantContent: pendingProposal.assistant_explanation ?? '',
            goalCard: goalCardFromSummary(
              pendingProposal,
              pendingProposal.revision_index > 0 ? 'revised_proposal' : 'proposal',
              {
                copySource: route.raw,
              }
            ),
          })
          return true
        }

        if (route.kind === 'goal_lifecycle' && route.action === 'stop' && pendingProposal) {
          await persistSessionGoalState(sessionId, {
            active_goal_id: null,
            active_goal_status: null,
            execution_mode: latestGoalSummary?.execution_mode ?? null,
            interaction_mode: latestGoalSummary?.interaction_mode ?? null,
            execution_topology: latestGoalSummary?.execution_topology ?? null,
            bound_run_id: latestGoalSummary?.bound_run_id ?? null,
            protocol_selection: latestGoalSummary?.protocol_selection ?? null,
            selection_rationale: latestGoalSummary?.selection_rationale ?? null,
            default_route: 'chat',
            last_goal_summary: latestGoalSummary,
            pending_proposal: null,
          })
          await persistGoalConversation({
            sessionId,
            attachments,
            userContent: route.raw,
            assistantContent: buildGoalLifecycleMessage(
              pendingProposal.objective || route.raw,
              'pending_cleared'
            ),
          })
          return true
        }

        await persistGoalConversation({
          sessionId,
          attachments,
          userContent: route.kind === 'goal_lifecycle' || route.kind === 'goal_confirmation' ? route.raw : requestText,
          assistantContent: buildGoalLifecycleMessage(
            latestGoalSummary?.objective ||
              (route.kind === 'goal_lifecycle' || route.kind === 'goal_confirmation'
                ? route.raw
                : requestText) ||
              '',
            'no_active_goal'
          ),
        })
        return true
      }

      if (route.kind !== 'goal_lifecycle') {
        return false
      }

      const nextGoal =
        route.action === 'status'
          ? await api.fetchGoal(activeGoalId)
          : route.action === 'pause'
            ? await api.pauseGoal(activeGoalId)
            : route.action === 'resume'
              ? await api.resumeGoal(activeGoalId)
              : await api.cancelGoal(activeGoalId)
      const nextGoalSummary = buildGoalSummaryFromGoal(nextGoal, latestGoalSummary)
      const nextRunId = getGoalAttemptRunId(nextGoal)
      const nextGoalTerminal = isGoalTerminalStatus(nextGoal.status)

      await syncWorkflowStateForGoal({
        sessionId,
        baseWorkflow,
        executionMode: nextGoal.execution_mode,
        interactionMode: nextGoal.interaction_mode,
        executionTopology: nextGoal.execution_topology,
        goalStatus: nextGoal.status,
        runId: nextRunId,
      })
      await persistSessionGoalState(sessionId, {
        active_goal_id: nextGoalTerminal ? null : nextGoal.goal_id,
        active_goal_status: nextGoal.status,
        execution_mode: nextGoal.execution_mode,
        interaction_mode: nextGoal.interaction_mode,
        execution_topology: nextGoal.execution_topology,
        bound_run_id: nextGoalSummary.bound_run_id,
        protocol_selection: nextGoalSummary.protocol_selection,
        selection_rationale: nextGoalSummary.selection_rationale,
        default_route: nextGoalTerminal ? 'chat' : nextGoal.execution_mode === 'workflow' ? 'workflow' : 'goal',
        last_goal_summary: nextGoalSummary,
        pending_proposal: null,
      })

      const lifecycleCopy = buildGoalCardChromeCopy(route.raw)
      const lifecycleLabel =
        route.action === 'status'
          ? lifecycleCopy.goalStatusLabel
          : route.action === 'pause'
            ? lifecycleCopy.goalPausedLabel
            : route.action === 'resume'
              ? lifecycleCopy.goalResumedLabel
              : lifecycleCopy.goalStoppedLabel
      const lifecycleContent =
        nextGoal.latest_error?.trim() ||
        buildGoalLifecycleMessage(
          nextGoalSummary.objective || route.raw || '',
          route.action === 'status'
            ? 'status_fetched'
            : route.action === 'pause'
              ? 'goal_paused'
              : route.action === 'resume'
                ? 'goal_resumed'
                : 'goal_stopped'
        )

      await persistGoalConversation({
        sessionId,
        attachments,
        userContent: route.raw,
        assistantContent: lifecycleContent,
        goalCard: goalCardFromSummary(nextGoalSummary, 'started', {
          label: lifecycleLabel,
          goalId: nextGoal.goal_id,
          status: nextGoal.status,
          copySource: route.raw,
        }),
      })
      return true
    },
    [
      persistGoalConversation,
      buildGoalProposalState,
      buildGoalSummaryFromGoal,
      currentModel,
      generateGoalProposalAssistantExplanation,
      modelOptions,
      persistSessionGoalState,
      probeGoalProposalModels,
      syncWorkflowStateForGoal,
    ]
  )

  const submitDirectChatTurn = React.useCallback(
    async ({
      targetSessionId,
      requestText,
      attachments,
      selectedSkillIds,
      toolMode,
      normalizedWorkflow,
      sessionScope,
      systemPromptOverride,
    }: DirectChatTurnInput) => {
      let sessionId = sessionScope.getSessionId()
      let sessionContext = sessionScope.getContext()
      const turnKey = `turn-${Date.now()}`
      const placeholderContent = isLocalModelId(currentModel) && currentModelLoaded === false
        ? t('chat.loadingLocalModel')
        : ''
      appendOptimisticConversationTurn(
        sessionId,
        requestText,
        attachments,
        turnKey,
        placeholderContent
      )
      const abortController = new AbortController()

      try {
        await sessionScope.materializeIfNeeded()
        sessionId = sessionScope.getSessionId()
        sessionContext = sessionScope.getContext()
        startStreaming(sessionId, abortController)

        if (normalizedWorkflow.enabled) {
          setWorkflowBusy(true)
          setWorkflowError(null)
          const workflowSessionProjectId =
            sessionContext.sessionAfterMaterialize?.projectId ?? activeProjectId ?? null
          const effectiveWorkspaceDir =
            normalizedWorkflow.workspace_dir_override ||
            normalizedWorkflow.config?.workspace_dir_override ||
            projects.find((project) => project.id === workflowSessionProjectId)?.workspaceDir ||
            uploadTargetDir ||
            null

          let runId = normalizedWorkflow.bound_run_id ?? null
          if (!runId) {
            const workflowSchedule = normalizeWorkflowScheduleConfig(
              (normalizedWorkflow.config?.schedule ?? {}) as Record<string, unknown>
            )
            const workflowTemplate =
              normalizedWorkflow.config?.template === 'research_debate'
                ? 'research_debate'
                : 'standard'
            const workflowEvidenceConfig = isRecord(normalizedWorkflow.config?.evidence)
              ? normalizedWorkflow.config?.evidence
              : {}
            const workflowResearchConfig = isRecord(normalizedWorkflow.config?.research)
              ? normalizedWorkflow.config?.research
              : {}
            const workflowExecutionPolicy = isRecord(normalizedWorkflow.config?.execution_policy)
              ? normalizedWorkflow.config?.execution_policy
              : {}
            const evidenceQueries = getStringArray(workflowEvidenceConfig.queries)
            const ragMcpServers = getStringArray(workflowEvidenceConfig.rag_mcp_servers)
            const controlledExecutionEnabled = getString(workflowExecutionPolicy.mode) === 'controlled'
            const manualSelectedRoles = normalizeSelectedModelRoles(
              normalizedWorkflow.config?.selected_models_roles
            )
            const fallbackControlledModelId =
              Object.values(manualSelectedRoles).find((value) => value.trim().length > 0) ??
              currentModel ??
              modelOptions[0]?.id ??
              ''
            const researchDebateRounds = parsePositiveInteger(
              workflowResearchConfig.debate_rounds,
              2
            )
            const researchMaxResultsPerQuery = parsePositiveInteger(
              workflowEvidenceConfig.max_results_per_query,
              4
            )
            const researchBaseRoles =
              workflowTemplate === 'research_debate'
                ? {
                    debater_a: getString(workflowResearchConfig.local_worker_model_id) ?? '',
                    debater_b: getString(workflowResearchConfig.local_worker_model_id) ?? '',
                    judge: getString(workflowResearchConfig.smart_model_id) ?? '',
                    verifier: getString(workflowResearchConfig.smart_model_id) ?? '',
                    planner: getString(workflowResearchConfig.smart_model_id) ?? '',
                    synthesizer: getString(workflowResearchConfig.smart_model_id) ?? '',
                    local_worker: getString(workflowResearchConfig.local_worker_model_id) ?? '',
                    skeptic: getString(workflowResearchConfig.local_worker_model_id) ?? '',
                    ...(controlledExecutionEnabled
                      ? {
                          executor:
                            getString(workflowResearchConfig.local_worker_model_id) ??
                            getString(workflowResearchConfig.smart_model_id) ??
                            '',
                          controller: getString(workflowResearchConfig.smart_model_id) ?? '',
                          evaluator: getString(workflowResearchConfig.smart_model_id) ?? '',
                        }
                      : {}),
                  }
                : {}
            const controlledExecutionRoles: Record<string, string> =
              workflowTemplate === 'research_debate' || !controlledExecutionEnabled || !fallbackControlledModelId
                ? {}
                : {
                    planner: fallbackControlledModelId,
                    executor: fallbackControlledModelId,
                    controller: fallbackControlledModelId,
                    evaluator: fallbackControlledModelId,
                  }
            const selectedModelsRoles = mergeSelectedModelRoles(
              researchBaseRoles,
              controlledExecutionRoles,
              manualSelectedRoles
            )
            const createdRun = await api.createAgentRun({
              protocol_id:
                workflowTemplate === 'research_debate'
                  ? 'multi_agent_debate'
                  : normalizedWorkflow.config?.protocol_id ?? DEFAULT_WORKFLOW_PROTOCOL,
              title: normalizedWorkflow.config?.title ?? null,
              topic: requestText.trim() || null,
              projectId: workflowSessionProjectId,
              workspaceDir: effectiveWorkspaceDir,
              reasoning_effort:
                normalizedWorkflow.config?.reasoning_effort ?? effectiveInference.reasoningEffort ?? null,
              selected_models_roles:
                Object.keys(selectedModelsRoles).length > 0
                  ? buildSelectedModelsRolesPayload(selectedModelsRoles)
                  : {},
              run_policy: normalizedWorkflow.config?.run_policy ?? {},
              evaluation_policy: {
                evidence_collection: {
                  enabled: workflowEvidenceConfig.enabled !== false,
                  mode:
                    workflowTemplate === 'research_debate'
                      ? sourceModeToEvidenceMode(
                          getString(workflowResearchConfig.source_mode) ?? 'hybrid'
                        )
                      : normalizeEvidenceCollectionMode(workflowEvidenceConfig.mode),
                  rag_provider: getString(workflowEvidenceConfig.rag_provider) ?? 'memory',
                  rag_mcp_servers: ragMcpServers,
                  max_results_per_query:
                    workflowTemplate === 'research_debate'
                      ? researchMaxResultsPerQuery
                      : parsePositiveInteger(workflowEvidenceConfig.max_results_per_query, 3),
                  max_fetch_per_query: parsePositiveInteger(
                    workflowEvidenceConfig.max_fetch_per_query,
                    2
                  ),
                  max_content_chars: parsePositiveInteger(
                    workflowEvidenceConfig.max_content_chars,
                    2000
                  ),
                },
                ...(workflowTemplate === 'research_debate'
                  ? {
                      research: {
                        enabled: true,
                        preset: 'smart_judge_research_debate',
                        output_targets: getStringArray(workflowResearchConfig.output_targets),
                        source_mode: getString(workflowResearchConfig.source_mode) ?? 'hybrid',
                        citation_policy:
                          getString(workflowResearchConfig.citation_policy) ??
                          'claim_level_required',
                        local_worker_count: parsePositiveInteger(
                          workflowResearchConfig.local_worker_count,
                          3
                        ),
                        local_worker_count_max: 6,
                        max_research_queries: Math.max(
                          researchMaxResultsPerQuery,
                          evidenceQueries.length || 1
                        ),
                        max_sources_per_query: researchMaxResultsPerQuery,
                        debate_rounds: researchDebateRounds,
                      },
                    }
                  : {}),
              },
              summary: {
                operator_message: requestText,
                selected_skill_ids: selectedSkillIds,
                execution_policy: workflowExecutionPolicy,
                ...(evidenceQueries.length > 0 ? { evidence_queries: evidenceQueries } : {}),
                ...(normalizedWorkflow.config?.protocol_id === 'dr_zero_self_evolve' &&
                workflowTemplate !== 'research_debate'
                  ? {
                      protocol_config: {
                        iterations: 1,
                        proposal_sample_size: 3,
                        solver_rollouts_per_task: 1,
                        proposer_role_id: 'proposer',
                        solver_role_id: 'solver',
                        verifier_role_id: 'verifier',
                      },
                    }
                  : {}),
                ...((normalizedWorkflow.config?.protocol_id === 'multi_agent_debate' ||
                  workflowTemplate === 'research_debate')
                  ? {
                      protocol_config: {
                        rounds: researchDebateRounds,
                      },
                    }
                  : {}),
              },
              schedule: workflowSchedule,
            })
            runId = createdRun.run_id
          }

          await api.appendAgentRunMessage(runId, {
            role: 'operator',
            content: requestText,
            projectId: workflowSessionProjectId,
            workspaceDir: effectiveWorkspaceDir,
            attachments,
            metadata: {
              channel: 'workflow-chat',
              selected_skill_ids: selectedSkillIds,
            },
          })

          let nextWorkflowState = normalizeWorkflowState(
            {
              ...normalizedWorkflow,
              enabled: true,
              bound_run_id: runId,
              workspace_dir_override: normalizedWorkflow.workspace_dir_override ?? effectiveWorkspaceDir,
              config: {
                ...normalizedWorkflow.config,
                reasoning_effort:
                  normalizedWorkflow.config?.reasoning_effort ?? effectiveInference.reasoningEffort ?? null,
              },
            },
            effectiveInference.reasoningEffort
          )

          if (!workflowScheduleEnabled(nextWorkflowState)) {
            await api.startAgentRun(runId)
          }

          const refreshedRun = await api.fetchAgentRun(runId)
          nextWorkflowState = await syncWorkflowRunEventsToSession(sessionId, refreshedRun, nextWorkflowState)

          const replayMessages = api.buildMessagesFromSessionEvents(
            (useSessionStore.getState().currentSessionDetail?.id === sessionId
              ? useSessionStore.getState().currentSessionDetail?.events
              : sessionContext.latestSessionState.currentSessionDetail?.events) ?? []
          )
          if (replayMessages.length > 0) {
            setSessionMessages(sessionId, replayMessages)
          } else {
            await syncSessionFromServer(sessionId)
          }

          const finalAssistantMessage = [...resolveMessagesForSession(sessionId)]
            .reverse()
            .find((message) => message.type === 'assistant')

          if (finalAssistantMessage) {
            updateLastMessage(sessionId, finalAssistantMessage.content)
          }

          setWorkflowDraftBySessionId((current) => ({
            ...current,
            [sessionId]: nextWorkflowState,
          }))
          return
        }

        let streamed = false
        let latestAssistantContent = ''
        activeChatRunRef.current = {
          sessionId,
          turnId: null,
          chatRunId: null,
        }
        pendingChatCancelRef.current = null
        try {
          for await (const chunk of api.streamChatMessages(requestText, {
            sessionId,
            projectId:
              sessionContext.latestSessionState.sessions.find((session) => session.id === sessionId)?.projectId ??
              activeProjectId ??
              null,
            model: currentModel ?? undefined,
            toolMode,
            selectedSkillIds,
            attachments,
            systemPrompt: systemPromptOverride ?? effectiveInference.systemPrompt,
            temperature: effectiveInference.temperature,
            maxTokens: effectiveInference.maxTokens,
            reserveOutputTokens: effectiveInference.reserveOutputTokens,
            topP: effectiveInference.topP,
            minP: effectiveInference.minP,
            topK: effectiveInference.topK,
            frequencyPenalty: effectiveInference.frequencyPenalty,
            presencePenalty: effectiveInference.presencePenalty,
            repeatPenalty: effectiveInference.repeatPenalty,
            reasoningEffort: effectiveInference.reasoningEffort,
            signal: abortController.signal,
            onSessionId: (nextSessionId) => {
              if (nextSessionId && nextSessionId !== targetSessionId) {
                void selectSession(nextSessionId)
              }
            },
            onResponseMeta: (meta) => {
              activeChatRunRef.current = {
                sessionId: meta.sessionId ?? activeChatRunRef.current.sessionId,
                turnId: meta.turnId ?? activeChatRunRef.current.turnId,
                chatRunId: meta.chatRunId ?? activeChatRunRef.current.chatRunId,
              }
            },
          })) {
            streamed = true

            const subagentEvent = normalizeStreamSubagentEvent(chunk.rawEvent)
            if (subagentEvent) {
              setExecutionTimelineEvents((current) => mergeTranscriptEvents(current, [subagentEvent]))
              const subagentSummary = buildSubagentSummaryFromEvent({
                ...subagentEvent,
                sessionId: subagentEvent.sessionId ?? sessionId,
              })
              if (subagentSummary) {
                if (subagentSummary.agentRunId) {
                  setAgentRunSubagents((current) => mergeSubagentSummaries(current, [subagentSummary]))
                } else {
                  setSessionSubagents((current) => mergeSubagentSummaries(current, [subagentSummary]))
                }
              }
              if (activeSubagentId && subagentEvent.subagentId === activeSubagentId) {
                setActiveSubagentDetail((current) => current
                  ? {
                      ...current,
                      events: mergeTranscriptEvents(current.events, [subagentEvent]),
                      updatedAt: subagentEvent.createdAt ?? current.updatedAt ?? null,
                    }
                  : current)
              }
            }

            if (chunk.event?.type === 'assistant' && chunk.event.content) {
              latestAssistantContent = chunk.event.content
            }
            if (chunk.model) {
              setCurrentModel(chunk.model)
            }

            updateSessionMessages(sessionId, (prev) => applyStreamChunk(prev, chunk, turnKey))
          }
        } catch (streamError) {
          if (!isStreamUnavailable(streamError)) {
            throw streamError
          }
        }

        if (!streamed) {
          const response = await requestChat(
            requestText,
            sessionId,
            sessionContext.latestSessionState.sessions.find((session) => session.id === sessionId)?.projectId ??
              activeProjectId ??
              null,
            currentModel,
            selectedSkillIds,
            attachments,
            toolMode,
            {
              ...effectiveInference,
              systemPrompt: systemPromptOverride ?? effectiveInference.systemPrompt,
            }
          )
          const eventMessages = response.events?.length
            ? api.buildMessagesFromChatEvents(response.events)
            : [
                {
                  id: `assistant-${Date.now()}`,
                  type: 'assistant' as const,
                  eventType: 'final_answer' as const,
                  content:
                    response.final_answer ??
                    response.content ??
                    t('chat.emptyAssistantResponse'),
                  timestamp: new Date(),
                  turnKey,
                },
              ]

          setSessionMessages(sessionId, [
            ...resolveMessagesForSession(sessionId).filter((message) => message.turnKey !== turnKey),
            ...eventMessages,
          ])

          const finalAssistantMessage = [...eventMessages]
            .reverse()
            .find((message) => message.type === 'assistant')

          if (finalAssistantMessage) {
            updateLastMessage(sessionId, finalAssistantMessage.content)
          }

          if (response.model) {
            setCurrentModel(response.model)
          }
        } else if (latestAssistantContent) {
          updateLastMessage(sessionId, latestAssistantContent)
        }

        await syncSessionFromServer(sessionId)
      } catch (error) {
        if (error instanceof DOMException && error.name === 'AbortError') {
          const cancelResponse = pendingChatCancelRef.current
          updateSessionMessages(sessionId, (prev) => prev.map((message) =>
            message.turnKey === turnKey ? { ...message, isStreaming: false } : message
          ))
          if (
            cancelResponse?.status === 'already_completed' ||
            cancelResponse?.status === 'cancel_requested'
          ) {
            try {
              await syncSessionFromServer(sessionId)
            } catch {
              // Transport stop already succeeded; leave the optimistic state as-is.
            }
          }
          return
        }
        const detail = error instanceof Error ? error.message : null
        if (normalizedWorkflow.enabled) {
          setWorkflowError(detail ?? 'Workflow request failed.')
        }
        updateSessionMessages(sessionId, (prev) => [
          ...prev.filter((message) => message.turnKey !== turnKey),
          {
            id: `error-${Date.now()}`,
            type: 'error',
            eventType: 'error',
            content: t('chat.requestFailed'),
            errorCode: detail ?? 'CHAT_REQUEST_FAILED',
            timestamp: new Date(),
          },
        ])
      } finally {
        setWorkflowBusy(false)
        activeChatRunRef.current = { sessionId: null, turnId: null, chatRunId: null }
        pendingChatCancelRef.current = null
        finishStreaming(sessionId)
      }
    },
    [
      activeProjectId,
      activeSubagentId,
      appendOptimisticConversationTurn,
      currentModel,
      currentModelLoaded,
      effectiveInference,
      finishStreaming,
      modelOptions,
      projects,
      resolveMessagesForSession,
      selectSession,
      setSessionMessages,
      startStreaming,
      syncSessionFromServer,
      syncWorkflowRunEventsToSession,
      t,
      updateSessionMessages,
      updateLastMessage,
      uploadTargetDir,
    ]
  )

  const handleSend = React.useCallback(
    async (
      text: string,
      options?: {
        forceSessionId?: string
        selectedSkillIds?: string[]
        attachments?: ChatAttachment[]
      }
    ) => {
      if (hasActiveStream) {
        return
      }

      const targetSessionId = options?.forceSessionId ?? currentSessionId
      const selectedSkillIds = options?.selectedSkillIds ?? []
      const attachments = options?.attachments ?? []
      const initialSessionId = targetSessionId ?? createDraftSession(activeProjectId)
      const targetSession = sessions.find((session) => session.id === initialSessionId)
      const sessionScope = createSendSessionScope(initialSessionId, targetSession)
      let sessionId = sessionScope.getSessionId()
      let sessionContext = sessionScope.getContext()
      const pendingProposal = sessionContext.baseGoalState.pending_proposal
      let {
        modeCommand,
        requestText,
        route,
        workflowModeRequested,
        requiresSessionMaterialization,
        shouldHandleGoalWorkflowRouting,
      } = resolveChatGoalWorkflowRouting({
        text,
        attachmentCount: attachments.length,
        hasPendingProposal: pendingProposal !== null,
        hasActiveGoal:
          sessionContext.baseGoalState.active_goal_id !== null &&
            !isGoalTerminalStatus(sessionContext.baseGoalState.active_goal_status),
      })
      const workflowProposalRequested = route.kind === 'workflow_proposal'

      if (requiresSessionMaterialization) {
        await sessionScope.materializeIfNeeded()
        sessionId = sessionScope.getSessionId()
        sessionContext = sessionScope.getContext()
      }

      if (route.kind === 'goal_pending_follow_up') {
        const currentPendingProposal = sessionContext.baseGoalState.pending_proposal
        if (currentPendingProposal) {
          let intentResult: api.PendingGoalProposalIntentResult
          try {
            intentResult = await api.classifyPendingGoalProposalIntent({
              message: route.requestText,
              proposalObjective: currentPendingProposal.objective,
              executionMode: currentPendingProposal.execution_mode,
            })
          } catch (error) {
            intentResult = {
              intent: 'ambiguous',
              confidence: null,
              rationale: error instanceof Error ? error.message : 'Goal intent classification failed.',
            }
          }

          if (intentResult.intent === 'confirm_start') {
            route = {
              kind: 'goal_confirmation',
              requestText: route.requestText,
              raw: route.raw,
            }
          } else if (intentResult.intent === 'revise_proposal') {
            route = {
              kind: 'goal_revision',
              requestText: route.requestText,
            }
          } else if (intentResult.intent === 'exit_goal_lane') {
            route = { kind: 'direct_chat' }
            shouldHandleGoalWorkflowRouting = false
          } else {
            await persistGoalConversation({
              sessionId,
              attachments,
              userContent: route.raw,
              assistantContent:
                currentPendingProposal.assistant_explanation ??
                buildLocalGoalProposalAssistantExplanation(
                  route.requestText || route.raw || currentPendingProposal.objective,
                  currentPendingProposal
                ),
              goalCard: goalCardFromSummary(
                currentPendingProposal,
                currentPendingProposal.revision_index > 0 ? 'revised_proposal' : 'proposal',
                {
                  copySource: route.raw,
                }
              ),
            })
            return
          }
        } else {
          route = { kind: 'direct_chat' }
          shouldHandleGoalWorkflowRouting = false
        }
      }

      const normalizedWorkflow = modeCommand
        ? normalizeWorkflowState(
            {
              ...sessionContext.baseWorkflow,
              enabled: workflowModeRequested,
            },
            effectiveInference.reasoningEffort
          )
        : sessionContext.baseWorkflow

      if (modeCommand && route.kind === 'direct_chat') {
        if (workflowModeRequested) {
          setTaskPanelOpen(false)
          setTaskPanelMode('default')
          setTaskPanelFocusedTaskId(null)
          setPanelOpen(false)
          setMobileInferenceOpen(false)
          setWorkflowPanelOpen(true)
        } else {
          setWorkflowPanelOpen(false)
        }
        await persistWorkflowState(sessionId, normalizedWorkflow)
        if (requestText.length === 0 && attachments.length === 0) {
          return
        }
      }
      if (workflowProposalRequested) {
        setTaskPanelOpen(false)
        setTaskPanelMode('default')
        setTaskPanelFocusedTaskId(null)
        setPanelOpen(false)
        setMobileInferenceOpen(false)
        setWorkflowPanelOpen(true)
      }

      if (shouldHandleGoalWorkflowRouting && route.kind !== 'direct_chat') {
        const routeTurnKey = `turn-${Date.now()}`
        const routeUserContent = resolveGoalWorkflowRouteUserContent(route, requestText)
        appendOptimisticConversationTurn(sessionId, routeUserContent, attachments, routeTurnKey)
        try {
          const handledGoalWorkflowRouting = await handleGoalWorkflowRouting({
            sessionId,
            attachments,
            selectedSkillIds,
            route,
            requestText,
            baseWorkflow: sessionContext.baseWorkflow,
            baseGoalState: sessionContext.baseGoalState,
            workflowSessionProjectId: sessionContext.workflowSessionProjectId,
            effectiveWorkspaceDir: sessionContext.effectiveWorkspaceDir,
          })
          if (handledGoalWorkflowRouting) {
            return
          }
          updateSessionMessages(sessionId, (prev) => prev.map((message) =>
            message.turnKey === routeTurnKey ? { ...message, isStreaming: false } : message
          ))
          return
        } catch (error) {
          const detail = error instanceof Error ? error.message : null
          updateSessionMessages(sessionId, (prev) => [
            ...prev.filter((message) => message.turnKey !== routeTurnKey),
            {
              id: `error-${Date.now()}`,
              type: 'error',
              eventType: 'error',
              content: t('chat.requestFailed'),
              errorCode: detail ?? 'CHAT_REQUEST_FAILED',
              timestamp: new Date(),
            },
          ])
          return
        }
      }

      const activeGoalId = sessionContext.baseGoalState.active_goal_id
      const hasActiveNonTerminalGoal =
        activeGoalId !== null && !isGoalTerminalStatus(sessionContext.baseGoalState.active_goal_status)
      if (
        route.kind === 'direct_chat' &&
        hasActiveNonTerminalGoal &&
        sessionContext.baseGoalState.pending_proposal === null &&
        requestText.trim().length > 0
      ) {
        const handledActiveGoalDecision = await handleActiveGoalDirectTurnDecision({
          sessionId,
          requestText,
          attachments,
          activeGoalId,
          baseWorkflow: sessionContext.baseWorkflow,
          latestGoalSummary: sessionContext.baseGoalState.last_goal_summary,
          selectedSkillIds,
          sessionScope,
        })
        if (handledActiveGoalDecision) {
          return
        }
      }

      await submitDirectChatTurn({
        targetSessionId,
        requestText,
        attachments,
        selectedSkillIds,
        toolMode: undefined,
        normalizedWorkflow,
        sessionScope,
      })
    },
    [
      activeProjectId,
      appendOptimisticConversationTurn,
      createSendSessionScope,
      createDraftSession,
      currentSessionId,
      effectiveInference,
      handleGoalWorkflowRouting,
      hasActiveStream,
      persistGoalConversation,
      persistWorkflowState,
      setPanelOpen,
      setTaskPanelFocusedTaskId,
      setTaskPanelMode,
      setTaskPanelOpen,
      setMobileInferenceOpen,
      setWorkflowPanelOpen,
      sessions,
      submitDirectChatTurn,
      t,
      updateSessionMessages,
    ]
  )

  const handleSearchSkills = React.useCallback(async (query: string) => {
    return api.fetchSkills({ q: query, limit: 20 })
  }, [])

  const headerModelLabel =
    modelOptions.find((option) => option.id === currentModel)?.label ??
    (currentModel ? formatModelLabel(currentModel) : 'configured')

  const handleVoiceEntry = React.useCallback(async () => {
    setVoiceOpen(true)
    setVoiceErrorMessage(null)
    try {
      const status = await refreshVoiceRuntimeStatus()
      const runtimePhase = resolveVoicePhaseFromRuntime(status)
      if (runtimePhase) {
        setVoicePhase(runtimePhase)
      }
      if (status?.error) {
        setVoiceErrorMessage(status.error)
        return
      }
      const sessionId = currentSessionId ?? createDraftSession(activeProjectId)
      voiceSessionIdRef.current = sessionId
      const client = ensureVoiceClient(sessionId)
      await client.connect()
    } catch (error) {
      const detail = error instanceof Error ? error.message : 'Voice connect failed.'
      setVoiceErrorMessage(detail)
      setVoicePhase('error')
    }
  }, [activeProjectId, createDraftSession, currentSessionId, ensureVoiceClient, refreshVoiceRuntimeStatus])

  const handleVoiceToggleRecording = React.useCallback(async () => {
    const sessionId = voiceSessionIdRef.current ?? currentSessionId ?? createDraftSession(activeProjectId)
    voiceSessionIdRef.current = sessionId
    const client = ensureVoiceClient(sessionId)
    if (voiceRecording) {
      await client.stopRecording()
      setVoiceRecording(false)
      return
    }
    setVoicePartialTranscription('')
    setVoiceFinalTranscription('')
    setVoiceAssistantText('')
    setVoiceInputLevel(0)
    setVoiceVadState(null)
    setVoiceCaptureDiagnostics(null)
    setVoiceCaptureWarning(null)
    setVoiceErrorMessage(null)
    try {
      setVoicePhase('connecting')
      const preparedStatus = await api.prepareVoiceRuntime(sessionId)
      setVoiceRuntimeStatus(preparedStatus)
      const preparedPhase = resolveVoicePhaseFromRuntime(preparedStatus)
      if (preparedPhase) {
        setVoicePhase(preparedPhase)
      }
      if (preparedStatus.error) {
        setVoiceErrorMessage(preparedStatus.error)
        setVoicePhase('error')
        setVoiceRecording(false)
        return
      }
      await client.startRecording()
      setVoiceRecording(true)
    } catch (error) {
      const detail = error instanceof Error ? error.message : 'Unable to start recording.'
      setVoiceErrorMessage(detail)
      setVoicePhase('error')
      setVoiceRecording(false)
    }
  }, [activeProjectId, createDraftSession, currentSessionId, ensureVoiceClient, voiceRecording])

  const handleVoiceInterrupt = React.useCallback(() => {
    const client = voiceClientRef.current
    if (!client) {
      return
    }
    client.interrupt()
    setVoiceRecording(false)
    setVoicePartialTranscription('')
    setVoiceInputLevel(0)
    setVoiceVadState(null)
    setVoiceCaptureWarning(null)
  }, [])

  const handleVoiceClose = React.useCallback(() => {
    setVoiceOpen(false)
    const client = voiceClientRef.current
    if (!client) {
      voiceSessionIdRef.current = null
      return
    }
    void client.disconnect()
    voiceClientRef.current = null
    voiceSessionIdRef.current = null
    setVoiceRecording(false)
    setVoicePhase('idle')
    setVoicePartialTranscription('')
    setVoiceFinalTranscription('')
    setVoiceAssistantText('')
    setVoiceInputLevel(0)
    setVoiceVadState(null)
    setVoiceCaptureDiagnostics(null)
    setVoiceCaptureWarning(null)
    setVoiceErrorMessage(null)
  }, [])

  const handleVoiceShortcut = React.useCallback(() => {
    if (!voiceOpen) {
      void handleVoiceEntry().then(() => {
        window.setTimeout(() => {
          void handleVoiceToggleRecording()
        }, 50)
      })
      return
    }
    void handleVoiceToggleRecording()
  }, [handleVoiceEntry, handleVoiceToggleRecording, voiceOpen])

  React.useEffect(() => {
    window.addEventListener('mochi:voice-toggle', handleVoiceShortcut)
    return () => {
      window.removeEventListener('mochi:voice-toggle', handleVoiceShortcut)
    }
  }, [handleVoiceShortcut])

  const handleStopGeneration = React.useCallback(() => {
    void (async () => {
      const activeRun = activeChatRunRef.current
      if (activeRun.sessionId) {
        try {
          pendingChatCancelRef.current = await api.cancelChatRun(activeRun.sessionId, {
            turnId: activeRun.turnId,
          })
        } catch {
          pendingChatCancelRef.current = null
        }
      }
      abortStreaming()
    })()
  }, [abortStreaming])

  const handleBuiltinCommand = React.useCallback(async (
    command: 'clear' | 'settings' | 'voice' | 'model' | 'export' | 'workflow' | 'chat'
  ) => {
    if (command === 'clear') {
      if (currentSessionId) {
        setSessionMessages(currentSessionId, createInitialMessages(t))
      }
      return
    }
    if (command === 'settings') {
      router.push('/settings')
      return
    }
    if (command === 'voice') {
      void handleVoiceEntry()
      return
    }
    if (command === 'model') {
      const modelButton = document.querySelector<HTMLButtonElement>('#chat-model-selector,[data-chat-model-selector="true"]')
      modelButton?.focus()
      modelButton?.click()
      return
    }
    if (command === 'export') {
      setExportOpen(true)
      return
    }
    if (command === 'workflow' || command === 'chat') {
      return
    }
  }, [
    currentSessionId,
    handleVoiceEntry,
    router,
    setSessionMessages,
    t,
  ])

  const handleUndoFileChange = React.useCallback(async (change: FileChangeSummary) => {
    if (!change.undoAvailable || !change.undoAction) {
      return
    }

    await api.undoFileWrite({
      file_path: change.filePath,
      original_content: change.originalContent,
      session_id: currentSessionId ?? undefined,
      action: change.undoAction,
      encoding: 'utf-8',
    })
    const workspaceState = useWorkspaceStore.getState()
    await workspaceState.loadChanges()
    await workspaceState.loadTree(workspaceState.currentPath)
    if (workspaceState.diff && workspaceState.selectedFilePath === change.filePath) {
      await workspaceState.loadDiff(change.filePath)
      return
    }
    if (workspaceState.preview && workspaceState.selectedFilePath === change.filePath) {
      await workspaceState.previewFile(change.filePath)
    }
  }, [currentSessionId])

  const handleRegenerate = React.useCallback((message: Message) => {
    const prompt = findRegeneratePrompt(messages, message.id)
    if (!prompt) {
      return
    }
    void handleSend(prompt)
  }, [handleSend, messages])

  const handleQueueWorkspaceAttachment = React.useCallback((attachment: ChatAttachment) => {
    setQueuedWorkspaceAttachments([attachment])
    setQueuedWorkspaceAttachmentsKey(`${attachment.id ?? attachment.path}-${Date.now()}`)
  }, [])

  const handleEditAndResend = React.useCallback((message: Message) => {
    const selectedSkillIds = (() => {
      if (!currentSessionDetail || !message.turnId) {
        return []
      }
      const matched = currentSessionDetail.events.find(
        (event) =>
          event.type === 'message' &&
          event.role === 'user' &&
          String(
            ('turn_id' in event ? event.turn_id : undefined) ??
            ('turnId' in event ? event.turnId : undefined) ??
            ''
          ) === message.turnId
      ) as Record<string, unknown> | undefined
      return getStringArray(matched?.selected_skill_ids)
    })()

    setEditState({
      messageId: message.id,
      turnId: message.turnId ?? null,
      resetKey: `${message.id}-${message.turnId ?? 'no-turn'}-${Date.now()}`,
      seed: {
        text: message.content,
        attachments: [...(message.attachments ?? [])],
        selectedSkills: selectedSkillIds.map((id) => ({ id, name: id })),
      },
    })
  }, [currentSessionDetail])

  const handleCancelEdit = React.useCallback(() => {
    setEditState(null)
  }, [])

  const handleSubmitEdit = React.useCallback(async (
    nextContent: string,
    options?: {
      selectedSkillIds?: string[]
      attachments?: ChatAttachment[]
    }
  ) => {
    const attachments = options?.attachments ?? []
    const selectedSkillIds = options?.selectedSkillIds ?? []

    if (!editState?.turnId || !currentSessionId) {
      setEditState(null)
      await handleSend(nextContent, { attachments, selectedSkillIds })
      return
    }

    const rewrittenSession = await api.rewriteSessionFromTurn(currentSessionId, editState.turnId)
    const rewrittenMessages = api.buildMessagesFromSessionEvents(rewrittenSession.events)
    const baseMessages = rewrittenMessages.length > 0 ? rewrittenMessages : createInitialMessages(t)
    const lastRetainedMessage = [...baseMessages]
      .reverse()
      .find((entry) => entry.type === 'user' || entry.type === 'assistant')

    setSessionMessages(currentSessionId, baseMessages)
    updateLastMessage(currentSessionId, lastRetainedMessage?.content ?? '')
    upsertSessionDetail(rewrittenSession)
    setEditState(null)
    void selectSession(currentSessionId)
    await handleSend(nextContent, {
      forceSessionId: currentSessionId,
      attachments,
      selectedSkillIds,
    })
  }, [
    currentSessionId,
    editState,
    handleSend,
    selectSession,
    setSessionMessages,
    t,
    updateLastMessage,
    upsertSessionDetail,
  ])

  const handleStarterPrompt = React.useCallback((prompt: string) => {
    void handleSend(prompt)
  }, [handleSend])

  const handleWorkflowToggle = React.useCallback(async (enabled: boolean) => {
    const initialSessionId = currentSessionId ?? createDraftSession(activeProjectId)
    const targetSession = sessions.find((session) => session.id === initialSessionId)
    const sessionId = initialSessionId
    const nextWorkflow = normalizeWorkflowState(
      workflowDraftBySessionId[sessionId] ?? targetSession?.workflow ?? null,
      effectiveInference.reasoningEffort
    )
    nextWorkflow.enabled = enabled
    if (enabled) {
      setTaskPanelOpen(false)
      setTaskPanelMode('default')
      setTaskPanelFocusedTaskId(null)
      setPanelOpen(false)
      setMobileInferenceOpen(false)
    }
    setWorkflowPanelOpen(enabled)
    await persistWorkflowState(sessionId, nextWorkflow)
  }, [
    activeProjectId,
    createDraftSession,
    currentSessionId,
    effectiveInference.reasoningEffort,
    persistWorkflowState,
    sessions,
    setPanelOpen,
    workflowDraftBySessionId,
  ])

  const handleWorkflowFieldChange = React.useCallback((
    patch: Partial<api.SessionWorkflowState>
  ) => {
    if (!currentSessionId) {
      return
    }
    const nextWorkflow = normalizeWorkflowState(
      {
        ...workflowState,
        ...patch,
        config: {
          ...(workflowState.config ?? {}),
          ...(patch.config ?? {}),
        },
      },
      effectiveInference.reasoningEffort
    )
    setWorkflowDraftBySessionId((current) => ({
      ...current,
      [currentSessionId]: nextWorkflow,
    }))
    setWorkflowSaveState((current) => (current === 'saving' ? current : 'idle'))
  }, [currentSessionId, effectiveInference.reasoningEffort, workflowState])

  const handleWorkflowConfigPatch = React.useCallback((
    patch: Partial<api.SessionWorkflowConfig>
  ) => {
    handleWorkflowFieldChange({
      config: {
        ...(workflowState.config ?? {}),
        ...patch,
      },
    })
  }, [handleWorkflowFieldChange, workflowState.config])

  const handleWorkflowTemplateChange = React.useCallback((template: WorkflowTemplate) => {
    handleWorkflowConfigPatch({
      template,
      protocol_id: template === 'research_debate' ? 'multi_agent_debate' : workflowProtocolId,
      research:
        template === 'research_debate'
          ? {
              smart_model_id: currentModel ?? modelOptions[0]?.id ?? '',
              local_worker_model_id: currentModel ?? modelOptions[0]?.id ?? '',
              output_targets: ['research_brief', 'dataset_package'],
              source_mode: 'hybrid',
              citation_policy: 'claim_level_required',
              local_worker_count: 3,
              debate_rounds: 2,
              ...(workflowConfig.research ?? {}),
            }
          : (workflowConfig.research ?? {}),
    })
  }, [
    currentModel,
    handleWorkflowConfigPatch,
    modelOptions,
    workflowConfig.research,
    workflowProtocolId,
  ])

  const handleWorkflowRunPolicyPresetChange = React.useCallback((preset: WorkflowRunPolicyPreset) => {
    if (preset === 'custom') {
      handleWorkflowConfigPatch({
        run_policy_preset: preset,
      })
      return
    }
    handleWorkflowConfigPatch({
      run_policy_preset: preset,
      run_policy: runPolicyPresetValues(preset),
    })
  }, [handleWorkflowConfigPatch])

  const handleWorkflowSave = React.useCallback(async () => {
    if (!currentSessionId) {
      return
    }
    const targetSession = sessions.find((session) => session.id === currentSessionId)
    const isDraftWorkflowSession = Boolean(targetSession?.isDraft || currentSessionId.startsWith('draft-'))
    setWorkflowError(null)
    setWorkflowSaveState('saving')
    try {
      await persistWorkflowState(currentSessionId, workflowState)
      setWorkflowSaveState('saved')
      setWorkflowLastSavedAt(new Date().toISOString())
      setWorkflowLastSaveScope(isDraftWorkflowSession ? 'draft' : 'persisted')
    } catch (error) {
      const detail = error instanceof Error ? error.message : 'Unable to save workflow settings.'
      setWorkflowError(detail)
      setWorkflowSaveState('error')
    }
  }, [currentSessionId, persistWorkflowState, sessions, workflowState])

  const handleWorkflowProjectChange = React.useCallback(async (projectId: string | null) => {
    const initialSessionId = currentSessionId ?? createDraftSession(projectId)
    const targetSession = sessions.find((session) => session.id === initialSessionId)
    const sessionId = targetSession?.isDraft
      ? initialSessionId
      : initialSessionId
    try {
      await moveSessionToProject(sessionId, projectId)
    } catch (error) {
      const detail = error instanceof Error ? error.message : 'Unable to update workflow project.'
      setWorkflowError(detail)
      return
    }

    handleWorkflowFieldChange({
      workspace_dir_override: null,
      config: {
        ...(workflowState.config ?? {}),
        workspace_dir_override: null,
      },
    })
  }, [
    createDraftSession,
    currentSessionId,
    handleWorkflowFieldChange,
    moveSessionToProject,
    sessions,
    workflowState.config,
  ])

  React.useEffect(() => {
    setWorkflowSaveState('idle')
    setWorkflowLastSavedAt(null)
    setWorkflowLastSaveScope(null)
  }, [currentSessionId])

  React.useEffect(() => {
    if (workflowSaveState !== 'saving' && workflowHasUnsavedChanges) {
      setWorkflowSaveState('idle')
    }
  }, [workflowHasUnsavedChanges, workflowSaveState])

  const handleWorkflowNewRun = React.useCallback(async () => {
    if (!currentSessionId) {
      return
    }
    const nextWorkflow = normalizeWorkflowState(
      {
        ...workflowState,
        bound_run_id: null,
        synced_run_event_count: 0,
      },
      effectiveInference.reasoningEffort
    )
    await persistWorkflowState(currentSessionId, nextWorkflow)
  }, [currentSessionId, effectiveInference.reasoningEffort, persistWorkflowState, workflowState])

  const handleSessionInferenceChange = React.useCallback(<K extends keyof typeof effectiveInference>(
    key: K,
    value: (typeof effectiveInference)[K]
  ) => {
    const sessionId = currentSessionId ?? createDraftSession(activeProjectId)
    setSessionOverride(sessionId, key, value)
  }, [activeProjectId, createDraftSession, currentSessionId, setSessionOverride])

  const handleApplyPresetToSession = React.useCallback(() => {
    if (!currentSessionId || !activeAgentSettings) {
      return
    }
    const preset =
      activeAgentSettings.presets.find((item) => item.name === selectedPresetName) ??
      getActivePreset(activeAgentSettings)
    if (!preset) {
      return
    }
    replaceSessionOverride(currentSessionId, {
      ...resolveEffectiveInferenceParams(undefined, activeAgentSettings),
      systemPrompt: preset.system_prompt,
      temperature: preset.temperature,
      maxTokens: preset.max_tokens,
      reserveOutputTokens: preset.reserve_output_tokens,
      topP: preset.top_p,
      minP: preset.min_p,
      topK: preset.top_k,
      frequencyPenalty: preset.frequency_penalty,
      presencePenalty: preset.presence_penalty,
      repeatPenalty: preset.repeat_penalty,
      reasoningEffort: preset.reasoning_effort ?? null,
    })
  }, [activeAgentSettings, currentSessionId, replaceSessionOverride, selectedPresetName])

  const handleResetSessionInference = React.useCallback(() => {
    if (!currentSessionId) {
      return
    }
    resetSessionOverride(currentSessionId)
  }, [currentSessionId, resetSessionOverride])

  const handleSaveInferencePreset = React.useCallback(async () => {
    if (!activeAgentSettings) {
      return
    }

    const targetPreset =
      activeAgentSettings.presets.find((preset) => preset.name === selectedPresetName) ??
      getActivePreset(activeAgentSettings)
    if (!targetPreset) {
      return
    }

    const nextPresets = activeAgentSettings.presets.map((preset) =>
      preset.name === targetPreset.name
        ? {
            ...preset,
            system_prompt: effectiveInference.systemPrompt,
            temperature: effectiveInference.temperature,
            max_tokens: effectiveInference.maxTokens,
            reserve_output_tokens: effectiveInference.reserveOutputTokens,
            top_p: effectiveInference.topP,
            min_p: effectiveInference.minP,
            top_k: effectiveInference.topK,
            frequency_penalty: effectiveInference.frequencyPenalty,
            presence_penalty: effectiveInference.presencePenalty,
            repeat_penalty: effectiveInference.repeatPenalty,
            reasoning_effort: effectiveInference.reasoningEffort,
          }
        : preset
    )

    setSavingPreset(true)
    try {
      const nextSettings = await api.updateSettings({
        agent: {
          presets: nextPresets.map((preset) => ({
            name: preset.name,
            system_prompt: preset.system_prompt,
            temperature: preset.temperature,
            max_tokens: preset.max_tokens,
            reserve_output_tokens: preset.reserve_output_tokens,
            top_p: preset.top_p,
            min_p: preset.min_p,
            top_k: preset.top_k,
            frequency_penalty: preset.frequency_penalty,
            presence_penalty: preset.presence_penalty,
            repeat_penalty: preset.repeat_penalty,
            reasoning_effort: preset.reasoning_effort ?? null,
          })),
          active_preset: activeAgentSettings.active_preset,
        },
      })
      setSettings(nextSettings)
      window.dispatchEvent(new Event('mochi:settings-updated'))
    } finally {
      setSavingPreset(false)
    }
  }, [activeAgentSettings, effectiveInference, selectedPresetName])

  const closeRightPanels = React.useCallback((except?: 'goal' | 'inference' | 'tasks' | 'workflow' | 'subagent') => {
    setWorkspaceMobileOpen(false)
    if (except !== 'goal') {
      setGoalDrawerOpen(false)
    }
    if (except !== 'subagent') {
      setSubagentDrawerOpen(false)
    }
    if (except !== 'inference') {
      setPanelOpen(false)
      setMobileInferenceOpen(false)
    }
    if (except !== 'tasks') {
      setTaskPanelOpen(false)
      setTaskPanelMode('default')
      setTaskPanelFocusedTaskId(null)
      setTaskPanelFocusedApprovalIds(null)
    }
    if (except !== 'workflow') {
      setWorkflowPanelOpen(false)
    }
  }, [setPanelOpen])

  const handleGoalDrawerToggle = React.useCallback(() => {
    const nextOpen = !goalDrawerOpen
    closeRightPanels('goal')
    setGoalDrawerOpen(nextOpen)
  }, [closeRightPanels, goalDrawerOpen])

  const loadSubagentDetail = React.useCallback(async (
    target: api.SubagentTranscriptSummary,
    subagentId: string
  ) => {
    setSubagentDrawerLoading(true)
    setSubagentDrawerError(null)
    try {
      const detail = target.agentRunId
        ? await api.fetchAgentRunSubagent(target.agentRunId, subagentId)
        : target.sessionId
          ? await api.fetchSessionSubagent(target.sessionId, subagentId)
          : null
      if (!detail) {
        throw new Error('Subagent transcript not found.')
      }
      setActiveSubagentDetail(detail)
      if (detail.agentRunId) {
        setAgentRunSubagents((current) => mergeSubagentSummaries(current, [detail]))
      } else {
        setSessionSubagents((current) => mergeSubagentSummaries(current, [detail]))
      }
    } catch (error) {
      setSubagentDrawerError(error instanceof Error ? error.message : 'Failed to load subagent.')
    } finally {
      setSubagentDrawerLoading(false)
    }
  }, [])

  const handleOpenSubagent = React.useCallback((subagentId: string) => {
    const summary =
      allVisibleSubagents.find((item) => item.subagentId === subagentId) ??
      null
    if (!summary) {
      return
    }

    closeRightPanels('subagent')
    setActiveSubagentId(subagentId)
    setSubagentGuidance('')
    setSubagentDrawerOpen(true)
    setActiveSubagentDetail(
      summary
        ? {
            ...summary,
            systemPrompt: null,
            userPrompt: null,
            events: [],
          }
        : null
    )
    void loadSubagentDetail(summary, subagentId)
  }, [allVisibleSubagents, closeRightPanels, loadSubagentDetail])

  const handleSendSubagentGuidance = React.useCallback(async () => {
    if (!activeSubagentId || subagentGuidance.trim().length === 0) {
      return
    }
    const target =
      allVisibleSubagents.find((item) => item.subagentId === activeSubagentId) ??
      activeSubagentDetail
    if (!target) {
      return
    }

    setSubagentGuidanceSending(true)
    setSubagentDrawerError(null)
    try {
      if (target.agentRunId) {
        try {
          await api.sendAgentRunSubagentMessage(target.agentRunId, activeSubagentId, {
            content: subagentGuidance.trim(),
            projectId: effectiveProjectId,
            workspaceDir: effectiveWorkflowWorkspace || null,
            metadata: {
              target_subagent_id: activeSubagentId,
            },
          })
        } catch {
          await api.appendAgentRunSubagentMessage(target.agentRunId, target.roleId ?? activeSubagentId, {
            content: subagentGuidance.trim(),
            projectId: effectiveProjectId,
            workspaceDir: effectiveWorkflowWorkspace || null,
            metadata: {
              target_subagent_id: activeSubagentId,
            },
          })
        }
      } else if (target.sessionId) {
        const baseInput: api.AgentRunSubagentMessageInput = {
          content: subagentGuidance.trim(),
          projectId: effectiveProjectId,
          workspaceDir: effectiveWorkflowWorkspace || null,
          metadata: {
            target_subagent_id: activeSubagentId,
          },
        }
        const deliveryDefaults = getSubagentMessageDeliveryDefaults(target.status)
        const inputWithDelivery: api.AgentRunSubagentMessageInput = {
          ...baseInput,
          ...deliveryDefaults,
        }
        let detail: api.SubagentTranscriptDetail | null = null
        try {
          detail = await api.sendSessionSubagentMessage(target.sessionId, activeSubagentId, inputWithDelivery)
        } catch (error) {
          if (deliveryDefaults.deliveryMode && isLegacySubagentMessageDeliveryError(error)) {
            detail = await api.sendSessionSubagentMessage(target.sessionId, activeSubagentId, baseInput)
          } else {
            throw error
          }
        }
        if (detail) {
          setActiveSubagentDetail(detail)
          setSessionSubagents((current) => mergeSubagentSummaries(current, [detail]))
        }
      } else {
        throw new Error('Subagent guidance target is unavailable.')
      }
      setSubagentGuidance('')
      await loadSubagentDetail(target, activeSubagentId)
    } catch (error) {
      setSubagentDrawerError(error instanceof Error ? error.message : 'Failed to send guidance.')
    } finally {
      setSubagentGuidanceSending(false)
    }
  }, [
    activeSubagentDetail,
    activeSubagentId,
    allVisibleSubagents,
    effectiveProjectId,
    effectiveWorkflowWorkspace,
    loadSubagentDetail,
    subagentGuidance,
  ])

  const runSessionSubagentAction = React.useCallback(async (action: 'cancel' | 'resume') => {
    if (!activeSubagentId || subagentBusyAction) {
      return
    }
    const target =
      allVisibleSubagents.find((item) => item.subagentId === activeSubagentId) ??
      activeSubagentDetail
    if (!target?.sessionId) {
      setSubagentDrawerError('Session subagent target is unavailable.')
      return
    }

    const guidance = action === 'resume' ? subagentGuidance.trim() : ''
    setSubagentBusyAction(action)
    setSubagentDrawerError(null)
    try {
      const actionDetail = action === 'cancel'
        ? await api.cancelSessionSubagent(target.sessionId, activeSubagentId)
        : await api.resumeSessionSubagent(target.sessionId, activeSubagentId, {
            guidance: guidance.length > 0 ? guidance : null,
          })
      if (actionDetail) {
        setActiveSubagentDetail(actionDetail)
        setSessionSubagents((current) => mergeSubagentSummaries(current, [actionDetail]))
      }

      const [subagents, detail] = await Promise.all([
        api.fetchSessionSubagents(target.sessionId),
        api.fetchSessionSubagent(target.sessionId, activeSubagentId),
      ])
      setSessionSubagents(subagents)
      if (detail) {
        setActiveSubagentDetail(detail)
      }
      if (action === 'resume' && guidance.length > 0) {
        setSubagentGuidance('')
      }
    } catch (error) {
      setSubagentDrawerError(
        error instanceof Error
          ? error.message
          : action === 'cancel'
            ? 'Failed to cancel subagent.'
            : 'Failed to resume subagent.'
      )
    } finally {
      setSubagentBusyAction(null)
    }
  }, [
    activeSubagentDetail,
    activeSubagentId,
    allVisibleSubagents,
    subagentBusyAction,
    subagentGuidance,
  ])

  const refreshCurrentGoalBinding = React.useCallback(async (goalId: string) => {
    if (!currentSessionId) {
      return null
    }

    const refreshedGoal = await api.fetchGoal(goalId)
    const refreshedGoalSummary = buildGoalSummaryFromGoal(
      refreshedGoal,
      currentSessionGoalState.last_goal_summary
    )
    const refreshedRunId = getGoalAttemptRunId(refreshedGoal)

    await syncWorkflowStateForGoal({
      sessionId: currentSessionId,
      baseWorkflow: workflowState,
      executionMode: refreshedGoal.execution_mode,
      interactionMode: refreshedGoal.interaction_mode,
      executionTopology: refreshedGoal.execution_topology,
      goalStatus: refreshedGoal.status,
      runId: refreshedRunId,
    })
    await persistSessionGoalState(currentSessionId, {
      active_goal_id: isGoalTerminalStatus(refreshedGoal.status) ? null : refreshedGoal.goal_id,
      active_goal_status: refreshedGoal.status,
      execution_mode: refreshedGoal.execution_mode,
      interaction_mode: refreshedGoal.interaction_mode,
      execution_topology: refreshedGoal.execution_topology,
      bound_run_id: refreshedGoalSummary.bound_run_id,
      protocol_selection: refreshedGoalSummary.protocol_selection,
      selection_rationale: refreshedGoalSummary.selection_rationale,
      default_route: isGoalTerminalStatus(refreshedGoal.status)
        ? 'chat'
        : refreshedGoal.execution_mode === 'workflow'
          ? 'workflow'
          : 'goal',
      last_goal_summary: refreshedGoalSummary,
      pending_proposal: null,
    })

    return refreshedGoal
  }, [
    buildGoalSummaryFromGoal,
    currentSessionGoalState.last_goal_summary,
    currentSessionId,
    persistSessionGoalState,
    syncWorkflowStateForGoal,
    workflowState,
  ])

  const loadGoalDrawerContext = React.useCallback(async (goalId: string) => {
    const copySource = resolveGoalSurfaceCopySource(
      goalId,
      currentSessionGoalState.last_goal_summary?.objective ?? null
    )
    const drawerCopy = buildGoalCardChromeCopy(copySource)
    setGoalDrawerHealthLoading(true)
    setGoalDrawerHealthError(null)
    setGoalDrawerApprovalError(null)
    setGoalDrawerApprovalsLoading(false)

    try {
      const health = await api.fetchGoalHealth(goalId)
      setGoalDrawerHealth(health)

      const approvalIds = getStringArray(health.approval_state?.approval_ids)
      if (approvalIds.length === 0) {
        setGoalDrawerApprovals([])
        setGoalDrawerApprovalsLoading(false)
        return health
      }

      setGoalDrawerApprovalsLoading(true)
      try {
        const approvals = await api.fetchApprovals()
        const approvalIdSet = new Set(approvalIds)
        setGoalDrawerApprovals(
          approvals.filter((approval) => approvalIdSet.has(approval.approval_id))
        )
      } catch (error) {
        const detail = buildGoalUiErrorMessage(
          copySource,
          error instanceof Error ? error.message : null,
          drawerCopy.pendingApprovalsLoadFailedLabel
        )
        setGoalDrawerApprovalError(detail)
        setGoalDrawerApprovals([])
      } finally {
        setGoalDrawerApprovalsLoading(false)
      }

      return health
    } catch (error) {
      const detail = buildGoalUiErrorMessage(
        copySource,
        error instanceof Error ? error.message : null,
        drawerCopy.goalStatusLoadFailedLabel
      )
      setGoalDrawerHealthError(detail)
      setGoalDrawerHealth(null)
      setGoalDrawerApprovals([])
      setGoalDrawerApprovalsLoading(false)
      return null
    } finally {
      setGoalDrawerHealthLoading(false)
    }
  }, [currentSessionGoalState.last_goal_summary, resolveGoalSurfaceCopySource])

  const handleGoalDrawerRefresh = React.useCallback(async () => {
    const goalId =
      currentSessionGoalState.active_goal_id ??
      currentSessionGoalState.last_goal_summary?.goal_id ??
      null
    if (!goalId) {
      return
    }
    setGoalDrawerBusyAction('status')
    try {
      await Promise.all([
        refreshCurrentGoalBinding(goalId),
        loadGoalDrawerContext(goalId),
      ])
    } catch (error) {
      const copySource = resolveGoalSurfaceCopySource(
        goalId,
        currentSessionGoalState.last_goal_summary?.objective ?? null
      )
      const detail = buildGoalUiErrorMessage(
        copySource,
        error instanceof Error ? error.message : null,
        buildGoalCardChromeCopy(copySource).goalStatusRefreshFailedLabel
      )
      setGoalDrawerHealthError(detail)
    } finally {
      setGoalDrawerBusyAction(null)
    }
  }, [currentSessionGoalState.active_goal_id, currentSessionGoalState.last_goal_summary, loadGoalDrawerContext, refreshCurrentGoalBinding, resolveGoalSurfaceCopySource])

  const handleGoalDrawerResolveApproval = React.useCallback(async (
    approvalId: string,
    decision: 'approve_once' | 'reject'
  ) => {
    const goalId =
      currentSessionGoalState.active_goal_id ??
      currentSessionGoalState.last_goal_summary?.goal_id ??
      null
    setGoalDrawerResolvingApprovalKey(`${approvalId}:${decision}`)
    setGoalDrawerApprovalError(null)
    try {
      await api.resolveApproval(approvalId, { decision })
      if (goalId) {
        await Promise.all([
          refreshCurrentGoalBinding(goalId),
          loadGoalDrawerContext(goalId),
        ])
      }
    } catch (error) {
      const copySource = resolveGoalSurfaceCopySource(
        goalId,
        currentSessionGoalState.last_goal_summary?.objective ?? null
      )
      const detail = buildGoalUiErrorMessage(
        copySource,
        error instanceof Error ? error.message : null,
        buildGoalCardChromeCopy(copySource).approvalResolveFailedLabel
      )
      setGoalDrawerApprovalError(detail)
    } finally {
      setGoalDrawerResolvingApprovalKey(null)
    }
  }, [currentSessionGoalState.active_goal_id, currentSessionGoalState.last_goal_summary, loadGoalDrawerContext, refreshCurrentGoalBinding, resolveGoalSurfaceCopySource])

  const runGoalDrawerCommand = React.useCallback(async (
    action: 'status' | 'pause' | 'resume' | 'stop'
  ) => {
    const sessionId = currentSessionId
    if (!sessionId) {
      return
    }
    setGoalDrawerBusyAction(action)
    try {
      await handleSend(`/goal ${action}`, { forceSessionId: sessionId })
    } finally {
      setGoalDrawerBusyAction(null)
    }
  }, [currentSessionId, handleSend])

  const handleWorkflowPanelToggle = React.useCallback(() => {
    const nextOpen = !workflowPanelOpen
    closeRightPanels('workflow')
    setWorkflowPanelOpen(nextOpen)
  }, [closeRightPanels, workflowPanelOpen])

  const handleInferencePanelToggle = React.useCallback(() => {
    closeRightPanels('inference')
    if (window.innerWidth < 768) {
      setPanelOpen(false)
      setMobileInferenceOpen((open) => !open)
      return
    }
    setMobileInferenceOpen(false)
    setPanelOpen(!panelOpen)
  }, [closeRightPanels, panelOpen, setPanelOpen])

  const handleTaskPanelToggle = React.useCallback(() => {
    const nextOpen = !taskPanelOpen
    closeRightPanels('tasks')
    if (nextOpen) {
      setTaskPanelMode('default')
      setTaskPanelFocusedTaskId(null)
      setTaskPanelFocusedApprovalIds(null)
    } else {
      setTaskPanelMode('default')
      setTaskPanelFocusedTaskId(null)
      setTaskPanelFocusedApprovalIds(null)
    }
    setTaskPanelOpen(nextOpen)
  }, [closeRightPanels, taskPanelOpen])

  const handleSessionAutonomyModeChange = React.useCallback(async (
    value: api.SessionSecurityOverride['autonomy_mode']
  ) => {
    const sessionId = currentSessionId ?? createDraftSession(activeProjectId)
    await persistSessionSecurityOverride(sessionId, value)
  }, [activeProjectId, createDraftSession, currentSessionId, persistSessionSecurityOverride])

  const handleOpenTaskPanel = React.useCallback(() => {
    closeRightPanels('tasks')
    setTaskPanelMode('default')
    setTaskPanelFocusedTaskId(null)
    setTaskPanelFocusedApprovalIds(null)
    setTaskPanelOpen(true)
  }, [closeRightPanels])

  const handleOpenRuntimeTask = React.useCallback((taskId: string) => {
    closeRightPanels('tasks')
    setTaskPanelMode('subagent')
    setTaskPanelFocusedTaskId(taskId)
    setTaskPanelFocusedApprovalIds(null)
    setTaskPanelOpen(true)
    void useTaskStore.getState().selectTask(taskId)
  }, [closeRightPanels])

  const handleOpenSubagentApprovals = React.useCallback((approvalIds: string[]) => {
    const uniqueApprovalIds = Array.from(new Set(approvalIds.map((item) => item.trim()).filter(Boolean)))
    if (uniqueApprovalIds.length === 0) {
      return
    }
    closeRightPanels('tasks')
    setTaskPanelMode('default')
    setTaskPanelFocusedTaskId(null)
    setTaskPanelFocusedApprovalIds(uniqueApprovalIds)
    setTaskPanelOpen(true)
    void useTaskStore.getState().load()
  }, [closeRightPanels])

  const handleTaskPanelOpenChange = React.useCallback((open: boolean) => {
    setTaskPanelOpen(open)
    if (!open) {
      setTaskPanelMode('default')
      setTaskPanelFocusedTaskId(null)
      setTaskPanelFocusedApprovalIds(null)
    }
  }, [])

  const handleOpenWorkflowPanel = React.useCallback(() => {
    closeRightPanels('workflow')
    setWorkflowPanelOpen(true)
  }, [closeRightPanels])

  const handleWorkspacePanelToggle = React.useCallback(() => {
    if (window.innerWidth < 1024) {
      closeRightPanels()
      setWorkspacePanelOpen(false)
      setWorkspaceMobileOpen((open) => !open)
      return
    }
    setWorkspaceMobileOpen(false)
    setWorkspacePanelOpen(!workspacePanelOpen)
  }, [closeRightPanels, setWorkspacePanelOpen, workspacePanelOpen])

  const headerGoal = React.useMemo<GoalHeaderChipView | null>(() => {
    if (currentSessionGoalState.pending_proposal) {
      return null
    }

    const summary = currentSessionGoalState.last_goal_summary
    if (!summary) {
      return null
    }

    const status = (currentSessionGoalState.active_goal_status ?? summary.status ?? '').trim()
    if (!status) {
      return null
    }

    const isCompleted = isGoalCompletedStatus(status)
    const isBlocked = isGoalBlockedStatus(status)
    const hasTerminalIssue =
      isGoalTerminalStatus(status) &&
      !isCompleted

    const isActiveGoalBound =
      currentSessionGoalState.active_goal_id !== null &&
      !isGoalTerminalStatus(status)

    if (!isCompleted && !isBlocked && !hasTerminalIssue && !isActiveGoalBound) {
      return null
    }

    const goalId = currentSessionGoalState.active_goal_id ?? summary.goal_id ?? null
    const copySource = resolveGoalSurfaceCopySource(goalId, summary.objective)

    return {
      title: summary.objective,
      goalId,
      status,
      executionMode: summary.execution_mode,
      copySource,
      protocolId: summary.protocol_id,
      modelCount: summary.models.length,
      runtimeMode: summary.runtime_mode,
      pendingApprovalCount: isCompleted ? pendingApprovalCount : goalSurfacePendingApprovalCount,
      displayState:
        isCompleted ? 'completed' : isBlocked ? 'blocked' : hasTerminalIssue ? 'failed' : 'active',
    }
  }, [currentSessionGoalState, goalSurfacePendingApprovalCount, pendingApprovalCount, resolveGoalSurfaceCopySource])

  const goalSurfaceCopySource =
    headerGoal?.copySource ??
    resolveGoalSurfaceCopySource(
      currentSessionGoalState.active_goal_id ?? currentSessionGoalState.last_goal_summary?.goal_id ?? null,
      currentSessionGoalState.last_goal_summary?.objective ?? null
    )

  const goalDrawerBlocker = React.useMemo<GoalDrawerBlockerView | null>(() => {
    if (!goalDrawerHealth) {
      return null
    }

    return {
      summary:
        getString(goalDrawerHealth.recommended_next_action?.summary) ??
        goalDrawerHealth.latest_error ??
        null,
      recommendedAction: getString(goalDrawerHealth.recommended_next_action?.action),
      latestError: goalDrawerHealth.latest_error,
      approvalCount:
        typeof goalDrawerHealth.approval_state?.pending_count === 'number' &&
        Number.isFinite(goalDrawerHealth.approval_state.pending_count)
          ? goalDrawerHealth.approval_state.pending_count
          : getStringArray(goalDrawerHealth.approval_state?.approval_ids).length,
      approvalIds: getStringArray(goalDrawerHealth.approval_state?.approval_ids),
      approvalToolNames: getStringArray(goalDrawerHealth.approval_state?.tool_names),
      blockedTools: goalDrawerHealth.operator_controls.blocked_tools,
      blockedDomains: goalDrawerHealth.operator_controls.blocked_domains,
      blockNetworkUsage: goalDrawerHealth.operator_controls.block_network_usage,
    }
  }, [goalDrawerHealth])

  const goalPendingApprovalIds = React.useMemo(
    () =>
      Array.from(
        new Set([
          ...pendingGoalApprovalIds,
          ...(goalDrawerBlocker?.approvalIds ?? []),
        ])
      ),
    [goalDrawerBlocker, pendingGoalApprovalIds]
  )

  const handleOpenGoalApprovals = React.useCallback(() => {
    if (goalPendingApprovalIds.length > 0) {
      handleOpenSubagentApprovals(goalPendingApprovalIds)
      return
    }
    handleOpenTaskPanel()
  }, [goalPendingApprovalIds, handleOpenSubagentApprovals, handleOpenTaskPanel])

  React.useEffect(() => {
    if (!headerGoal) {
      setGoalDrawerOpen(false)
      setGoalDrawerHealth(null)
      setGoalDrawerHealthLoading(false)
      setGoalDrawerHealthError(null)
      setGoalDrawerApprovals([])
      setGoalDrawerApprovalsLoading(false)
      setGoalDrawerApprovalError(null)
      setGoalDrawerResolvingApprovalKey(null)
    }
  }, [headerGoal])

  React.useEffect(() => {
    if (!subagentDrawerOpen) {
      setSubagentGuidance('')
    }
  }, [subagentDrawerOpen])

  React.useEffect(() => {
    const goalId = headerGoal?.goalId
    if (!goalId) {
      return
    }
    const needsGoalSurfaceContext =
      goalDrawerOpen ||
      headerGoal.displayState === 'blocked' ||
      headerGoal.displayState === 'failed' ||
      headerGoal.pendingApprovalCount > 0
    if (!needsGoalSurfaceContext) {
      return
    }

    void loadGoalDrawerContext(goalId)
  }, [goalDrawerOpen, headerGoal, loadGoalDrawerContext])

  const goalSurfaceCallout = React.useMemo(
    () =>
      headerGoal
        ? buildGoalFocusCallout({
            userMessage: goalSurfaceCopySource,
            pendingApprovalCount: goalSurfacePendingApprovalCount,
            blocker: goalDrawerBlocker,
            errorMessage:
              goalDrawerHealthError ??
              goalDrawerApprovalError ??
              (headerGoal.executionMode === 'workflow' ? workflowError : null),
            goalDisplayState: headerGoal.displayState,
          })
        : null,
    [
      goalDrawerApprovalError,
      goalDrawerBlocker,
      goalDrawerHealthError,
      goalSurfacePendingApprovalCount,
      goalSurfaceCopySource,
      headerGoal,
      workflowError,
    ]
  )

  const handleActiveGoalDirectTurnDecision = React.useCallback(
    async ({
      sessionId,
      requestText,
      attachments,
      activeGoalId,
      baseWorkflow,
      latestGoalSummary,
      selectedSkillIds,
      sessionScope,
      systemPromptOverride,
    }: ActiveGoalDirectTurnDecisionInput): Promise<boolean> => {
      let decision: api.ActiveGoalTurnDecision
      try {
        decision = await api.fetchGoalTurnDecision(activeGoalId, { message: requestText })
      } catch {
        await submitDirectChatTurn({
          targetSessionId: sessionId,
          requestText,
          attachments,
          selectedSkillIds,
          toolMode: undefined,
          normalizedWorkflow: baseWorkflow,
          sessionScope,
          systemPromptOverride,
        })
        return true
      }
      const decisionMetadata = {
        active_goal_turn_decision: {
          lane: decision.lane,
          kind: decision.kind,
          confidence: decision.confidence,
          selection_source: decision.selection_source,
          selection_reason: decision.selection_reason,
          requires_confirmation: decision.requires_confirmation,
          goal_status: decision.goal_status ?? null,
          linked_run_status: decision.linked_run_status ?? null,
          recommended_action: decision.recommended_action ?? null,
        },
      }

      if (
        decision.kind === 'answer_question' ||
        decision.kind === 'explain_goal_state' ||
        decision.kind === 'exit_to_chat' ||
        (decision.kind === 'clarify' && decision.requires_confirmation)
      ) {
        await submitDirectChatTurn({
          targetSessionId: sessionId,
          requestText,
          attachments,
          selectedSkillIds,
          toolMode: undefined,
          normalizedWorkflow: baseWorkflow,
          sessionScope,
          systemPromptOverride:
            systemPromptOverride ??
            `${effectiveInference.systemPrompt ?? ''}\n\nActive goal decision metadata: ${JSON.stringify(decisionMetadata)}`
              .trim(),
        })
        return true
      }

      if (decision.kind === 'steer') {
        try {
          const goalHealth = await api.fetchGoalHealth(activeGoalId)
          const continuation = resolveGoalContinuationDecision(goalHealth)
          const goalObjective = latestGoalSummary?.objective ?? goalHealth.goal_id ?? requestText
          const continuationRunId = continuation.runId
          const continuationResumeStrategy = continuation.resumeStrategy ?? 'continue_from_checkpoint'
          let refreshedGoal: api.GoalSummary | null = null
          if (continuation.action === 'refresh_then_forward') {
            refreshedGoal = await api.refreshGoal(activeGoalId, { strategy: continuationResumeStrategy })
            const refreshedRunId =
              getGoalAttemptRunId(refreshedGoal) ??
              getGoalAttemptRunId(await api.fetchGoal(activeGoalId))
            if (refreshedRunId) {
              await api.appendAgentRunGuidance(refreshedRunId, {
                guidance: requestText,
                author: 'operator',
                metadata: decisionMetadata,
              })
            } else {
              await api.resumeGoal(activeGoalId, {
                strategy: continuationResumeStrategy,
                guidanceMessage: requestText,
              })
            }
          } else if (
            continuation.action === 'resume_then_forward' ||
            continuation.action === 'restart_attempt_with_context' ||
            continuation.action === 'no_live_attempt_recoverable'
          ) {
            await api.resumeGoal(activeGoalId, {
              strategy: continuationResumeStrategy,
              guidanceMessage: requestText,
            })
          } else if (continuationRunId) {
            await api.appendAgentRunGuidance(continuationRunId, {
              guidance: requestText,
              author: 'operator',
              metadata: decisionMetadata,
            })
          } else {
            await api.resumeGoal(activeGoalId, {
              strategy: continuationResumeStrategy,
              guidanceMessage: requestText,
            })
          }

          refreshedGoal = await api.fetchGoal(activeGoalId)
          const refreshedGoalSummary = buildGoalSummaryFromGoal(refreshedGoal, latestGoalSummary)
          const refreshedRunId = getGoalAttemptRunId(refreshedGoal)
          await syncWorkflowStateForGoal({
            sessionId,
            baseWorkflow,
            executionMode: refreshedGoal.execution_mode,
            interactionMode: refreshedGoal.interaction_mode,
            executionTopology: refreshedGoal.execution_topology,
            goalStatus: refreshedGoal.status,
            runId: refreshedRunId,
          })
          await persistSessionGoalState(sessionId, {
            active_goal_id: isGoalTerminalStatus(refreshedGoal.status) ? null : refreshedGoal.goal_id,
            active_goal_status: refreshedGoal.status,
            execution_mode: refreshedGoal.execution_mode,
            interaction_mode: refreshedGoal.interaction_mode,
            execution_topology: refreshedGoal.execution_topology,
            bound_run_id: refreshedGoalSummary.bound_run_id,
            protocol_selection: refreshedGoalSummary.protocol_selection,
            selection_rationale: refreshedGoalSummary.selection_rationale,
            default_route:
              isGoalTerminalStatus(refreshedGoal.status)
                ? 'chat'
                : refreshedGoal.execution_mode === 'workflow'
                  ? 'workflow'
                  : 'goal',
            last_goal_summary: refreshedGoalSummary,
            pending_proposal: null,
          })

          const followUpKind =
            continuation.action === 'refresh_then_forward'
              ? 'refreshed_forwarded'
              : continuation.action === 'resume_then_forward'
                ? 'resumed_forwarded'
                : continuation.action === 'restart_attempt_with_context' ||
                    continuation.action === 'no_live_attempt_recoverable'
                  ? 'restarted_forwarded'
                  : 'forwarded'
          const assistantCopy = await api.generateGoalFollowUpAssistantCopy({
            message: requestText,
            kind: followUpKind,
            goalObjective,
            goalStatus: refreshedGoal.status,
            linkedRunStatus: decision.linked_run_status ?? null,
            continuationAction: continuation.action,
            continuationSummary: continuation.summary,
            approvalCount: continuation.approvalIds.length,
            toolNames: continuation.toolNames,
            recommendedAction: continuation.recommendedAction,
            latestError: refreshedGoal.latest_error ?? null,
          })

          await persistGoalConversation({
            sessionId,
            attachments,
            userContent: requestText,
            assistantContent:
              assistantCopy.explanation.trim().length > 0
                ? assistantCopy.explanation
                : buildGoalFollowUpMessage(requestText, followUpKind),
            userMetadata: decisionMetadata,
            assistantMetadata: decisionMetadata,
            goalCard: goalCardFromSummary(refreshedGoalSummary, 'started', {
              goalId: refreshedGoal.goal_id,
              status: refreshedGoal.status,
              copySource: requestText,
            }),
          })
          return true
        } catch (error) {
          const detail = error instanceof Error ? error.message : null
          updateSessionMessages(sessionId, (prev) => [
            ...prev,
            {
              id: `error-${Date.now()}`,
              type: 'error',
              eventType: 'error',
              content: t('chat.requestFailed'),
              errorCode: detail ?? 'CHAT_REQUEST_FAILED',
              timestamp: new Date(),
            },
          ])
          return true
        }
      }

      if (decision.kind === 'replan') {
        try {
          const goalHealth = await api.fetchGoalHealth(activeGoalId)
          const continuation = resolveGoalContinuationDecision(goalHealth)
          const goalObjective = latestGoalSummary?.objective ?? goalHealth.goal_id ?? requestText
          const continuationResumeStrategy = continuation.resumeStrategy ?? 'continue_from_checkpoint'
          let refreshedGoal: api.GoalSummary | null = null
          if (continuation.action === 'refresh_then_forward') {
            refreshedGoal = await api.refreshGoal(activeGoalId, { strategy: continuationResumeStrategy })
            const refreshedRunId =
              getGoalAttemptRunId(refreshedGoal) ??
              getGoalAttemptRunId(await api.fetchGoal(activeGoalId))
            if (refreshedRunId) {
              await api.appendAgentRunGuidance(refreshedRunId, {
                guidance: requestText,
                author: 'operator',
                metadata: decisionMetadata,
              })
            } else {
              await api.resumeGoal(activeGoalId, {
                strategy: continuationResumeStrategy,
                guidanceMessage: requestText,
              })
            }
          } else {
            await api.resumeGoal(activeGoalId, {
              strategy: 'restart_attempt',
              guidanceMessage: requestText,
            })
          }

          refreshedGoal = await api.fetchGoal(activeGoalId)
          const refreshedGoalSummary = buildGoalSummaryFromGoal(refreshedGoal, latestGoalSummary)
          const refreshedRunId = getGoalAttemptRunId(refreshedGoal)
          await syncWorkflowStateForGoal({
            sessionId,
            baseWorkflow,
            executionMode: refreshedGoal.execution_mode,
            interactionMode: refreshedGoal.interaction_mode,
            executionTopology: refreshedGoal.execution_topology,
            goalStatus: refreshedGoal.status,
            runId: refreshedRunId,
          })
          await persistSessionGoalState(sessionId, {
            active_goal_id: isGoalTerminalStatus(refreshedGoal.status) ? null : refreshedGoal.goal_id,
            active_goal_status: refreshedGoal.status,
            execution_mode: refreshedGoal.execution_mode,
            interaction_mode: refreshedGoal.interaction_mode,
            execution_topology: refreshedGoal.execution_topology,
            bound_run_id: refreshedGoalSummary.bound_run_id,
            protocol_selection: refreshedGoalSummary.protocol_selection,
            selection_rationale: refreshedGoalSummary.selection_rationale,
            default_route:
              isGoalTerminalStatus(refreshedGoal.status)
                ? 'chat'
                : refreshedGoal.execution_mode === 'workflow'
                  ? 'workflow'
                  : 'goal',
            last_goal_summary: refreshedGoalSummary,
            pending_proposal: null,
          })
          const followUpKind =
            continuation.action === 'refresh_then_forward'
              ? 'refreshed_forwarded'
              : 'restarted_forwarded'
          const assistantCopy = await api.generateGoalFollowUpAssistantCopy({
            message: requestText,
            kind: followUpKind,
            goalObjective,
            goalStatus: refreshedGoal.status,
            linkedRunStatus: decision.linked_run_status ?? null,
            continuationAction: continuation.action,
            continuationSummary: continuation.summary,
            approvalCount: continuation.approvalIds.length,
            toolNames: continuation.toolNames,
            recommendedAction: continuation.recommendedAction,
            latestError: refreshedGoal.latest_error ?? null,
          })
          await persistGoalConversation({
            sessionId,
            attachments,
            userContent: requestText,
            assistantContent:
              assistantCopy.explanation.trim().length > 0
                ? assistantCopy.explanation
                : buildGoalFollowUpMessage(requestText, followUpKind),
            userMetadata: decisionMetadata,
            assistantMetadata: decisionMetadata,
            goalCard: goalCardFromSummary(refreshedGoalSummary, 'started', {
              goalId: refreshedGoal.goal_id,
              status: refreshedGoal.status,
              copySource: requestText,
            }),
          })
          return true
        } catch (error) {
          const detail = error instanceof Error ? error.message : null
          updateSessionMessages(sessionId, (prev) => [
            ...prev,
            {
              id: `error-${Date.now()}`,
              type: 'error',
              eventType: 'error',
              content: t('chat.requestFailed'),
              errorCode: detail ?? 'CHAT_REQUEST_FAILED',
              timestamp: new Date(),
            },
          ])
          return true
        }
      }

      return false
    },
    [
      buildGoalSummaryFromGoal,
      effectiveInference.systemPrompt,
      persistGoalConversation,
      persistSessionGoalState,
      submitDirectChatTurn,
      syncWorkflowStateForGoal,
    ]
  )

  const footerRuntimeNotice =
    !headerGoal && pendingApprovalCount > 0
      ? {
          tone: 'warning' as const,
          message: buildGoalPendingApprovalNotice(goalSurfaceCopySource, pendingApprovalCount),
          actionLabel: buildGoalReviewApprovalsLabel(goalSurfaceCopySource),
          onAction: handleOpenTaskPanel,
        }
      : !headerGoal && workflowError
        ? {
            tone: 'error' as const,
            message: workflowError,
            actionLabel: buildGoalOpenWorkflowLabel(goalSurfaceCopySource),
            onAction: handleOpenWorkflowPanel,
          }
        : null

  const showEmptyState = isConversationEffectivelyEmpty(displayMessages)
  const showExecutionTranscript =
    goalSurfaceTimelineEvents.length > 0 ||
    goalSurfaceSubagents.length > 0 ||
    Boolean(executionTimelineError)
  const goalConversationBodyLayout = resolveGoalConversationBodyLayout({
    showEmptyState,
    showExecutionTranscript,
  })

  return (
    <div className="flex h-full flex-col">
      <header className="border-b border-border bg-canvas/95 backdrop-blur">
        <div className="mx-auto flex h-14 w-full max-w-5xl items-center justify-between gap-4 px-4">
          <h1 className="min-w-0 truncate text-sm font-semibold text-foreground">
            {displaySessionTitle(currentSession?.title, t('chat.newChat'))}
          </h1>
          <div className="flex items-center gap-1">
            {headerGoal ? (
              <div className="mr-1 flex">
                <GoalHeaderChip
                  goal={headerGoal}
                  open={goalDrawerOpen}
                  onClick={handleGoalDrawerToggle}
                />
              </div>
            ) : null}
            <div className="mr-2 hidden max-w-[220px] items-center gap-1.5 text-xs text-muted-foreground sm:flex">
              {isStreaming ? (
                <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin" />
              ) : (
                <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-success" />
              )}
              <span className="truncate">{headerModelLabel}</span>
            </div>
            <div className="relative shrink-0">
              <Button
                variant={workflowEnabled || workflowPanelOpen ? 'secondary' : 'ghost'}
                size="sm"
                title={workflowShortcutTitle}
                aria-label={workflowShortcutTitle}
                onClick={handleWorkflowPanelToggle}
                className="max-sm:w-8 max-sm:px-0"
              >
                <Workflow className="h-4 w-4" />
                <span className="hidden sm:inline">{workflowShortcutLabel}</span>
              </Button>
              {workflowError ? (
                <HeaderRuntimeIndicator tone="error" pulse />
              ) : workflowBoundRunId ? (
                <HeaderRuntimeIndicator tone="warning" />
              ) : workflowEnabled ? (
                <HeaderRuntimeIndicator tone="info" />
              ) : null}
            </div>
            <Button
              variant="ghost"
              size="icon-sm"
              title={t('chat.moreOptions')}
              onClick={() => setExportOpen(true)}
            >
              <MoreHorizontal className="h-4 w-4" />
            </Button>
            <Button
              variant={workspacePanelOpen || workspaceMobileOpen ? 'secondary' : 'ghost'}
              size="icon-sm"
              title="Workspace"
              onClick={handleWorkspacePanelToggle}
            >
              <FolderTree className="h-4 w-4" />
            </Button>
            <Button
              variant={panelOpen || mobileInferenceOpen ? 'secondary' : 'ghost'}
              size="icon-sm"
              title="Inference"
              onClick={handleInferencePanelToggle}
            >
              <SlidersHorizontal className="h-4 w-4" />
            </Button>
            <div className="relative shrink-0">
              <Button
                variant={taskPanelOpen ? 'secondary' : 'ghost'}
                size="icon-sm"
                title={taskShortcutTitle}
                aria-label={taskShortcutTitle}
                onClick={handleTaskPanelToggle}
              >
                <ListTodo className="h-4 w-4" />
              </Button>
              {pendingApprovalCount > 0 ? (
                <HeaderRuntimeIndicator tone="error" count={pendingApprovalCount} pulse />
              ) : failedTaskCount > 0 ? (
                <HeaderRuntimeIndicator tone="error" count={failedTaskCount} />
              ) : activeTaskCount > 0 ? (
                <HeaderRuntimeIndicator tone="warning" count={activeTaskCount} />
              ) : null}
            </div>
            <Button
              variant="ghost"
              size="icon-sm"
              title={t('chat.settingsShortcut')}
              onClick={() => router.push('/settings')}
            >
              <Settings className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </header>

      <div className="relative flex flex-1 overflow-hidden">
        <FloatingPanelShell
          open={workspacePanelOpen}
          onOpenChange={setWorkspacePanelOpen}
          desktopSide="left"
          desktopWidthClass="w-[min(40vw,44rem)] min-w-[24rem] max-w-[48rem]"
          desktopBreakpoint="lg"
          mobileSide="left"
          mobileClassName="w-[92vw] max-w-[92vw] p-0 sm:max-w-[92vw]"
          renderMobile={false}
        >
          <WorkspacePanel
            onAttachAttachment={handleQueueWorkspaceAttachment}
            onClose={() => setWorkspacePanelOpen(false)}
          />
        </FloatingPanelShell>
        {headerGoal ? (
          <FloatingPanelShell
            open={goalDrawerOpen}
            onOpenChange={setGoalDrawerOpen}
            desktopSide="right"
            desktopWidthClass="w-[min(34vw,26rem)] min-w-[22rem] max-w-[30rem]"
            desktopBreakpoint="lg"
            mobileSide="right"
            mobileClassName="w-[92vw] max-w-[92vw] p-0 sm:max-w-[28rem]"
          >
            <GoalDrawerContent
              goal={headerGoal}
              busyAction={goalDrawerBusyAction}
              blocker={goalDrawerBlocker}
              approvals={goalDrawerApprovals}
              approvalLoading={goalDrawerHealthLoading || goalDrawerApprovalsLoading}
              approvalError={goalDrawerHealthError ?? goalDrawerApprovalError}
              resolvingApprovalKey={goalDrawerResolvingApprovalKey}
              onRefresh={() => {
                void handleGoalDrawerRefresh()
              }}
              onPause={
                headerGoal.goalId
                  ? () => {
                      void runGoalDrawerCommand('pause')
                    }
                  : undefined
              }
              onResume={
                headerGoal.goalId
                  ? () => {
                      void runGoalDrawerCommand('resume')
                    }
                  : undefined
              }
              onStop={
                headerGoal.goalId
                  ? () => {
                      void runGoalDrawerCommand('stop')
                    }
                  : undefined
              }
              onResolveApproval={(approvalId, decision) => {
                void handleGoalDrawerResolveApproval(approvalId, decision)
              }}
              onOpenConsole={() => router.push('/goals')}
              onClose={() => setGoalDrawerOpen(false)}
            />
          </FloatingPanelShell>
        ) : null}
        <SubagentDrawer
          open={subagentDrawerOpen}
          onOpenChange={setSubagentDrawerOpen}
          subagent={activeSubagentDetail}
          loading={subagentDrawerLoading}
          error={subagentDrawerError}
          guidanceValue={subagentGuidance}
          onGuidanceChange={setSubagentGuidance}
          onSendGuidance={handleSendSubagentGuidance}
          onCancel={
            activeSubagentDetail?.sessionId
              ? () => {
                  void runSessionSubagentAction('cancel')
                }
              : undefined
          }
          onResume={
            activeSubagentDetail?.sessionId
              ? () => {
                  void runSessionSubagentAction('resume')
                }
              : undefined
          }
          onOpenApprovals={handleOpenSubagentApprovals}
          sendingGuidance={subagentGuidanceSending}
          canceling={subagentBusyAction === 'cancel'}
          resuming={subagentBusyAction === 'resume'}
        />
        <div
          className={cn(
            'min-w-0 flex-1 transition-[padding] duration-300 ease-out-smooth',
            workspacePanelOpen ? 'lg:pl-[calc(min(40vw,44rem)+0.75rem)]' : ''
          )}
        >
          <ScrollToBottom visible={showScrollToBottom} onClick={scrollToBottom} />
          <div ref={scrollRef} className="h-full overflow-y-auto">
            <div className="mx-auto flex w-full max-w-4xl flex-col px-4 py-8 sm:px-6">
              <div className="space-y-6">
                {headerGoal ? (
                  <GoalFocusPanel
                    goal={headerGoal}
                    blocker={goalDrawerBlocker}
                    callout={goalSurfaceCallout}
                    timelineEvents={goalSurfaceTimelineEvents}
                    subagents={goalSurfaceSubagents}
                    timelineError={executionTimelineError}
                    activeSubagentId={activeSubagentId}
                    expandedSubagentIds={expandedSubagentIds}
                    onExpandedChange={(subagentId, expanded) => {
                      setExpandedSubagentIds((current) => {
                        const next = new Set(current)
                        if (expanded) {
                          next.add(subagentId)
                        } else {
                          next.delete(subagentId)
                        }
                        return next
                      })
                    }}
                    onOpenSubagent={handleOpenSubagent}
                    onReviewApprovals={handleOpenGoalApprovals}
                    onOpenDetails={handleGoalDrawerToggle}
                    onOpenConsole={() => router.push('/goals')}
                  />
                ) : null}
                {!headerGoal && layoutShowsExecutionHighlights(goalConversationBodyLayout) ? (
                  <section className="space-y-4">
                    {executionTimelineError ? (
                      <div className="rounded-lg border border-warning/30 bg-warning/10 px-3 py-2 text-xs text-warning-foreground">
                        {executionTimelineError}
                      </div>
                    ) : null}
                    {goalSurfaceSubagents.length > 0 ? (
                      <div className="grid gap-3 sm:grid-cols-2">
                        {goalSurfaceSubagents.map((subagent) => (
                          <SubagentTimelineCard
                            key={subagent.subagentId}
                            subagent={subagent}
                            active={subagent.subagentId === activeSubagentId}
                            expanded={expandedSubagentIds.has(subagent.subagentId)}
                            onExpandedChange={(subagentId, expanded) => {
                              setExpandedSubagentIds((current) => {
                                const next = new Set(current)
                                if (expanded) {
                                  next.add(subagentId)
                                } else {
                                  next.delete(subagentId)
                                }
                                return next
                              })
                            }}
                            onOpen={handleOpenSubagent}
                            onOpenThread={handleOpenSubagent}
                            recentEvents={subagentTimelineEventsById.get(subagent.subagentId) ?? []}
                          />
                        ))}
                      </div>
                    ) : null}
                    <ExecutionTimeline
                      events={goalSurfaceTimelineEvents}
                      subagents={allVisibleSubagents}
                      activeSubagentId={activeSubagentId}
                      onOpenSubagent={handleOpenSubagent}
                    />
                  </section>
                ) : null}
                {layoutShowsEmptyState(goalConversationBodyLayout) ? (
                  <EmptyState
                    onPrompt={handleStarterPrompt}
                    onVoice={() => void handleVoiceEntry()}
                    onSettings={() => router.push('/settings')}
                  />
                ) : (
                  displayMessages.map((message) => (
                    <ChatMessage
                      key={message.id}
                      message={
                        message.type === 'assistant' && !effectiveInference.showTokenStats
                          ? { ...message, tokenStats: undefined }
                          : message
                      }
                      sessionId={currentSessionId}
                      projectId={effectiveProjectId}
                      onRegenerate={
                        message.type === 'assistant' &&
                        !message.workflowCard &&
                        !message.workflowCompletion &&
                        !message.subagentTaskCard
                          ? handleRegenerate
                          : undefined
                      }
                      onEditAndResend={message.type === 'user' ? (message) => handleEditAndResend(message) : undefined}
                      onUndoFileChange={handleUndoFileChange}
                      onOpenTask={handleOpenRuntimeTask}
                    />
                  ))
                )}
              </div>
            </div>
          </div>
        </div>
        <TaskPanel
          open={taskPanelOpen}
          onOpenChange={handleTaskPanelOpenChange}
          mode={taskPanelMode}
          focusedTaskId={taskPanelFocusedTaskId}
          focusedApprovalIds={taskPanelFocusedApprovalIds}
          workflowRunId={workflowBoundRunId}
          onOpenWorkflowRun={(runId) => router.push(`/agent-runs/${runId}`)}
        />
        <InferencePanel
          open={panelOpen}
          mobileOpen={mobileInferenceOpen}
          onOpenChange={setPanelOpen}
          onMobileOpenChange={setMobileInferenceOpen}
          presets={activeAgentSettings?.presets ?? []}
          activePresetName={activeAgentSettings?.active_preset ?? 'default'}
          selectedPresetName={selectedPresetName}
          onSelectedPresetChange={setSelectedPresetName}
          value={effectiveInference}
          onChange={handleSessionInferenceChange}
          onApplyPreset={handleApplyPresetToSession}
        onReset={handleResetSessionInference}
          onSavePreset={handleSaveInferencePreset}
          isSavingPreset={savingPreset}
          supportsReasoningEffort={supportsReasoningEffort}
          showReasoningEffort={false}
          disabledKeys={disabledInferenceKeys}
          disabledReason={disabledReason}
          agent={activeAgentSettings}
          settings={settings}
          activeModelInfo={activeModelInfo}
          onSettingsUpdated={setSettings}
        />
      </div>

      {modelSwitchError ? (
        <div className="border-t border-border bg-canvas/95 py-2 backdrop-blur">
          <div className="mx-auto max-w-4xl px-4 sm:px-6">
            <div className="flex items-start gap-2 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
              <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              <span className="min-w-0 break-words">{modelSwitchError}</span>
            </div>
          </div>
        </div>
      ) : null}

      {footerRuntimeNotice ? (
        <div className="border-t border-border bg-canvas/95 py-2 backdrop-blur">
          <div className="mx-auto max-w-4xl px-4 sm:px-6">
            <div
              className={cn(
                'flex items-center justify-between gap-3 rounded-md px-3 py-2 text-xs',
                footerRuntimeNotice.tone === 'error'
                  ? 'border border-destructive/30 bg-destructive/10 text-destructive'
                  : 'border border-warning/30 bg-warning/10 text-warning-foreground'
              )}
            >
              <div className="flex min-w-0 items-start gap-2">
              <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                <span className="min-w-0 break-words">{footerRuntimeNotice.message}</span>
              </div>
              <Button
                type="button"
                size="sm"
                variant="ghost"
                onClick={footerRuntimeNotice.onAction}
                className={cn(
                  'h-6 shrink-0 rounded-full px-2.5 text-[11px]',
                  footerRuntimeNotice.tone === 'error'
                    ? 'text-destructive hover:bg-destructive/10'
                    : 'text-warning-foreground hover:bg-warning/10'
                )}
              >
                {footerRuntimeNotice.actionLabel}
              </Button>
            </div>
          </div>
        </div>
      ) : null}

      <ChatInput
        sessionId={currentSessionId}
        projectId={effectiveProjectId}
        uploadTargetDir={uploadTargetDir}
        onSend={handleSend}
        onSubmitEdit={handleSubmitEdit}
        onCancelEdit={handleCancelEdit}
        onStop={handleStopGeneration}
        onVoice={handleVoiceEntry}
        onBuiltinCommand={handleBuiltinCommand}
        isStreaming={isStreaming}
        disabled={false}
        models={modelOptions}
        currentModel={currentModel}
        inference={effectiveInference}
        onSearchSkills={handleSearchSkills}
        onSwitchModel={handleSwitchModel}
        activeLocalRuntimeStatus={activeLocalRuntimeStatus}
        onUnloadCurrentModel={handleUnloadCurrentModel}
        isUnloadingCurrentModel={isUnloadingCurrentModel}
        reasoningOptions={supportedReasoningEfforts}
        onReasoningEffortChange={(value) => handleSessionInferenceChange('reasoningEffort', value)}
        approvalMode={effectiveAutonomyMode}
        approvalModeSourceLabel={autonomyModeSourceLabel}
        approvalModeSourceDescription={autonomyModeSourceDescription}
        onApprovalModeChange={(value) => void handleSessionAutonomyModeChange(value)}
        composerMode={editState ? 'edit' : 'compose'}
        composerSeed={editState?.seed ?? null}
        composerResetKey={editState?.resetKey}
        queuedAttachments={queuedWorkspaceAttachments}
        queuedAttachmentsKey={queuedWorkspaceAttachmentsKey}
      />

      <ExportDialog
        open={exportOpen}
        onOpenChange={setExportOpen}
        messages={messages}
      />

      <FloatingPanelShell
        open={workspaceMobileOpen}
        onOpenChange={setWorkspaceMobileOpen}
        desktopSide="left"
        desktopWidthClass="w-[min(40vw,44rem)] min-w-[24rem] max-w-[48rem]"
        desktopBreakpoint="lg"
        mobileSide="left"
        mobileClassName="w-[92vw] max-w-[92vw] p-0 sm:max-w-[92vw]"
        renderDesktop={false}
      >
        <WorkspacePanel
          onAttachAttachment={handleQueueWorkspaceAttachment}
          onClose={() => setWorkspaceMobileOpen(false)}
        />
      </FloatingPanelShell>

      <WorkflowPanel
        open={workflowPanelOpen}
        onOpenChange={setWorkflowPanelOpen}
        sessionId={currentSessionId}
        workflowEnabled={workflowEnabled}
        workflowBusy={workflowBusy}
        workflowError={workflowError}
        workflowBoundRunId={workflowBoundRunId}
        workflowState={workflowState}
        workflowConfig={workflowConfig}
        workflowTemplate={workflowTemplate}
        workflowProtocolId={workflowProtocolId}
        workflowReasoningEffort={workflowReasoningEffort}
        workflowRunPolicyPreset={workflowRunPolicyPreset}
        workflowRunPolicy={workflowRunPolicy}
        workflowExecutionPolicy={workflowExecutionPolicy}
        workflowEvidenceConfig={workflowEvidenceConfig}
        workflowResearchConfig={workflowResearchConfig}
        workflowScheduleConfig={workflowScheduleConfig}
        workflowScheduleType={workflowScheduleType}
        workflowScheduleEnabled={workflowScheduleEnabled(workflowState)}
        workflowProtocolOptions={WORKFLOW_PROTOCOL_OPTIONS}
        modelOptions={modelOptions}
        supportedReasoningEfforts={supportedReasoningEfforts}
        effectiveProjectId={effectiveProjectId}
        projects={projects}
        workflowProjectWorkspace={workflowProject?.workspaceDir ?? null}
        uploadTargetDir={uploadTargetDir ?? null}
        effectiveWorkflowWorkspace={effectiveWorkflowWorkspace}
        workflowHasUnsavedChanges={workflowHasUnsavedChanges}
        workflowSaveState={workflowSaveState}
        workflowLastSavedLabel={workflowLastSavedLabel}
        workflowLastSaveScope={workflowLastSaveScope}
        onWorkflowToggle={(enabled) => {
          void handleWorkflowToggle(enabled)
        }}
        onWorkflowNewRun={() => {
          void handleWorkflowNewRun()
        }}
        onOpenRunDetail={(runId) => router.push(`/agent-runs/${runId}`)}
        onWorkflowProjectChange={(projectId) => {
          void handleWorkflowProjectChange(projectId)
        }}
        onWorkflowFieldChange={handleWorkflowFieldChange}
        onWorkflowTemplateChange={handleWorkflowTemplateChange}
        onWorkflowRunPolicyPresetChange={handleWorkflowRunPolicyPresetChange}
        onWorkflowConfigPatch={handleWorkflowConfigPatch}
        onWorkflowSave={() => {
          void handleWorkflowSave()
        }}
        buildWorkflowScheduleConfig={buildWorkflowScheduleConfig}
        formatWorkflowScheduleRunAt={formatWorkflowScheduleRunAt}
        defaultScheduleTimezone={defaultScheduleTimezone}
      />

      <VoiceOverlay
        open={voiceOpen}
        phase={resolveVoiceOverlayPhase(voicePhase, voiceRuntimeStatus)}
        isRecording={voiceRecording}
        inputLevel={voiceInputLevel}
        hasInputSignal={voiceCaptureDiagnostics?.hasInputSignal ?? false}
        microphoneLabel={voiceCaptureDiagnostics?.microphoneLabel ?? null}
        vadState={voiceVadState}
        partialTranscription={voicePartialTranscription}
        finalTranscription={voiceFinalTranscription}
        assistantText={voiceAssistantText}
        captureWarning={voiceCaptureWarning}
        errorMessage={voiceErrorMessage ?? voiceRuntimeStatus?.error ?? null}
        onToggleRecording={handleVoiceToggleRecording}
        onInterrupt={handleVoiceInterrupt}
        onClose={handleVoiceClose}
      />
    </div>
  )
}
