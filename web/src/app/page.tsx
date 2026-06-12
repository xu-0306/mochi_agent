'use client'

import * as React from 'react'
import { useRouter } from 'next/navigation'
import {
  AlertCircle,
  ExternalLink,
  ListTodo,
  Loader2,
  MoreHorizontal,
  RotateCcw,
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
import { ChatMessage } from '@/components/chat/ChatMessage'
import { EmptyState } from '@/components/chat/EmptyState'
import { ExportDialog } from '@/components/chat/ExportDialog'
import { InferencePanel } from '@/components/chat/InferencePanel'
import { ScrollToBottom } from '@/components/chat/ScrollToBottom'
import { TaskPanel } from '@/components/chat/TaskPanel'
import { Input } from '@/components/ui/input'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Separator } from '@/components/ui/separator'
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from '@/components/ui/sheet'
import { Switch } from '@/components/ui/switch'
import { VoiceOverlay } from '@/components/voice/VoiceOverlay'
import * as api from '@/lib/api'
import type { ChatAttachment, Message, ReasoningStep } from '@/lib/chat'
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
import { useSessionStore } from '@/lib/stores/session-store'
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
      selectedSkillIds?: string[]
      attachments?: ChatAttachment[]
      systemPrompt?: string
      temperature?: number
      maxTokens?: number
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
    selected_skill_ids?: string[]
    selectedSkillIds?: string[]
    attachments?: ChatAttachment[]
    system_prompt?: string
    temperature?: number
    max_tokens?: number
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
  sessionId?: string
  trajectoryId?: string | null
  model?: string | null
  done?: boolean
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

function buildDefaultWorkflowState(
  reasoningEffort: api.ReasoningEffort | null
): api.SessionWorkflowState {
  return {
    enabled: false,
    bound_run_id: null,
    synced_run_event_count: 0,
    workspace_dir_override: null,
    config: {
      title: null,
      protocol_id: DEFAULT_WORKFLOW_PROTOCOL,
      workspace_dir_override: null,
      reasoning_effort: reasoningEffort,
      selected_models_roles: {},
      run_policy: {},
      execution_policy: {},
      schedule: {},
      evidence: {},
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
      protocol_id: config.protocol_id ?? defaults.config?.protocol_id ?? DEFAULT_WORKFLOW_PROTOCOL,
      reasoning_effort: config.reasoning_effort ?? reasoningEffort,
      selected_models_roles: config.selected_models_roles ?? {},
      run_policy: config.run_policy ?? {},
      execution_policy: config.execution_policy ?? {},
      schedule: normalizeWorkflowScheduleConfig(
        isRecord(config.schedule) ? config.schedule : {}
      ),
      evidence: config.evidence ?? {},
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

function formatWorkflowLifecycleMessage(event: Record<string, unknown>): string {
  const type = getString(event.type) ?? 'workflow_event'
  const status = getString(event.status)

  if (type === 'run_created') {
    return 'Workflow run created.'
  }
  if (type === 'run_started') {
    return 'Workflow run started.'
  }
  if (type === 'run_completed') {
    return 'Workflow run completed.'
  }
  if (type === 'run_failed') {
    return `Workflow run failed${status ? `: ${status}.` : '.'}`
  }
  if (type === 'run_paused') {
    return 'Workflow run paused.'
  }
  if (type === 'run_resumed') {
    return 'Workflow run resumed.'
  }
  if (type === 'run_finalized_partial') {
    return 'Workflow run finalized as partial.'
  }
  if (type === 'run_scheduled') {
    return 'Workflow run scheduled.'
  }
  if (type === 'run_status') {
    return status ? `Workflow status: ${status}.` : 'Workflow status updated.'
  }

  return type.replaceAll('_', ' ')
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
    provider === 'gemini' ||
    provider === 'anthropic'
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
    return toolMode ? `OpenAI Codex · ${toolMode}` : 'OpenAI Codex'
  }

  if (provider === 'openai_compat' || backendType === 'openai_compat') {
    const source = baseUrl ?? 'OpenAI-Compatible'
    return toolMode ? `${source} · ${toolMode}` : source
  }

  if (provider === 'ollama' || backendType === 'ollama') {
    return toolMode ? `Ollama · ${toolMode}` : 'Ollama'
  }

  if (provider === 'local') {
    const source = backendType ? `Local ${backendType}` : 'Local'
    return toolMode ? `${source} · ${toolMode}` : source
  }

  const source = provider ?? backendType ?? baseUrl
  return toolMode && source ? `${source} · ${toolMode}` : source
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
        'connected',
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
        'connected',
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

function resolveActiveModelId(
  payload: ModelsResponse,
  options: ChatInputModelOption[]
): string | null {
  const configuredModel = getString(payload.configured_model)
  if (configuredModel) {
    const configuredOption = options.find((option) => option.id === configuredModel)
    if (configuredOption) {
      return configuredOption.id
    }
  }

  const activeName = resolveModelOptionId(payload.active_model ?? undefined)
  if (activeName) {
    const activeOption = options.find(
      (option) => option.id === activeName || option.id.endsWith(`:${activeName}`)
    )
    return activeOption?.id ?? activeName
  }

  return configuredModel ?? options[0]?.id ?? null
}

async function requestChat(
  text: string,
  sessionId: string | undefined,
  projectId: string | null | undefined,
  model: string | null,
  selectedSkillIds: string[],
  attachments: ChatAttachment[],
  inference: {
    systemPrompt: string
    temperature: number
    maxTokens: number
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
      selected_skill_ids: selectedSkillIds,
      selectedSkillIds,
      attachments,
      system_prompt: inference.systemPrompt,
      temperature: inference.temperature,
      max_tokens: inference.maxTokens,
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
      selectedSkillIds,
      attachments,
      systemPrompt: inference.systemPrompt,
      temperature: inference.temperature,
      maxTokens: inference.maxTokens,
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

function buildTurnPlaceholder(
  turnKey: string,
  options?: {
    content?: string
  }
): Message {
  return {
    id: `assistant-turn-${turnKey}`,
    type: 'assistant',
    content: options?.content ?? '',
    timestamp: new Date(),
    turnKey,
    reasoningSteps: [],
    isStreaming: true,
  }
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
  const [workflowPanelOpen, setWorkflowPanelOpen] = React.useState(false)
  const [selectedPresetName, setSelectedPresetName] = React.useState('default')
  const [savingPreset, setSavingPreset] = React.useState(false)
  const [workflowBusy, setWorkflowBusy] = React.useState(false)
  const [workflowError, setWorkflowError] = React.useState<string | null>(null)
  const [workflowDraftBySessionId, setWorkflowDraftBySessionId] = React.useState<
    Record<string, api.SessionWorkflowState>
  >({})
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
  const startStreaming = useChatRuntimeStore((state) => state.startStreaming)
  const finishStreaming = useChatRuntimeStore((state) => state.finishStreaming)
  const abortStreaming = useChatRuntimeStore((state) => state.abortStreaming)
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
  const workflowEnabled = Boolean(workflowState.enabled)
  const workflowBoundRunId = workflowState.bound_run_id ?? null
  const workflowConfig = workflowState.config ?? {}
  const workflowProject = projects.find((project) => project.id === effectiveProjectId) ?? null
  const workflowProtocolId = workflowConfig.protocol_id ?? DEFAULT_WORKFLOW_PROTOCOL
  const workflowReasoningEffort =
    workflowConfig.reasoning_effort ?? effectiveInference.reasoningEffort ?? null
  const workflowRunPolicy = React.useMemo(
    () => (workflowConfig.run_policy ?? {}) as api.AgentRunRunPolicy,
    [workflowConfig.run_policy]
  )
  const workflowExecutionPolicy = React.useMemo(
    () => (workflowConfig.execution_policy ?? {}) as Record<string, unknown>,
    [workflowConfig.execution_policy]
  )
  const workflowEvidenceConfig = React.useMemo(
    () => (workflowConfig.evidence ?? {}) as Record<string, unknown>,
    [workflowConfig.evidence]
  )
  const workflowScheduleConfig = React.useMemo(
    () => normalizeWorkflowScheduleConfig((workflowConfig.schedule ?? {}) as Record<string, unknown>),
    [workflowConfig.schedule]
  )
  const workflowScheduleType = React.useMemo(
    () => resolveWorkflowScheduleType(workflowScheduleConfig),
    [workflowScheduleConfig]
  )
  const uploadTargetDir =
    projects.find((project) => project.id === effectiveProjectId)?.workspaceDir ??
    getString(settings?.paths?.workspace_dir) ??
    undefined
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
  }, [messages, scrollToBottom])

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
    const needsCanonicalTurnIds =
      runtimeMessages.length > 0 &&
      !hasActiveStream &&
      runtimeMessages.some(
        (message) =>
          (message.type === 'user' || message.type === 'assistant') &&
          !message.turnId
      )

    if (needsCanonicalTurnIds) {
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
        .map((event) => {
          const type = getString(event.type)
          const timestamp = getString(event.timestamp) ?? new Date().toISOString()
          if (type === 'operator_message') {
            return {
              type: 'message',
              role: 'user',
              content: getString(event.content) ?? '',
              attachments: Array.isArray(event.attachments) ? event.attachments : [],
              timestamp,
              turn_id: `${runDetail.run_id}:${syncedCount}`,
              metadata: {
                channel: 'workflow',
                workflow_run_id: runDetail.run_id,
              },
            }
          }
          if (type === 'assistant_message') {
            return {
              type: 'message',
              role: 'assistant',
              content: getString(event.content) ?? '',
              timestamp,
              turn_id: `${runDetail.run_id}:${syncedCount}`,
              metadata: {
                channel: 'workflow',
                workflow_run_id: runDetail.run_id,
              },
            }
          }
          if (type === 'artifact') {
            return {
              type: 'turn_event',
              phase: 'workflow_artifact',
              timestamp,
              payload: {
                content:
                  getString(event.title) ??
                  getString(event.artifact_type) ??
                  'Workflow artifact recorded.',
                artifact_type: getString(event.artifact_type),
                workflow_run_id: runDetail.run_id,
              },
            }
          }
          if (type === 'exec_update' || type === 'detached_exec_reattached' || type === 'detached_exec_stop') {
            return {
              type: 'turn_event',
              phase: 'workflow_exec_update',
              timestamp,
              payload: {
                content:
                  getString(event.content) ??
                  getString(event.status) ??
                  'Workflow execution updated.',
                workflow_run_id: runDetail.run_id,
              },
            }
          }
          return {
            type: 'turn_event',
            phase: 'workflow_status',
            timestamp,
            payload: {
              content: formatWorkflowLifecycleMessage(event),
              status: getString(event.status),
              event_type: type,
              workflow_run_id: runDetail.run_id,
            },
          }
        })
        .filter((event) => {
          if (event.type === 'message') {
            return Boolean(event.content) || (Array.isArray(event.attachments) && event.attachments.length > 0)
          }
          return true
        })

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

  const syncSessionFromServer = React.useCallback(async (sessionId: string) => {
    try {
      const detail = await api.fetchSession(sessionId)
      const replayMessages = api.buildMessagesFromSessionEvents(detail.events)
      const lastRetainedMessage = [...replayMessages]
        .reverse()
        .find((message) => message.type === 'user' || message.type === 'assistant')

      if (replayMessages.length > 0) {
        setSessionMessages(sessionId, replayMessages)
      }
      if (lastRetainedMessage) {
        updateLastMessage(sessionId, lastRetainedMessage.content)
      }

      upsertSessionDetail(detail)
    } catch {
      // Keep the optimistic transcript if canonical session refresh fails.
    }
  }, [setSessionMessages, updateLastMessage, upsertSessionDetail])

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
    const nextOptions = deriveModelOptions(payload)
    const activeModel = resolveActiveModelId(payload, nextOptions)
    const nextActiveModelInfo = isRecord(payload.active_model) ? payload.active_model : null
    const activeModelMetadata = isRecord(payload.active_model?.metadata) ? payload.active_model?.metadata : null
    const loaded =
      activeModelMetadata && typeof activeModelMetadata.loaded === 'boolean'
        ? activeModelMetadata.loaded
        : null

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
  }, [loadModels])

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
  }, [t])

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
      const sessionId = (
        targetSession?.isDraft
      ) || initialSessionId.startsWith('draft-')
        ? await materializeDraftSession(initialSessionId)
        : initialSessionId
      const latestSessionState = useSessionStore.getState()
      const sessionAfterMaterialize = latestSessionState.sessions.find((session) => session.id === sessionId)
      const normalizedWorkflow = normalizeWorkflowState(
        workflowDraftBySessionId[sessionId] ??
          sessionAfterMaterialize?.workflow ??
          (latestSessionState.currentSessionDetail?.id === sessionId
            ? latestSessionState.currentSessionDetail.workflow
            : null),
        effectiveInference.reasoningEffort
      )
      const turnKey = `turn-${Date.now()}`
      const userMessage: Message = {
        id: `user-${Date.now()}`,
        type: 'user',
        content: text,
        attachments,
        timestamp: new Date(),
      }
      const lastMessageSummary =
        text.trim() ||
        (attachments.length > 0
          ? attachments.slice(0, 2).map((attachment) => attachment.name).join(', ')
          : '')

      const placeholderContent = isLocalModelId(currentModel) && currentModelLoaded === false
        ? t('chat.loadingLocalModel')
        : ''

      setSessionMessages(sessionId, [
        ...resolveMessagesForSession(sessionId),
        userMessage,
        buildTurnPlaceholder(turnKey, { content: placeholderContent }),
      ])
      updateLastMessage(sessionId, lastMessageSummary)
      const abortController = new AbortController()
      startStreaming(sessionId, abortController)

      try {
        if (normalizedWorkflow.enabled) {
          setWorkflowBusy(true)
          setWorkflowError(null)
          const workflowSessionProjectId =
            sessionAfterMaterialize?.projectId ?? activeProjectId ?? null
          const effectiveWorkspaceDir =
            normalizedWorkflow.workspace_dir_override ||
            normalizedWorkflow.config?.workspace_dir_override ||
            projects.find((project) => project.id === workflowSessionProjectId)?.workspaceDir ||
            uploadTargetDir ||
            null

          let runId = normalizedWorkflow.bound_run_id ?? null
          let appendedDetail: api.AgentRunDetail
          if (!runId) {
            const workflowSchedule = normalizeWorkflowScheduleConfig(
              (normalizedWorkflow.config?.schedule ?? {}) as Record<string, unknown>
            )
            const createdRun = await api.createAgentRun({
              protocol_id: normalizedWorkflow.config?.protocol_id ?? DEFAULT_WORKFLOW_PROTOCOL,
              title: normalizedWorkflow.config?.title ?? null,
              topic: text.trim() || null,
              projectId: workflowSessionProjectId,
              workspaceDir: effectiveWorkspaceDir,
              reasoning_effort:
                normalizedWorkflow.config?.reasoning_effort ?? effectiveInference.reasoningEffort ?? null,
              selected_models_roles:
                normalizedWorkflow.config?.selected_models_roles &&
                Object.keys(normalizedWorkflow.config.selected_models_roles).length > 0
                  ? {
                      by_role: normalizedWorkflow.config.selected_models_roles,
                      entries: Object.entries(normalizedWorkflow.config.selected_models_roles).map(
                        ([role, model_id]) => ({ role, model_id })
                      ),
                    }
                  : {},
              run_policy: normalizedWorkflow.config?.run_policy ?? {},
              evaluation_policy: {
                evidence_collection: normalizedWorkflow.config?.evidence ?? {},
              },
              summary: {
                operator_message: text,
                selected_skill_ids: selectedSkillIds,
                execution_policy: normalizedWorkflow.config?.execution_policy ?? {},
              },
              schedule: workflowSchedule,
            })
            runId = createdRun.run_id
          }

          appendedDetail = await api.appendAgentRunMessage(runId, {
            role: 'operator',
            content: text,
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
              : latestSessionState.currentSessionDetail?.events) ?? []
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
        try {
          for await (const chunk of api.streamChatMessages(text, {
            sessionId,
            projectId:
              sessions.find((session) => session.id === sessionId)?.projectId ?? activeProjectId ?? null,
            model: currentModel ?? undefined,
            selectedSkillIds,
            attachments,
            systemPrompt: effectiveInference.systemPrompt,
            temperature: effectiveInference.temperature,
            maxTokens: effectiveInference.maxTokens,
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
          })) {
            streamed = true

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
            text,
            sessionId,
            sessions.find((session) => session.id === sessionId)?.projectId ?? activeProjectId ?? null,
            currentModel,
            selectedSkillIds,
            attachments,
            effectiveInference
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
        } else {
          if (latestAssistantContent) {
            updateLastMessage(sessionId, latestAssistantContent)
          }
        }

        await syncSessionFromServer(sessionId)
      } catch (error) {
        if (error instanceof DOMException && error.name === 'AbortError') {
          updateSessionMessages(sessionId, (prev) => prev.map((message) =>
            message.turnKey === turnKey ? { ...message, isStreaming: false } : message
          ))
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
        finishStreaming(sessionId)
      }
    },
    [
      activeProjectId,
      createDraftSession,
      currentModel,
      currentModelLoaded,
      currentSessionId,
      effectiveInference,
      finishStreaming,
      hasActiveStream,
      materializeDraftSession,
      persistWorkflowState,
      projects,
      resolveMessagesForSession,
      selectSession,
      setSessionMessages,
      sessions,
      startStreaming,
      syncWorkflowRunEventsToSession,
      t,
      syncSessionFromServer,
      uploadTargetDir,
      updateSessionMessages,
      updateLastMessage,
      workflowDraftBySessionId,
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
    abortStreaming()
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
      const initialSessionId = currentSessionId ?? createDraftSession(activeProjectId)
      const targetSession = sessions.find((session) => session.id === initialSessionId)
      const sessionId = targetSession?.isDraft ? initialSessionId : initialSessionId
      const nextWorkflow = normalizeWorkflowState(
        workflowDraftBySessionId[sessionId] ?? targetSession?.workflow ?? null,
        effectiveInference.reasoningEffort
      )
      nextWorkflow.enabled = command === 'workflow'
      setWorkflowPanelOpen(command === 'workflow')
      await persistWorkflowState(sessionId, nextWorkflow)
    }
  }, [
    activeProjectId,
    createDraftSession,
    currentSessionId,
    effectiveInference.reasoningEffort,
    handleVoiceEntry,
    persistWorkflowState,
    router,
    sessions,
    setSessionMessages,
    t,
    workflowDraftBySessionId,
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
  }, [currentSessionId])

  const handleRegenerate = React.useCallback((message: Message) => {
    const prompt = findRegeneratePrompt(messages, message.id)
    if (!prompt) {
      return
    }
    void handleSend(prompt)
  }, [handleSend, messages])

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
    setWorkflowPanelOpen(enabled)
    await persistWorkflowState(sessionId, nextWorkflow)
  }, [
    activeProjectId,
    createDraftSession,
    currentSessionId,
    effectiveInference.reasoningEffort,
    persistWorkflowState,
    sessions,
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

  const handleWorkflowSave = React.useCallback(async () => {
    if (!currentSessionId) {
      return
    }
    setWorkflowError(null)
    try {
      await persistWorkflowState(currentSessionId, workflowState)
    } catch (error) {
      const detail = error instanceof Error ? error.message : 'Unable to save workflow settings.'
      setWorkflowError(detail)
    }
  }, [currentSessionId, persistWorkflowState, workflowState])

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

  const showEmptyState = isConversationEffectivelyEmpty(messages)

  return (
    <div className="flex h-full flex-col">
      <header className="border-b border-border bg-canvas/95 backdrop-blur">
        <div className="mx-auto flex h-14 w-full max-w-5xl items-center justify-between gap-4 px-4">
          <h1 className="min-w-0 truncate text-sm font-semibold text-foreground">
            {displaySessionTitle(currentSession?.title, t('chat.newChat'))}
          </h1>
          <div className="flex items-center gap-1">
            <div className="mr-2 hidden max-w-[220px] items-center gap-1.5 text-xs text-muted-foreground sm:flex">
              {isStreaming ? (
                <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin" />
              ) : (
                <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-success" />
              )}
              <span className="truncate">{headerModelLabel}</span>
            </div>
            <Button
              variant={workflowEnabled || workflowPanelOpen ? 'secondary' : 'ghost'}
              size="sm"
              title={t('sidebar.workflows')}
              onClick={() => setWorkflowPanelOpen(true)}
              className="max-sm:w-8 max-sm:px-0"
            >
              <Workflow className="h-4 w-4" />
              <span className="hidden sm:inline">{t('sidebar.workflows')}</span>
            </Button>
            <Button
              variant="ghost"
              size="icon-sm"
              title={t('chat.moreOptions')}
              onClick={() => setExportOpen(true)}
            >
              <MoreHorizontal className="h-4 w-4" />
            </Button>
            <Button
              variant={panelOpen || mobileInferenceOpen ? 'secondary' : 'ghost'}
              size="icon-sm"
              title="Inference"
              onClick={() => {
                setTaskPanelOpen(false)
                if (window.innerWidth < 768) {
                  setMobileInferenceOpen(true)
                } else {
                  setMobileInferenceOpen(false)
                  setPanelOpen(!panelOpen)
                }
              }}
            >
              <SlidersHorizontal className="h-4 w-4" />
            </Button>
            <Button
              variant={taskPanelOpen ? 'secondary' : 'ghost'}
              size="icon-sm"
              title="Tasks"
              onClick={() => {
                setMobileInferenceOpen(false)
                setPanelOpen(false)
                setTaskPanelOpen((open) => !open)
              }}
            >
              <ListTodo className="h-4 w-4" />
            </Button>
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
        <div className="min-w-0 flex-1">
          <ScrollToBottom visible={showScrollToBottom} onClick={scrollToBottom} />
          <div ref={scrollRef} className="h-full overflow-y-auto">
            <div className="mx-auto flex w-full max-w-4xl flex-col px-4 py-8 sm:px-6">
              {showEmptyState ? (
                <EmptyState
                  onPrompt={handleStarterPrompt}
                  onVoice={() => void handleVoiceEntry()}
                  onSettings={() => router.push('/settings')}
                />
              ) : (
                <div className="space-y-6">
                  {messages.map((message) => (
                    <ChatMessage
                      key={message.id}
                      message={
                        message.type === 'assistant' && !effectiveInference.showTokenStats
                          ? { ...message, tokenStats: undefined }
                          : message
                      }
                      onRegenerate={message.type === 'assistant' ? handleRegenerate : undefined}
                      onEditAndResend={message.type === 'user' ? (message) => handleEditAndResend(message) : undefined}
                      onUndoFileChange={handleUndoFileChange}
                    />
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
        <TaskPanel open={taskPanelOpen} onOpenChange={setTaskPanelOpen} />
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

      {workflowError ? (
        <div className="border-t border-border bg-canvas/95 py-2 backdrop-blur">
          <div className="mx-auto max-w-4xl px-4 sm:px-6">
            <div className="flex items-start gap-2 rounded-md border border-warning/30 bg-warning/10 px-3 py-2 text-xs text-warning-foreground">
              <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              <span className="min-w-0 break-words">{workflowError}</span>
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
        composerMode={editState ? 'edit' : 'compose'}
        composerSeed={editState?.seed ?? null}
        composerResetKey={editState?.resetKey}
      />

      <ExportDialog
        open={exportOpen}
        onOpenChange={setExportOpen}
        messages={messages}
      />

      <Sheet open={workflowPanelOpen} onOpenChange={setWorkflowPanelOpen}>
        <SheetContent side="right" className="w-[420px] max-w-[92vw] overflow-y-auto">
          <SheetHeader>
            <SheetTitle>Workflow</SheetTitle>
            <SheetDescription>
              Chat-first workflow mode keeps this conversation bound to one main run.
            </SheetDescription>
          </SheetHeader>

          <div className="space-y-5 pr-1">
            <section className="space-y-3">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="text-sm font-medium text-foreground">Workflow mode</p>
                  <p className="text-xs text-muted-foreground">
                    Route new messages through the workflow runtime for this session.
                  </p>
                </div>
                <Switch
                  checked={workflowEnabled}
                  onCheckedChange={(checked) => {
                    void handleWorkflowToggle(checked)
                  }}
                />
              </div>
              <div className="rounded-lg border border-border bg-surface-layer px-3 py-2 text-xs text-muted-foreground">
                {workflowEnabled
                  ? 'Workflow mode is active for this chat session.'
                  : 'Workflow mode is off. Use /workflow or this switch to enable it.'}
              </div>
            </section>

            <Separator />

            <section className="space-y-3">
              <div className="flex items-center justify-between gap-2">
                <div>
                  <p className="text-sm font-medium text-foreground">Bound run</p>
                  <p className="text-xs text-muted-foreground">
                    This chat keeps appending to the same workflow run unless you start a new one.
                  </p>
                </div>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => void handleWorkflowNewRun()}
                >
                  <RotateCcw className="h-3.5 w-3.5" />
                  New run
                </Button>
              </div>
              <div className="rounded-lg border border-border bg-surface-layer px-3 py-3">
                <p className="text-sm text-foreground">
                  {workflowBoundRunId ?? 'No run bound yet'}
                </p>
                <p className="mt-1 text-xs text-muted-foreground">
                  {workflowBoundRunId
                    ? `Synced events: ${workflowState.synced_run_event_count ?? 0}`
                    : 'The first workflow message will create and bind a run.'}
                </p>
                {workflowBoundRunId ? (
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="mt-2 px-0"
                    onClick={() => router.push(`/agent-runs/${workflowBoundRunId}`)}
                  >
                    <ExternalLink className="h-3.5 w-3.5" />
                    Open run detail
                  </Button>
                ) : null}
              </div>
            </section>

            <Separator />

            <section className="space-y-3">
              <div>
                <p className="text-sm font-medium text-foreground">Project / workspace</p>
                <p className="text-xs text-muted-foreground">
                  Files and execution resolve from the selected project unless you override the path.
                </p>
              </div>
              <div className="space-y-2">
                <span className="text-xs font-medium text-muted-foreground">Project</span>
                <Select
                  value={effectiveProjectId ?? '__none__'}
                  onValueChange={(value) => {
                    void handleWorkflowProjectChange(value === '__none__' ? null : value)
                  }}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="No project" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="__none__">No project</SelectItem>
                    {projects.map((project) => (
                      <SelectItem key={project.id} value={project.id}>
                        {project.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <span className="text-xs font-medium text-muted-foreground">Workspace override</span>
                <Input
                  value={workflowState.workspace_dir_override ?? ''}
                  placeholder={workflowProject?.workspaceDir ?? uploadTargetDir ?? 'Use project workspace'}
                  onChange={(event) =>
                    handleWorkflowFieldChange({
                      workspace_dir_override: event.target.value || null,
                      config: {
                        ...(workflowState.config ?? {}),
                        workspace_dir_override: event.target.value || null,
                      },
                    })
                  }
                />
              </div>
              <div className="rounded-lg border border-border bg-muted/40 px-3 py-2 text-xs text-muted-foreground">
                Effective workspace: <span className="break-all text-foreground">{effectiveWorkflowWorkspace || 'Not set'}</span>
              </div>
            </section>

            <Separator />

            <section className="space-y-3">
              <div>
                <p className="text-sm font-medium text-foreground">Protocol / reasoning</p>
                <p className="text-xs text-muted-foreground">
                  Session-scoped workflow defaults used when creating a new bound run.
                </p>
              </div>
              <div className="space-y-2">
                <span className="text-xs font-medium text-muted-foreground">Title</span>
                <Input
                  value={workflowConfig.title ?? ''}
                  placeholder="Optional workflow title"
                  onChange={(event) =>
                    handleWorkflowFieldChange({
                      config: {
                        ...(workflowState.config ?? {}),
                        title: event.target.value || null,
                      },
                    })
                  }
                />
              </div>
              <div className="space-y-2">
                <span className="text-xs font-medium text-muted-foreground">Protocol</span>
                <Select
                  value={workflowProtocolId}
                  onValueChange={(value) =>
                    handleWorkflowFieldChange({
                      config: {
                        ...(workflowState.config ?? {}),
                        protocol_id: value as api.AgentRunProtocolId,
                      },
                    })
                  }
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {WORKFLOW_PROTOCOL_OPTIONS.map((option) => (
                      <SelectItem key={option.value} value={option.value}>
                        {option.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground">
                  {WORKFLOW_PROTOCOL_OPTIONS.find((option) => option.value === workflowProtocolId)?.description}
                </p>
              </div>
              <div className="space-y-2">
                <span className="text-xs font-medium text-muted-foreground">Reasoning effort</span>
                <Select
                  value={workflowReasoningEffort ?? '__inherit__'}
                  onValueChange={(value) =>
                    handleWorkflowFieldChange({
                      config: {
                        ...(workflowState.config ?? {}),
                        reasoning_effort:
                          value === '__inherit__' ? effectiveInference.reasoningEffort : value as api.ReasoningEffort,
                      },
                    })
                  }
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="__inherit__">Inherit chat setting</SelectItem>
                    {supportedReasoningEfforts.map((option) => (
                      <SelectItem key={option} value={option}>
                        {option}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </section>

            <Separator />

            <section className="space-y-3">
              <div>
                <p className="text-sm font-medium text-foreground">Execution / schedule</p>
                <p className="text-xs text-muted-foreground">
                  Existing controlled execution boundaries stay in the runtime. This panel stores defaults only.
                </p>
              </div>
              <div className="space-y-3 rounded-lg border border-border bg-surface-layer px-3 py-3">
                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-2">
                    <span className="text-xs font-medium text-muted-foreground">Max wall clock (sec)</span>
                    <Input
                      value={String(workflowRunPolicy.max_wall_clock_sec ?? '')}
                      placeholder="1800"
                      onChange={(event) =>
                        handleWorkflowConfigPatch({
                          run_policy: {
                            ...workflowRunPolicy,
                            max_wall_clock_sec: event.target.value
                              ? Number.parseInt(event.target.value, 10)
                              : null,
                          },
                        })
                      }
                    />
                  </div>
                  <div className="space-y-2">
                    <span className="text-xs font-medium text-muted-foreground">Heartbeat timeout (sec)</span>
                    <Input
                      value={String(workflowRunPolicy.heartbeat_timeout_sec ?? '')}
                      placeholder="90"
                      onChange={(event) =>
                        handleWorkflowConfigPatch({
                          run_policy: {
                            ...workflowRunPolicy,
                            heartbeat_timeout_sec: event.target.value
                              ? Number.parseInt(event.target.value, 10)
                              : null,
                          },
                        })
                      }
                    />
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-2">
                    <span className="text-xs font-medium text-muted-foreground">Checkpoint steps</span>
                    <Input
                      value={String(workflowRunPolicy.checkpoint_interval_steps ?? '')}
                      placeholder="1"
                      onChange={(event) =>
                        handleWorkflowConfigPatch({
                          run_policy: {
                            ...workflowRunPolicy,
                            checkpoint_interval_steps: event.target.value
                              ? Number.parseInt(event.target.value, 10)
                              : undefined,
                          },
                        })
                      }
                    />
                  </div>
                  <div className="space-y-2">
                    <span className="text-xs font-medium text-muted-foreground">Max subagent failures</span>
                    <Input
                      value={String(workflowRunPolicy.max_subagent_failures_per_role ?? '')}
                      placeholder="2"
                      onChange={(event) =>
                        handleWorkflowConfigPatch({
                          run_policy: {
                            ...workflowRunPolicy,
                            max_subagent_failures_per_role: event.target.value
                              ? Number.parseInt(event.target.value, 10)
                              : undefined,
                          },
                        })
                      }
                    />
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-2">
                    <span className="text-xs font-medium text-muted-foreground">On budget exhausted</span>
                    <Select
                      value={workflowRunPolicy.on_budget_exhausted ?? 'finalize_partial'}
                      onValueChange={(value) =>
                        handleWorkflowConfigPatch({
                          run_policy: {
                            ...workflowRunPolicy,
                            on_budget_exhausted: value as 'pause' | 'finalize_partial',
                          },
                        })
                      }
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="finalize_partial">Finalize partial</SelectItem>
                        <SelectItem value="pause">Pause</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <span className="text-xs font-medium text-muted-foreground">On subagent disconnect</span>
                    <Select
                      value={workflowRunPolicy.on_subagent_disconnect ?? 'pause'}
                      onValueChange={(value) =>
                        handleWorkflowConfigPatch({
                          run_policy: {
                            ...workflowRunPolicy,
                            on_subagent_disconnect: value as 'retry_then_degrade' | 'pause' | 'fail',
                          },
                        })
                      }
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="pause">Pause</SelectItem>
                        <SelectItem value="retry_then_degrade">Retry then degrade</SelectItem>
                        <SelectItem value="fail">Fail</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </div>

              <div className="space-y-3 rounded-lg border border-border bg-surface-layer px-3 py-3">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="text-sm font-medium text-foreground">Controlled execution</p>
                    <p className="text-xs text-muted-foreground">
                      Keep subagents proposal-only while the controller owns runtime execution.
                    </p>
                  </div>
                  <Switch
                    checked={workflowExecutionPolicy.mode === 'controlled'}
                    onCheckedChange={(checked) =>
                      handleWorkflowConfigPatch({
                        execution_policy: checked
                          ? {
                              ...workflowExecutionPolicy,
                              mode: 'controlled',
                              max_execution_requests:
                                Number(workflowExecutionPolicy.max_execution_requests) || 1,
                              max_commands_per_request:
                                Number(workflowExecutionPolicy.max_commands_per_request) || 1,
                              default_timeout_sec:
                                Number(workflowExecutionPolicy.default_timeout_sec) || 300,
                              background_allowed:
                                workflowExecutionPolicy.background_allowed !== false,
                            }
                          : {},
                      })
                    }
                  />
                </div>
                {workflowExecutionPolicy.mode === 'controlled' ? (
                  <div className="grid grid-cols-2 gap-3">
                    <div className="space-y-2">
                      <span className="text-xs font-medium text-muted-foreground">Max exec requests</span>
                      <Input
                        value={String(workflowExecutionPolicy.max_execution_requests ?? '')}
                        placeholder="1"
                        onChange={(event) =>
                          handleWorkflowConfigPatch({
                            execution_policy: {
                              ...workflowExecutionPolicy,
                              mode: 'controlled',
                              max_execution_requests: event.target.value
                                ? Number.parseInt(event.target.value, 10)
                                : 1,
                            },
                          })
                        }
                      />
                    </div>
                    <div className="space-y-2">
                      <span className="text-xs font-medium text-muted-foreground">Max commands / request</span>
                      <Input
                        value={String(workflowExecutionPolicy.max_commands_per_request ?? '')}
                        placeholder="1"
                        onChange={(event) =>
                          handleWorkflowConfigPatch({
                            execution_policy: {
                              ...workflowExecutionPolicy,
                              mode: 'controlled',
                              max_commands_per_request: event.target.value
                                ? Number.parseInt(event.target.value, 10)
                                : 1,
                            },
                          })
                        }
                      />
                    </div>
                    <div className="space-y-2">
                      <span className="text-xs font-medium text-muted-foreground">Default exec timeout</span>
                      <Input
                        value={String(workflowExecutionPolicy.default_timeout_sec ?? '')}
                        placeholder="300"
                        onChange={(event) =>
                          handleWorkflowConfigPatch({
                            execution_policy: {
                              ...workflowExecutionPolicy,
                              mode: 'controlled',
                              default_timeout_sec: event.target.value
                                ? Number.parseInt(event.target.value, 10)
                                : 300,
                            },
                          })
                        }
                      />
                    </div>
                    <div className="flex items-center justify-between gap-3 rounded-md border border-border px-3 py-2">
                      <span className="text-xs text-muted-foreground">Allow background execution</span>
                      <Switch
                        checked={workflowExecutionPolicy.background_allowed !== false}
                        onCheckedChange={(checked) =>
                          handleWorkflowConfigPatch({
                            execution_policy: {
                              ...workflowExecutionPolicy,
                              mode: 'controlled',
                              background_allowed: checked,
                            },
                          })
                        }
                      />
                    </div>
                  </div>
                ) : null}
              </div>

              <div className="space-y-3 rounded-lg border border-border bg-surface-layer px-3 py-3">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="text-sm font-medium text-foreground">Evidence collection</p>
                    <p className="text-xs text-muted-foreground">
                      Configure retrieval defaults for new workflow runs.
                    </p>
                  </div>
                  <Switch
                    checked={workflowEvidenceConfig.enabled !== false}
                    onCheckedChange={(checked) =>
                      handleWorkflowConfigPatch({
                        evidence: {
                          ...workflowEvidenceConfig,
                          enabled: checked,
                          mode: String(workflowEvidenceConfig.mode ?? 'hybrid'),
                        },
                      })
                    }
                  />
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-2">
                    <span className="text-xs font-medium text-muted-foreground">Mode</span>
                    <Select
                      value={String(workflowEvidenceConfig.mode ?? 'hybrid')}
                      onValueChange={(value) =>
                        handleWorkflowConfigPatch({
                          evidence: {
                            ...workflowEvidenceConfig,
                            enabled: workflowEvidenceConfig.enabled !== false,
                            mode: value,
                          },
                        })
                      }
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="hybrid">Hybrid</SelectItem>
                        <SelectItem value="web_only">Web only</SelectItem>
                        <SelectItem value="local_only">Local only</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <span className="text-xs font-medium text-muted-foreground">Max results / query</span>
                    <Input
                      value={String(workflowEvidenceConfig.max_results_per_query ?? '')}
                      placeholder="3"
                      onChange={(event) =>
                        handleWorkflowConfigPatch({
                          evidence: {
                            ...workflowEvidenceConfig,
                            enabled: workflowEvidenceConfig.enabled !== false,
                            max_results_per_query: event.target.value
                              ? Number.parseInt(event.target.value, 10)
                              : 3,
                          },
                        })
                      }
                    />
                  </div>
                </div>
              </div>

              <div className="space-y-3 rounded-lg border border-border bg-surface-layer px-3 py-3">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="text-sm font-medium text-foreground">Schedule</p>
                    <p className="text-xs text-muted-foreground">
                      Let this workflow execute via backend scheduling instead of immediate start.
                    </p>
                  </div>
                  <Switch
                    checked={workflowScheduleEnabled(workflowState)}
                    onCheckedChange={(checked) =>
                      handleWorkflowConfigPatch({
                        schedule: buildWorkflowScheduleConfig(
                          workflowScheduleConfig,
                          workflowScheduleType,
                          checked
                        ),
                      })
                    }
                  />
                </div>
                {workflowScheduleEnabled(workflowState) ? (
                  <>
                    <div className="space-y-2">
                      <span className="text-xs font-medium text-muted-foreground">Schedule type</span>
                      <Select
                        value={workflowScheduleType}
                        onValueChange={(value) =>
                          handleWorkflowConfigPatch({
                            schedule: buildWorkflowScheduleConfig(
                              workflowScheduleConfig,
                              value as WorkflowScheduleType,
                              true
                            ),
                          })
                        }
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="interval">Interval</SelectItem>
                          <SelectItem value="once">One-shot</SelectItem>
                          <SelectItem value="cron">Cron</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    {workflowScheduleType === 'interval' ? (
                      <div className="space-y-2">
                        <span className="text-xs font-medium text-muted-foreground">Interval seconds</span>
                        <Input
                          value={String(workflowScheduleConfig.interval_seconds ?? '')}
                          placeholder="3600"
                          onChange={(event) =>
                            handleWorkflowConfigPatch({
                              schedule: {
                                ...workflowScheduleConfig,
                                enabled: true,
                                interval_seconds: event.target.value
                                  ? Number.parseInt(event.target.value, 10)
                                  : 3600,
                              },
                            })
                          }
                        />
                      </div>
                    ) : null}
                    {workflowScheduleType === 'once' ? (
                      <div className="space-y-2">
                        <span className="text-xs font-medium text-muted-foreground">Run at</span>
                        <Input
                          type="datetime-local"
                          value={formatWorkflowScheduleRunAt(workflowScheduleConfig.run_at)}
                          onChange={(event) =>
                            handleWorkflowConfigPatch({
                              schedule: {
                                ...workflowScheduleConfig,
                                enabled: true,
                                run_at: event.target.value,
                              },
                            })
                          }
                        />
                      </div>
                    ) : null}
                    {workflowScheduleType === 'cron' ? (
                      <div className="space-y-2">
                        <span className="text-xs font-medium text-muted-foreground">Cron</span>
                        <Input
                          value={String(workflowScheduleConfig.cron ?? '')}
                          placeholder="0 9 * * 1"
                          onChange={(event) =>
                            handleWorkflowConfigPatch({
                              schedule: {
                                ...workflowScheduleConfig,
                                enabled: true,
                                cron: event.target.value,
                              },
                            })
                          }
                        />
                      </div>
                    ) : null}
                    <div className="grid grid-cols-2 gap-3">
                      <div className="space-y-2">
                        <span className="text-xs font-medium text-muted-foreground">Timezone</span>
                        <Input
                          value={String(workflowScheduleConfig.timezone ?? '')}
                          placeholder={defaultScheduleTimezone()}
                          onChange={(event) =>
                            handleWorkflowConfigPatch({
                              schedule: {
                                ...workflowScheduleConfig,
                                enabled: true,
                                timezone: event.target.value,
                              },
                            })
                          }
                        />
                      </div>
                      <div className="space-y-2">
                        <span className="text-xs font-medium text-muted-foreground">Max runs</span>
                        <Input
                          value={
                            workflowScheduleConfig.max_runs === null || workflowScheduleConfig.max_runs === undefined
                              ? ''
                              : String(workflowScheduleConfig.max_runs)
                          }
                          placeholder="Optional"
                          onChange={(event) =>
                            handleWorkflowConfigPatch({
                              schedule: {
                                ...workflowScheduleConfig,
                                enabled: true,
                                max_runs: event.target.value
                                  ? Number.parseInt(event.target.value, 10)
                                  : null,
                              },
                            })
                          }
                        />
                      </div>
                    </div>
                    {workflowScheduleType !== 'once' ? (
                      <div className="flex items-center justify-between gap-3 rounded-md border border-border px-3 py-2">
                        <span className="text-xs text-muted-foreground">Start immediately</span>
                        <Switch
                          checked={workflowScheduleConfig.start_immediately !== false}
                          onCheckedChange={(checked) =>
                            handleWorkflowConfigPatch({
                              schedule: {
                                ...workflowScheduleConfig,
                                enabled: true,
                                start_immediately: checked,
                              },
                            })
                          }
                        />
                      </div>
                    ) : null}
                    <div className="flex items-center justify-between gap-3 rounded-md border border-border px-3 py-2">
                      <span className="text-xs text-muted-foreground">Auto-pause on failure</span>
                      <Switch
                        checked={workflowScheduleConfig.auto_pause_on_failure !== false}
                        onCheckedChange={(checked) =>
                          handleWorkflowConfigPatch({
                            schedule: {
                              ...workflowScheduleConfig,
                              enabled: true,
                              auto_pause_on_failure: checked,
                            },
                          })
                        }
                      />
                    </div>
                  </>
                ) : null}
              </div>
            </section>

            <Separator />

            <section className="space-y-3">
              <div>
                <p className="text-sm font-medium text-foreground">Status / recovery</p>
                <p className="text-xs text-muted-foreground">
                  Full artifacts, logs, and recovery actions stay available from the run detail page.
                </p>
              </div>
              <div className="rounded-lg border border-border bg-surface-layer px-3 py-3 text-xs text-muted-foreground">
                <p>Workflow busy: {workflowBusy ? 'Yes' : 'No'}</p>
                <p>Last error: {workflowError ?? 'None'}</p>
              </div>
            </section>

            <div className="flex items-center justify-end gap-2 pt-2">
              <Button
                type="button"
                variant="outline"
                onClick={() => setWorkflowPanelOpen(false)}
              >
                Close
              </Button>
              <Button
                type="button"
                onClick={() => void handleWorkflowSave()}
              >
                Save settings
              </Button>
            </div>
          </div>
        </SheetContent>
      </Sheet>

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
