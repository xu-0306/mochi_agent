'use client'

import * as React from 'react'
import { useRouter } from 'next/navigation'
import { AlertCircle, ListTodo, Loader2, MoreHorizontal, Settings, SlidersHorizontal, Workflow } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { ChatInput, type ChatInputModelOption } from '@/components/chat/ChatInput'
import { ChatMessage } from '@/components/chat/ChatMessage'
import { EmptyState } from '@/components/chat/EmptyState'
import { ExportDialog } from '@/components/chat/ExportDialog'
import { InferencePanel } from '@/components/chat/InferencePanel'
import { ScrollToBottom } from '@/components/chat/ScrollToBottom'
import { TaskPanel } from '@/components/chat/TaskPanel'
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
  const [selectedPresetName, setSelectedPresetName] = React.useState('default')
  const [savingPreset, setSavingPreset] = React.useState(false)
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
  const uploadTargetDir =
    projects.find((project) => project.id === effectiveProjectId)?.workspaceDir ??
    getString(settings?.paths?.workspace_dir) ??
    undefined
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

      useSessionStore.setState((state) => ({
        sessions: state.sessions.map((session) =>
          session.id === sessionId
            ? {
                ...session,
                title: detail.title || session.title,
                lastMessageAt: new Date(detail.updatedAt),
                messageCount: detail.eventCount,
                projectId: detail.projectId,
                isDraft: false,
              }
            : session
        ),
        currentSessionDetail:
          state.currentSessionDetail?.id === sessionId || state.currentSessionId === sessionId
            ? detail
            : state.currentSessionDetail,
      }))
    } catch {
      // Keep the optimistic transcript if canonical session refresh fails.
    }
  }, [setSessionMessages, updateLastMessage])

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
      resolveMessagesForSession,
      selectSession,
      setSessionMessages,
      sessions,
      startStreaming,
      t,
      syncSessionFromServer,
      updateSessionMessages,
      updateLastMessage,
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

  const handleBuiltinCommand = React.useCallback((command: 'clear' | 'settings' | 'voice' | 'model' | 'export') => {
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
    }
  }, [currentSessionId, handleVoiceEntry, router, setSessionMessages, t])

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

  const handleEditAndResend = React.useCallback(async (message: Message, nextContent: string) => {
    if (!currentSessionId || !message.turnId) {
      await handleSend(nextContent, { attachments: message.attachments ?? [] })
      return
    }

    const rewrittenSession = await api.rewriteSessionFromTurn(currentSessionId, message.turnId)
    const rewrittenMessages = api.buildMessagesFromSessionEvents(rewrittenSession.events)
    const baseMessages = rewrittenMessages.length > 0 ? rewrittenMessages : createInitialMessages(t)
    const lastRetainedMessage = [...baseMessages]
      .reverse()
      .find((entry) => entry.type === 'user' || entry.type === 'assistant')

    setSessionMessages(currentSessionId, baseMessages)
    updateLastMessage(currentSessionId, lastRetainedMessage?.content ?? '')
    void selectSession(currentSessionId)
    await handleSend(nextContent, {
      forceSessionId: currentSessionId,
      attachments: message.attachments ?? [],
    })
  }, [currentSessionId, handleSend, selectSession, setSessionMessages, t, updateLastMessage])

  const handleStarterPrompt = React.useCallback((prompt: string) => {
    void handleSend(prompt)
  }, [handleSend])

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
              variant="ghost"
              size="sm"
              title={t('sidebar.workflows')}
              onClick={() => router.push('/agent-runs')}
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
                      onEditAndResend={message.type === 'user' ? handleEditAndResend : undefined}
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

      <ChatInput
        sessionId={currentSessionId}
        projectId={effectiveProjectId}
        uploadTargetDir={uploadTargetDir}
        onSend={handleSend}
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
      />

      <ExportDialog
        open={exportOpen}
        onOpenChange={setExportOpen}
        messages={messages}
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
