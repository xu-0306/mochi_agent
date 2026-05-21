'use client'

import * as React from 'react'
import { useRouter } from 'next/navigation'
import { AlertCircle, Loader2, MoreHorizontal, Settings, SlidersHorizontal } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { ChatInput, type ChatInputModelOption } from '@/components/chat/ChatInput'
import { ChatMessage } from '@/components/chat/ChatMessage'
import { EmptyState } from '@/components/chat/EmptyState'
import { ExportDialog } from '@/components/chat/ExportDialog'
import { InferencePanel } from '@/components/chat/InferencePanel'
import { ScrollToBottom } from '@/components/chat/ScrollToBottom'
import { VoiceOverlay } from '@/components/voice/VoiceOverlay'
import * as api from '@/lib/api'
import type { Message, ReasoningStep } from '@/lib/chat'
import {
  findRegeneratePrompt,
  isConversationEffectivelyEmpty,
  type FileChangeSummary,
} from '@/lib/chat-p2'
import { useI18n } from '@/lib/i18n'
import {
  getActivePreset,
  resolveEffectiveInferenceParams,
  useInferenceStore,
} from '@/lib/stores/inference-store'
import { useProjectStore } from '@/lib/stores/project-store'
import { useSessionStore } from '@/lib/stores/session-store'
import {
  VoiceWsClient,
  type VoiceRuntimePhase,
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
      systemPrompt?: string
      temperature?: number
      maxTokens?: number
      topP?: number
      minP?: number
      topK?: number
      frequencyPenalty?: number
      presencePenalty?: number
      repeatPenalty?: number
    }
  ) => Promise<unknown>
  postChat?: (payload: {
    message: string
    session_id?: string
    sessionId?: string
    project_id?: string | null
    projectId?: string | null
    model?: string
    system_prompt?: string
    temperature?: number
    max_tokens?: number
    top_p?: number
    min_p?: number
    top_k?: number
    frequency_penalty?: number
    presence_penalty?: number
    repeat_penalty?: number
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
    label?: string | null
  ) => {
    if (!modelId || seen.has(modelId)) {
      return
    }
    seen.add(modelId)
    candidates.push({
      id: modelId,
      label: label ?? formatModelLabel(modelId),
      status,
    })
  }

  if (Array.isArray(payload.available_models)) {
    for (const entry of payload.available_models) {
      if (!isRecord(entry)) {
        continue
      }
      const modelId = resolveModelOptionId(entry)
      pushModel(modelId, 'connected', modelId ? resolveModelLabel(entry, modelId) : null)
    }
  }

  if (candidates.length === 0 && Array.isArray(payload.models)) {
    for (const entry of payload.models) {
      if (!isRecord(entry)) {
        continue
      }
      const modelId = resolveModelOptionId(entry)
      pushModel(modelId, 'connected', modelId ? resolveModelLabel(entry, modelId) : null)
    }
  }

  if (candidates.length === 0) {
    pushModel(resolveModelOptionId(payload.active_model ?? undefined), 'connected')
    pushModel(getString(payload.configured_model), 'configured')
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
      system_prompt: inference.systemPrompt,
      temperature: inference.temperature,
      max_tokens: inference.maxTokens,
      top_p: inference.topP,
      min_p: inference.minP,
      top_k: inference.topK,
      frequency_penalty: inference.frequencyPenalty,
      presence_penalty: inference.presencePenalty,
      repeat_penalty: inference.repeatPenalty,
    })
    return response as BackendChatResponse
  }

  if (typeof client.sendMessage === 'function') {
    const response = await client.sendMessage(text, {
      sessionId,
      projectId: projectId ?? undefined,
      model: model ?? undefined,
      systemPrompt: inference.systemPrompt,
      temperature: inference.temperature,
      maxTokens: inference.maxTokens,
      topP: inference.topP,
      minP: inference.minP,
      topK: inference.topK,
      frequencyPenalty: inference.frequencyPenalty,
      presencePenalty: inference.presencePenalty,
      repeatPenalty: inference.repeatPenalty,
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

function resolveVoicePhaseFromRuntime(status: api.VoiceRuntimeStatus | null): VoiceRuntimePhase | null {
  if (!status) {
    return null
  }

  if (status.error) {
    return 'error'
  }

  const phase = status.phase?.toLowerCase()
  if (
    phase === 'idle' ||
    phase === 'connecting' ||
    phase === 'ready' ||
    phase === 'listening' ||
    phase === 'transcribing' ||
    phase === 'thinking' ||
    phase === 'synthesizing' ||
    phase === 'error'
  ) {
    return phase
  }

  return status.ready ? 'ready' : 'connecting'
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

function mergeReasoningStep(steps: ReasoningStep[], nextStep: ReasoningStep): ReasoningStep[] {
  if (!nextStep.toolCallId) {
    return [...steps, nextStep]
  }

  const existingIndex = steps.findIndex(
    (step) =>
      step.toolCallId === nextStep.toolCallId &&
      (step.type === 'tool_call' || step.type === 'tool_result')
  )

  if (existingIndex === -1) {
    return [...steps, nextStep]
  }

  const existing = steps[existingIndex]
  const merged: ReasoningStep =
    nextStep.type === 'tool_result'
      ? {
          ...existing,
          ...nextStep,
          type: 'tool_result',
          status: nextStep.status,
        }
      : { ...existing, ...nextStep }

  return steps.map((step, index) => (index === existingIndex ? merged : step))
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
  const [messages, setMessages] = React.useState<Message[]>(() => createInitialMessages(t))
  const [isStreaming, setIsStreaming] = React.useState(false)
  const [modelOptions, setModelOptions] = React.useState<ChatInputModelOption[]>([])
  const [currentModel, setCurrentModel] = React.useState<string | null>(null)
  const [currentModelLoaded, setCurrentModelLoaded] = React.useState<boolean | null>(null)
  const [modelSwitchError, setModelSwitchError] = React.useState<string | null>(null)
  const [skills, setSkills] = React.useState<api.Skill[]>([])
  const [settings, setSettings] = React.useState<api.Settings | null>(null)
  const [mobileInferenceOpen, setMobileInferenceOpen] = React.useState(false)
  const [selectedPresetName, setSelectedPresetName] = React.useState('default')
  const [savingPreset, setSavingPreset] = React.useState(false)
  const [voiceOpen, setVoiceOpen] = React.useState(false)
  const [voicePhase, setVoicePhase] = React.useState<VoiceRuntimePhase>('idle')
  const [voiceRecording, setVoiceRecording] = React.useState(false)
  const [voicePartialTranscription, setVoicePartialTranscription] = React.useState('')
  const [voiceFinalTranscription, setVoiceFinalTranscription] = React.useState('')
  const [voiceAssistantText, setVoiceAssistantText] = React.useState('')
  const [voiceErrorMessage, setVoiceErrorMessage] = React.useState<string | null>(null)
  const [voiceRuntimeStatus, setVoiceRuntimeStatus] = React.useState<api.VoiceRuntimeStatus | null>(null)
  const [, setVoiceRuntimeLoading] = React.useState(false)
  const [exportOpen, setExportOpen] = React.useState(false)
  const [showScrollToBottom, setShowScrollToBottom] = React.useState(false)
  const scrollRef = React.useRef<HTMLDivElement>(null)
  const shouldAutoScrollRef = React.useRef(true)
  const voiceClientRef = React.useRef<VoiceWsClient | null>(null)
  const voiceSessionIdRef = React.useRef<string | null>(null)
  const activeAbortControllerRef = React.useRef<AbortController | null>(null)

  const {
    sessions,
    currentSessionId,
    currentSessionDetail,
    isLoadingDetail,
    createDraftSession,
    forkSessionFromTurn,
    materializeDraftSession,
    selectSession,
    updateLastMessage,
  } = useSessionStore()
  const activeProjectId = useProjectStore((state) => state.activeProjectId)
  const {
    panelOpen,
    setPanelOpen,
    sessionOverridesById,
    setSessionOverride,
    replaceSessionOverride,
    resetSessionOverride,
  } = useInferenceStore()
  const currentSession = sessions.find((session) => session.id === currentSessionId)
  const activeAgentSettings = settings?.agent
  const activePreset = getActivePreset(activeAgentSettings)
  const sessionOverride = currentSessionId ? sessionOverridesById[currentSessionId] : undefined
  const effectiveInference = React.useMemo(
    () => resolveEffectiveInferenceParams(sessionOverride, activeAgentSettings),
    [activeAgentSettings, sessionOverride]
  )

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
        const [nextSettings, nextSkills] = await Promise.all([
          api.fetchSettings(),
          api.fetchSkills({ limit: 50 }),
        ])
        if (cancelled) {
          return
        }
        setSettings(nextSettings)
        setSkills(nextSkills)
      } catch {
        if (!cancelled) {
          setSkills([])
        }
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
    if (!currentSessionId) {
      setMessages(createInitialMessages(t))
      return
    }

    if (currentSessionDetail?.id === currentSessionId) {
      const replayMessages = api.buildMessagesFromSessionEvents(currentSessionDetail.events)
      setMessages(replayMessages.length > 0 ? replayMessages : createInitialMessages(t))
      return
    }

    if (isLoadingDetail) {
      setMessages([
        {
          id: `loading-${currentSessionId}`,
          type: 'system',
          content: t('chat.loadingSession'),
          timestamp: new Date(),
        },
      ])
    }
  }, [currentSessionDetail, currentSessionId, isLoadingDetail, t])

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
        setMessages((prev) => [...prev, ...newMessages])
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
    [activeProjectId, createDraftSession, currentSessionId, selectSession, updateLastMessage]
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
      onError: (message, code) => {
        setVoiceErrorMessage(code ? `${message} (${code})` : message)
        setVoiceRecording(false)
      },
    })
    voiceClientRef.current = client
    return client
  }, [appendVoiceMessages])

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
      activeAbortControllerRef.current?.abort()
      const client = voiceClientRef.current
      voiceClientRef.current = null
      if (client) {
        void client.disconnect()
      }
    }
  }, [])

  const loadModels = React.useCallback(async (signal?: AbortSignal) => {
    const response = await fetch('/v1/models', {
      cache: 'no-store',
      signal,
    })
    if (!response.ok) {
      throw new Error(`GET /v1/models failed: ${response.status}`)
    }

    const payload = (await response.json()) as ModelsResponse
    const nextOptions = deriveModelOptions(payload)
    const activeModel = resolveActiveModelId(payload, nextOptions)
    const activeModelMetadata = isRecord(payload.active_model?.metadata) ? payload.active_model?.metadata : null
    const loaded =
      activeModelMetadata && typeof activeModelMetadata.loaded === 'boolean'
        ? activeModelMetadata.loaded
        : null

    setModelOptions(nextOptions)
    setCurrentModel(activeModel)
    setCurrentModelLoaded(loaded)
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

      await response.json()
      const nextSettings = await api.fetchSettings()
      const nextModel = modelId

      setCurrentModel(nextModel)
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

  const handleSend = React.useCallback(
    async (
      text: string,
      options?: {
        forceSessionId?: string
      }
    ) => {
      if (isStreaming) {
        return
      }

      const targetSessionId = options?.forceSessionId ?? currentSessionId
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
        timestamp: new Date(),
      }

      const placeholderContent = isLocalModelId(currentModel) && currentModelLoaded === false
        ? t('chat.loadingLocalModel')
        : ''

      setMessages((prev) => [
        ...prev,
        userMessage,
        buildTurnPlaceholder(turnKey, { content: placeholderContent }),
      ])
      setIsStreaming(true)
      updateLastMessage(sessionId, text)
      const abortController = new AbortController()
      activeAbortControllerRef.current = abortController

      try {
        let streamed = false
        let latestAssistantContent = ''
        try {
          for await (const chunk of api.streamChatMessages(text, {
            sessionId,
            projectId:
              sessions.find((session) => session.id === sessionId)?.projectId ?? activeProjectId ?? null,
            model: currentModel ?? undefined,
            systemPrompt: effectiveInference.systemPrompt,
            temperature: effectiveInference.temperature,
            maxTokens: effectiveInference.maxTokens,
            topP: effectiveInference.topP,
            minP: effectiveInference.minP,
            topK: effectiveInference.topK,
            frequencyPenalty: effectiveInference.frequencyPenalty,
            presencePenalty: effectiveInference.presencePenalty,
            repeatPenalty: effectiveInference.repeatPenalty,
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

            setMessages((prev) => applyStreamChunk(prev, chunk, turnKey))
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

          setMessages((prev) => [
            ...prev.filter((message) => message.turnKey !== turnKey),
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
      } catch (error) {
        if (error instanceof DOMException && error.name === 'AbortError') {
          setMessages((prev) => prev.map((message) =>
            message.turnKey === turnKey ? { ...message, isStreaming: false } : message
          ))
          return
        }
        const detail = error instanceof Error ? error.message : null
        setMessages((prev) => [
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
        activeAbortControllerRef.current = null
        setIsStreaming(false)
      }
    },
    [
      activeProjectId,
      createDraftSession,
      currentModel,
      currentSessionId,
      effectiveInference,
      isStreaming,
      materializeDraftSession,
      selectSession,
      sessions,
      t,
      updateLastMessage,
    ]
  )

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
    setVoiceErrorMessage(null)
    try {
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
    activeAbortControllerRef.current?.abort()
  }, [])

  const handleBuiltinCommand = React.useCallback((command: 'clear' | 'settings' | 'voice' | 'model' | 'export') => {
    if (command === 'clear') {
      setMessages(createInitialMessages(t))
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
  }, [handleVoiceEntry, router, t])

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
      await handleSend(nextContent)
      return
    }

    const currentSession = sessions.find((session) => session.id === currentSessionId)
    const forkedSessionId = await forkSessionFromTurn(
      currentSessionId,
      message.turnId,
      currentSession?.projectId
    )
    await handleSend(nextContent, { forceSessionId: forkedSessionId })
  }, [currentSessionId, forkSessionFromTurn, handleSend, sessions])

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
              size="icon-sm"
              title={t('chat.moreOptions')}
              onClick={() => setExportOpen(true)}
            >
              <MoreHorizontal className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon-sm"
              title="Inference"
              onClick={() => {
                if (window.innerWidth < 768) {
                  setMobileInferenceOpen(true)
                } else {
                  setPanelOpen(!panelOpen)
                }
              }}
            >
              <SlidersHorizontal className="h-4 w-4" />
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

      <div className="flex flex-1 overflow-hidden">
        <div className="relative flex-1">
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
        onSend={handleSend}
        onStop={handleStopGeneration}
        onVoice={handleVoiceEntry}
        onBuiltinCommand={handleBuiltinCommand}
        isStreaming={isStreaming}
        disabled={false}
        models={modelOptions}
        currentModel={currentModel}
        skills={skills}
        onSwitchModel={handleSwitchModel}
      />

      <ExportDialog
        open={exportOpen}
        onOpenChange={setExportOpen}
        messages={messages}
      />

      <VoiceOverlay
        open={voiceOpen}
        phase={resolveVoicePhaseFromRuntime(voiceRuntimeStatus) ?? voicePhase}
        isRecording={voiceRecording}
        partialTranscription={voicePartialTranscription}
        finalTranscription={voiceFinalTranscription}
        assistantText={voiceAssistantText}
        errorMessage={voiceErrorMessage ?? voiceRuntimeStatus?.error ?? null}
        onToggleRecording={handleVoiceToggleRecording}
        onInterrupt={handleVoiceInterrupt}
        onClose={handleVoiceClose}
      />
    </div>
  )
}
