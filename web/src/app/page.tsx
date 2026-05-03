'use client'

import * as React from 'react'
import { useRouter } from 'next/navigation'
import { Loader2, MoreHorizontal, Settings } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { ChatInput, type ChatInputModelOption } from '@/components/chat/ChatInput'
import { ChatMessage, type Message } from '@/components/chat/ChatMessage'
import { VoiceOverlay } from '@/components/voice/VoiceOverlay'
import * as api from '@/lib/api'
import { useI18n } from '@/lib/i18n'
import { useSessionStore } from '@/lib/stores/session-store'
import {
  VoiceWsClient,
  type VoiceRuntimePhase,
  type VoiceTurnResult,
} from '@/lib/voice-ws'

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
      model?: string
    }
  ) => Promise<unknown>
  postChat?: (payload: {
    message: string
    session_id?: string
    sessionId?: string
    model?: string
  }) => Promise<unknown>
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

function resolveModelName(model: Record<string, unknown> | null | undefined): string | null {
  if (!model) {
    return null
  }
  return (
    getString(model.name) ??
    getString(model.model) ??
    getString(model.id) ??
    getString(model.label)
  )
}

function formatModelLabel(modelId: string): string {
  return modelId.replace(/^ollama:/, '').replace(/^openai:/, '')
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
    status: ChatInputModelOption['status']
  ) => {
    if (!modelId || seen.has(modelId)) {
      return
    }
    seen.add(modelId)
    candidates.push({
      id: modelId,
      label: formatModelLabel(modelId),
      status,
    })
  }

  if (Array.isArray(payload.models)) {
    for (const entry of payload.models) {
      if (!isRecord(entry)) {
        continue
      }
      pushModel(resolveModelName(entry), 'connected')
    }
  }

  if (Array.isArray(payload.available_models)) {
    for (const entry of payload.available_models) {
      if (!isRecord(entry)) {
        continue
      }
      pushModel(resolveModelName(entry), 'connected')
    }
  }

  pushModel(resolveModelName(payload.active_model ?? undefined), 'connected')
  pushModel(getString(payload.configured_model), 'configured')

  return candidates
}

async function requestChat(
  text: string,
  sessionId: string | undefined,
  model: string | null
): Promise<BackendChatResponse> {
  const client = api as ApiCompat

  if (typeof client.postChat === 'function') {
    const response = await client.postChat({
      message: text,
      session_id: sessionId,
      sessionId,
      model: model ?? undefined,
    })
    return response as BackendChatResponse
  }

  if (typeof client.sendMessage === 'function') {
    const response = await client.sendMessage(text, {
      sessionId,
      model: model ?? undefined,
    })
    return response as BackendChatResponse
  }

  throw new Error('Chat API client is unavailable.')
}

export default function ChatPage() {
  const router = useRouter()
  const { t } = useI18n()
  const [messages, setMessages] = React.useState<Message[]>(() => createInitialMessages(t))
  const [isStreaming, setIsStreaming] = React.useState(false)
  const [modelOptions, setModelOptions] = React.useState<ChatInputModelOption[]>([])
  const [currentModel, setCurrentModel] = React.useState<string | null>(null)
  const [voiceOpen, setVoiceOpen] = React.useState(false)
  const [voicePhase, setVoicePhase] = React.useState<VoiceRuntimePhase>('idle')
  const [voiceRecording, setVoiceRecording] = React.useState(false)
  const [voicePartialTranscription, setVoicePartialTranscription] = React.useState('')
  const [voiceFinalTranscription, setVoiceFinalTranscription] = React.useState('')
  const [voiceAssistantText, setVoiceAssistantText] = React.useState('')
  const [voiceErrorMessage, setVoiceErrorMessage] = React.useState<string | null>(null)
  const scrollRef = React.useRef<HTMLDivElement>(null)
  const voiceClientRef = React.useRef<VoiceWsClient | null>(null)
  const voiceSessionIdRef = React.useRef<string | null>(null)

  const {
    sessions,
    currentSessionId,
    currentSessionDetail,
    isLoadingDetail,
    createSession,
    selectSession,
    updateLastMessage,
  } = useSessionStore()
  const currentSession = sessions.find((session) => session.id === currentSessionId)

  const scrollToBottom = React.useCallback(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [])

  React.useEffect(() => {
    scrollToBottom()
  }, [messages, scrollToBottom])

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

      const sessionId = voiceSessionIdRef.current ?? currentSessionId ?? createSession()
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
    [createSession, currentSessionId, selectSession, updateLastMessage]
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

  React.useEffect(() => {
    return () => {
      const client = voiceClientRef.current
      voiceClientRef.current = null
      if (client) {
        void client.disconnect()
      }
    }
  }, [])

  React.useEffect(() => {
    let cancelled = false

    const loadModels = async () => {
      try {
        const response = await fetch('/v1/models', { cache: 'no-store' })
        if (!response.ok) {
          throw new Error(`GET /v1/models failed: ${response.status}`)
        }
        const payload = (await response.json()) as ModelsResponse
        if (cancelled) {
          return
        }

        const nextOptions = deriveModelOptions(payload)
        const activeModel =
          resolveModelName(payload.active_model ?? undefined) ??
          getString(payload.configured_model)

        setModelOptions(nextOptions)
        setCurrentModel(activeModel)
      } catch {
        if (cancelled) {
          return
        }
        setModelOptions((prev) => prev)
      }
    }

    void loadModels()

    return () => {
      cancelled = true
    }
  }, [])

  const handleSwitchModel = React.useCallback(async (modelId: string) => {
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

    const payload = (await response.json()) as { active_model?: Record<string, unknown> }
    const nextModel = resolveModelName(payload.active_model ?? undefined) ?? modelId

    setCurrentModel(nextModel)
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
  }, [])

  const handleSend = React.useCallback(
    async (text: string) => {
      if (isStreaming) {
        return
      }

      const sessionId = currentSessionId ?? createSession()
      const userMessage: Message = {
        id: `user-${Date.now()}`,
        type: 'user',
        content: text,
        timestamp: new Date(),
      }

      setMessages((prev) => [...prev, userMessage])
      setIsStreaming(true)
      updateLastMessage(sessionId, text)

      try {
        const response = await requestChat(text, sessionId, currentModel)
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
              },
            ]

        setMessages((prev) => [...prev, ...eventMessages])

        const finalAssistantMessage = [...eventMessages]
          .reverse()
          .find((message) => message.type === 'assistant')

        if (finalAssistantMessage) {
          updateLastMessage(sessionId, finalAssistantMessage.content)
        }

        if (response.model) {
          setCurrentModel(response.model)
        }
      } catch (error) {
        const detail = error instanceof Error ? error.message : null
        setMessages((prev) => [
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
      } finally {
        setIsStreaming(false)
      }
    },
    [createSession, currentModel, currentSessionId, isStreaming, t, updateLastMessage]
  )

  const headerModelLabel =
    modelOptions.find((option) => option.id === currentModel)?.label ??
    (currentModel ? formatModelLabel(currentModel) : 'configured')

  const handleVoiceEntry = React.useCallback(async () => {
    setVoiceOpen(true)
    setVoiceErrorMessage(null)
    try {
      const sessionId = currentSessionId ?? createSession()
      voiceSessionIdRef.current = sessionId
      const client = ensureVoiceClient(sessionId)
      await client.connect()
    } catch (error) {
      const detail = error instanceof Error ? error.message : 'Voice connect failed.'
      setVoiceErrorMessage(detail)
      setVoicePhase('error')
    }
  }, [createSession, currentSessionId, ensureVoiceClient])

  const handleVoiceToggleRecording = React.useCallback(async () => {
    const sessionId = voiceSessionIdRef.current ?? currentSessionId ?? createSession()
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
  }, [createSession, currentSessionId, ensureVoiceClient, voiceRecording])

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

  return (
    <div className="flex h-full flex-col">
      <header className="flex h-12 shrink-0 items-center justify-between border-b border-border bg-canvas px-4">
        <h1 className="truncate text-sm font-semibold text-foreground">
          {displaySessionTitle(currentSession?.title, t('chat.newChat'))}
        </h1>
        <div className="flex items-center gap-1">
          <div className="mr-2 flex max-w-[180px] items-center gap-1.5 text-xs text-muted-foreground">
            {isStreaming ? (
              <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin" />
            ) : (
              <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-success" />
            )}
            <span className="truncate">{headerModelLabel}</span>
          </div>
          <Button variant="ghost" size="icon-sm" title={t('chat.moreOptions')}>
            <MoreHorizontal className="h-4 w-4" />
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
      </header>

      <div
        ref={scrollRef}
        className="flex-1 space-y-5 overflow-y-auto px-4 py-6"
      >
        {messages.map((message) => (
          <ChatMessage key={message.id} message={message} />
        ))}
      </div>

      <ChatInput
        onSend={handleSend}
        onVoice={handleVoiceEntry}
        isStreaming={isStreaming}
        disabled={false}
        models={modelOptions}
        currentModel={currentModel}
        onSwitchModel={handleSwitchModel}
      />

      <VoiceOverlay
        open={voiceOpen}
        phase={voicePhase}
        isRecording={voiceRecording}
        partialTranscription={voicePartialTranscription}
        finalTranscription={voiceFinalTranscription}
        assistantText={voiceAssistantText}
        errorMessage={voiceErrorMessage}
        onToggleRecording={handleVoiceToggleRecording}
        onInterrupt={handleVoiceInterrupt}
        onClose={handleVoiceClose}
      />
    </div>
  )
}
