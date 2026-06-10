'use client'

import * as React from 'react'
import {
  BrainCircuit,
  Check,
  ChevronDown,
  Loader2,
  Mic,
  Paperclip,
  Send,
  Square,
  X,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import { useI18n } from '@/lib/i18n'
import * as api from '@/lib/api'
import type { InferenceParams } from '@/lib/stores/inference-store'
import {
  buildBuiltinActions,
  CommandPalette,
  type CommandPaletteAction,
} from './CommandPalette'
import { ChatAttachments } from './ChatAttachments'
import type {
  ChatContextSnapshot,
  LocalActiveModelRuntimeStatus,
  Skill,
} from '@/lib/api'
import type { ChatAttachment } from '@/lib/chat'

export interface ChatInputModelOption {
  id: string
  label: string
  detail?: string | null
  status?: 'connected' | 'configured' | 'disconnected'
}

interface ChatInputProps {
  sessionId?: string | null
  projectId?: string | null
  uploadTargetDir?: string
  onSend: (text: string, options?: { selectedSkillIds?: string[]; attachments?: ChatAttachment[] }) => void
  onStop?: () => void
  onVoice?: () => void
  onBuiltinCommand?: (command: 'clear' | 'settings' | 'voice' | 'model' | 'export') => void
  disabled?: boolean
  isStreaming?: boolean
  models?: ChatInputModelOption[]
  currentModel?: string | null
  onSearchSkills?: (query: string) => Promise<Skill[]>
  onSwitchModel?: (modelId: string) => void | Promise<void>
  activeLocalRuntimeStatus?: LocalActiveModelRuntimeStatus | null
  onUnloadCurrentModel?: () => void | Promise<void>
  isUnloadingCurrentModel?: boolean
  inference: InferenceParams
  reasoningOptions?: api.ReasoningEffort[]
  onReasoningEffortChange?: (value: api.ReasoningEffort | null) => void
}

interface AttachedFile extends ChatAttachment {}

function parseSlashQuery(nextValue: string, caret: number | null): string | null {
  const textBeforeCursor = caret === null ? nextValue : nextValue.slice(0, caret)
  const lineStart = Math.max(0, textBeforeCursor.lastIndexOf('\n') + 1)
  const currentToken = textBeforeCursor.slice(lineStart).trimStart()
  if (!currentToken.startsWith('/')) {
    return null
  }
  return currentToken.slice(1).trim()
}

function buildSkillAction(skill: Skill): CommandPaletteAction {
  return {
    kind: 'skill',
    id: skill.id,
    label: `/${skill.name}`,
    description: skill.description,
  }
}

function formatTokenCount(value: number | null | undefined): string {
  return new Intl.NumberFormat().format(value ?? 0)
}

function formatSelectedModelText(model: ChatInputModelOption | null, fallback: string): string {
  if (!model) {
    return fallback
  }
  return model.detail ? `${model.label} · ${model.detail}` : model.label
}

function formatMountedModelName(modelSpec: string | null | undefined, fallback: string | null | undefined): string {
  const raw = modelSpec ?? fallback ?? 'Local model'
  return raw.split(/[\\/]/).pop() ?? raw
}

function formatReasoningEffortLabel(value: api.ReasoningEffort): string {
  if (value === 'xhigh') {
    return 'Extra High'
  }
  if (value === 'minimal') {
    return 'Minimal'
  }
  if (value === 'none') {
    return 'None'
  }
  return value.charAt(0).toUpperCase() + value.slice(1)
}

export function ChatInput({
  sessionId,
  projectId,
  uploadTargetDir,
  onSend,
  onStop,
  onVoice,
  onBuiltinCommand,
  disabled = false,
  isStreaming = false,
  models,
  currentModel,
  onSearchSkills,
  onSwitchModel,
  activeLocalRuntimeStatus,
  onUnloadCurrentModel,
  isUnloadingCurrentModel = false,
  inference,
  reasoningOptions,
  onReasoningEffortChange,
}: ChatInputProps) {
  const { t } = useI18n()
  const [value, setValue] = React.useState('')
  const [showModelMenu, setShowModelMenu] = React.useState(false)
  const [paletteIndex, setPaletteIndex] = React.useState(0)
  const [paletteSkillActions, setPaletteSkillActions] = React.useState<CommandPaletteAction[]>([])
  const [paletteLoading, setPaletteLoading] = React.useState(false)
  const [selectedSkills, setSelectedSkills] = React.useState<Array<{ id: string; name: string }>>([])
  const [paletteDismissed, setPaletteDismissed] = React.useState(false)
  const [contextSnapshot, setContextSnapshot] = React.useState<ChatContextSnapshot | null>(null)
  const [contextLoading, setContextLoading] = React.useState(false)
  const [contextError, setContextError] = React.useState<string | null>(null)
  const [contextOpen, setContextOpen] = React.useState(false)
  const [attachedFiles, setAttachedFiles] = React.useState<AttachedFile[]>([])
  const [isUploadingFiles, setIsUploadingFiles] = React.useState(false)
  const [uploadError, setUploadError] = React.useState<string | null>(null)
  const [dragActive, setDragActive] = React.useState(false)
  const textareaRef = React.useRef<HTMLTextAreaElement>(null)
  const fileInputRef = React.useRef<HTMLInputElement>(null)
  const modelMenuRef = React.useRef<HTMLDivElement>(null)
  const contextMenuRef = React.useRef<HTMLDivElement>(null)
  const contextRequestSeqRef = React.useRef(0)
  const dragDepthRef = React.useRef(0)
  const slashQuery = React.useMemo(() => parseSlashQuery(value, null), [value])
  const paletteOpen = slashQuery !== null && !paletteDismissed
  const paletteQuery = slashQuery ?? ''

  const paletteActions = React.useMemo(() => {
    const q = paletteQuery.trim().toLowerCase()
    const builtins = buildBuiltinActions().filter((action) => {
      if (!q) {
        return true
      }
      return action.label.toLowerCase().includes(`/${q}`) || action.description.toLowerCase().includes(q)
    })
    return [...builtins, ...paletteSkillActions]
  }, [paletteQuery, paletteSkillActions])

  const selectedPaletteAction = paletteActions[
    Math.min(paletteIndex, Math.max(0, paletteActions.length - 1))
  ]
  const contextUsagePercent = contextSnapshot
    ? Math.max(0, Math.min(100, Math.round(contextSnapshot.usage_ratio * 100)))
    : 0
  const contextDisplayPercent =
    activeLocalRuntimeStatus?.hasActiveLocalModel && activeLocalRuntimeStatus.loaded === false
      ? 100
      : contextUsagePercent
  const hasReasoningSelector = Boolean(reasoningOptions && reasoningOptions.length > 0)

  const availableModels = React.useMemo<ChatInputModelOption[]>(() => {
    if (models && models.length > 0) {
      return models
    }
    if (currentModel) {
      return [{ id: currentModel, label: currentModel, status: 'configured' }]
    }
    return []
  }, [currentModel, models])

  const selectedModel = React.useMemo<ChatInputModelOption | null>(() => {
    if (availableModels.length === 0) {
      return currentModel
        ? { id: currentModel, label: currentModel, status: 'configured' }
        : null
    }
    return (
      availableModels.find((model) => model.id === currentModel) ?? availableModels[0]
    )
  }, [availableModels, currentModel])

  React.useEffect(() => {
    const handlePointerDown = (event: MouseEvent) => {
      if (!modelMenuRef.current?.contains(event.target as Node)) {
        setShowModelMenu(false)
      }
    }

    document.addEventListener('mousedown', handlePointerDown)
    return () => document.removeEventListener('mousedown', handlePointerDown)
  }, [])

  React.useEffect(() => {
    if (!contextOpen) {
      return
    }

    const handlePointerDown = (event: MouseEvent) => {
      if (!contextMenuRef.current?.contains(event.target as Node)) {
        setContextOpen(false)
      }
    }

    document.addEventListener('mousedown', handlePointerDown)
    return () => document.removeEventListener('mousedown', handlePointerDown)
  }, [contextOpen])

  React.useEffect(() => {
    if (!paletteOpen) {
      setPaletteSkillActions([])
      setPaletteLoading(false)
      return
    }

    let cancelled = false
    const nextQuery = paletteQuery.trim()
    const timer = window.setTimeout(() => {
      setPaletteLoading(true)
      void (async () => {
        try {
          const results = await onSearchSkills?.(nextQuery)
          if (cancelled) {
            return
          }
          setPaletteSkillActions((results ?? []).map(buildSkillAction))
        } catch {
          if (!cancelled) {
            setPaletteSkillActions([])
          }
        } finally {
          if (!cancelled) {
            setPaletteLoading(false)
          }
        }
      })()
    }, 150)

    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [onSearchSkills, paletteOpen, paletteQuery])

  React.useEffect(() => {
    const controller = new AbortController()
    const sequence = ++contextRequestSeqRef.current
    const timer = window.setTimeout(() => {
      setContextLoading(true)
      setContextError(null)
      void (async () => {
        try {
          const snapshot = await api.fetchChatContextPreview({
            message: value,
            session_id: sessionId ?? undefined,
            project_id: projectId,
            model: currentModel ?? undefined,
            selected_skill_ids: selectedSkills.map((skill) => skill.id),
            attachments: attachedFiles,
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
            signal: controller.signal,
          })
          if (sequence !== contextRequestSeqRef.current) {
            return
          }
          setContextSnapshot(snapshot)
        } catch (error) {
          if (sequence !== contextRequestSeqRef.current) {
            return
          }
          setContextSnapshot(null)
          setContextError(error instanceof Error ? error.message : 'Failed to load context preview.')
        } finally {
          if (sequence === contextRequestSeqRef.current) {
            setContextLoading(false)
          }
        }
      })()
    }, 250)

    return () => {
      controller.abort()
      window.clearTimeout(timer)
    }
  }, [
    currentModel,
    inference.frequencyPenalty,
    inference.maxTokens,
    inference.minP,
    inference.presencePenalty,
    inference.reasoningEffort,
    inference.repeatPenalty,
    inference.systemPrompt,
    inference.temperature,
    inference.topK,
    inference.topP,
    projectId,
    selectedSkills,
    sessionId,
    value,
    attachedFiles,
  ])

  const handleAttachFiles = React.useCallback(async (incomingFiles: FileList | File[]) => {
    const files = Array.from(incomingFiles)
    if (files.length === 0) {
      return
    }

    if (!uploadTargetDir) {
      setUploadError('Workspace path is not ready yet. Please try again in a moment.')
      return
    }

    setUploadError(null)
    setIsUploadingFiles(true)
    try {
      const uploadedFiles: AttachedFile[] = []
      for (const [index, file] of files.entries()) {
        const result = await api.importFilesystemFiles({
          files: [file],
          targetDir: uploadTargetDir,
          packageName: `${Date.now()}-${index + 1}-${file.name}`,
        })
        uploadedFiles.push({
          id: `${Date.now()}-${index}-${file.name}`,
          name: file.name,
          path: result.importedPath,
          size: file.size,
          contentType: file.type || null,
        })
      }

      setAttachedFiles((prev) => [...prev, ...uploadedFiles])
    } catch (error) {
      setUploadError(error instanceof Error ? error.message : 'Failed to upload files.')
    } finally {
      setIsUploadingFiles(false)
    }
  }, [uploadTargetDir])

  const handleSend = () => {
    const trimmed = value.trim()
    if ((trimmed.length === 0 && attachedFiles.length === 0) || disabled || isStreaming || isUploadingFiles) {
      return
    }
    onSend(trimmed, {
      selectedSkillIds: selectedSkills.map((skill) => skill.id),
      attachments: attachedFiles,
    })
    setValue('')
    setSelectedSkills([])
    setAttachedFiles([])
    setUploadError(null)
    setPaletteDismissed(false)
    textareaRef.current?.focus()
  }

  const handlePaletteSelect = React.useCallback((action: CommandPaletteAction) => {
    if (action.kind === 'builtin') {
      setPaletteDismissed(true)
      onBuiltinCommand?.(action.id)
      return
    }

    setSelectedSkills((prev) => {
      if (prev.some((skill) => skill.id === action.id)) {
        return prev
      }
      return [...prev, { id: action.id, name: action.label.replace(/^\//, '') }]
    })

    const textarea = textareaRef.current
    if (textarea) {
      const start = textarea.selectionStart
      const end = textarea.selectionEnd
      const before = value.slice(0, start)
      const after = value.slice(end)
      const lineStart = Math.max(0, before.lastIndexOf('\n') + 1)
      const updated = `${before.slice(0, lineStart)}${after}`.replace(/^\s+/, '')
      setValue(updated)
    }

    setPaletteDismissed(false)
    textareaRef.current?.focus()
  }, [onBuiltinCommand, value])

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (isStreaming && e.key === 'Escape') {
      e.preventDefault()
      onStop?.()
      return
    }

    if (paletteOpen) {
      if (e.key === 'ArrowDown') {
        e.preventDefault()
        setPaletteIndex((current) =>
          paletteActions.length === 0 ? 0 : Math.min(current + 1, paletteActions.length - 1)
        )
        return
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault()
        setPaletteIndex((current) => Math.max(0, current - 1))
        return
      }
      if (e.key === 'Enter' && selectedPaletteAction) {
        e.preventDefault()
        handlePaletteSelect(selectedPaletteAction)
        return
      }
      if (e.key === 'Escape') {
        e.preventDefault()
        setPaletteDismissed(true)
        return
      }
    }

    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      if (paletteOpen) {
        return
      }
      if (isStreaming) {
        onStop?.()
      } else {
        handleSend()
      }
    }
  }

  return (
    <div className="relative border-t border-border bg-canvas pt-3 pb-4">
      <div className="mx-auto w-full max-w-[960px] px-4">
        <CommandPalette
          open={paletteOpen}
          query={paletteQuery}
          actions={paletteActions}
          loading={paletteLoading}
          selectedIndex={paletteIndex}
          onSelectedIndexChange={setPaletteIndex}
          onSelect={handlePaletteSelect}
        />
        <div
          className={cn(
            'flex flex-col rounded-xl border border-border bg-surface-layer',
            'transition-all duration-150',
            'focus-within:border-primary-500 focus-within:ring-[3px] focus-within:ring-primary-500/20',
            dragActive && 'border-primary-500 ring-[3px] ring-primary-500/20'
          )}
          onDragEnter={(event) => {
            event.preventDefault()
            dragDepthRef.current += 1
            setDragActive(true)
          }}
          onDragOver={(event) => {
            event.preventDefault()
            event.dataTransfer.dropEffect = 'copy'
            setDragActive(true)
          }}
          onDragLeave={(event) => {
            event.preventDefault()
            dragDepthRef.current = Math.max(0, dragDepthRef.current - 1)
            if (dragDepthRef.current === 0) {
              setDragActive(false)
            }
          }}
          onDrop={(event) => {
            event.preventDefault()
            dragDepthRef.current = 0
            setDragActive(false)
            if (event.dataTransfer.files.length > 0) {
              void handleAttachFiles(event.dataTransfer.files)
            }
          }}
        >
          {selectedSkills.length > 0 ? (
            <div className="flex flex-wrap gap-1.5 px-3 pt-2">
              {selectedSkills.map((skill) => (
                <button
                  key={skill.id}
                  type="button"
                  onClick={() => {
                    setSelectedSkills((prev) => prev.filter((item) => item.id !== skill.id))
                  }}
                  className="inline-flex items-center gap-1 rounded-full border border-border bg-elevated-layer px-2 py-0.5 text-[11px] text-foreground"
                >
                  <span className="max-w-[180px] truncate">{skill.name}</span>
                  <X className="h-3 w-3 text-muted-foreground" />
                </button>
              ))}
            </div>
          ) : null}

          {attachedFiles.length > 0 ? (
            <div className="px-3 pt-2">
              <ChatAttachments
                attachments={attachedFiles}
                variant="composer"
                onRemove={(attachment) => {
                  setAttachedFiles((prev) => prev.filter((item) => item.id !== attachment.id))
                }}
              />
            </div>
          ) : null}

          <Textarea
            id="chat-input-textarea"
            ref={textareaRef}
            value={value}
            onChange={(e) => {
              const nextValue = e.target.value
              setValue(nextValue)
              setPaletteDismissed(false)
            }}
            onKeyDown={handleKeyDown}
            placeholder={t('chat.input.placeholder')}
            disabled={disabled}
            autoResize
            minRows={1}
            maxRows={8}
            className={cn(
              'border-0 bg-transparent px-4 pt-3 pb-2 text-sm resize-none',
              'focus-visible:ring-0 focus-visible:border-0 focus-visible:ring-offset-0',
              'min-h-[44px]'
            )}
          />

          <input
            ref={fileInputRef}
            type="file"
            multiple
            className="hidden"
            onChange={(event) => {
              const files = event.target.files
              if (files && files.length > 0) {
                void handleAttachFiles(files)
              }
              event.target.value = ''
            }}
          />

          <div className="flex items-center justify-between px-3 pb-2">
            <div className="flex items-center gap-1">
              <Button
                variant="ghost"
                size="icon-sm"
                title={t('chat.input.attachFile')}
                aria-label={t('chat.input.attachFile')}
                onClick={() => fileInputRef.current?.click()}
                disabled={disabled || isUploadingFiles}
                className="h-7 w-7 text-muted-foreground hover:text-foreground"
              >
                {isUploadingFiles ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <Paperclip className="h-3.5 w-3.5" />
                )}
              </Button>

              <Button
                variant="ghost"
                size="icon-sm"
                title={t('chat.input.voice')}
                aria-label={t('chat.input.voice')}
                onClick={onVoice}
                disabled={disabled}
                className="h-7 w-7 text-muted-foreground hover:text-foreground"
              >
                <Mic className="h-3.5 w-3.5" />
              </Button>

              <div className="relative" ref={contextMenuRef}>
                <button
                  type="button"
                  onClick={() => setContextOpen((open) => !open)}
                  title="Context budget"
                  aria-label="Context budget"
                  className={cn(
                    'flex h-7 items-center gap-1 rounded-md border border-border/70 bg-canvas/80 px-2 text-[11px]',
                    'text-muted-foreground transition-colors duration-150 hover:bg-elevated-layer hover:text-foreground'
                  )}
                >
                  {contextLoading && !contextSnapshot ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  ) : (
                    <BrainCircuit className="h-3.5 w-3.5" />
                  )}
                  <span className="font-mono tabular-nums">
                    {contextSnapshot ? `${contextDisplayPercent}%` : '--'}
                  </span>
                </button>

                {contextOpen ? (
                  <div
                    className={cn(
                      'absolute bottom-full left-0 z-50 mb-2 w-80 max-w-[calc(100vw-2rem)]',
                      'rounded-lg border border-border bg-elevated-layer p-3 text-left shadow-md',
                      'animate-slide-up'
                    )}
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
                          Context Tracker
                        </p>
                        <p className="mt-1 max-w-[190px] truncate text-xs text-foreground">
                          {contextSnapshot?.model ?? currentModel ?? 'Current model'}
                        </p>
                      </div>
                      <span className="font-mono text-lg font-semibold tabular-nums text-foreground">
                        {contextSnapshot ? `${contextDisplayPercent}%` : '--'}
                      </span>
                    </div>

                    {contextSnapshot ? (
                      <div className="mt-3 space-y-3">
                        <div className="grid grid-cols-2 gap-2 text-xs">
                          <div className="rounded-md border border-border bg-canvas px-2 py-1.5">
                            <p className="text-[10px] text-muted-foreground">Next prompt</p>
                            <p className="font-mono text-foreground">
                              {formatTokenCount(contextSnapshot.estimated_prompt_tokens)}
                            </p>
                          </div>
                          <div className="rounded-md border border-border bg-canvas px-2 py-1.5">
                            <p className="text-[10px] text-muted-foreground">Output reserve</p>
                            <p className="font-mono text-foreground">
                              {formatTokenCount(contextSnapshot.reserved_output_tokens)}
                            </p>
                          </div>
                          <div className="rounded-md border border-border bg-canvas px-2 py-1.5">
                            <p className="text-[10px] text-muted-foreground">Remaining</p>
                            <p className="font-mono text-foreground">
                              {formatTokenCount(contextSnapshot.remaining_tokens)}
                            </p>
                          </div>
                          <div className="rounded-md border border-border bg-canvas px-2 py-1.5">
                            <p className="text-[10px] text-muted-foreground">Window</p>
                            <p className="font-mono text-foreground">
                              {formatTokenCount(contextSnapshot.context_length)}
                            </p>
                          </div>
                        </div>

                        <div className="space-y-1.5 text-[11px]">
                          <div className="flex items-center justify-between gap-3 text-muted-foreground">
                            <span>Session cumulative</span>
                            <span>
                              {contextSnapshot.compaction_triggered
                                ? `Compacted (${contextSnapshot.compaction_reason ?? 'history'})`
                                : 'Not compacted'}
                            </span>
                          </div>
                          {[
                            ['History', contextSnapshot.history_tokens],
                            ['Summary', contextSnapshot.summary_tokens],
                            ['Memory', contextSnapshot.memory_tokens],
                            ['Skills', contextSnapshot.skills_tokens],
                            ['Tools', contextSnapshot.tool_tokens],
                            ['Draft', contextSnapshot.draft_tokens],
                          ].map(([label, tokens]) => (
                            <div key={label} className="flex items-center justify-between gap-3">
                              <span className="text-muted-foreground">{label}</span>
                              <span className="font-mono text-foreground">{formatTokenCount(Number(tokens))}</span>
                            </div>
                          ))}
                        </div>

                        {contextSnapshot.approximate ? (
                          <p className="text-[10px] leading-4 text-muted-foreground">
                            Token counts are estimated from the active backend and may vary slightly from provider billing.
                          </p>
                        ) : null}
                      </div>
                    ) : (
                      <p className="mt-3 text-xs leading-5 text-muted-foreground">
                        {contextLoading ? 'Loading context preview...' : contextError ?? 'Context preview unavailable.'}
                      </p>
                    )}
                  </div>
                ) : null}
              </div>

              {hasReasoningSelector ? (
                <div
                  className={cn(
                    'flex h-7 items-center gap-1 rounded-md border border-border/70 bg-canvas/80 px-1.5',
                    'transition-colors duration-150 hover:bg-elevated-layer'
                  )}
                >
                  <span className="whitespace-nowrap px-1 text-[11px] text-muted-foreground">
                    Reasoning
                  </span>
                  <Select
                    value={inference.reasoningEffort ?? 'auto'}
                    onValueChange={(next) =>
                      onReasoningEffortChange?.(
                        next === 'auto' ? null : (next as api.ReasoningEffort)
                      )
                    }
                  >
                    <SelectTrigger
                      aria-label="Reasoning Effort"
                      disabled={disabled}
                      className={cn(
                        'h-6 min-w-[92px] border-0 bg-transparent px-1.5 py-0 text-[11px] text-foreground shadow-none',
                        'focus:border-0 focus:ring-0 focus:ring-transparent'
                      )}
                    >
                      <SelectValue placeholder="Auto" />
                    </SelectTrigger>
                    <SelectContent className="min-w-[120px]">
                      <SelectItem value="auto">Auto</SelectItem>
                      {reasoningOptions?.map((option) => (
                        <SelectItem key={option} value={option}>
                          {formatReasoningEffortLabel(option)}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              ) : null}

              <div className="relative" ref={modelMenuRef}>
                <button
                  type="button"
                  onClick={() => setShowModelMenu((v) => !v)}
                  title={selectedModel?.label ?? t('chat.input.currentModel')}
                  aria-label={t('chat.input.currentModel')}
                  id="chat-model-selector"
                  data-chat-model-selector="true"
                  className={cn(
                    'flex h-7 max-w-[180px] items-center gap-1.5 rounded-md px-2 text-xs',
                    'text-muted-foreground hover:text-foreground hover:bg-elevated-layer',
                    'transition-colors duration-150'
                  )}
                  disabled={disabled || availableModels.length === 0}
                >
                  <span
                    className={cn(
                      'h-1.5 w-1.5 shrink-0 rounded-full',
                      selectedModel?.status === 'connected' ? 'bg-success' : 'bg-muted-foreground'
                    )}
                  />
                  <span className="min-w-0 flex-1 truncate">
                    {formatSelectedModelText(selectedModel, t('chat.input.currentModel'))}
                  </span>
                  <ChevronDown className="h-3 w-3 shrink-0" />
                </button>

                {showModelMenu && availableModels.length > 0 && (
                  <div
                    className={cn(
                      'absolute bottom-full left-0 mb-1 z-50 w-60 overflow-hidden',
                      'bg-elevated-layer border border-border rounded-lg shadow-md py-1',
                      'animate-slide-up'
                    )}
                  >
                    {availableModels.map((model) => (
                      <button
                        key={model.id}
                        type="button"
                        onClick={() => {
                          setShowModelMenu(false)
                          if (model.id !== selectedModel?.id) {
                            void onSwitchModel?.(model.id)
                          }
                        }}
                        className={cn(
                          'flex w-full items-center gap-2 px-3 py-1.5 text-sm',
                          'hover:bg-muted transition-colors duration-100',
                          selectedModel?.id === model.id
                            ? 'text-foreground'
                            : 'text-muted-foreground'
                        )}
                      >
                        <span
                          className={cn(
                            'h-1.5 w-1.5 rounded-full shrink-0',
                            model.status === 'connected' ? 'bg-success' : 'bg-muted-foreground/40'
                          )}
                        />
                        <span className="min-w-0 flex-1 text-left">
                          <span className="block truncate">
                            {model.label}
                          </span>
                          {model.detail ? (
                            <span className="mt-0.5 block truncate text-[11px] text-muted-foreground/80">
                              {model.detail}
                            </span>
                          ) : null}
                        </span>
                        {selectedModel?.id === model.id ? (
                          <Check className="h-3.5 w-3.5 shrink-0 text-primary-400" />
                        ) : null}
                      </button>
                    ))}
                    {activeLocalRuntimeStatus?.hasActiveLocalModel ? (
                      <div className="mt-1 border-t border-border px-2 pt-2 pb-1">
                        <div
                          className={cn(
                            'mb-2 flex items-center justify-between gap-3 rounded-md border px-3 py-2',
                            activeLocalRuntimeStatus.loaded
                              ? 'border-border bg-canvas'
                              : 'border-border/70 bg-muted/25'
                          )}
                        >
                          <div className="flex min-w-0 items-center gap-2">
                            <span
                              className={cn(
                                'h-1.5 w-1.5 shrink-0 rounded-full',
                                activeLocalRuntimeStatus.loaded ? 'bg-success' : 'bg-muted-foreground/40'
                              )}
                            />
                            <span className="min-w-0 truncate text-[11px] font-medium text-foreground">
                              {formatMountedModelName(
                                activeLocalRuntimeStatus.modelSpec,
                                selectedModel?.label ?? currentModel
                              )}
                            </span>
                          </div>
                          <span
                            className={cn(
                              'inline-flex shrink-0 items-center rounded-full border px-2 py-0.5 text-[10px] font-medium tracking-[0.04em]',
                              activeLocalRuntimeStatus.loaded
                                ? 'border-border/70 bg-canvas text-foreground/80'
                                : 'border-border/70 bg-muted/40 text-muted-foreground'
                            )}
                          >
                            {activeLocalRuntimeStatus.loaded ? 'Mounted' : 'Not mounted'}
                          </span>
                        </div>
                        <Button
                          type="button"
                          variant="outline"
                          size="sm"
                          className="h-8 w-full justify-center text-xs"
                          onClick={() => {
                            setShowModelMenu(false)
                            void onUnloadCurrentModel?.()
                          }}
                          disabled={
                            !activeLocalRuntimeStatus.canUnload ||
                            !activeLocalRuntimeStatus.loaded ||
                            isUnloadingCurrentModel
                          }
                        >
                          {isUnloadingCurrentModel ? (
                            <Loader2 className="h-3.5 w-3.5 animate-spin" />
                          ) : null}
                          Unload Current Model
                        </Button>
                      </div>
                    ) : null}
                  </div>
                )}
              </div>
            </div>

            <Button
              variant={isStreaming || value.trim() || attachedFiles.length > 0 ? 'primary' : 'ghost'}
              size="icon-sm"
              onClick={isStreaming ? onStop : handleSend}
              disabled={disabled || isUploadingFiles || (isStreaming ? !onStop : (!value.trim() && attachedFiles.length === 0))}
              title={isStreaming ? 'Stop generation' : t('chat.input.send')}
              aria-label={isStreaming ? 'Stop generation' : t('chat.input.send')}
              className="h-7 w-7 shrink-0"
            >
              {isStreaming ? (
                <Square className="h-3.5 w-3.5 fill-current" />
              ) : (
                <Send className="h-3.5 w-3.5" />
              )}
            </Button>
          </div>

          {(dragActive || isUploadingFiles || uploadError || attachedFiles.length > 0) ? (
            <div className="flex min-h-8 items-center justify-between gap-3 border-t border-border/70 px-3 py-2 text-[11px]">
              <span className={cn(
                'truncate',
                uploadError ? 'text-destructive' : 'text-muted-foreground'
              )}>
                {uploadError
                  ? uploadError
                  : dragActive
                    ? 'Drop files to attach them to this chat.'
                    : isUploadingFiles
                      ? 'Uploading files into the current workspace...'
                      : `${attachedFiles.length} file${attachedFiles.length === 1 ? '' : 's'} attached`}
              </span>
              {attachedFiles.length > 0 ? (
                <button
                  type="button"
                  onClick={() => {
                    setAttachedFiles([])
                    setUploadError(null)
                  }}
                  className="shrink-0 text-muted-foreground transition-colors hover:text-foreground"
                >
                  Clear
                </button>
              ) : null}
            </div>
          ) : null}
        </div>

        <p className="mt-2 text-center text-[10px] text-muted-foreground/60">
          {t('chat.disclaimer')}
        </p>
      </div>
    </div>
  )
}
