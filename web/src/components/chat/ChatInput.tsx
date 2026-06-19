'use client'

import * as React from 'react'
import {
  AlertCircle,
  BrainCircuit,
  Check,
  CheckCircle2,
  ChevronDown,
  Loader2,
  Mic,
  Paperclip,
  Send,
  Shield,
  Sparkles,
  Square,
  X,
  Zap,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
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
import { ThinkingLevelChipControl } from './ThinkingLevelControls'
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

export interface ChatComposerSeed {
  text: string
  attachments: ChatAttachment[]
  selectedSkills?: Array<{ id: string; name: string }>
}

interface ChatInputProps {
  sessionId?: string | null
  projectId?: string | null
  uploadTargetDir?: string
  queuedAttachments?: ChatAttachment[]
  queuedAttachmentsKey?: string
  onSend: (text: string, options?: { selectedSkillIds?: string[]; attachments?: ChatAttachment[] }) => void
  onSubmitEdit?: (text: string, options?: { selectedSkillIds?: string[]; attachments?: ChatAttachment[] }) => void
  onCancelEdit?: () => void
  onStop?: () => void
  onVoice?: () => void
  onBuiltinCommand?: (command: 'clear' | 'settings' | 'voice' | 'model' | 'export' | 'workflow' | 'chat') => void
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
  approvalMode?: api.SessionSecurityOverride['autonomy_mode']
  approvalModeSourceLabel?: string | null
  approvalModeSourceDescription?: string | null
  onApprovalModeChange?: (value: api.SessionSecurityOverride['autonomy_mode']) => void | Promise<void>
  composerMode?: 'compose' | 'edit'
  composerSeed?: ChatComposerSeed | null
  composerResetKey?: string
}

interface AttachedFile extends ChatAttachment {}

function extractClipboardFiles(data: DataTransfer | null): File[] {
  if (!data) {
    return []
  }

  const filesFromItems = Array.from(data.items ?? [])
    .filter((item) => item.kind === 'file')
    .map((item) => item.getAsFile())
    .filter((file): file is File => file instanceof File)

  if (filesFromItems.length > 0) {
    return filesFromItems
  }

  return Array.from(data.files ?? [])
}

function normalizeUploadFile(file: File, fallbackIndex: number): File {
  if (file.name.trim().length > 0) {
    return file
  }

  const extension = file.type === 'image/png'
    ? 'png'
    : file.type === 'image/jpeg'
      ? 'jpg'
      : file.type === 'image/webp'
        ? 'webp'
        : file.type === 'image/gif'
          ? 'gif'
          : 'bin'

  return new File([file], `pasted-image-${Date.now()}-${fallbackIndex + 1}.${extension}`, {
    type: file.type,
    lastModified: file.lastModified,
  })
}

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

function attachmentIdentity(attachment: ChatAttachment): string {
  return [
    attachment.id ?? '',
    attachment.path,
    attachment.source ?? '',
    attachment.lineStart ?? '',
    attachment.lineEnd ?? '',
    attachment.quote ?? '',
  ].join('::')
}

type ApprovalMode = api.SessionSecurityOverride['autonomy_mode']

const APPROVAL_MODE_OPTIONS: Array<{
  value: ApprovalMode
  label: string
  description: string
  icon: React.ComponentType<{ className?: string }>
}> = [
  {
    value: 'strict',
    label: 'Ask every time',
    description: 'Pause before Mochi runs commands or applies file changes.',
    icon: AlertCircle,
  },
  {
    value: 'trusted_workspace',
    label: 'Approve for me',
    description: 'Auto-run clearly safe workspace actions and only ask for risky ones.',
    icon: CheckCircle2,
  },
  {
    value: 'auto_review',
    label: 'Auto review',
    description: 'Keep trusted-workspace thresholds while surfacing review metadata more explicitly.',
    icon: Sparkles,
  },
  {
    value: 'high_autonomy',
    label: 'High autonomy',
    description: 'Allow broader in-workspace execution before pausing for approval.',
    icon: Zap,
  },
]

const APPROVAL_MODE_BY_VALUE = Object.fromEntries(
  APPROVAL_MODE_OPTIONS.map((option) => [option.value, option])
) as Record<ApprovalMode, (typeof APPROVAL_MODE_OPTIONS)[number]>

function ApprovalModeControl({
  value,
  sourceLabel,
  sourceDescription,
  disabled,
  onChange,
}: {
  value: ApprovalMode
  sourceLabel: string | null | undefined
  sourceDescription: string | null | undefined
  disabled?: boolean
  onChange?: (value: ApprovalMode) => void | Promise<void>
}) {
  const [open, setOpen] = React.useState(false)
  const [isSaving, setIsSaving] = React.useState(false)
  const menuRef = React.useRef<HTMLDivElement>(null)
  const activeOption = APPROVAL_MODE_BY_VALUE[value]

  React.useEffect(() => {
    if (!open) {
      return
    }

    const handlePointerDown = (event: MouseEvent) => {
      if (!menuRef.current?.contains(event.target as Node)) {
        setOpen(false)
      }
    }

    document.addEventListener('mousedown', handlePointerDown)
    return () => document.removeEventListener('mousedown', handlePointerDown)
  }, [open])

  const handleSelect = React.useCallback(async (nextValue: ApprovalMode) => {
    if (!onChange) {
      setOpen(false)
      return
    }
    if (nextValue === value) {
      setOpen(false)
      return
    }

    setIsSaving(true)
    try {
      await onChange(nextValue)
      setOpen(false)
    } finally {
      setIsSaving(false)
    }
  }, [onChange, value])

  return (
    <div className="relative" ref={menuRef}>
      <button
        type="button"
        onClick={() => setOpen((current) => !current)}
        disabled={disabled || isSaving}
        aria-haspopup="menu"
        aria-expanded={open}
        title={activeOption.label}
        className={cn(
          'flex h-7 items-center gap-1.5 rounded-full border border-white/10 bg-canvas/85 pl-2 pr-2.5',
          'text-[11px] text-foreground transition-all duration-150 hover:bg-elevated-layer',
          'focus:outline-none focus:ring-2 focus:ring-primary-500/35',
          'disabled:cursor-not-allowed disabled:opacity-50'
        )}
      >
        <span className="flex h-4 w-4 items-center justify-center rounded-full bg-primary-500/12 text-primary-300">
          {isSaving ? <Loader2 className="h-3 w-3 animate-spin" /> : <Shield className="h-3 w-3" />}
        </span>
        <span className="whitespace-nowrap">{activeOption.label}</span>
        <ChevronDown className={cn('h-3 w-3 shrink-0 text-muted-foreground transition-transform', open && 'rotate-180')} />
      </button>

      {open ? (
        <div
          className={cn(
            'absolute bottom-full left-0 z-50 mb-2 w-[22rem] max-w-[calc(100vw-2rem)] overflow-hidden rounded-[1.15rem]',
            'border border-white/8 bg-[linear-gradient(180deg,rgba(28,28,31,0.98),rgba(18,18,21,0.98))] shadow-[0_24px_64px_rgba(0,0,0,0.42)]',
            'animate-slide-up'
          )}
        >
          <div className="border-b border-white/8 px-4 py-3">
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <p className="text-[12px] font-medium text-slate-100">
                  How should Mochi actions be approved?
                </p>
                <p className="mt-1 text-[11px] leading-4 text-slate-400">
                  Pick the review posture for this chat without leaving the composer.
                </p>
              </div>
              {sourceLabel ? (
                <span className="shrink-0 rounded-full border border-white/10 bg-white/[0.04] px-2 py-0.5 text-[10px] uppercase tracking-[0.14em] text-slate-300">
                  {sourceLabel}
                </span>
              ) : null}
            </div>
          </div>

          <div className="space-y-1.5 px-2 py-2">
            {APPROVAL_MODE_OPTIONS.map((option) => {
              const Icon = option.icon
              const selected = option.value === value

              return (
                <button
                  key={option.value}
                  type="button"
                  onClick={() => void handleSelect(option.value)}
                  disabled={isSaving}
                  className={cn(
                    'flex w-full items-start gap-3 rounded-[0.95rem] px-3 py-2.5 text-left transition-colors duration-150',
                    selected
                      ? 'bg-primary-500/12 text-slate-50'
                      : 'text-slate-200 hover:bg-white/[0.04]'
                  )}
                >
                  <span
                    className={cn(
                      'mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full border',
                      selected
                        ? 'border-primary-400/30 bg-primary-500/14 text-primary-200'
                        : 'border-white/8 bg-white/[0.03] text-slate-400'
                    )}
                  >
                    <Icon className="h-4 w-4" />
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="flex items-center gap-2">
                      <span className="text-sm font-medium">{option.label}</span>
                      {selected ? <Check className="h-3.5 w-3.5 text-primary-300" /> : null}
                    </span>
                    <span className="mt-1 block text-[11px] leading-4 text-slate-400">
                      {option.description}
                    </span>
                  </span>
                </button>
              )
            })}
          </div>

          <div className="border-t border-white/8 px-4 py-3 text-[11px] leading-4 text-slate-400">
            Effective now: <span className="text-slate-100">{activeOption.label}</span>. {sourceDescription ?? 'This chat follows the current workspace safety default.'}
          </div>
        </div>
      ) : null}
    </div>
  )
}

export function ChatInput({
  sessionId,
  projectId,
  uploadTargetDir,
  queuedAttachments,
  queuedAttachmentsKey,
  onSend,
  onSubmitEdit,
  onCancelEdit,
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
  approvalMode,
  approvalModeSourceLabel,
  approvalModeSourceDescription,
  onApprovalModeChange,
  composerMode = 'compose',
  composerSeed = null,
  composerResetKey,
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

  React.useEffect(() => {
    if (composerResetKey === undefined) {
      return
    }
    setValue(composerSeed?.text ?? '')
    setAttachedFiles([...(composerSeed?.attachments ?? [])])
    setSelectedSkills([...(composerSeed?.selectedSkills ?? [])])
    setUploadError(null)
    setPaletteDismissed(false)
  }, [composerResetKey, composerSeed])

  React.useEffect(() => {
    if (!queuedAttachmentsKey || !queuedAttachments || queuedAttachments.length === 0) {
      return
    }
    setAttachedFiles((prev) => {
      const known = new Set(prev.map((attachment) => attachmentIdentity(attachment)))
      const next = [...prev]
      for (const attachment of queuedAttachments) {
        const key = attachmentIdentity(attachment)
        if (!known.has(key)) {
          known.add(key)
          next.push(attachment)
        }
      }
      return next
    })
    setUploadError(null)
  }, [queuedAttachments, queuedAttachmentsKey])

  const handleAttachFiles = React.useCallback(async (incomingFiles: FileList | File[]) => {
    const files = Array.from(incomingFiles).map((file, index) => normalizeUploadFile(file, index))
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
        const resolvedImportedFilePath = result.files[0]?.path ?? result.importedPath
        uploadedFiles.push({
          id: `${Date.now()}-${index}-${file.name}`,
          name: file.name,
          path: resolvedImportedFilePath,
          size: result.files[0]?.size ?? file.size,
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

  const handlePaste = React.useCallback((event: React.ClipboardEvent<HTMLTextAreaElement>) => {
    const clipboardFiles = extractClipboardFiles(event.clipboardData)
    if (clipboardFiles.length === 0) {
      return
    }

    event.preventDefault()
    void handleAttachFiles(clipboardFiles)
  }, [handleAttachFiles])

  const handleSend = () => {
    const trimmed = value.trim()
    if ((trimmed.length === 0 && attachedFiles.length === 0) || disabled || isStreaming || isUploadingFiles) {
      return
    }
    const submit = composerMode === 'edit' && onSubmitEdit ? onSubmitEdit : onSend
    submit(trimmed, {
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
          {composerMode === 'edit' ? (
            <div className="flex items-center justify-between gap-3 border-b border-border/70 px-3 py-2 text-[11px] text-muted-foreground">
              <span>Editing message. Attachments stay until you remove them.</span>
              {onCancelEdit ? (
                <button
                  type="button"
                  onClick={onCancelEdit}
                  className="shrink-0 text-foreground transition-colors hover:text-primary-300"
                >
                  Cancel edit
                </button>
              ) : null}
            </div>
          ) : null}

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
                sessionId={sessionId}
                projectId={projectId}
                onRemove={(attachment) => {
                  const targetKey = attachmentIdentity(attachment)
                  setAttachedFiles((prev) =>
                    prev.filter((item) => attachmentIdentity(item) !== targetKey)
                  )
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
            onPaste={handlePaste}
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

          <div className="flex items-center justify-between gap-2 px-3 pb-2">
            <div className="flex min-w-0 flex-1 flex-wrap items-center gap-1">
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

              {approvalMode ? (
                <ApprovalModeControl
                  value={approvalMode}
                  sourceLabel={approvalModeSourceLabel}
                  sourceDescription={approvalModeSourceDescription}
                  disabled={disabled}
                  onChange={onApprovalModeChange}
                />
              ) : null}

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
                <ThinkingLevelChipControl
                  supportedEfforts={reasoningOptions}
                  value={inference.reasoningEffort}
                  disabled={disabled}
                  onChange={onReasoningEffortChange}
                />
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
              title={isStreaming ? 'Stop generation' : composerMode === 'edit' ? 'Resend edited message' : t('chat.input.send')}
              aria-label={isStreaming ? 'Stop generation' : composerMode === 'edit' ? 'Resend edited message' : t('chat.input.send')}
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
