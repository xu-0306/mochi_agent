'use client'

import * as React from 'react'
import { Check, ChevronDown, Mic, Paperclip, Send, Square } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { useI18n } from '@/lib/i18n'
import {
  CommandPalette,
  filterPaletteActions,
  type CommandPaletteAction,
} from './CommandPalette'
import type { Skill } from '@/lib/api'

export interface ChatInputModelOption {
  id: string
  label: string
  status?: 'connected' | 'configured' | 'disconnected'
}

interface ChatInputProps {
  onSend: (text: string) => void
  onStop?: () => void
  onVoice?: () => void
  onBuiltinCommand?: (command: 'clear' | 'settings' | 'voice' | 'model' | 'export') => void
  disabled?: boolean
  isStreaming?: boolean
  models?: ChatInputModelOption[]
  currentModel?: string | null
  skills?: Skill[]
  onSwitchModel?: (modelId: string) => void | Promise<void>
}

export function ChatInput({
  onSend,
  onStop,
  onVoice,
  onBuiltinCommand,
  disabled = false,
  isStreaming = false,
  models,
  currentModel,
  skills = [],
  onSwitchModel,
}: ChatInputProps) {
  const { t } = useI18n()
  const [value, setValue] = React.useState('')
  const [showModelMenu, setShowModelMenu] = React.useState(false)
  const [paletteOpen, setPaletteOpen] = React.useState(false)
  const [paletteQuery, setPaletteQuery] = React.useState('')
  const [paletteIndex, setPaletteIndex] = React.useState(0)
  const textareaRef = React.useRef<HTMLTextAreaElement>(null)
  const modelMenuRef = React.useRef<HTMLDivElement>(null)
  const paletteActions = React.useMemo(
    () => filterPaletteActions(skills, paletteQuery),
    [paletteQuery, skills]
  )
  const selectedPaletteAction = paletteActions[
    Math.min(paletteIndex, Math.max(0, paletteActions.length - 1))
  ]

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

  const handleSend = () => {
    const trimmed = value.trim()
    if (!trimmed || disabled || isStreaming) return
    onSend(trimmed)
    setValue('')
    setPaletteOpen(false)
    setPaletteQuery('')
    textareaRef.current?.focus()
  }

  const maybeOpenPalette = React.useCallback((nextValue: string, caret: number | null) => {
    const textBeforeCursor = caret === null ? nextValue : nextValue.slice(0, caret)
    const lineStart = Math.max(0, textBeforeCursor.lastIndexOf('\n') + 1)
    const currentToken = textBeforeCursor.slice(lineStart)
    const shouldOpen = currentToken.startsWith('/')

    setPaletteOpen(shouldOpen)
    if (!shouldOpen) {
      setPaletteQuery('')
      setPaletteIndex(0)
      return
    }
    setPaletteQuery(currentToken)
    setPaletteIndex(0)
  }, [])

  const handlePaletteSelect = React.useCallback((action: CommandPaletteAction) => {
    if (action.kind === 'builtin') {
      setPaletteOpen(false)
      setPaletteQuery('')
      onBuiltinCommand?.(action.id)
      return
    }

    const insertion = `${action.label} `
    const textarea = textareaRef.current
    if (!textarea) {
      setValue(insertion)
    } else {
      const start = textarea.selectionStart
      const end = textarea.selectionEnd
      const before = value.slice(0, start)
      const after = value.slice(end)
      const lineStart = Math.max(0, before.lastIndexOf('\n') + 1)
      const updated = `${before.slice(0, lineStart)}${insertion}${after}`
      setValue(updated)
    }
    setPaletteOpen(false)
    setPaletteQuery('')
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
        setPaletteOpen(false)
        setPaletteQuery('')
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
          skills={skills}
          selectedIndex={paletteIndex}
          onSelectedIndexChange={setPaletteIndex}
          onSelect={handlePaletteSelect}
        />
        <div
          className={cn(
            'flex flex-col rounded-xl border border-border bg-surface-layer',
            'transition-all duration-150',
            'focus-within:border-primary-500 focus-within:ring-[3px] focus-within:ring-primary-500/20'
          )}
        >
          <Textarea
            id="chat-input-textarea"
            ref={textareaRef}
            value={value}
            onChange={(e) => {
              const nextValue = e.target.value
              setValue(nextValue)
              maybeOpenPalette(nextValue, e.target.selectionStart)
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

          <div className="flex items-center justify-between px-3 pb-2">
            <div className="flex items-center gap-1">
              <Button
                variant="ghost"
                size="icon-sm"
                title={t('chat.input.attachFile')}
                aria-label={t('chat.input.attachFile')}
                disabled={disabled}
                className="h-7 w-7 text-muted-foreground hover:text-foreground"
              >
                <Paperclip className="h-3.5 w-3.5" />
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
                  <span className="truncate">
                    {selectedModel?.label ?? t('chat.input.currentModel')}
                  </span>
                  <ChevronDown className="h-3 w-3 shrink-0" />
                </button>

                {showModelMenu && availableModels.length > 0 && (
                  <div
                    className={cn(
                      'absolute bottom-full left-0 mb-1 z-50 w-52',
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
                        <span className="min-w-0 flex-1 truncate text-left">
                          {model.label}
                        </span>
                        {selectedModel?.id === model.id ? (
                          <Check className="h-3.5 w-3.5 shrink-0 text-primary-400" />
                        ) : null}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>

            <Button
              variant={isStreaming || value.trim() ? 'primary' : 'ghost'}
              size="icon-sm"
              onClick={isStreaming ? onStop : handleSend}
              disabled={disabled || (isStreaming ? !onStop : !value.trim())}
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
        </div>

        <p className="mt-2 text-center text-[10px] text-muted-foreground/60">
          {t('chat.disclaimer')}
        </p>
      </div>
    </div>
  )
}
