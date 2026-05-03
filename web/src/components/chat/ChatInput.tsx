'use client'

import * as React from 'react'
import { Check, ChevronDown, Mic, Paperclip, Send } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { useI18n } from '@/lib/i18n'

export interface ChatInputModelOption {
  id: string
  label: string
  status?: 'connected' | 'configured' | 'disconnected'
}

interface ChatInputProps {
  onSend: (text: string) => void
  onVoice?: () => void
  disabled?: boolean
  isStreaming?: boolean
  models?: ChatInputModelOption[]
  currentModel?: string | null
  onSwitchModel?: (modelId: string) => void | Promise<void>
}

export function ChatInput({
  onSend,
  onVoice,
  disabled = false,
  isStreaming = false,
  models,
  currentModel,
  onSwitchModel,
}: ChatInputProps) {
  const { t } = useI18n()
  const [value, setValue] = React.useState('')
  const [showModelMenu, setShowModelMenu] = React.useState(false)
  const textareaRef = React.useRef<HTMLTextAreaElement>(null)
  const modelMenuRef = React.useRef<HTMLDivElement>(null)

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
    textareaRef.current?.focus()
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="relative border-t border-border bg-canvas px-4 pt-3 pb-4">
      <div
        className={cn(
          'flex flex-col rounded-xl border border-border bg-surface-layer',
          'transition-all duration-150',
          'focus-within:border-primary-500 focus-within:ring-[3px] focus-within:ring-primary-500/20'
        )}
      >
        {/* Textarea */}
        <Textarea
          id="chat-input-textarea"
          ref={textareaRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
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

        {/* Toolbar */}
        <div className="flex items-center justify-between px-3 pb-2">
          <div className="flex items-center gap-1">
            {/* File attach */}
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

            {/* Voice */}
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

            {/* Model selector */}
            <div className="relative" ref={modelMenuRef}>
              <button
                type="button"
                onClick={() => setShowModelMenu((v) => !v)}
                title={selectedModel?.label ?? t('chat.input.currentModel')}
                aria-label={t('chat.input.currentModel')}
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

          {/* Send button */}
          <Button
            variant={value.trim() ? 'primary' : 'ghost'}
            size="icon-sm"
            onClick={handleSend}
            disabled={!value.trim() || disabled || isStreaming}
            title={t('chat.input.send')}
            aria-label={t('chat.input.send')}
            className="h-7 w-7 shrink-0"
          >
            {isStreaming ? (
              <span className="h-3 w-3 rounded-sm bg-current" />
            ) : (
              <Send className="h-3.5 w-3.5" />
            )}
          </Button>
        </div>
      </div>

      <p className="text-center text-[10px] text-muted-foreground/60 mt-2">
        {t('chat.disclaimer')}
      </p>
    </div>
  )
}
