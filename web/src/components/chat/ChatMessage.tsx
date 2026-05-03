'use client'

import * as React from 'react'
import { AlertCircle, Brain, Bot } from 'lucide-react'
import { cn } from '@/lib/utils'
import { useI18n } from '@/lib/i18n'
import { ToolCallCard } from './ToolCallCard'

export type MessageType =
  | 'user'
  | 'assistant'
  | 'system'
  | 'thinking'
  | 'tool_call'
  | 'tool_result'
  | 'error'

export interface Message {
  id: string
  type: MessageType
  content: string
  eventType?:
    | 'thinking'
    | 'tool_call_request'
    | 'tool_call_result'
    | 'final_answer'
    | 'error'
  toolName?: string
  toolArgs?: Record<string, unknown>
  toolResult?: unknown
  toolCallId?: string
  toolError?: string
  errorCode?: string
  timestamp: Date
  isStreaming?: boolean
}

interface ChatMessageProps {
  message: Message
}

export function ChatMessage({ message }: ChatMessageProps) {
  const { t } = useI18n()
  const {
    type,
    content,
    toolName,
    toolArgs,
    toolResult,
    toolCallId,
    toolError,
    errorCode,
    isStreaming,
  } = message

  if (type === 'tool_call' || type === 'tool_result') {
    return (
      <div className="flex justify-start animate-slide-up">
        <ToolCallCard
          toolName={toolName ?? 'unknown_tool'}
          args={toolArgs}
          result={toolResult}
          type={type}
          callId={toolCallId}
          status={type === 'tool_call' ? 'calling' : toolError ? 'error' : 'success'}
          errorMessage={toolError}
        />
      </div>
    )
  }

  if (type === 'system') {
    return (
      <div className="flex justify-center animate-fade-in">
        <div className="max-w-md bg-elevated-layer border border-dashed border-border rounded-lg px-4 py-2 text-xs text-muted-foreground text-center">
          {content}
        </div>
      </div>
    )
  }

  if (type === 'thinking') {
    return (
      <div className="flex justify-start animate-slide-up">
        <div className="flex max-w-[560px] items-start gap-3 rounded-lg border border-border bg-surface-layer px-3 py-2.5">
          <div className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full border border-primary-500/30 bg-primary-500/10">
            <Brain className="h-3.5 w-3.5 text-primary-400" />
          </div>
          <div className="min-w-0 flex-1">
            <p className="mb-1 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
              {t('chat.thinking')}
            </p>
            <p className="text-sm leading-relaxed text-foreground whitespace-pre-wrap break-words">
              {content}
            </p>
          </div>
        </div>
      </div>
    )
  }

  if (type === 'error') {
    return (
      <div className="flex justify-start animate-slide-up">
        <div className="flex max-w-[560px] items-start gap-3 rounded-lg border border-error/40 bg-error/10 px-3 py-2.5">
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-error" />
          <div className="min-w-0 flex-1">
            <p className="text-sm leading-relaxed text-foreground whitespace-pre-wrap break-words">
              {content}
            </p>
            {errorCode ? (
              <p className="mt-1 text-[11px] text-muted-foreground break-all">
                {errorCode}
              </p>
            ) : null}
          </div>
        </div>
      </div>
    )
  }

  if (type === 'user') {
    return (
      <div className="flex justify-end animate-slide-up">
        <div
          className={cn(
            'max-w-[480px] bg-primary-500 text-white px-4 py-2.5',
            'rounded-[16px_16px_4px_16px]',
            'text-sm leading-relaxed shadow-sm whitespace-pre-wrap break-words'
          )}
        >
          {content}
        </div>
      </div>
    )
  }

  // assistant
  return (
    <div className="flex justify-start gap-3 animate-slide-up">
      <div className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full border border-primary-500/30 bg-primary-500/20">
        <Bot className="h-3.5 w-3.5 text-primary-400" />
      </div>
      <div className="max-w-[560px] min-w-0 flex-1">
        <p className="text-sm leading-relaxed text-foreground whitespace-pre-wrap break-words">
          {content}
          {isStreaming && <span className="animate-blink text-primary-400 ml-0.5">▍</span>}
        </p>
      </div>
    </div>
  )
}
