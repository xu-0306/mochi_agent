'use client'

import * as React from 'react'
import { AlertCircle, Bot, Pencil, RefreshCcw } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeHighlight from 'rehype-highlight'
import { cn, formatDate, formatRelativeTime } from '@/lib/utils'
import type { Message } from '@/lib/chat'
import { ReasoningPanel } from './ReasoningPanel'
import { CopyButton } from './CopyButton'
import type { FileChangeSummary } from '@/lib/chat-p2'
import { extractFileChangeGroupFromReasoningStep } from '@/lib/chat-p2'
import { Button } from '@/components/ui/button'
import { createMarkdownCodeComponents } from '@/components/code/markdown-code'
import { ChatAttachments } from './ChatAttachments'
import { FileChangeCard } from './FileChangeCard'
import type { FileChangeGroupSummary } from '@/lib/file-change-preview'
import { WorkflowProgressCard } from './WorkflowProgressCard'
import { SubagentTaskCard } from './SubagentTaskCard'
import { GoalCard } from './GoalCard'

interface ChatMessageProps {
  message: Message
  sessionId?: string | null
  projectId?: string | null
  onRegenerate?: (message: Message) => void
  onEditAndResend?: (message: Message, nextContent: string) => Promise<void> | void
  onUndoFileChange?: (change: FileChangeSummary) => Promise<void> | void
  onOpenTask?: (taskId: string) => void
}

function hasWideMarkdownContent(content: string): boolean {
  if (!content.trim()) {
    return false
  }

  if (/```[\s\S]*?```/.test(content)) {
    return true
  }

  if (/^\|.+\|\s*$/m.test(content)) {
    return true
  }

  return content
    .split(/\r?\n/)
    .some((line) => line.trim().length >= 96 && !/\s/.test(line.trim()))
}

function formatTokenStats(message: Message): string | null {
  if (!message.tokenStats) {
    return null
  }

  const { inputTokens, outputTokens, generationTimeMs, finishReason } = message.tokenStats
  const stats: string[] = []

  if (generationTimeMs > 0) {
    const tokensPerSecond = outputTokens / (generationTimeMs / 1000)
    if (Number.isFinite(tokensPerSecond) && tokensPerSecond > 0) {
      stats.push(`${tokensPerSecond.toFixed(2)} tok/sec`)
    }
  }

  stats.push(`input tokens ${inputTokens}`)
  stats.push(`output tokens ${outputTokens}`)
  stats.push(`generation time ${(generationTimeMs / 1000).toFixed(2)}s`)

  if (finishReason) {
    stats.push(`finish reason ${finishReason}`)
  }

  return stats.join('  ')
}

export function ChatMessage({
  message,
  sessionId,
  projectId,
  onRegenerate,
  onEditAndResend,
  onUndoFileChange,
  onOpenTask,
}: ChatMessageProps) {
  const { type, content, errorCode, isStreaming, reasoningSteps } = message
  const tokenStatsLabel = type === 'assistant' ? formatTokenStats(message) : null
  const timestampLabel = formatRelativeTime(message.timestamp)
  const timestampTitle = formatDate(message.timestamp, {
    format: {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    },
  })
  const markdownCodeComponents = React.useMemo(
    () => createMarkdownCodeComponents({ showCopyButton: true }),
    []
  )
  const prefersWideAssistantLayout = type === 'assistant' && hasWideMarkdownContent(content)
  const fileChanges = React.useMemo(
    () =>
      (reasoningSteps ?? [])
        .map((step) => extractFileChangeGroupFromReasoningStep(step))
        .filter((change): change is FileChangeGroupSummary => change !== null),
    [reasoningSteps]
  )
  const goalCardSupplementaryContent =
    message.goalCard &&
    content.trim().length > 0 &&
    content.trim() !== message.goalCard.objective.trim()
      ? content
      : null

  if (type === 'system') {
    return (
      <div className="flex justify-center animate-fade-in">
        <div className="max-w-md rounded-lg border border-dashed border-border bg-elevated-layer px-4 py-2 text-center text-xs text-muted-foreground">
          {content}
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
            <p className="whitespace-pre-wrap break-words text-sm leading-relaxed text-foreground">
              {content}
            </p>
            {errorCode ? (
              <p className="mt-1 break-all text-[11px] text-muted-foreground">
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
      <div className="group flex justify-end animate-slide-up">
        <div
          className={cn(
            'flex max-w-[720px] flex-col gap-1',
            'items-end'
          )}
        >
          <div className="relative w-full">
            <div className="pointer-events-none absolute top-2 right-2 z-10 opacity-0 transition-opacity duration-150 group-hover:opacity-100">
              <div className="pointer-events-auto flex items-center gap-1 rounded-full border border-white/15 bg-black/30 p-1 shadow-sm backdrop-blur-sm">
                {onEditAndResend ? (
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon-sm"
                    className="h-6 w-6 rounded-full text-white/75 hover:bg-white/12 hover:text-white"
                    title="Edit and resend"
                    aria-label="Edit and resend"
                    onClick={() => void onEditAndResend(message, content)}
                  >
                    <Pencil className="h-3.5 w-3.5" />
                  </Button>
                ) : null}
                <CopyButton
                  text={content}
                  label="Copy message"
                  className="h-6 w-6 rounded-full text-white/75 hover:bg-white/12 hover:text-white"
                />
              </div>
            </div>

            <div
              className={cn(
                'bg-primary-500 px-4 py-3 text-white',
                'rounded-[18px_18px_6px_18px]',
                'whitespace-pre-wrap break-words text-sm leading-relaxed shadow-sm',
                onEditAndResend ? 'pr-16' : null
              )}
            >
              <div className="space-y-3">
                {content ? <div>{content}</div> : null}
                {message.attachments && message.attachments.length > 0 ? (
                  <ChatAttachments
                    attachments={message.attachments}
                    variant="message"
                    sessionId={sessionId}
                    projectId={projectId}
                  />
                ) : null}
              </div>
            </div>
          </div>
          <span
            className={cn(
              'px-1 text-[11px] text-muted-foreground'
            )}
            title={timestampTitle}
          >
            {timestampLabel}
          </span>
        </div>
      </div>
    )
  }

  if (message.workflowCard) {
    return (
      <div className="group animate-slide-up">
        <div className="flex justify-start">
          <div className="w-full max-w-[860px]">
            <div className="flex gap-3">
              <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-primary-500/30 bg-primary-500/15">
                <Bot className="h-3.5 w-3.5 text-primary-400" />
              </div>
              <div className="min-w-0 flex-1 pt-0.5">
                <WorkflowProgressCard card={message.workflowCard} />
                <div className="mt-2 opacity-0 transition-opacity group-hover:opacity-100">
                  <span className="text-[11px] text-muted-foreground" title={timestampTitle}>
                    {timestampLabel}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    )
  }

  if (message.goalCard) {
    return (
      <div className="group animate-slide-up">
        <div className="flex justify-start">
          <div className="w-full max-w-[860px]">
            <ReasoningPanel
              steps={reasoningSteps ?? []}
              isStreaming={isStreaming}
              tokenStats={message.tokenStats}
              onUndoFileChange={onUndoFileChange}
              onOpenTask={onOpenTask}
            />
            <div className="flex gap-3">
              <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-primary-500/30 bg-primary-500/15">
                <Bot className="h-3.5 w-3.5 text-primary-400" />
              </div>
              <div className="min-w-0 flex-1 pt-0.5">
                <GoalCard card={message.goalCard} />
                {goalCardSupplementaryContent ? (
                  <div className="mt-4 prose prose-invert max-w-none text-sm leading-7 text-foreground">
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      rehypePlugins={[rehypeHighlight]}
                      components={markdownCodeComponents}
                    >
                      {goalCardSupplementaryContent}
                    </ReactMarkdown>
                  </div>
                ) : null}
                {message.attachments && message.attachments.length > 0 ? (
                  <div className="mt-4">
                    <ChatAttachments
                      attachments={message.attachments}
                      variant="message"
                      sessionId={sessionId}
                      projectId={projectId}
                    />
                  </div>
                ) : null}
                <div className="mt-2 opacity-0 transition-opacity group-hover:opacity-100">
                  <span className="text-[11px] text-muted-foreground" title={timestampTitle}>
                    {timestampLabel}
                  </span>
                </div>
                {tokenStatsLabel ? (
                  <p className="mt-1 text-xs text-muted-foreground">{tokenStatsLabel}</p>
                ) : null}
              </div>
            </div>
          </div>
        </div>
      </div>
    )
  }

  if (message.subagentTaskCard) {
    return (
      <div className="group animate-slide-up">
        <div className="flex justify-start">
          <div className="w-full max-w-[800px]">
            <div className="flex gap-3">
              <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-primary-500/30 bg-primary-500/15">
                <Bot className="h-3.5 w-3.5 text-primary-400" />
              </div>
              <div className="min-w-0 flex-1 pt-0.5">
                <SubagentTaskCard card={message.subagentTaskCard} onOpenTask={onOpenTask} />
                <div className="mt-2 opacity-0 transition-opacity group-hover:opacity-100">
                  <span className="text-[11px] text-muted-foreground" title={timestampTitle}>
                    {timestampLabel}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="group animate-slide-up">
      <div className="flex justify-start">
        <div
          className={cn(
            'w-full',
            prefersWideAssistantLayout
              ? 'max-w-[min(1120px,calc(100vw-10rem))]'
              : 'max-w-[760px]'
          )}
        >
          <ReasoningPanel
            steps={reasoningSteps ?? []}
            isStreaming={isStreaming}
            tokenStats={message.tokenStats}
            onUndoFileChange={onUndoFileChange}
            onOpenTask={onOpenTask}
          />
          <div className="flex gap-3">
            <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-primary-500/30 bg-primary-500/15">
              <Bot className="h-3.5 w-3.5 text-primary-400" />
            </div>
            <div className="min-w-0 flex-1 pt-0.5">
              <div className="prose prose-invert max-w-none text-sm leading-7 text-foreground">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  rehypePlugins={[rehypeHighlight]}
                  components={markdownCodeComponents}
                >
                  {content}
                </ReactMarkdown>
                {isStreaming ? <span className="ml-0.5 animate-blink text-primary-400">|</span> : null}
              </div>
              {fileChanges.length > 0 ? (
                <div className="mt-4 space-y-3">
                  {fileChanges.map((change, index) => (
                    <FileChangeCard
                      key={`${change.id}:${index}`}
                      group={change}
                      onUndo={onUndoFileChange}
                    />
                  ))}
                </div>
              ) : null}
              <div className="mt-2 flex items-center justify-between gap-2 opacity-0 transition-opacity group-hover:opacity-100">
                <span className="text-[11px] text-muted-foreground" title={timestampTitle}>
                  {timestampLabel}
                </span>
                <div className="flex items-center gap-1">
                  {onRegenerate && !isStreaming ? (
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      onClick={() => onRegenerate(message)}
                    >
                      <RefreshCcw className="h-3.5 w-3.5" />
                      Regenerate
                    </Button>
                  ) : null}
                  <CopyButton text={content} label="Copy message" />
                </div>
              </div>
              {tokenStatsLabel ? (
                <p className="mt-1 text-xs text-muted-foreground">{tokenStatsLabel}</p>
              ) : null}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
