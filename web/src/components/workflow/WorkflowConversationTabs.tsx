'use client'

import * as React from 'react'
import { SendHorizontal } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Textarea } from '@/components/ui/textarea'
import { cn } from '@/lib/utils'
import type { WorkflowConversationTab } from '@/components/workflow/types'

function messageTone(role: WorkflowConversationTab['messages'][number]['role']): string {
  if (role === 'assistant') {
    return 'border-primary-500/20 bg-primary-500/8'
  }
  if (role === 'user') {
    return 'border-border bg-surface-layer'
  }
  return 'border-white/8 bg-background/50'
}

function tabStatusVariant(status: WorkflowConversationTab['status']): 'neutral' | 'warning' | 'success' | 'error' | 'outline' {
  if (status === 'error' || status === 'blocked') {
    return 'error'
  }
  if (status === 'done') {
    return 'success'
  }
  if (status === 'thinking' || status === 'running_tool' || status === 'waiting') {
    return 'warning'
  }
  return 'outline'
}

interface WorkflowConversationTabsProps {
  tabs: WorkflowConversationTab[]
  value: string
  onValueChange: (value: string) => void
  onSendRoleMessage?: (roleId: string, content: string) => Promise<void>
}

function isTerminalRoleStatus(status?: WorkflowConversationTab['status']): boolean {
  return status === 'done' || status === 'error' || status === 'blocked'
}

export function WorkflowConversationTabs({
  tabs,
  value,
  onValueChange,
  onSendRoleMessage,
}: WorkflowConversationTabsProps) {
  const [draft, setDraft] = React.useState('')
  const [pending, setPending] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)
  const [success, setSuccess] = React.useState<string | null>(null)
  const activeTab = tabs.find((tab) => tab.id === value) ?? tabs[0] ?? null

  React.useEffect(() => {
    setDraft('')
    setPending(false)
    setError(null)
    setSuccess(null)
  }, [value])

  const handleSend = React.useCallback(async () => {
    if (!activeTab || activeTab.kind !== 'role' || !onSendRoleMessage) {
      return
    }
    const content = draft.trim()
    if (!content) {
      return
    }
    setPending(true)
    setError(null)
    setSuccess(null)
    try {
      await onSendRoleMessage(activeTab.roleId ?? activeTab.id.slice(5), content)
      setDraft('')
      setSuccess('Guidance queued for the next applicable turn.')
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : 'Failed to queue guidance.')
    } finally {
      setPending(false)
    }
  }, [activeTab, draft, onSendRoleMessage])

  return (
    <Card className="border-white/8 bg-[linear-gradient(180deg,rgba(255,255,255,0.04),rgba(255,255,255,0.02))] p-0">
      <CardHeader className="border-b border-white/8 px-4 py-3">
        <CardTitle className="text-sm">Conversations</CardTitle>
      </CardHeader>
      <CardContent className="p-4">
        <Tabs value={value} onValueChange={onValueChange}>
          <div className="overflow-x-auto pb-1">
            <TabsList className="min-w-max bg-background/60">
              {tabs.map((tab) => (
                <TabsTrigger key={tab.id} value={tab.id} className="gap-2">
                  <span>{tab.label}</span>
                  {tab.kind === 'role' && tab.status ? (
                    <Badge variant={tabStatusVariant(tab.status)} className="h-5 px-1.5 text-[10px]">
                      {tab.status}
                    </Badge>
                  ) : null}
                </TabsTrigger>
              ))}
            </TabsList>
          </div>

          {tabs.map((tab) => (
            <TabsContent key={tab.id} value={tab.id} className="mt-4">
              <div className="space-y-3">
                {tab.messages.map((message) => (
                  <div
                    key={message.id}
                    className={cn('rounded-xl border px-3 py-3', messageTone(message.role))}
                  >
                    <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-semibold uppercase tracking-[0.14em] text-muted-foreground">
                          {message.label}
                        </span>
                        {message.meta ? (
                          <span className="text-[11px] text-muted-foreground">{message.meta}</span>
                        ) : null}
                      </div>
                      {message.timestamp ? (
                        <span className="text-[11px] text-muted-foreground">
                          {new Date(message.timestamp).toLocaleString()}
                        </span>
                      ) : null}
                    </div>
                    <p className="whitespace-pre-wrap text-sm leading-6 text-foreground/95">{message.content}</p>
                  </div>
                ))}
              </div>
              {tab.kind === 'role' && onSendRoleMessage ? (
                <div className="mt-4 rounded-xl border border-white/8 bg-surface-layer/60 px-3 py-3">
                  <div className="flex flex-wrap items-start justify-between gap-2">
                    <div>
                      <p className="text-sm font-semibold text-foreground">Queue guidance</p>
                      <p className="mt-1 text-[11px] leading-relaxed text-muted-foreground">
                        Guidance queued here is additive context for the next applicable turn. It does not directly control the subagent.
                      </p>
                    </div>
                    <Badge variant="outline">Guidance queued</Badge>
                  </div>
                  <Textarea
                    value={draft}
                    onChange={(event) => setDraft(event.target.value)}
                    placeholder={
                      isTerminalRoleStatus(tab.status)
                        ? 'This role is finished. Guidance queue is disabled.'
                        : `Add context for ${tab.label}...`
                    }
                    minRows={3}
                    autoResize
                    disabled={pending || isTerminalRoleStatus(tab.status)}
                    className="mt-3 rounded-2xl border-white/10 bg-[#090d19] text-[12px] leading-6 text-slate-100 placeholder:text-slate-400"
                  />
                  {error ? (
                    <p className="mt-3 rounded-xl border border-destructive/30 bg-destructive/10 px-3 py-2 text-[11px] text-destructive">
                      {error}
                    </p>
                  ) : null}
                  {success ? (
                    <p className="mt-3 rounded-xl border border-emerald-400/25 bg-emerald-400/10 px-3 py-2 text-[11px] text-emerald-100">
                      {success}
                    </p>
                  ) : null}
                  <div className="mt-3 flex justify-end">
                    <Button
                      size="sm"
                      variant="secondary"
                      className="h-8 rounded-full px-3 text-xs"
                      loading={pending}
                      disabled={pending || isTerminalRoleStatus(tab.status) || draft.trim().length === 0}
                      onClick={() => void handleSend()}
                    >
                      <SendHorizontal className="h-3.5 w-3.5" />
                      Queue guidance
                    </Button>
                  </div>
                </div>
              ) : null}
            </TabsContent>
          ))}
        </Tabs>
      </CardContent>
    </Card>
  )
}
