'use client'

import { Bot, Clock3 } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { cn } from '@/lib/utils'
import type { WorkflowAgentCard } from '@/components/workflow/types'

function statusVariant(status: WorkflowAgentCard['status']): 'neutral' | 'warning' | 'success' | 'error' | 'outline' {
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

interface AgentBoardProps {
  agents: WorkflowAgentCard[]
  selectedRoleId: string | null
  onSelectRole: (roleId: string) => void
}

export function AgentBoard({ agents, selectedRoleId, onSelectRole }: AgentBoardProps) {
  return (
    <Card className="border-white/8 bg-[linear-gradient(180deg,rgba(255,255,255,0.04),rgba(255,255,255,0.02))] p-0">
      <CardHeader className="border-b border-white/8 px-4 py-3">
        <CardTitle className="text-sm">Agent Board</CardTitle>
      </CardHeader>
      <CardContent className="p-4">
        {agents.length === 0 ? (
          <div className="rounded-xl border border-dashed border-border bg-background/40 px-4 py-6 text-sm text-muted-foreground">
            No visible subagent roles yet.
          </div>
        ) : (
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {agents.map((agent) => {
              const isSelected = selectedRoleId === agent.roleId
              return (
                <button
                  key={agent.roleId}
                  type="button"
                  aria-pressed={isSelected}
                  onClick={() => onSelectRole(agent.roleId)}
                  className={cn(
                    'rounded-xl border px-3 py-3 text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
                    isSelected
                      ? 'border-primary-500/50 bg-primary-500/10'
                      : 'border-border bg-surface-layer hover:border-primary-500/30 hover:bg-elevated-layer'
                  )}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <Bot className="h-4 w-4 text-primary-300" />
                        <span className="truncate text-sm font-semibold text-foreground">{agent.label}</span>
                      </div>
                      {agent.modelId ? (
                        <p className="mt-1 truncate text-[11px] text-muted-foreground">{agent.modelId}</p>
                      ) : null}
                    </div>
                    <Badge variant={statusVariant(agent.status)}>{agent.status}</Badge>
                  </div>

                  <div className="mt-3 space-y-2 text-xs text-muted-foreground">
                    <p>{agent.currentAction}</p>
                    {agent.lastOutputSummary ? (
                      <p className="line-clamp-3 text-foreground/90">{agent.lastOutputSummary}</p>
                    ) : (
                      <p>Waiting for the first readable role output.</p>
                    )}
                    <div className="flex flex-wrap items-center gap-3 pt-1">
                      <span>{agent.outputCount} outputs</span>
                      {agent.updatedAt ? (
                        <span className="inline-flex items-center gap-1">
                          <Clock3 className="h-3 w-3" />
                          {new Date(agent.updatedAt).toLocaleString()}
                        </span>
                      ) : null}
                    </div>
                  </div>
                </button>
              )
            })}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
