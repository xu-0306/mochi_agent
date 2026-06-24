'use client'

import * as React from 'react'
import { Activity, Bug, FileStack, GitBranch, ListTree, Sparkles } from 'lucide-react'
import type { AgentRunDetail, AgentRunSummary } from '@/lib/api'
import * as api from '@/lib/api'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import { AgentBoard } from '@/components/workflow/AgentBoard'
import type { WorkflowDebugEntry } from '@/components/workflow/types'
import { buildWorkflowDeskView } from '@/components/workflow/utils'
import { WorkflowConversationTabs } from '@/components/workflow/WorkflowConversationTabs'
import { cn } from '@/lib/utils'

interface WorkflowDeskProps {
  run: AgentRunDetail | null
  summary?: AgentRunSummary | null
  debugEntries?: WorkflowDebugEntry[]
  onRunUpdated?: (run: AgentRunDetail) => void
}

function statusVariant(status: string): 'neutral' | 'warning' | 'success' | 'error' | 'outline' {
  const normalized = status.toLowerCase()
  if (normalized.includes('fail') || normalized.includes('error') || normalized.includes('cancel')) {
    return 'error'
  }
  if (normalized.includes('complete') || normalized.includes('done') || normalized.includes('success')) {
    return 'success'
  }
  if (normalized.includes('run') || normalized.includes('queue') || normalized.includes('await')) {
    return 'warning'
  }
  return 'outline'
}

function stageClassName(status: 'pending' | 'active' | 'completed' | 'blocked'): string {
  if (status === 'completed') {
    return 'border-emerald-500/25 bg-emerald-500/10'
  }
  if (status === 'active') {
    return 'border-primary-500/35 bg-primary-500/10'
  }
  if (status === 'blocked') {
    return 'border-destructive/35 bg-destructive/10'
  }
  return 'border-border bg-surface-layer'
}

function isTerminalWorkflowStatus(status: string | null | undefined): boolean {
  const normalized = (status ?? '').toLowerCase()
  return (
    normalized === 'succeeded' ||
    normalized === 'failed' ||
    normalized === 'cancelled' ||
    normalized === 'completed' ||
    normalized === 'done' ||
    normalized === 'error'
  )
}

export function WorkflowDesk({ run, summary, debugEntries = [], onRunUpdated }: WorkflowDeskProps) {
  const view = React.useMemo(
    () => buildWorkflowDeskView({ run, summary, debugEntries }),
    [debugEntries, run, summary]
  )
  const [selectedTab, setSelectedTab] = React.useState<string>('main')
  const canSendRoleGuidance = Boolean(run?.run_id) && !isTerminalWorkflowStatus(run?.status)

  const handleSendRoleMessage = async (roleId: string, content: string) => {
    if (!run?.run_id) {
      throw new Error('No workflow run is selected.')
    }
    if (isTerminalWorkflowStatus(run.status)) {
      throw new Error('This workflow run is finished. Role guidance is disabled.')
    }
    const detail = await api.appendAgentRunSubagentMessage(run.run_id, roleId, {
      content,
      projectId: run.project_id,
      workspaceDir: run.workspace_dir,
      metadata: {
        channel: 'workflow_role_tab',
        delivery: 'guidance_only',
        target_role_id: roleId,
      },
    })
    onRunUpdated?.(detail)
  }

  React.useEffect(() => {
    const availableIds = new Set(view.conversations.map((tab) => tab.id))
    if (!availableIds.has(selectedTab)) {
      setSelectedTab('main')
    }
  }, [selectedTab, view.conversations])

  if (!view.runId) {
    return (
      <Card className="border-white/8 bg-[linear-gradient(180deg,rgba(255,255,255,0.04),rgba(255,255,255,0.02))]">
        <CardHeader>
          <CardTitle className="text-sm">Workflow Desk</CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-muted-foreground">
          No run selected yet. Create a run or choose one from the recent runs sidebar to load the desk shell.
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-4">
      <Card className="overflow-hidden border-white/8 bg-[radial-gradient(circle_at_top_right,rgba(94,106,210,0.18),transparent_34%),linear-gradient(180deg,rgba(255,255,255,0.04),rgba(255,255,255,0.02))] p-0">
        <CardHeader className="border-b border-white/8 px-4 py-4">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
            <div className="min-w-0 space-y-2">
              <div className="inline-flex items-center gap-2 rounded-full border border-primary-500/20 bg-primary-500/10 px-2.5 py-1 text-[11px] font-medium uppercase tracking-[0.18em] text-primary-300">
                <Sparkles className="h-3.5 w-3.5" />
                Workflow Desk
              </div>
              <div>
                <h3 className="truncate text-base font-semibold text-foreground">{view.title}</h3>
                <p className="mt-1 text-sm text-muted-foreground">
                  Phase: <span className="text-foreground">{view.phaseLabel}</span>
                  {view.protocolId ? <span> / Protocol: {view.protocolId}</span> : null}
                </p>
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant={statusVariant(view.status)}>{view.status}</Badge>
              <Badge variant="outline">{view.agents.length} agents</Badge>
              <Badge variant="outline">{view.rawEventCount} events</Badge>
              <Badge variant="outline">{view.artifacts.length} artifacts</Badge>
            </div>
          </div>

          <div className="mt-3 grid gap-3 text-xs text-muted-foreground sm:grid-cols-3">
            <div>
              <span className="font-medium text-foreground">Started:</span>{' '}
              {view.startedAt ? new Date(view.startedAt).toLocaleString() : 'Not started'}
            </div>
            <div>
              <span className="font-medium text-foreground">Updated:</span>{' '}
              {view.updatedAt ? new Date(view.updatedAt).toLocaleString() : 'Unknown'}
            </div>
            <div>
              <span className="font-medium text-foreground">Finished:</span>{' '}
              {view.finishedAt ? new Date(view.finishedAt).toLocaleString() : 'In progress'}
            </div>
          </div>
        </CardHeader>

        <CardContent className="space-y-4 px-4 py-4">
          <AgentBoard
            agents={view.agents}
            selectedRoleId={selectedTab.startsWith('role:') ? selectedTab.slice(5) : null}
            onSelectRole={(roleId) => setSelectedTab(`role:${roleId}`)}
          />

          <Card className="border-white/8 bg-[linear-gradient(180deg,rgba(255,255,255,0.03),rgba(255,255,255,0.015))] p-0">
            <CardHeader className="border-b border-white/8 px-4 py-3">
              <CardTitle className="flex items-center gap-2 text-sm">
                <GitBranch className="h-4 w-4" />
                Stage Map
              </CardTitle>
            </CardHeader>
            <CardContent className="p-4">
              <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
                {view.stages.map((stage) => (
                  <div key={stage.id} className={cn('rounded-xl border px-3 py-3', stageClassName(stage.status))}>
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-sm font-semibold text-foreground">{stage.label}</span>
                      <Badge variant="outline">{stage.status}</Badge>
                    </div>
                    <p className="mt-2 text-xs text-muted-foreground">{stage.summary}</p>
                    <div className="mt-3 flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
                      <span>{stage.outputCount} outputs</span>
                      <span>{stage.blockerCount} blockers</span>
                    </div>
                    {stage.activeRoles.length > 0 ? (
                      <div className="mt-3 flex flex-wrap gap-1.5">
                        {stage.activeRoles.map((role) => (
                          <span key={`${stage.id}-${role}`} className="rounded-full border border-white/8 bg-background/40 px-2 py-0.5 text-[11px] text-foreground/85">
                            {role}
                          </span>
                        ))}
                      </div>
                    ) : null}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <WorkflowConversationTabs
            tabs={view.conversations}
            value={selectedTab}
            onValueChange={setSelectedTab}
            onSendRoleMessage={canSendRoleGuidance ? handleSendRoleMessage : undefined}
          />

          <div className="grid gap-4 xl:grid-cols-[minmax(0,1.05fr)_minmax(280px,0.95fr)]">
            <Card className="border-white/8 bg-[linear-gradient(180deg,rgba(255,255,255,0.03),rgba(255,255,255,0.015))] p-0">
              <CardHeader className="border-b border-white/8 px-4 py-3">
                <CardTitle className="flex items-center gap-2 text-sm">
                  <Activity className="h-4 w-4" />
                  Narrative
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 p-4">
                {view.narrative.length === 0 ? (
                  <p className="text-sm text-muted-foreground">Readable workflow highlights will appear here as the run emits outputs and artifacts.</p>
                ) : (
                  view.narrative.map((item) => (
                    <div key={item.id} className="rounded-xl border border-white/8 bg-background/40 px-3 py-3">
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <span className="text-sm font-semibold text-foreground">{item.title}</span>
                        {item.timestamp ? (
                          <span className="text-[11px] text-muted-foreground">{new Date(item.timestamp).toLocaleString()}</span>
                        ) : null}
                      </div>
                      <p className="mt-2 text-sm text-muted-foreground">{item.content}</p>
                    </div>
                  ))
                )}
              </CardContent>
            </Card>

            <Card className="border-white/8 bg-[linear-gradient(180deg,rgba(255,255,255,0.03),rgba(255,255,255,0.015))] p-0">
              <CardHeader className="border-b border-white/8 px-4 py-3">
                <CardTitle className="flex items-center gap-2 text-sm">
                  <FileStack className="h-4 w-4" />
                  Artifacts
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 p-4">
                {view.artifacts.length === 0 ? (
                  <p className="text-sm text-muted-foreground">No artifacts have been recorded for this run yet.</p>
                ) : (
                  view.artifacts.map((artifact) => (
                    <div key={artifact.id} className="rounded-xl border border-white/8 bg-background/40 px-3 py-3">
                      <div className="flex items-start justify-between gap-3">
                        <div className="min-w-0">
                          <p className="text-sm font-semibold text-foreground">{artifact.label}</p>
                          <p className="mt-1 text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
                            {artifact.artifactType}
                          </p>
                        </div>
                        {artifact.uri ? <Badge variant="outline">linked</Badge> : null}
                      </div>
                      <p className="mt-2 text-sm text-muted-foreground">{artifact.summary}</p>
                    </div>
                  ))
                )}
              </CardContent>
            </Card>
          </div>

          <Separator className="bg-white/8" />

          <details className="rounded-xl border border-white/8 bg-background/35 px-4 py-3 text-sm text-muted-foreground">
            <summary className="flex cursor-pointer list-none items-center gap-2 font-medium text-foreground">
              <ListTree className="h-4 w-4" />
              Raw Debug
              <Badge variant="outline" className="ml-1">
                <Bug className="h-3 w-3" />
                {view.debugEntries.length} entries
              </Badge>
            </summary>
            <div className="mt-3 space-y-3">
              {view.debugEntries.length === 0 ? (
                <p>No secondary debug rows were provided by the page integration yet.</p>
              ) : (
                view.debugEntries.map((entry) => (
                  <div key={entry.id} className="rounded-lg border border-white/8 bg-black/20 px-3 py-3">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <span className="text-xs font-semibold uppercase tracking-[0.14em] text-foreground/85">
                        {entry.role} / {entry.title}
                      </span>
                      <span className="text-[11px]">{new Date(entry.timestamp).toLocaleString()}</span>
                    </div>
                    <p className="mt-2 whitespace-pre-wrap text-xs leading-6">{entry.body}</p>
                    {entry.meta ? <p className="mt-2 text-[11px] text-muted-foreground">{entry.meta}</p> : null}
                  </div>
                ))
              )}
            </div>
          </details>
        </CardContent>
      </Card>
    </div>
  )
}
