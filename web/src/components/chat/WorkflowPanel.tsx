'use client'

import * as React from 'react'
import { ExternalLink, PanelRightClose, RotateCcw, Workflow } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  FloatingPanelShell,
} from '@/components/chat/FloatingPanelShell'
import { PanelSectionCard } from '@/components/chat/PanelSectionCard'
import { Switch } from '@/components/ui/switch'
import type {
  AgentRunProtocolId,
  AgentRunRunPolicy,
  ProjectSummary,
  ReasoningEffort,
  SessionWorkflowConfig,
  SessionWorkflowState,
} from '@/lib/api'
type WorkflowScheduleType = 'interval' | 'once' | 'cron'

interface WorkflowProtocolOption {
  value: AgentRunProtocolId
  label: string
  description: string
}

interface WorkflowPanelProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  workflowEnabled: boolean
  workflowBusy: boolean
  workflowError: string | null
  workflowBoundRunId: string | null
  workflowState: SessionWorkflowState
  workflowConfig: SessionWorkflowConfig
  workflowProtocolId: AgentRunProtocolId
  workflowReasoningEffort: ReasoningEffort | null
  workflowRunPolicy: AgentRunRunPolicy
  workflowExecutionPolicy: Record<string, unknown>
  workflowEvidenceConfig: Record<string, unknown>
  workflowScheduleConfig: Record<string, unknown>
  workflowScheduleType: WorkflowScheduleType
  workflowScheduleEnabled: boolean
  workflowProtocolOptions: WorkflowProtocolOption[]
  supportedReasoningEfforts: ReasoningEffort[]
  effectiveInferenceReasoningEffort: ReasoningEffort | null
  effectiveProjectId: string | null
  projects: ProjectSummary[]
  workflowProjectWorkspace: string | null
  uploadTargetDir: string | null
  effectiveWorkflowWorkspace: string | null
  onWorkflowToggle: (enabled: boolean) => void
  onWorkflowNewRun: () => void
  onOpenRunDetail: (runId: string) => void
  onWorkflowProjectChange: (projectId: string | null) => void
  onWorkflowFieldChange: (patch: Partial<SessionWorkflowState>) => void
  onWorkflowConfigPatch: (patch: Partial<SessionWorkflowConfig>) => void
  onWorkflowSave: () => void
  buildWorkflowScheduleConfig: (
    schedule: Record<string, unknown>,
    type: WorkflowScheduleType,
    enabled: boolean
  ) => Record<string, unknown>
  formatWorkflowScheduleRunAt: (value: unknown) => string
  defaultScheduleTimezone: () => string
}

function WorkflowPanelBody({
  workflowEnabled,
  workflowBusy,
  workflowError,
  workflowBoundRunId,
  workflowState,
  workflowConfig,
  workflowProtocolId,
  workflowReasoningEffort,
  workflowRunPolicy,
  workflowExecutionPolicy,
  workflowEvidenceConfig,
  workflowScheduleConfig,
  workflowScheduleType,
  workflowScheduleEnabled,
  workflowProtocolOptions,
  supportedReasoningEfforts,
  effectiveInferenceReasoningEffort,
  effectiveProjectId,
  projects,
  workflowProjectWorkspace,
  uploadTargetDir,
  effectiveWorkflowWorkspace,
  onWorkflowToggle,
  onWorkflowNewRun,
  onOpenRunDetail,
  onWorkflowProjectChange,
  onWorkflowFieldChange,
  onWorkflowConfigPatch,
  onWorkflowSave,
  buildWorkflowScheduleConfig,
  formatWorkflowScheduleRunAt,
  defaultScheduleTimezone,
  onClose,
}: Omit<WorkflowPanelProps, 'open' | 'onOpenChange'> & {
  onClose?: () => void
}) {
  return (
    <div className="flex h-full flex-col overflow-hidden rounded-[inherit] bg-[radial-gradient(circle_at_top_right,rgba(94,106,210,0.18),transparent_34%),linear-gradient(180deg,rgba(255,255,255,0.04),transparent_24%)]">
      <div className="border-b border-white/8 bg-canvas/40 px-4 py-4 backdrop-blur-xl">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="mb-2 inline-flex items-center gap-2 rounded-full border border-primary-500/20 bg-primary-500/10 px-2.5 py-1 text-[11px] font-medium uppercase tracking-[0.18em] text-primary-300">
              <Workflow className="h-3.5 w-3.5" />
              Workflow Desk
            </div>
            <h2 className="text-base font-semibold text-foreground">Workflow controls</h2>
            <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
              Keep this conversation bound to one orchestrated runtime without leaving the main chat.
            </p>
          </div>
          {onClose ? (
            <Button
              type="button"
              size="icon-sm"
              variant="ghost"
              onClick={onClose}
              title="Hide workflow controls"
              aria-label="Hide workflow controls"
              className="mt-0.5 rounded-full border border-white/8 bg-canvas/55 text-muted-foreground hover:bg-elevated-layer hover:text-foreground"
            >
              <PanelRightClose className="h-4 w-4" />
            </Button>
          ) : null}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-4">
        <div className="space-y-4">
          <PanelSectionCard
            title="Workflow mode"
            description="Route new chat turns into the workflow runtime for this session."
          >
            <div className="space-y-3">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="text-sm font-medium text-foreground">Enable workflow</p>
                  <p className="text-xs text-muted-foreground">
                    Switch between direct chat replies and the bound workflow lane.
                  </p>
                </div>
                <Switch checked={workflowEnabled} onCheckedChange={onWorkflowToggle} />
              </div>
              <div className="rounded-xl border border-white/8 bg-surface-layer/60 px-3 py-2 text-xs text-muted-foreground">
                {workflowEnabled
                  ? 'Workflow mode is active for this chat session.'
                  : 'Workflow mode is off. Use /workflow or this switch to enable it.'}
              </div>
            </div>
          </PanelSectionCard>

          <PanelSectionCard
            title="Bound run"
            description="This chat keeps appending to the same workflow run unless you intentionally rotate it."
          >
            <div className="space-y-3">
              <div className="flex items-center justify-between gap-2">
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium text-foreground">
                    {workflowBoundRunId ?? 'No run bound yet'}
                  </p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    {workflowBoundRunId
                      ? `Synced events: ${workflowState.synced_run_event_count ?? 0}`
                      : 'The first workflow message will create and bind a run.'}
                  </p>
                </div>
                <Button type="button" variant="outline" size="sm" onClick={onWorkflowNewRun}>
                  <RotateCcw className="h-3.5 w-3.5" />
                  New run
                </Button>
              </div>
              {workflowBoundRunId ? (
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="rounded-full px-0"
                  onClick={() => onOpenRunDetail(workflowBoundRunId)}
                >
                  <ExternalLink className="h-3.5 w-3.5" />
                  Open run detail
                </Button>
              ) : null}
            </div>
          </PanelSectionCard>

          <PanelSectionCard
            title="Project / workspace"
            description="Files and execution resolve from the selected project unless you override the path."
          >
            <div className="space-y-3">
              <div className="space-y-2">
                <span className="text-xs font-medium text-muted-foreground">Project</span>
                <Select
                  value={effectiveProjectId ?? '__none__'}
                  onValueChange={(value) => onWorkflowProjectChange(value === '__none__' ? null : value)}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="No project" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="__none__">No project</SelectItem>
                    {projects.map((project) => (
                      <SelectItem key={project.id} value={project.id}>
                        {project.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <span className="text-xs font-medium text-muted-foreground">Workspace override</span>
                <Input
                  value={workflowState.workspace_dir_override ?? ''}
                  placeholder={workflowProjectWorkspace ?? uploadTargetDir ?? 'Use project workspace'}
                  onChange={(event) =>
                    onWorkflowFieldChange({
                      workspace_dir_override: event.target.value || null,
                      config: {
                        ...(workflowState.config ?? {}),
                        workspace_dir_override: event.target.value || null,
                      },
                    })
                  }
                />
              </div>
              <div className="rounded-xl border border-white/8 bg-surface-layer/60 px-3 py-2 text-xs text-muted-foreground">
                Effective workspace:{' '}
                <span className="break-all text-foreground">{effectiveWorkflowWorkspace || 'Not set'}</span>
              </div>
            </div>
          </PanelSectionCard>

          <PanelSectionCard
            title="Protocol / reasoning"
            description="Session-scoped defaults applied whenever a new bound run is created."
          >
            <div className="space-y-3">
              <div className="space-y-2">
                <span className="text-xs font-medium text-muted-foreground">Title</span>
                <Input
                  value={workflowConfig.title ?? ''}
                  placeholder="Optional workflow title"
                  onChange={(event) =>
                    onWorkflowFieldChange({
                      config: {
                        ...(workflowState.config ?? {}),
                        title: event.target.value || null,
                      },
                    })
                  }
                />
              </div>
              <div className="space-y-2">
                <span className="text-xs font-medium text-muted-foreground">Protocol</span>
                <Select
                  value={workflowProtocolId}
                  onValueChange={(value) =>
                    onWorkflowFieldChange({
                      config: {
                        ...(workflowState.config ?? {}),
                        protocol_id: value as AgentRunProtocolId,
                      },
                    })
                  }
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {workflowProtocolOptions.map((option) => (
                      <SelectItem key={option.value} value={option.value}>
                        {option.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground">
                  {workflowProtocolOptions.find((option) => option.value === workflowProtocolId)?.description}
                </p>
              </div>
              <div className="space-y-2">
                <span className="text-xs font-medium text-muted-foreground">Reasoning effort</span>
                <Select
                  value={workflowReasoningEffort ?? '__inherit__'}
                  onValueChange={(value) =>
                    onWorkflowFieldChange({
                      config: {
                        ...(workflowState.config ?? {}),
                        reasoning_effort:
                          value === '__inherit__'
                            ? null
                            : value as ReasoningEffort,
                      },
                    })
                  }
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="__inherit__">Inherit chat setting</SelectItem>
                    {supportedReasoningEfforts.map((option) => (
                      <SelectItem key={option} value={option}>
                        {option}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
          </PanelSectionCard>

          <PanelSectionCard
            title="Execution / schedule"
            description="These are defaults only. Controlled execution boundaries remain enforced by runtime policy."
          >
            <div className="space-y-3">
              <div className="rounded-xl border border-white/8 bg-surface-layer/70 px-3 py-3">
                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-2">
                    <span className="text-xs font-medium text-muted-foreground">Max wall clock (sec)</span>
                    <Input
                      value={String(workflowRunPolicy.max_wall_clock_sec ?? '')}
                      placeholder="1800"
                      onChange={(event) =>
                        onWorkflowConfigPatch({
                          run_policy: {
                            ...workflowRunPolicy,
                            max_wall_clock_sec: event.target.value
                              ? Number.parseInt(event.target.value, 10)
                              : null,
                          },
                        })
                      }
                    />
                  </div>
                  <div className="space-y-2">
                    <span className="text-xs font-medium text-muted-foreground">Heartbeat timeout (sec)</span>
                    <Input
                      value={String(workflowRunPolicy.heartbeat_timeout_sec ?? '')}
                      placeholder="90"
                      onChange={(event) =>
                        onWorkflowConfigPatch({
                          run_policy: {
                            ...workflowRunPolicy,
                            heartbeat_timeout_sec: event.target.value
                              ? Number.parseInt(event.target.value, 10)
                              : null,
                          },
                        })
                      }
                    />
                  </div>
                </div>
                <div className="mt-3 grid grid-cols-2 gap-3">
                  <div className="space-y-2">
                    <span className="text-xs font-medium text-muted-foreground">Checkpoint steps</span>
                    <Input
                      value={String(workflowRunPolicy.checkpoint_interval_steps ?? '')}
                      placeholder="1"
                      onChange={(event) =>
                        onWorkflowConfigPatch({
                          run_policy: {
                            ...workflowRunPolicy,
                            checkpoint_interval_steps: event.target.value
                              ? Number.parseInt(event.target.value, 10)
                              : undefined,
                          },
                        })
                      }
                    />
                  </div>
                  <div className="space-y-2">
                    <span className="text-xs font-medium text-muted-foreground">Max subagent failures</span>
                    <Input
                      value={String(workflowRunPolicy.max_subagent_failures_per_role ?? '')}
                      placeholder="2"
                      onChange={(event) =>
                        onWorkflowConfigPatch({
                          run_policy: {
                            ...workflowRunPolicy,
                            max_subagent_failures_per_role: event.target.value
                              ? Number.parseInt(event.target.value, 10)
                              : undefined,
                          },
                        })
                      }
                    />
                  </div>
                </div>
                <div className="mt-3 grid grid-cols-2 gap-3">
                  <div className="space-y-2">
                    <span className="text-xs font-medium text-muted-foreground">On budget exhausted</span>
                    <Select
                      value={workflowRunPolicy.on_budget_exhausted ?? 'finalize_partial'}
                      onValueChange={(value) =>
                        onWorkflowConfigPatch({
                          run_policy: {
                            ...workflowRunPolicy,
                            on_budget_exhausted: value as 'pause' | 'finalize_partial',
                          },
                        })
                      }
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="finalize_partial">Finalize partial</SelectItem>
                        <SelectItem value="pause">Pause</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <span className="text-xs font-medium text-muted-foreground">On subagent disconnect</span>
                    <Select
                      value={workflowRunPolicy.on_subagent_disconnect ?? 'pause'}
                      onValueChange={(value) =>
                        onWorkflowConfigPatch({
                          run_policy: {
                            ...workflowRunPolicy,
                            on_subagent_disconnect: value as 'retry_then_degrade' | 'pause' | 'fail',
                          },
                        })
                      }
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="pause">Pause</SelectItem>
                        <SelectItem value="retry_then_degrade">Retry then degrade</SelectItem>
                        <SelectItem value="fail">Fail</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </div>

              <div className="rounded-xl border border-white/8 bg-surface-layer/70 px-3 py-3">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="text-sm font-medium text-foreground">Controlled execution</p>
                    <p className="text-xs text-muted-foreground">
                      Keep subagents proposal-only while the controller owns runtime execution.
                    </p>
                  </div>
                  <Switch
                    checked={workflowExecutionPolicy.mode === 'controlled'}
                    onCheckedChange={(checked) =>
                      onWorkflowConfigPatch({
                        execution_policy: checked
                          ? {
                              ...workflowExecutionPolicy,
                              mode: 'controlled',
                              max_execution_requests:
                                Number(workflowExecutionPolicy.max_execution_requests) || 1,
                              max_commands_per_request:
                                Number(workflowExecutionPolicy.max_commands_per_request) || 1,
                              default_timeout_sec:
                                Number(workflowExecutionPolicy.default_timeout_sec) || 300,
                              background_allowed:
                                workflowExecutionPolicy.background_allowed !== false,
                            }
                          : {},
                      })
                    }
                  />
                </div>
                {workflowExecutionPolicy.mode === 'controlled' ? (
                  <div className="mt-3 grid grid-cols-2 gap-3">
                    <div className="space-y-2">
                      <span className="text-xs font-medium text-muted-foreground">Max exec requests</span>
                      <Input
                        value={String(workflowExecutionPolicy.max_execution_requests ?? '')}
                        placeholder="1"
                        onChange={(event) =>
                          onWorkflowConfigPatch({
                            execution_policy: {
                              ...workflowExecutionPolicy,
                              mode: 'controlled',
                              max_execution_requests: event.target.value
                                ? Number.parseInt(event.target.value, 10)
                                : 1,
                            },
                          })
                        }
                      />
                    </div>
                    <div className="space-y-2">
                      <span className="text-xs font-medium text-muted-foreground">Max commands / request</span>
                      <Input
                        value={String(workflowExecutionPolicy.max_commands_per_request ?? '')}
                        placeholder="1"
                        onChange={(event) =>
                          onWorkflowConfigPatch({
                            execution_policy: {
                              ...workflowExecutionPolicy,
                              mode: 'controlled',
                              max_commands_per_request: event.target.value
                                ? Number.parseInt(event.target.value, 10)
                                : 1,
                            },
                          })
                        }
                      />
                    </div>
                    <div className="space-y-2">
                      <span className="text-xs font-medium text-muted-foreground">Default exec timeout</span>
                      <Input
                        value={String(workflowExecutionPolicy.default_timeout_sec ?? '')}
                        placeholder="300"
                        onChange={(event) =>
                          onWorkflowConfigPatch({
                            execution_policy: {
                              ...workflowExecutionPolicy,
                              mode: 'controlled',
                              default_timeout_sec: event.target.value
                                ? Number.parseInt(event.target.value, 10)
                                : 300,
                            },
                          })
                        }
                      />
                    </div>
                    <div className="flex items-center justify-between gap-3 rounded-md border border-border px-3 py-2">
                      <span className="text-xs text-muted-foreground">Allow background execution</span>
                      <Switch
                        checked={workflowExecutionPolicy.background_allowed !== false}
                        onCheckedChange={(checked) =>
                          onWorkflowConfigPatch({
                            execution_policy: {
                              ...workflowExecutionPolicy,
                              mode: 'controlled',
                              background_allowed: checked,
                            },
                          })
                        }
                      />
                    </div>
                  </div>
                ) : null}
              </div>

              <div className="rounded-xl border border-white/8 bg-surface-layer/70 px-3 py-3">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="text-sm font-medium text-foreground">Evidence collection</p>
                    <p className="text-xs text-muted-foreground">
                      Configure retrieval defaults for new workflow runs.
                    </p>
                  </div>
                  <Switch
                    checked={workflowEvidenceConfig.enabled !== false}
                    onCheckedChange={(checked) =>
                      onWorkflowConfigPatch({
                        evidence: {
                          ...workflowEvidenceConfig,
                          enabled: checked,
                          mode: String(workflowEvidenceConfig.mode ?? 'hybrid'),
                        },
                      })
                    }
                  />
                </div>
                <div className="mt-3 grid grid-cols-2 gap-3">
                  <div className="space-y-2">
                    <span className="text-xs font-medium text-muted-foreground">Mode</span>
                    <Select
                      value={String(workflowEvidenceConfig.mode ?? 'hybrid')}
                      onValueChange={(value) =>
                        onWorkflowConfigPatch({
                          evidence: {
                            ...workflowEvidenceConfig,
                            enabled: workflowEvidenceConfig.enabled !== false,
                            mode: value,
                          },
                        })
                      }
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="hybrid">Hybrid</SelectItem>
                        <SelectItem value="web_only">Web only</SelectItem>
                        <SelectItem value="local_only">Local only</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <span className="text-xs font-medium text-muted-foreground">Max results / query</span>
                    <Input
                      value={String(workflowEvidenceConfig.max_results_per_query ?? '')}
                      placeholder="3"
                      onChange={(event) =>
                        onWorkflowConfigPatch({
                          evidence: {
                            ...workflowEvidenceConfig,
                            enabled: workflowEvidenceConfig.enabled !== false,
                            max_results_per_query: event.target.value
                              ? Number.parseInt(event.target.value, 10)
                              : 3,
                          },
                        })
                      }
                    />
                  </div>
                </div>
              </div>

              <div className="rounded-xl border border-white/8 bg-surface-layer/70 px-3 py-3">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="text-sm font-medium text-foreground">Schedule</p>
                    <p className="text-xs text-muted-foreground">
                      Let this workflow execute via backend scheduling instead of immediate start.
                    </p>
                  </div>
                  <Switch
                    checked={workflowScheduleEnabled}
                    onCheckedChange={(checked) =>
                      onWorkflowConfigPatch({
                        schedule: buildWorkflowScheduleConfig(
                          workflowScheduleConfig,
                          workflowScheduleType,
                          checked
                        ),
                      })
                    }
                  />
                </div>
                {workflowScheduleEnabled ? (
                  <>
                    <div className="mt-3 space-y-2">
                      <span className="text-xs font-medium text-muted-foreground">Schedule type</span>
                      <Select
                        value={workflowScheduleType}
                        onValueChange={(value) =>
                          onWorkflowConfigPatch({
                            schedule: buildWorkflowScheduleConfig(
                              workflowScheduleConfig,
                              value as WorkflowScheduleType,
                              true
                            ),
                          })
                        }
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="interval">Interval</SelectItem>
                          <SelectItem value="once">One-shot</SelectItem>
                          <SelectItem value="cron">Cron</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    {workflowScheduleType === 'interval' ? (
                      <div className="mt-3 space-y-2">
                        <span className="text-xs font-medium text-muted-foreground">Interval seconds</span>
                        <Input
                          value={String(workflowScheduleConfig.interval_seconds ?? '')}
                          placeholder="3600"
                          onChange={(event) =>
                            onWorkflowConfigPatch({
                              schedule: {
                                ...workflowScheduleConfig,
                                enabled: true,
                                interval_seconds: event.target.value
                                  ? Number.parseInt(event.target.value, 10)
                                  : 3600,
                              },
                            })
                          }
                        />
                      </div>
                    ) : null}
                    {workflowScheduleType === 'once' ? (
                      <div className="mt-3 space-y-2">
                        <span className="text-xs font-medium text-muted-foreground">Run at</span>
                        <Input
                          type="datetime-local"
                          value={formatWorkflowScheduleRunAt(workflowScheduleConfig.run_at)}
                          onChange={(event) =>
                            onWorkflowConfigPatch({
                              schedule: {
                                ...workflowScheduleConfig,
                                enabled: true,
                                run_at: event.target.value,
                              },
                            })
                          }
                        />
                      </div>
                    ) : null}
                    {workflowScheduleType === 'cron' ? (
                      <div className="mt-3 space-y-2">
                        <span className="text-xs font-medium text-muted-foreground">Cron</span>
                        <Input
                          value={String(workflowScheduleConfig.cron ?? '')}
                          placeholder="0 9 * * 1"
                          onChange={(event) =>
                            onWorkflowConfigPatch({
                              schedule: {
                                ...workflowScheduleConfig,
                                enabled: true,
                                cron: event.target.value,
                              },
                            })
                          }
                        />
                      </div>
                    ) : null}
                    <div className="mt-3 grid grid-cols-2 gap-3">
                      <div className="space-y-2">
                        <span className="text-xs font-medium text-muted-foreground">Timezone</span>
                        <Input
                          value={String(workflowScheduleConfig.timezone ?? '')}
                          placeholder={defaultScheduleTimezone()}
                          onChange={(event) =>
                            onWorkflowConfigPatch({
                              schedule: {
                                ...workflowScheduleConfig,
                                enabled: true,
                                timezone: event.target.value,
                              },
                            })
                          }
                        />
                      </div>
                      <div className="space-y-2">
                        <span className="text-xs font-medium text-muted-foreground">Max runs</span>
                        <Input
                          value={
                            workflowScheduleConfig.max_runs === null || workflowScheduleConfig.max_runs === undefined
                              ? ''
                              : String(workflowScheduleConfig.max_runs)
                          }
                          placeholder="Optional"
                          onChange={(event) =>
                            onWorkflowConfigPatch({
                              schedule: {
                                ...workflowScheduleConfig,
                                enabled: true,
                                max_runs: event.target.value
                                  ? Number.parseInt(event.target.value, 10)
                                  : null,
                              },
                            })
                          }
                        />
                      </div>
                    </div>
                    {workflowScheduleType !== 'once' ? (
                      <div className="mt-3 flex items-center justify-between gap-3 rounded-md border border-border px-3 py-2">
                        <span className="text-xs text-muted-foreground">Start immediately</span>
                        <Switch
                          checked={workflowScheduleConfig.start_immediately !== false}
                          onCheckedChange={(checked) =>
                            onWorkflowConfigPatch({
                              schedule: {
                                ...workflowScheduleConfig,
                                enabled: true,
                                start_immediately: checked,
                              },
                            })
                          }
                        />
                      </div>
                    ) : null}
                    <div className="mt-3 flex items-center justify-between gap-3 rounded-md border border-border px-3 py-2">
                      <span className="text-xs text-muted-foreground">Auto-pause on failure</span>
                      <Switch
                        checked={workflowScheduleConfig.auto_pause_on_failure !== false}
                        onCheckedChange={(checked) =>
                          onWorkflowConfigPatch({
                            schedule: {
                              ...workflowScheduleConfig,
                              enabled: true,
                              auto_pause_on_failure: checked,
                            },
                          })
                        }
                      />
                    </div>
                  </>
                ) : null}
              </div>
            </div>
          </PanelSectionCard>

          <PanelSectionCard
            title="Status / recovery"
            description="Full artifacts, logs, and recovery actions stay available from the run detail page."
          >
            <div className="rounded-xl border border-white/8 bg-surface-layer/60 px-3 py-3 text-xs text-muted-foreground">
              <p>Workflow busy: {workflowBusy ? 'Yes' : 'No'}</p>
              <p>Last error: {workflowError ?? 'None'}</p>
            </div>
          </PanelSectionCard>

          <div className="flex items-center justify-end gap-2 pt-1">
            {onClose ? (
              <Button type="button" variant="outline" onClick={onClose}>
                Close
              </Button>
            ) : null}
            <Button type="button" onClick={onWorkflowSave}>
              Save settings
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}

export function WorkflowPanel({
  open,
  onOpenChange,
  ...bodyProps
}: WorkflowPanelProps) {
  return (
    <FloatingPanelShell
      open={open}
      onOpenChange={onOpenChange}
      desktopSide="right"
      desktopWidthClass="w-[26rem]"
      desktopBreakpoint="lg"
    >
      <WorkflowPanelBody {...bodyProps} onClose={() => onOpenChange(false)} />
    </FloatingPanelShell>
  )
}
