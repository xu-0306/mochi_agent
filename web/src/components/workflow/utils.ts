import type { AgentRunDetail, AgentRunSummary } from '@/lib/api'
import type {
  WorkflowAgentCard,
  WorkflowAgentStatus,
  WorkflowArtifactCard,
  WorkflowConversationMessage,
  WorkflowConversationTab,
  WorkflowDebugEntry,
  WorkflowDeskView,
  WorkflowNarrativeItem,
  WorkflowProgressCardResult,
  WorkflowProgressCardView,
  WorkflowStageCard,
  WorkflowStageStatus,
} from '@/components/workflow/types'

const ROLE_LABELS: Record<string, string> = {
  planner: 'Planner',
  teacher: 'Teacher',
  student: 'Student',
  proposer: 'Proposer',
  solver: 'Solver',
  verifier: 'Verifier',
  synthesizer: 'Synthesizer',
  judge: 'Judge',
  debater_a: 'Debater A',
  debater_b: 'Debater B',
  local_worker: 'Research Worker',
  executor: 'Executor',
  controller: 'Controller',
  evaluator: 'Evaluator',
}

const STAGE_DEFS = [
  { id: 'planning', label: 'Planning', roles: ['planner', 'teacher', 'proposer'] },
  { id: 'research', label: 'Research', roles: ['local_worker', 'solver'] },
  { id: 'debate', label: 'Debate', roles: ['debater_a', 'debater_b', 'judge'] },
  { id: 'verification', label: 'Verification', roles: ['verifier', 'judge'] },
  { id: 'synthesis', label: 'Synthesis', roles: ['synthesizer', 'student'] },
  { id: 'finalization', label: 'Finalization', roles: [] },
] as const

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : null
}

function asString(value: unknown): string | null {
  return typeof value === 'string' && value.trim().length > 0 ? value : null
}

function asStringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string' && item.trim().length > 0) : []
}

function excerpt(value: string | null, max = 160): string | null {
  if (!value) {
    return null
  }
  const compact = value.replace(/\s+/g, ' ').trim()
  if (compact.length <= max) {
    return compact
  }
  return `${compact.slice(0, max - 1)}...`
}

function humanizeToken(value: string | null | undefined): string {
  if (!value) {
    return 'Unknown'
  }
  return value
    .replace(/[_-]+/g, ' ')
    .split(' ')
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ')
}

function formatRoleLabel(roleId: string): string {
  return ROLE_LABELS[roleId] ?? humanizeToken(roleId)
}

function extractSelectedModels(summary: AgentRunSummary | AgentRunDetail | null): Record<string, string> {
  const source = asRecord(summary?.selected_models_roles)
  if (!source) {
    return {}
  }

  const byRole = asRecord(source.by_role)
  if (byRole) {
    return Object.fromEntries(
      Object.entries(byRole).filter((entry): entry is [string, string] => typeof entry[1] === 'string' && entry[1].trim().length > 0)
    )
  }

  return Object.fromEntries(
    Object.entries(source).filter((entry): entry is [string, string] => typeof entry[1] === 'string' && entry[1].trim().length > 0)
  )
}

function findArtifact(summary: AgentRunSummary | AgentRunDetail | null, artifactType: string): Record<string, unknown> | null {
  const artifact = summary?.artifacts.find((item) => item.artifact_type === artifactType)
  const metadata = asRecord(artifact?.metadata)
  return asRecord(metadata?.content)
}

function findRoleSnapshot(run: AgentRunDetail | null): Record<string, unknown> {
  const fromArtifact = findArtifact(run, 'role_task_snapshot')
  const roles = asRecord(fromArtifact?.roles)
  if (roles) {
    return roles
  }

  const resumePayload = asRecord(asRecord(asRecord(run?.summary)?.recovery_state)?.resume_payload)
  const fromSummary = asRecord(asRecord(resumePayload?.role_task_snapshot)?.roles)
  return fromSummary ?? {}
}

function findHealthSnapshot(run: AgentRunDetail | null): Record<string, unknown> {
  return findArtifact(run, 'subagent_health_snapshot') ?? {}
}

function eventTimestamp(event: Record<string, unknown>): string | null {
  return asString(event.timestamp) ?? asString(event.created_at) ?? asString(event.occurred_at)
}

function eventPayload(event: Record<string, unknown>): Record<string, unknown> {
  return asRecord(event.payload) ?? {}
}

function eventRoleId(event: Record<string, unknown>): string | null {
  const payload = eventPayload(event)
  return (
    asString(payload.target_role_id) ??
    asString(payload.role_id) ??
    asString(payload.candidate_id) ??
    asString(event.target_role_id) ??
    asString(event.role_id) ??
    asString(event.author_role_id)
  )
}

function eventStage(event: Record<string, unknown>): string | null {
  const payload = eventPayload(event)
  return asString(payload.stage) ?? asString(payload.current_stage) ?? asString(event.stage)
}

function eventContent(event: Record<string, unknown>): string | null {
  const payload = eventPayload(event)
  return (
    asString(payload.content) ??
    asString(payload.summary) ??
    asString(payload.message) ??
    asString(payload.current_action) ??
    asString(payload.detail) ??
    asString(payload.final_answer) ??
    asString(event.content) ??
    asString(event.guidance)
  )
}

function normalizeAgentStatus(rawStatus: string | null, runStatus: string, currentAction: string): WorkflowAgentStatus {
  const status = (rawStatus ?? '').toLowerCase()
  const action = currentAction.toLowerCase()
  const overall = runStatus.toLowerCase()

  if (status.includes('error') || status.includes('fail')) {
    return 'error'
  }
  if (status.includes('blocked') || status.includes('stalled')) {
    return 'blocked'
  }
  if (status.includes('wait') || overall.includes('awaiting')) {
    return 'waiting'
  }
  if (status.includes('done') || status.includes('complete') || status.includes('success')) {
    return 'done'
  }
  if (status.includes('run') || status.includes('think')) {
    return action.includes('exec') || action.includes('tool') ? 'running_tool' : 'thinking'
  }
  if (status.includes('queue') || status.includes('pending')) {
    return 'queued'
  }
  if (action.includes('exec') || action.includes('tool')) {
    return 'running_tool'
  }
  if (overall.includes('fail') || overall.includes('error')) {
    return 'error'
  }
  if (overall.includes('done') || overall.includes('complete') || overall.includes('success')) {
    return 'done'
  }
  if (overall.includes('run') || overall.includes('resume')) {
    return 'thinking'
  }
  return 'queued'
}

function inferStageId(value: string | null): (typeof STAGE_DEFS)[number]['id'] {
  const lower = (value ?? '').toLowerCase()
  if (lower.includes('plan') || lower.includes('teacher_generation') || lower.includes('proposer')) {
    return 'planning'
  }
  if (lower.includes('research') || lower.includes('source') || lower.includes('evidence')) {
    return 'research'
  }
  if (lower.includes('debate') || lower.includes('judge') || lower.includes('candidate')) {
    return 'debate'
  }
  if (lower.includes('verif')) {
    return 'verification'
  }
  if (lower.includes('synth') || lower.includes('student_generation') || lower.includes('brief')) {
    return 'synthesis'
  }
  if (lower.includes('final') || lower.includes('protocol_completed') || lower.includes('complete')) {
    return 'finalization'
  }
  return 'planning'
}

function buildRoles(summary: AgentRunSummary | AgentRunDetail | null, run: AgentRunDetail | null): string[] {
  const roleIds = new Set<string>()
  Object.keys(extractSelectedModels(summary)).forEach((roleId) => roleIds.add(roleId))
  Object.keys(findRoleSnapshot(run)).forEach((roleId) => roleIds.add(roleId))
  run?.events.forEach((event) => {
    const roleId = eventRoleId(event)
    if (roleId) {
      roleIds.add(roleId)
    }
  })

  if (roleIds.size === 0) {
    const protocolId = summary?.protocol_id ?? ''
    if (protocolId === 'multi_agent_debate') {
      ;['planner', 'debater_a', 'debater_b', 'verifier', 'judge', 'synthesizer'].forEach((roleId) => roleIds.add(roleId))
    } else if (protocolId === 'dr_zero_self_evolve') {
      ;['proposer', 'solver', 'verifier'].forEach((roleId) => roleIds.add(roleId))
    } else {
      ;['teacher', 'student'].forEach((roleId) => roleIds.add(roleId))
    }
  }

  const order = Object.keys(ROLE_LABELS)
  return Array.from(roleIds).sort((left, right) => {
    const leftIndex = order.indexOf(left)
    const rightIndex = order.indexOf(right)
    if (leftIndex === -1 && rightIndex === -1) {
      return left.localeCompare(right)
    }
    if (leftIndex === -1) {
      return 1
    }
    if (rightIndex === -1) {
      return -1
    }
    return leftIndex - rightIndex
  })
}

function buildAgents(summary: AgentRunSummary | AgentRunDetail | null, run: AgentRunDetail | null): WorkflowAgentCard[] {
  const selectedModels = extractSelectedModels(summary)
  const roleSnapshot = findRoleSnapshot(run)
  const healthSnapshot = findHealthSnapshot(run)
  const degradedRoles = new Set(asStringArray(healthSnapshot.degraded_role_ids))
  const failureCounts = asRecord(healthSnapshot.failure_counts) ?? {}

  return buildRoles(summary, run).map((roleId) => {
    const roleState = asRecord(roleSnapshot[roleId]) ?? {}
    const roleEvents = (run?.events ?? []).filter((event) => eventRoleId(event) === roleId)
    const lastEvent = roleEvents[roleEvents.length - 1] ?? null
    const lastPayload = eventPayload(lastEvent ?? {})
    const lastContent = excerpt(
      asString(asRecord(roleState.candidate)?.content) ?? eventContent(lastEvent ?? {})
    )
    const stage = asString(roleState.stage) ?? eventStage(lastEvent ?? {})
    const currentAction =
      asString(lastPayload.current_action) ??
      humanizeToken(stage ?? (roleEvents.length > 0 ? 'recent output recorded' : 'waiting for first turn'))
    const roleStatus = degradedRoles.has(roleId)
      ? 'blocked'
      : normalizeAgentStatus(
          asString(roleState.status) ?? asString(lastPayload.status),
          summary?.status ?? 'unknown',
          currentAction
        )

    return {
      roleId,
      label: formatRoleLabel(roleId),
      modelId: asString(roleState.assigned_model_id) ?? selectedModels[roleId] ?? asString(lastPayload.model_id),
      status: roleStatus,
      stage,
      currentAction,
      lastOutputSummary: lastContent,
      outputCount: roleEvents.filter((event) => asString(event.type) === 'role_output').length,
      updatedAt: eventTimestamp(lastEvent ?? {}) ?? summary?.updated_at ?? null,
    }
  }).map((agent) => {
    const failureCount = Number(failureCounts[agent.roleId] ?? 0)
    if (failureCount > 0 && agent.status !== 'error' && agent.status !== 'done') {
      return { ...agent, status: 'blocked' as WorkflowAgentStatus }
    }
    return agent
  })
}

function buildStages(summary: AgentRunSummary | AgentRunDetail | null, agents: WorkflowAgentCard[], run: AgentRunDetail | null): WorkflowStageCard[] {
  const currentStageId = inferStageId(
    asString(asRecord(summary?.recovery_state)?.stage) ??
      agents.find((agent) => agent.status === 'thinking' || agent.status === 'running_tool')?.stage ??
      eventStage((run?.events ?? [])[run?.events.length ? run.events.length - 1 : 0] ?? {})
  )
  const runStatus = summary?.status ?? 'unknown'
  const runBlocked = runStatus.toLowerCase().includes('awaiting') || runStatus.toLowerCase().includes('stall')
  const finished = Boolean(summary?.finished_at)

  return STAGE_DEFS.map((stageDef, index) => {
    const stageAgents = agents.filter((agent) => {
      if (stageDef.roles.some((role) => role === agent.roleId)) {
        return true
      }
      return inferStageId(agent.stage) === stageDef.id
    })
    const blockerCount = stageAgents.filter((agent) => agent.status === 'blocked' || agent.status === 'error').length
    const outputCount = stageAgents.reduce((sum, agent) => sum + agent.outputCount, 0)

    let status: WorkflowStageStatus = 'pending'
    if (finished && index <= STAGE_DEFS.findIndex((item) => item.id === currentStageId)) {
      status = 'completed'
    } else if (stageDef.id === currentStageId) {
      status = runBlocked ? 'blocked' : 'active'
    } else if (outputCount > 0 || stageAgents.some((agent) => agent.status === 'done')) {
      status = 'completed'
    }

    return {
      id: stageDef.id,
      label: stageDef.label,
      status,
      summary:
        stageAgents.length > 0
          ? `${stageAgents.length} active role${stageAgents.length === 1 ? '' : 's'} in this lane.`
          : 'No visible role activity yet.',
      activeRoles: stageAgents.map((agent) => agent.label),
      outputCount,
      blockerCount,
    }
  })
}

function buildMainConversation(run: AgentRunDetail | null): WorkflowConversationMessage[] {
  return (run?.events ?? [])
    .filter((event) => {
      const type = asString(event.type)
      return type === 'operator_message' || type === 'assistant_message' || type === 'guidance'
    })
    .map((event, index) => {
      const type = asString(event.type)
      return {
        id: `main-${index}`,
        role: type === 'assistant_message' ? 'assistant' : type === 'operator_message' ? 'user' : 'system',
        label: type === 'assistant_message' ? 'Workflow Assistant' : type === 'operator_message' ? 'Operator' : 'Guidance',
        content: eventContent(event) ?? 'No content recorded.',
        timestamp: eventTimestamp(event),
        meta: type === 'guidance' ? 'Queued for the next applicable turn.' : undefined,
      }
    })
}

function buildWorkflowMessages(run: AgentRunDetail | null): WorkflowConversationMessage[] {
  const messages: Array<WorkflowConversationMessage | null> = (run?.events ?? [])
    .map((event, index) => {
      const type = asString(event.type) ?? 'event'
      if (type === 'role_output') {
        const roleId = eventRoleId(event) ?? 'worker'
        return {
          id: `workflow-${index}`,
          role: 'assistant' as const,
          label: formatRoleLabel(roleId),
          content: eventContent(event) ?? `${formatRoleLabel(roleId)} returned an output.`,
          timestamp: eventTimestamp(event),
          meta: humanizeToken(eventStage(event)),
        }
      }
      if (type === 'role_started' || type === 'role_progress' || type === 'role_completed' || type === 'role_error') {
        const roleId = eventRoleId(event) ?? 'worker'
        return {
          id: `workflow-${index}`,
          role: 'system' as const,
          label: formatRoleLabel(roleId),
          content: eventContent(event) ?? `${formatRoleLabel(roleId)} updated progress.`,
          timestamp: eventTimestamp(event),
          meta: humanizeToken(type),
        }
      }
      if (type === 'verification') {
        return {
          id: `workflow-${index}`,
          role: 'system' as const,
          label: 'Verification',
          content: eventContent(event) ?? 'Verification reviewed the candidate outputs.',
          timestamp: eventTimestamp(event),
        }
      }
      if (type === 'evaluation') {
        const selected = asString(eventPayload(event).selected_candidate_id)
        return {
          id: `workflow-${index}`,
          role: 'system' as const,
          label: 'Selection',
          content: selected ? `Selected candidate: ${formatRoleLabel(selected)}.` : 'Evaluation updated the preferred candidate.',
          timestamp: eventTimestamp(event),
        }
      }
      if (type === 'run_started' || type === 'run_completed' || type === 'run_failed' || type === 'run_paused' || type === 'run_resumed') {
        return {
          id: `workflow-${index}`,
          role: 'system' as const,
          label: humanizeToken(type),
          content: eventContent(event) ?? `Run state changed to ${humanizeToken(type)}.`,
          timestamp: eventTimestamp(event),
        }
      }
      return null
    })

  return messages.filter((message): message is WorkflowConversationMessage => message !== null)
}

function buildRoleConversations(agents: WorkflowAgentCard[], run: AgentRunDetail | null): WorkflowConversationTab[] {
  return agents.map((agent) => {
    const messages: WorkflowConversationMessage[] = (run?.events ?? [])
      .filter((event) => {
        const type = asString(event.type)
        return (
          eventRoleId(event) === agent.roleId &&
          (
            type === 'role_output' ||
            type === 'subagent_message' ||
            type === 'role_started' ||
            type === 'role_progress' ||
            type === 'role_completed' ||
            type === 'role_error'
          )
        )
      })
      .map((event, index) => {
        const type = asString(event.type)
        if (type === 'subagent_message') {
          return {
            id: `${agent.roleId}-${index}`,
            role: 'user' as const,
            label: `You -> ${agent.label}`,
            content: eventContent(event) ?? `Guidance sent to ${agent.label}.`,
            timestamp: eventTimestamp(event),
            meta: 'Guidance queued for the next applicable turn.',
          }
        }
        if (type === 'role_started' || type === 'role_progress' || type === 'role_completed' || type === 'role_error') {
          return {
            id: `${agent.roleId}-${index}`,
            role: 'system' as const,
            label: agent.label,
            content: eventContent(event) ?? `${agent.label} updated progress.`,
            timestamp: eventTimestamp(event),
            meta: humanizeToken(type),
          }
        }

        return {
          id: `${agent.roleId}-${index}`,
          role: 'assistant' as const,
          label: agent.label,
          content: eventContent(event) ?? `${agent.label} produced an output.`,
          timestamp: eventTimestamp(event),
          meta: humanizeToken(eventStage(event)),
        }
      })

    if (messages.length === 0) {
      messages.push({
        id: `${agent.roleId}-empty`,
        role: 'system',
        label: 'Role status',
        content: agent.lastOutputSummary
          ? `Latest summary: ${agent.lastOutputSummary}`
          : `${agent.label} has not emitted a dedicated transcript item yet.`,
        timestamp: agent.updatedAt,
        meta: agent.currentAction,
      })
    }

    return {
      id: `role:${agent.roleId}`,
      label: agent.label,
      kind: 'role' as const,
      roleId: agent.roleId,
      status: agent.status,
      messages,
    }
  })
}

function buildNarrative(run: AgentRunDetail | null, agents: WorkflowAgentCard[], artifacts: WorkflowArtifactCard[]): WorkflowNarrativeItem[] {
  const items: WorkflowNarrativeItem[] = []

  agents.forEach((agent) => {
    if (!agent.lastOutputSummary) {
      return
    }
    items.push({
      id: `agent-${agent.roleId}`,
      title: `${agent.label} is ${humanizeToken(agent.status)}`,
      content: agent.lastOutputSummary,
      timestamp: agent.updatedAt,
    })
  })

  if (artifacts.length > 0) {
    artifacts.slice(0, 3).forEach((artifact) => {
      items.push({
        id: `artifact-${artifact.id}`,
        title: `${artifact.label} available`,
        content: artifact.summary,
        timestamp: run?.updated_at ?? null,
      })
    })
  }

  return items.slice(0, 6)
}

function buildArtifacts(run: AgentRunDetail | null): WorkflowArtifactCard[] {
  return (run?.artifacts ?? []).map((artifact, index) => {
    const content = asRecord(artifact.metadata)?.content
    const summary =
      excerpt(
        asString(asRecord(content)?.summary) ??
          asString(asRecord(content)?.description) ??
          asString(asRecord(content)?.status) ??
          asString(asRecord(content)?.final_answer)
      ) ??
      `Stored as ${humanizeToken(artifact.artifact_type)}.`

    return {
      id: artifact.artifact_id ?? `${artifact.artifact_type}-${index}`,
      label: artifact.title ?? humanizeToken(artifact.artifact_type),
      artifactType: artifact.artifact_type,
      summary,
      uri: artifact.uri,
    }
  })
}

export function buildWorkflowDeskView(input: {
  run: AgentRunDetail | null
  summary?: AgentRunSummary | null
  debugEntries?: WorkflowDebugEntry[]
}): WorkflowDeskView {
  const summary = input.run ?? input.summary ?? null
  const artifacts = buildArtifacts(input.run)
  const agents = buildAgents(summary, input.run)
  const stages = buildStages(summary, agents, input.run)
  const mainConversation = buildMainConversation(input.run)
  const workflowConversation = buildWorkflowMessages(input.run)
  const roleConversations = buildRoleConversations(agents, input.run)
  const phaseLabel = stages.find((stage) => stage.status === 'active' || stage.status === 'blocked')?.label
    ?? stages.filter((stage) => stage.status === 'completed').slice(-1)[0]?.label
    ?? 'Planning'

  return {
    runId: summary?.run_id ?? null,
    title: summary?.title ?? summary?.topic ?? summary?.run_id ?? 'Workflow Desk',
    status: summary?.status ?? 'idle',
    protocolId: summary?.protocol_id ?? null,
    phaseLabel,
    updatedAt: summary?.updated_at ?? null,
    startedAt: summary?.started_at ?? null,
    finishedAt: summary?.finished_at ?? null,
    agents,
    stages,
    conversations: [
      {
        id: 'main',
        label: 'Main',
        kind: 'main',
        roleId: null,
        messages: mainConversation.length > 0
          ? mainConversation
          : [{
              id: 'main-empty',
              role: 'system',
              label: 'Main',
              content: 'No operator or assistant messages have been recorded for this run yet.',
              timestamp: null,
            }],
      },
      {
        id: 'workflow',
        label: 'Workflow',
        kind: 'workflow',
        roleId: null,
        messages: workflowConversation.length > 0
          ? workflowConversation
          : [{
              id: 'workflow-empty',
              role: 'system',
              label: 'Workflow',
              content: 'Workflow-level progress updates will appear here once the run emits role, verification, or lifecycle events.',
              timestamp: null,
            }],
      },
      ...roleConversations,
    ],
    narrative: buildNarrative(input.run, agents, artifacts),
    artifacts,
    debugEntries: input.debugEntries ?? [],
    rawEventCount: input.run?.events.length ?? 0,
  }
}

function workflowResultStatus(status: string): WorkflowProgressCardResult['status'] {
  const normalized = status.toLowerCase()
  if (normalized === 'succeeded' || normalized === 'completed' || normalized === 'done') {
    return 'complete'
  }
  if (normalized === 'partial') {
    return 'partial'
  }
  if (normalized === 'failed' || normalized === 'error' || normalized === 'cancelled') {
    return 'error'
  }
  return 'pending'
}

export function buildWorkflowProgressCardView(run: AgentRunDetail | null): WorkflowProgressCardView | null {
  if (!run?.run_id) {
    return null
  }

  const view = buildWorkflowDeskView({ run })
  const resultStatus = workflowResultStatus(view.status)
  const finalAnswer = asString(run.summary?.final_answer)
  const latestError = asString(run.latest_error)
  const activeStage = view.stages.find((stage) => stage.status === 'active' || stage.status === 'blocked')
  const completedStageCount = view.stages.filter((stage) => stage.status === 'completed').length
  const activeAgents = view.agents.filter((agent) =>
    agent.status === 'thinking' ||
    agent.status === 'running_tool' ||
    agent.status === 'waiting' ||
    agent.status === 'blocked'
  )

  return {
    runId: run.run_id,
    title: view.title,
    status: view.status,
    phaseLabel: view.phaseLabel,
    summary: activeStage
      ? `${activeStage.label}: ${activeStage.summary}`
      : `${completedStageCount} workflow stage${completedStageCount === 1 ? '' : 's'} completed.`,
    startedAt: view.startedAt,
    updatedAt: view.updatedAt,
    finishedAt: view.finishedAt,
    latestError,
    roles: (activeAgents.length > 0 ? activeAgents : view.agents).slice(0, 4).map((agent) => ({
      roleId: agent.roleId,
      label: agent.label,
      status: agent.status,
      currentAction: agent.currentAction,
      lastOutputSummary: agent.lastOutputSummary,
      updatedAt: agent.updatedAt,
    })),
    transcriptSnippets: view.conversations
      .flatMap((conversation) =>
        conversation.kind === 'main' || conversation.kind === 'role' ? conversation.messages : []
      )
      .filter((message) => message.id !== 'main-empty')
      .slice(-4),
    finalResult: {
      status: resultStatus,
      content: finalAnswer ?? (resultStatus === 'error' ? latestError : null),
      source: finalAnswer ? 'run_summary' : latestError ? 'latest_error' : 'none',
    },
  }
}
