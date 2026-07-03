import type { ExecutionTranscriptEvent, SubagentTranscriptSummary } from '@/lib/api'
import {
  mergeSubagentSummaries,
  projectGoalSurfaceExecutionEvents,
} from '@/lib/execution-transcript'

export interface GoalExecutionProjectionView {
  allVisibleSubagents: SubagentTranscriptSummary[]
  goalSurfaceTimelineEvents: ExecutionTranscriptEvent[]
  goalSurfaceTimelineEventsById: Map<string, ExecutionTranscriptEvent[]>
  goalSurfaceSubagents: SubagentTranscriptSummary[]
  subagentTimelineEventsById: Map<string, ExecutionTranscriptEvent[]>
}

function transcriptEventsForSubagent(
  events: ExecutionTranscriptEvent[],
  subagent: SubagentTranscriptSummary
): ExecutionTranscriptEvent[] {
  return events.filter((event) => {
    if (event.subagentId && event.subagentId === subagent.subagentId) {
      return true
    }
    if (event.roleId && subagent.roleId && event.roleId === subagent.roleId) {
      return true
    }
    return false
  })
}

export function isGoalSurfaceSubagentVisible(
  subagent: SubagentTranscriptSummary,
  recentEvents: ExecutionTranscriptEvent[]
): boolean {
  const status = subagent.status.trim().toLowerCase()
  const hasIdentity = Boolean((subagent.title ?? '').trim() || (subagent.roleId ?? '').trim())
  const hasSummary = Boolean(
    (subagent.summary ?? '').trim() ||
      (subagent.promptPreview ?? '').trim() ||
      (subagent.outputPreview ?? '').trim()
  )

  if (recentEvents.length > 0) {
    return true
  }
  if (!hasIdentity) {
    return false
  }
  if (
    status === 'blocked' ||
    status === 'awaiting_approval' ||
    status === 'failed' ||
    status === 'cancelled' ||
    status === 'interrupted'
  ) {
    return true
  }
  if (status === 'completed' || status === 'running') {
    return hasSummary
  }
  return hasSummary
}

export function buildGoalExecutionProjectionView(input: {
  executionTimelineEvents: ExecutionTranscriptEvent[]
  sessionSubagents: SubagentTranscriptSummary[]
  agentRunSubagents: SubagentTranscriptSummary[]
}): GoalExecutionProjectionView {
  const allVisibleSubagents = mergeSubagentSummaries(
    input.sessionSubagents,
    input.agentRunSubagents
  )
  const goalSurfaceTimelineEvents = projectGoalSurfaceExecutionEvents(input.executionTimelineEvents)
  const goalSurfaceTimelineEventsById = new Map<string, ExecutionTranscriptEvent[]>()
  for (const subagent of allVisibleSubagents) {
    goalSurfaceTimelineEventsById.set(
      subagent.subagentId,
      transcriptEventsForSubagent(goalSurfaceTimelineEvents, subagent)
    )
  }

  const goalSurfaceSubagents = allVisibleSubagents.filter((subagent) =>
    isGoalSurfaceSubagentVisible(
      subagent,
      goalSurfaceTimelineEventsById.get(subagent.subagentId) ?? []
    )
  )

  const subagentTimelineEventsById = new Map<string, ExecutionTranscriptEvent[]>()
  for (const subagent of goalSurfaceSubagents) {
    subagentTimelineEventsById.set(
      subagent.subagentId,
      goalSurfaceTimelineEventsById.get(subagent.subagentId) ?? []
    )
  }

  return {
    allVisibleSubagents,
    goalSurfaceTimelineEvents,
    goalSurfaceTimelineEventsById,
    goalSurfaceSubagents,
    subagentTimelineEventsById,
  }
}
