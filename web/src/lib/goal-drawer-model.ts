export type GoalDrawerStatus =
  | 'proposed'
  | 'running'
  | 'paused'
  | 'stopped'
  | 'completed'
  | 'failed'

export interface GoalDrawerSnapshot {
  goalId: string
  title: string
  status: GoalDrawerStatus
  hasLinkedAgentRun: boolean
}

export interface GoalDrawerModel {
  goalId: string
  title: string
  status: GoalDrawerStatus
  canPause: boolean
  canResume: boolean
  canStop: boolean
  canEdit: boolean
  canClear: boolean
  canOpenExecutionRecord: boolean
}

type GoalDrawerActionEligibility = Pick<
  GoalDrawerModel,
  'canPause' | 'canResume' | 'canStop' | 'canEdit' | 'canClear'
>

const actionEligibilityByStatus: Record<GoalDrawerStatus, GoalDrawerActionEligibility> = {
  proposed: {
    canPause: false,
    canResume: false,
    canStop: false,
    canEdit: true,
    canClear: true,
  },
  running: {
    canPause: true,
    canResume: false,
    canStop: true,
    canEdit: true,
    canClear: true,
  },
  paused: {
    canPause: false,
    canResume: true,
    canStop: true,
    canEdit: true,
    canClear: true,
  },
  stopped: {
    canPause: false,
    canResume: true,
    canStop: false,
    canEdit: true,
    canClear: true,
  },
  completed: {
    canPause: false,
    canResume: false,
    canStop: false,
    canEdit: false,
    canClear: true,
  },
  failed: {
    canPause: false,
    canResume: true,
    canStop: false,
    canEdit: true,
    canClear: true,
  },
}

function isGoalDrawerStatus(value: unknown): value is GoalDrawerStatus {
  return (
    typeof value === 'string' &&
    Object.prototype.hasOwnProperty.call(actionEligibilityByStatus, value)
  )
}

export function projectGoalDrawerState(
  snapshot: GoalDrawerSnapshot | null | undefined
): GoalDrawerModel | null {
  if (
    !snapshot ||
    typeof snapshot.goalId !== 'string' ||
    snapshot.goalId.trim().length === 0 ||
    !isGoalDrawerStatus(snapshot.status)
  ) {
    return null
  }

  return {
    goalId: snapshot.goalId,
    title: snapshot.title,
    status: snapshot.status,
    ...actionEligibilityByStatus[snapshot.status],
    canOpenExecutionRecord: snapshot.hasLinkedAgentRun,
  }
}
