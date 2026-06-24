export type WorkflowAgentStatus =
  | 'queued'
  | 'thinking'
  | 'running_tool'
  | 'waiting'
  | 'blocked'
  | 'done'
  | 'error'

export type WorkflowStageStatus = 'pending' | 'active' | 'completed' | 'blocked'

export interface WorkflowDebugEntry {
  id: string
  role: 'system' | 'assistant' | 'operator'
  title: string
  body: string
  timestamp: string
  status?: 'ready' | 'pending' | 'success' | 'error'
  meta?: string
}

export interface WorkflowAgentCard {
  roleId: string
  label: string
  modelId: string | null
  status: WorkflowAgentStatus
  stage: string | null
  currentAction: string
  lastOutputSummary: string | null
  outputCount: number
  updatedAt: string | null
}

export interface WorkflowStageCard {
  id: string
  label: string
  status: WorkflowStageStatus
  summary: string
  activeRoles: string[]
  outputCount: number
  blockerCount: number
}

export interface WorkflowConversationMessage {
  id: string
  role: 'user' | 'assistant' | 'system'
  label: string
  content: string
  timestamp: string | null
  meta?: string
}

export interface WorkflowConversationTab {
  id: string
  label: string
  kind: 'main' | 'workflow' | 'role'
  roleId: string | null
  status?: WorkflowAgentStatus
  messages: WorkflowConversationMessage[]
}

export interface WorkflowNarrativeItem {
  id: string
  title: string
  content: string
  timestamp: string | null
}

export interface WorkflowArtifactCard {
  id: string
  label: string
  artifactType: string
  summary: string
  uri: string | null
}

export interface WorkflowDeskView {
  runId: string | null
  title: string
  status: string
  protocolId: string | null
  phaseLabel: string
  updatedAt: string | null
  startedAt: string | null
  finishedAt: string | null
  agents: WorkflowAgentCard[]
  stages: WorkflowStageCard[]
  conversations: WorkflowConversationTab[]
  narrative: WorkflowNarrativeItem[]
  artifacts: WorkflowArtifactCard[]
  debugEntries: WorkflowDebugEntry[]
  rawEventCount: number
}

export interface WorkflowProgressCardRole {
  roleId: string
  label: string
  status: WorkflowAgentStatus
  currentAction: string
  lastOutputSummary: string | null
  updatedAt: string | null
}

export interface WorkflowProgressCardResult {
  status: 'pending' | 'partial' | 'complete' | 'error'
  content: string | null
  source: 'run_summary' | 'latest_error' | 'none'
}

export interface WorkflowProgressCardView {
  runId: string
  title: string
  status: string
  phaseLabel: string
  summary: string
  startedAt: string | null
  updatedAt: string | null
  finishedAt: string | null
  latestError: string | null
  roles: WorkflowProgressCardRole[]
  transcriptSnippets: WorkflowConversationMessage[]
  finalResult: WorkflowProgressCardResult
}
