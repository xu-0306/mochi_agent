/**
 * Mochi FastAPI client.
 * All requests use relative /v1/* paths and rely on Next.js rewrites in development.
 */

import type { Message } from '@/components/chat/ChatMessage'

const API_BASE = '/v1'

type ApiPrimitive = string | number | boolean | null
type ApiValue = ApiPrimitive | ApiValue[] | { [key: string]: ApiValue }

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function getString(value: unknown): string | null {
  return typeof value === 'string' ? value : null
}

function getNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function getBoolean(value: unknown): boolean | null {
  return typeof value === 'boolean' ? value : null
}

function getStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return []
  }
  return value.filter((item): item is string => typeof item === 'string')
}

function getRecordArray(value: unknown): Record<string, unknown>[] {
  if (!Array.isArray(value)) {
    return []
  }
  return value.filter(isRecord)
}

function toIsoString(value: unknown): string {
  const numeric = getNumber(value)
  if (numeric !== null) {
    return new Date(numeric * 1000).toISOString()
  }

  const text = getString(value)
  if (text) {
    const parsed = Date.parse(text)
    if (!Number.isNaN(parsed)) {
      return new Date(parsed).toISOString()
    }
  }

  return new Date(0).toISOString()
}

function getApiMessage(payload: unknown, fallback: string): string {
  if (!isRecord(payload)) {
    return fallback
  }

  const detail = payload.detail
  if (typeof detail === 'string' && detail.length > 0) {
    return detail
  }

  const message = payload.message
  if (typeof message === 'string' && message.length > 0) {
    return message
  }

  const error = payload.error
  if (typeof error === 'string' && error.length > 0) {
    return error
  }

  return fallback
}

export class ApiError extends Error {
  readonly status: number
  readonly payload?: unknown

  constructor(status: number, message: string, payload?: unknown) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.payload = payload
  }
}

async function parseResponseBody(response: Response): Promise<unknown> {
  const contentType = response.headers.get('content-type') ?? ''

  if (contentType.includes('application/json')) {
    return response.json()
  }

  const text = await response.text()
  return text.length > 0 ? text : null
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  const isFormData = typeof FormData !== 'undefined' && init?.body instanceof FormData

  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: {
        Accept: 'application/json',
        ...(init?.body && !isFormData ? { 'Content-Type': 'application/json' } : {}),
        ...init?.headers,
      },
      cache: 'no-store',
    })
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : 'Network request failed'
    throw new ApiError(0, message, error)
  }

  const payload = await parseResponseBody(response)

  if (!response.ok) {
    throw new ApiError(
      response.status,
      getApiMessage(payload, response.statusText || 'Request failed'),
      payload
    )
  }

  return payload as T
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system'
  content: string
}

export interface SendMessageOptions {
  sessionId?: string
  model?: string
  temperature?: number
  maxTokens?: number
}

export type TurnEventPhase =
  | 'thinking'
  | 'tool_call_request'
  | 'tool_call_result'
  | 'error'
  | 'final_answer'

export interface LegacyChatEvent {
  type:
    | 'thinking'
    | 'tool_call_request'
    | 'tool_call_result'
    | 'error'
    | 'final_answer'
}

export interface TurnEventPayload extends Record<string, unknown> {
  content?: unknown
  final_answer?: unknown
  text?: unknown
  message?: unknown
  answer?: unknown
  call_id?: unknown
  toolCallId?: unknown
  tool_name?: unknown
  toolName?: unknown
  arguments?: unknown
  toolArgs?: unknown
  result?: unknown
  toolResult?: unknown
  error?: unknown
  toolError?: unknown
  code?: unknown
  errorCode?: unknown
  trajectory_id?: unknown
  trajectoryId?: unknown
}

export interface SessionMessageEvent {
  type: 'message'
  role?: string
  content?: string
  timestamp?: string
  turn_id?: string | number
  turnId?: string | number
}

export interface SessionTurnEvent {
  type: 'turn_event'
  phase?: TurnEventPhase | string
  payload?: TurnEventPayload
  content?: string
  timestamp?: string
  turn_id?: string | number
  turnId?: string | number
}

export interface UnknownSessionEvent {
  type: string
  role?: string
  content?: string
  timestamp?: string
}

export type BackendChatEvent = LegacyChatEvent | SessionTurnEvent

export interface BackendChatResponse {
  type: 'chat_response'
  session_id: string
  final_answer: string
  trajectory_id: string | null
  events: BackendChatEvent[]
}

export interface PostChatPayload {
  message: string
  session_id?: string
  sessionId?: string
  model?: string
}

export interface SendMessageResult {
  id: string
  content: string
  model: string
  sessionId: string
  createdAt: string
  trajectoryId: string | null
  events: BackendChatEvent[]
}

interface NormalizedMessageEvent {
  kind: 'message'
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp?: string
  turnKey: string | null
}

interface NormalizedTurnEvent {
  kind: 'turn_event'
  phase: TurnEventPhase
  content: string
  timestamp?: string
  turnKey: string | null
  toolCallId?: string
  toolName?: string
  toolArgs?: Record<string, unknown>
  toolResult?: unknown
  toolError?: string
  errorCode?: string
  trajectoryId?: string | null
}

type NormalizedTimelineEvent = NormalizedMessageEvent | NormalizedTurnEvent

function getNonEmptyString(value: unknown): string | null {
  return typeof value === 'string' && value.trim().length > 0 ? value : null
}

function getTurnKey(value: Record<string, unknown>): string | null {
  const direct =
    getString(value.turn_id) ??
    getString(value.turnId) ??
    getNumber(value.turn_id)?.toString() ??
    getNumber(value.turnId)?.toString()

  if (direct) {
    return direct
  }

  const payload = isRecord(value.payload) ? value.payload : null
  if (!payload) {
    return null
  }

  return (
    getString(payload.turn_id) ??
    getString(payload.turnId) ??
    getNumber(payload.turn_id)?.toString() ??
    getNumber(payload.turnId)?.toString() ??
    getString(payload.trajectory_id) ??
    null
  )
}

function getPayloadContent(payload: Record<string, unknown>): string {
  return (
    getNonEmptyString(payload.content) ??
    getNonEmptyString(payload.final_answer) ??
    getNonEmptyString(payload.text) ??
    getNonEmptyString(payload.message) ??
    getNonEmptyString(payload.answer) ??
    ''
  )
}

function getPayloadRecord(
  payload: Record<string, unknown>,
  snakeCaseKey: string,
  camelCaseKey: string
): Record<string, unknown> | undefined {
  const snakeCaseValue = payload[snakeCaseKey]
  if (isRecord(snakeCaseValue)) {
    return snakeCaseValue
  }

  const camelCaseValue = payload[camelCaseKey]
  return isRecord(camelCaseValue) ? camelCaseValue : undefined
}

function normalizeTimelineEvent(event: Record<string, unknown>): NormalizedTimelineEvent | null {
  const type = getString(event.type)
  const timestamp = getString(event.timestamp) ?? undefined
  const turnKey = getTurnKey(event)

  if (type === 'message') {
    const role = getString(event.role)
    const content = getNonEmptyString(event.content)

    if (!role || !content) {
      return null
    }

    if (role !== 'user' && role !== 'assistant' && role !== 'system') {
      return null
    }

    return {
      kind: 'message',
      role,
      content,
      timestamp,
      turnKey,
    }
  }

  if (type === 'turn_event') {
    const phase = getString(event.phase)
    const payload = isRecord(event.payload) ? event.payload : {}

    if (
      phase !== 'thinking' &&
      phase !== 'tool_call_request' &&
      phase !== 'tool_call_result' &&
      phase !== 'error' &&
      phase !== 'final_answer'
    ) {
      return null
    }

    return {
      kind: 'turn_event',
      phase,
      content: getPayloadContent(payload),
      timestamp,
      turnKey,
      toolCallId: getString(payload.call_id) ?? getString(payload.toolCallId) ?? undefined,
      toolName: getString(payload.tool_name) ?? getString(payload.toolName) ?? undefined,
      toolArgs: getPayloadRecord(payload, 'arguments', 'toolArgs'),
      toolResult: payload.result ?? payload.toolResult,
      toolError:
        getNonEmptyString(payload.error) ??
        getNonEmptyString(payload.toolError) ??
        (phase === 'error' ? getNonEmptyString(payload.message) ?? undefined : undefined),
      errorCode: getString(payload.code) ?? getString(payload.errorCode) ?? undefined,
      trajectoryId: getString(payload.trajectory_id) ?? getString(payload.trajectoryId),
    }
  }

  if (
    type === 'thinking' ||
    type === 'tool_call_request' ||
    type === 'tool_call_result' ||
    type === 'error' ||
    type === 'final_answer'
  ) {
    return {
      kind: 'turn_event',
      phase: type,
      content:
        getNonEmptyString(event.content) ??
        (type === 'error'
          ? getNonEmptyString(event.error) ?? getNonEmptyString(event.message) ?? ''
          : ''),
      timestamp,
      turnKey,
      toolCallId: getString(event.call_id) ?? undefined,
      toolName: getString(event.tool_name) ?? undefined,
      toolArgs: isRecord(event.arguments) ? event.arguments : undefined,
      toolResult: event.result,
      toolError: getNonEmptyString(event.error) ?? undefined,
      errorCode: getString(event.code) ?? undefined,
      trajectoryId: getString(event.trajectory_id),
    }
  }

  return null
}

function toMessageTimestamp(timestamp?: string): Date {
  if (timestamp) {
    const parsed = Date.parse(timestamp)
    if (!Number.isNaN(parsed)) {
      return new Date(parsed)
    }
  }
  return new Date()
}

function buildMessageId(prefix: string, index: number, turnKey: string | null, timestamp?: string): string {
  return [prefix, turnKey ?? 'na', timestamp ?? 'now', index.toString()].join('-')
}

export function buildMessagesFromTimelineEvents(events: ReadonlyArray<unknown>): Message[] {
  const normalized = events
    .filter(isRecord)
    .map(normalizeTimelineEvent)
    .filter((event): event is NormalizedTimelineEvent => event !== null)
  const toolResultsByCallId = new Map<string, NormalizedTurnEvent>()

  for (const event of normalized) {
    if (
      event.kind === 'turn_event' &&
      event.phase === 'tool_call_result' &&
      event.toolCallId
    ) {
      toolResultsByCallId.set(event.toolCallId, event)
    }
  }

  const turnsWithFinalAnswer = new Set(
    normalized
      .filter(
        (event): event is NormalizedTurnEvent =>
          event.kind === 'turn_event' &&
          event.phase === 'final_answer' &&
          event.turnKey !== null
      )
      .map((event) => event.turnKey)
  )

  const messages = normalized
    .map((event, index): Message | null => {
      if (event.kind === 'message') {
        if (
          event.role === 'assistant' &&
          event.turnKey !== null &&
          turnsWithFinalAnswer.has(event.turnKey)
        ) {
          return null
        }

        return {
          id: buildMessageId(`timeline-${event.role}`, index, event.turnKey, event.timestamp),
          type: event.role,
          content: event.content,
          timestamp: toMessageTimestamp(event.timestamp),
        }
      }

      const timestamp = toMessageTimestamp(event.timestamp)
      const id = buildMessageId(`timeline-${event.phase}`, index, event.turnKey, event.timestamp)

      switch (event.phase) {
        case 'thinking':
          return {
            id,
            type: 'thinking',
            eventType: 'thinking',
            content: event.content,
            timestamp,
          }
        case 'tool_call_request':
          if (event.toolCallId) {
            const resultEvent = toolResultsByCallId.get(event.toolCallId)
            if (resultEvent) {
              return {
                id,
                type: 'tool_result',
                eventType: 'tool_call_result',
                content: resultEvent.toolError ?? resultEvent.content,
                toolCallId: event.toolCallId,
                toolName: event.toolName ?? resultEvent.toolName,
                toolArgs: event.toolArgs,
                toolResult: resultEvent.toolResult,
                toolError: resultEvent.toolError,
                timestamp: toMessageTimestamp(resultEvent.timestamp ?? event.timestamp),
              }
            }
          }

          return {
            id,
            type: 'tool_call',
            eventType: 'tool_call_request',
            content: '',
            toolCallId: event.toolCallId,
            toolName: event.toolName,
            toolArgs: event.toolArgs,
            timestamp,
          }
        case 'tool_call_result':
          if (event.toolCallId && toolResultsByCallId.get(event.toolCallId) === event) {
            const hasRequest = normalized.some(
              (candidate) =>
                candidate.kind === 'turn_event' &&
                candidate.phase === 'tool_call_request' &&
                candidate.toolCallId === event.toolCallId
            )

            if (hasRequest) {
              return null
            }
          }

          return {
            id,
            type: 'tool_result',
            eventType: 'tool_call_result',
            content: event.toolError ?? event.content,
            toolCallId: event.toolCallId,
            toolName: event.toolName,
            toolResult: event.toolResult,
            toolError: event.toolError,
            timestamp,
          }
        case 'error':
          return {
            id,
            type: 'error',
            eventType: 'error',
            content: event.toolError ?? event.content ?? 'Unknown error.',
            errorCode: event.errorCode,
            timestamp,
          }
        case 'final_answer':
          return {
            id,
            type: 'assistant',
            eventType: 'final_answer',
            content: event.content,
            timestamp,
          }
      }
    })
    .filter((message): message is Message => message !== null)

  return messages
}

export function buildMessagesFromChatEvents(events: BackendChatEvent[]): Message[] {
  return buildMessagesFromTimelineEvents(events)
}

export function buildMessagesFromSessionEvents(events: SessionEvent[]): Message[] {
  return buildMessagesFromTimelineEvents(events)
}

export async function sendMessage(
  text: string,
  options: SendMessageOptions = {}
): Promise<SendMessageResult> {
  const payload = await requestJson<BackendChatResponse>('/chat', {
    method: 'POST',
    body: JSON.stringify({
      message: text,
      session_id: options.sessionId,
      model: options.model,
    }),
  })

  return {
    id: payload.trajectory_id ?? `chat-${payload.session_id}-${Date.now()}`,
    content: payload.final_answer,
    model: options.model ?? 'unknown',
    sessionId: payload.session_id,
    createdAt: new Date().toISOString(),
    trajectoryId: payload.trajectory_id,
    events: payload.events,
  }
}

export async function postChat(payload: PostChatPayload): Promise<BackendChatResponse> {
  return requestJson<BackendChatResponse>('/chat', {
    method: 'POST',
    body: JSON.stringify({
      message: payload.message,
      session_id: payload.session_id ?? payload.sessionId,
      model: payload.model,
    }),
  })
}

export async function* streamChat(
  text: string,
  options: SendMessageOptions = {}
): AsyncGenerator<string, void, unknown> {
  const response = await sendMessage(text, options)
  for (const char of response.content) {
    yield char
  }
}

interface BackendSkill {
  skill_id: string
  name: string
  description: string
  trigger_keywords: string[]
  preconditions: string
  steps: string[]
  tools_used: string[]
  source_trajectory_id: string
  times_used: number
  success_rate: number
  created_at: number | string
  updated_at: number | string
  version: number
}

export interface Skill {
  id: string
  name: string
  description: string
  category: 'general' | 'task-specific'
  tags: string[]
  useCount: number
  successRate: number
  version: string
  enabled: boolean
  createdAt: string
  updatedAt: string
  triggerKeywords: string[]
  preconditions: string[]
  steps: string[]
  toolsUsed: string[]
  sourceTrajectoryId: string | null
}

function normalizeSkill(skill: BackendSkill): Skill {
  const triggerKeywords = getStringArray(skill.trigger_keywords)
  const toolsUsed = getStringArray(skill.tools_used)
  const preconditions = skill.preconditions
    .split('\n')
    .map((item) => item.trim())
    .filter((item) => item.length > 0)

  return {
    id: skill.skill_id,
    name: skill.name,
    description: skill.description,
    category: skill.source_trajectory_id ? 'task-specific' : 'general',
    tags: [...triggerKeywords, ...toolsUsed].slice(0, 8),
    useCount: skill.times_used,
    successRate: Math.max(0, Math.min(100, Math.round(skill.success_rate * 100))),
    version: `v${skill.version}`,
    enabled: true,
    createdAt: toIsoString(skill.created_at),
    updatedAt: toIsoString(skill.updated_at),
    triggerKeywords,
    preconditions,
    steps: getStringArray(skill.steps),
    toolsUsed,
    sourceTrajectoryId: skill.source_trajectory_id || null,
  }
}

export async function fetchSkills(
  params: { q?: string; limit?: number } = {}
): Promise<Skill[]> {
  const searchParams = new URLSearchParams()

  if (params.q) {
    searchParams.set('q', params.q)
  }
  if (typeof params.limit === 'number') {
    searchParams.set('limit', String(params.limit))
  }

  const query = searchParams.toString()
  const payload = await requestJson<BackendSkill[]>(`/skills${query ? `?${query}` : ''}`)
  return payload.map(normalizeSkill)
}

export async function exportSkills(): Promise<Skill[]> {
  const payload = await requestJson<BackendSkill[]>('/skills/export')
  return payload.map(normalizeSkill)
}

export async function deleteSkill(skillId: string): Promise<void> {
  await requestJson<{ deleted: boolean }>(`/skills/${encodeURIComponent(skillId)}`, {
    method: 'DELETE',
  })
}

interface BackendSessionListItem {
  session_id: string
  title?: string
  event_count: number
  updated_at: string
}

interface BackendSessionListResponse {
  type: 'sessions'
  items: BackendSessionListItem[]
}

export interface SessionSummary {
  id: string
  title: string
  createdAt: string
  updatedAt: string
  eventCount: number
}

export type SessionEvent = SessionMessageEvent | SessionTurnEvent | UnknownSessionEvent

interface BackendSessionResponse {
  type: 'session'
  session_id: string
  title?: string
  events: Record<string, unknown>[]
}

export interface SessionDetail extends SessionSummary {
  events: SessionEvent[]
}

function normalizeSessionSummary(item: BackendSessionListItem): SessionSummary {
  return {
    id: item.session_id,
    title: getString(item.title) ?? item.session_id,
    createdAt: item.updated_at,
    updatedAt: item.updated_at,
    eventCount: item.event_count,
  }
}

function normalizeSessionEvent(event: Record<string, unknown>): SessionEvent {
  const type = getString(event.type) ?? 'unknown'

  if (type === 'message') {
    return {
      ...event,
      type,
      role: getString(event.role) ?? undefined,
      content: getString(event.content) ?? undefined,
      timestamp: getString(event.timestamp) ?? undefined,
      turn_id:
        getString(event.turn_id) ??
        getNumber(event.turn_id) ??
        getString(event.turnId) ??
        getNumber(event.turnId) ??
        undefined,
      turnId:
        getString(event.turnId) ??
        getNumber(event.turnId) ??
        getString(event.turn_id) ??
        getNumber(event.turn_id) ??
        undefined,
    }
  }

  if (type === 'turn_event') {
    return {
      ...event,
      type,
      phase: getString(event.phase) ?? undefined,
      payload: isRecord(event.payload) ? event.payload : undefined,
      content: getString(event.content) ?? undefined,
      timestamp: getString(event.timestamp) ?? undefined,
      turn_id:
        getString(event.turn_id) ??
        getNumber(event.turn_id) ??
        getString(event.turnId) ??
        getNumber(event.turnId) ??
        undefined,
      turnId:
        getString(event.turnId) ??
        getNumber(event.turnId) ??
        getString(event.turn_id) ??
        getNumber(event.turn_id) ??
        undefined,
    }
  }

  return {
    ...event,
    type,
    role: getString(event.role) ?? undefined,
    content: getString(event.content) ?? undefined,
    timestamp: getString(event.timestamp) ?? undefined,
  }
}

export async function fetchSessions(): Promise<SessionSummary[]> {
  const payload = await requestJson<BackendSessionListResponse>('/sessions')
  return payload.items.map(normalizeSessionSummary)
}

export async function fetchSession(sessionId: string): Promise<SessionDetail> {
  const payload = await requestJson<BackendSessionResponse>(
    `/sessions/${encodeURIComponent(sessionId)}`
  )

  const events = getRecordArray(payload.events).map(normalizeSessionEvent)
  const updatedAt =
    [...events]
      .reverse()
      .map((event) => getString(event.timestamp))
      .find((timestamp): timestamp is string => Boolean(timestamp)) ?? new Date().toISOString()

  return {
    id: payload.session_id,
    title: getString(payload.title) ?? payload.session_id,
    createdAt: updatedAt,
    updatedAt,
    eventCount: events.length,
    events,
  }
}

interface BackendCreateSessionResponse {
  type: 'session'
  session_id: string
}

export async function createSession(sessionId?: string): Promise<SessionSummary> {
  const payload = await requestJson<BackendCreateSessionResponse>('/sessions', {
    method: 'POST',
    body: JSON.stringify(sessionId ? { session_id: sessionId } : {}),
  })

  const now = new Date().toISOString()
  return {
    id: payload.session_id,
    title: payload.session_id,
    createdAt: now,
    updatedAt: now,
    eventCount: 1,
  }
}

interface BackendUpdateSessionResponse {
  type: 'session'
  session_id: string
  title: string
}

export async function renameSession(sessionId: string, title: string): Promise<SessionSummary> {
  const payload = await requestJson<BackendUpdateSessionResponse>(
    `/sessions/${encodeURIComponent(sessionId)}`,
    {
      method: 'PATCH',
      body: JSON.stringify({ title }),
    }
  )

  const now = new Date().toISOString()
  return {
    id: payload.session_id,
    title: payload.title,
    createdAt: now,
    updatedAt: now,
    eventCount: 0,
  }
}

export async function deleteSession(sessionId: string): Promise<void> {
  await requestJson<{ deleted: boolean }>(`/sessions/${encodeURIComponent(sessionId)}`, {
    method: 'DELETE',
  })
}

interface BackendModelInfo extends Record<string, ApiValue | undefined> {
  id?: string
  name?: string
  label?: string
  model?: string
  model_spec?: string
  provider?: string
  base_url?: string | null
  backend_type?: string
  context_length?: number
  supports_tool_calling?: boolean
  metadata?: Record<string, ApiValue>
}

interface BackendModelsStatus {
  type: 'models_status'
  configured_model: string
  supported_model_spec_formats: Array<{
    type: string
    pattern: string
    description: string
  }>
  active_model: BackendModelInfo | null
  available_models?: BackendModelInfo[]
  configured_remote_provider?: string | null
}

export interface ModelInfo {
  id: string
  name: string
  label: string
  provider: string | null
  modelSpec: string | null
  baseUrl: string | null
  backendType: string
  contextLength: number | null
  supportsToolCalling: boolean | null
  metadata: Record<string, ApiValue>
}

export type ModelProvider = 'ollama' | 'openai_compat' | 'gemini' | 'anthropic'

export interface ConfigureModelInput {
  provider: ModelProvider
  model: string
  baseUrl?: string
  apiKey?: string
  persist?: boolean
}

interface BackendConfigureModelResponse {
  type: 'model_configure'
  provider: ModelProvider
  active_model: BackendModelInfo
  available_models?: BackendModelInfo[]
  api_key_configured: boolean
  persisted?: boolean
  config_path?: string | null
}

export interface ConfigureModelResult {
  type: 'model_configure'
  provider: ModelProvider
  activeModel: ModelInfo
  availableModels: ModelInfo[]
  apiKeyConfigured: boolean
  persisted: boolean
  configPath: string | null
}

export interface ModelsStatus {
  type: 'models_status'
  configuredModel: string
  supportedModelSpecFormats: Array<{
    type: string
    pattern: string
    description: string
  }>
  activeModel: ModelInfo | null
  availableModels: ModelInfo[]
  configuredRemoteProvider: string | null
}

function normalizeModelInfo(model: BackendModelInfo | Record<string, unknown> | null): ModelInfo | null {
  if (!model) {
    return null
  }

  const metadata: Record<string, ApiValue> = {}
  if (isRecord(model.metadata)) {
    for (const [key, value] of Object.entries(model.metadata)) {
      if (value !== undefined) {
        metadata[key] = value as ApiValue
      }
    }
  }

  return {
    id: getString(model.id) ?? getString(model.model_spec) ?? getString(model.name) ?? getString(model.model) ?? '',
    name: getString(model.name) ?? getString(model.model) ?? getString(model.model_spec) ?? '',
    label: getString(model.label) ?? getString(model.name) ?? getString(model.model) ?? '',
    provider: getString(model.provider),
    modelSpec: getString(model.model_spec),
    baseUrl: getString(model.base_url),
    backendType: getString(model.backend_type) ?? '',
    contextLength: getNumber(model.context_length) ?? null,
    supportsToolCalling: getBoolean(model.supports_tool_calling) ?? null,
    metadata,
  }
}

export async function fetchModelsStatus(): Promise<ModelsStatus> {
  const payload = await requestJson<BackendModelsStatus>('/models')
  return {
    type: payload.type,
    configuredModel: payload.configured_model,
    supportedModelSpecFormats: payload.supported_model_spec_formats,
    activeModel: normalizeModelInfo(payload.active_model),
    availableModels: getRecordArray(payload.available_models).map((model) =>
      normalizeModelInfo(model)
    ).filter((model): model is ModelInfo => model !== null),
    configuredRemoteProvider: payload.configured_remote_provider ?? null,
  }
}

export async function fetchModels(): Promise<ModelInfo[]> {
  const status = await fetchModelsStatus()
  if (status.availableModels.length > 0) {
    return status.availableModels
  }
  return status.activeModel ? [status.activeModel] : []
}

interface BackendModelSwitchResponse {
  type: 'model_switch'
  active_model: BackendModelInfo
}

export async function switchModel(model: string): Promise<ModelInfo> {
  const payload = await requestJson<BackendModelSwitchResponse>('/models/switch', {
    method: 'POST',
    body: JSON.stringify({ model }),
  })

  const activeModel = normalizeModelInfo(payload.active_model)
  if (!activeModel) {
    throw new ApiError(500, 'Backend did not return an active model')
  }
  return activeModel
}

export async function configureModel(input: ConfigureModelInput): Promise<ConfigureModelResult> {
  const payload = await requestJson<BackendConfigureModelResponse>('/models/configure', {
    method: 'POST',
    body: JSON.stringify({
      provider: input.provider,
      model: input.model,
      base_url: input.baseUrl,
      api_key: input.apiKey,
      persist: input.persist,
    }),
  })

  const activeModel = normalizeModelInfo(payload.active_model)
  if (!activeModel) {
    throw new ApiError(500, 'Backend did not return an active model')
  }

  return {
    type: payload.type,
    provider: payload.provider,
    activeModel,
    availableModels: getRecordArray(payload.available_models).map((model) =>
      normalizeModelInfo(model)
    ).filter((model): model is ModelInfo => model !== null),
    apiKeyConfigured: payload.api_key_configured,
    persisted: Boolean(payload.persisted),
    configPath: payload.config_path ?? null,
  }
}

interface BackendOllamaModelsResponse {
  type: 'ollama_models'
  base_url: string
  models: string[]
}

export interface OllamaModelsResult {
  type: 'ollama_models'
  baseUrl: string
  models: string[]
}

export async function fetchOllamaModels(baseUrl: string): Promise<OllamaModelsResult> {
  const query = new URLSearchParams({ base_url: baseUrl })
  const payload = await requestJson<BackendOllamaModelsResponse>(`/models/ollama?${query}`)
  return {
    type: payload.type,
    baseUrl: payload.base_url,
    models: getStringArray(payload.models),
  }
}

interface BackendFilesystemRoot {
  name?: string
  label: string
  path: string
}

interface BackendFilesystemRootsResponse {
  roots?: BackendFilesystemRoot[]
  items?: BackendFilesystemRoot[]
}

interface BackendFilesystemListItem {
  name: string
  path: string
  is_dir: boolean
  is_file: boolean
}

interface BackendFilesystemListResponse {
  path?: string
  current_path?: string
  parent?: string | null
  parent_path?: string | null
  items: BackendFilesystemListItem[]
}

export interface FilesystemRoot {
  label: string
  path: string
}

export interface FilesystemListItem {
  name: string
  path: string
  isDir: boolean
  isFile: boolean
}

export interface FilesystemListResult {
  path: string
  parent: string | null
  items: FilesystemListItem[]
}

interface BackendFilesystemImportResponse {
  type: 'filesystem_import'
  import_root: string
  imported_path: string
  file_count: number
  total_bytes: number
}

export interface FilesystemImportInput {
  files: File[]
  relativePaths?: string[]
  targetDir?: string
  packageName?: string
}

export interface FilesystemImportResult {
  type: 'filesystem_import'
  importRoot: string
  importedPath: string
  fileCount: number
  totalBytes: number
}

export async function fetchFilesystemRoots(): Promise<FilesystemRoot[]> {
  const payload = await requestJson<BackendFilesystemRootsResponse>('/filesystem/roots')
  const roots = Array.isArray(payload.roots) ? payload.roots : payload.items

  return Array.isArray(roots)
    ? roots
        .filter((item) => typeof item?.path === 'string' && item.path.length > 0)
        .map((item) => ({
          label: getString(item.label) ?? getString(item.name) ?? item.path,
          path: item.path,
        }))
    : []
}

export async function fetchFilesystemList(path: string): Promise<FilesystemListResult> {
  const query = new URLSearchParams({ path })
  const payload = await requestJson<BackendFilesystemListResponse>(`/filesystem/list?${query.toString()}`)

  return {
    path: getString(payload.path) ?? getString(payload.current_path) ?? path,
    parent: getString(payload.parent) ?? getString(payload.parent_path) ?? null,
    items: Array.isArray(payload.items)
      ? payload.items
          .filter((item) => typeof item?.path === 'string' && item.path.length > 0)
          .map((item) => ({
            name: getString(item.name) ?? item.path,
            path: item.path,
            isDir: Boolean(item.is_dir),
            isFile: Boolean(item.is_file),
          }))
      : [],
  }
}

export async function importFilesystemFiles(
  input: FilesystemImportInput
): Promise<FilesystemImportResult> {
  const form = new FormData()
  for (const file of input.files) {
    form.append('files', file, file.name)
  }
  for (const relativePath of input.relativePaths ?? input.files.map((file) => file.name)) {
    form.append('relative_paths', relativePath)
  }
  if (input.targetDir) {
    form.append('target_dir', input.targetDir)
  }
  if (input.packageName) {
    form.append('package_name', input.packageName)
  }

  const payload = await requestJson<BackendFilesystemImportResponse>('/filesystem/import', {
    method: 'POST',
    body: form,
  })

  return {
    type: payload.type,
    importRoot: payload.import_root,
    importedPath: payload.imported_path,
    fileCount: payload.file_count,
    totalBytes: payload.total_bytes,
  }
}

interface BackendSettings {
  type: 'settings'
  model: string
  model_config?: Record<string, ApiValue>
  model_setup?: Record<string, ApiValue>
  voice: Record<string, ApiValue>
  memory: Record<string, ApiValue>
  learning: Record<string, ApiValue>
  channels: Record<string, ApiValue>
  web: Record<string, ApiValue>
  paths?: Record<string, ApiValue>
  update?: Record<string, ApiValue>
}

export interface Settings {
  type: 'settings'
  model: string
  model_config?: Record<string, ApiValue>
  model_setup?: Record<string, ApiValue>
  voice: Record<string, ApiValue>
  memory: Record<string, ApiValue>
  learning: Record<string, ApiValue>
  channels: Record<string, ApiValue>
  web: Record<string, ApiValue>
  paths?: Record<string, ApiValue>
  update?: Record<string, ApiValue>
}

export interface ChannelRuntimeStatus extends Record<string, unknown> {
  enabled?: boolean
  registered?: boolean
  running?: boolean
  token_configured?: boolean
  tokenConfigured?: boolean
  allowed_channel_ids?: Array<string | number>
  allowedChannelIds?: Array<string | number>
  allowed_chat_ids?: Array<string | number>
  allowedChatIds?: Array<string | number>
  allowed_user_ids?: Array<string | number>
  allowedUserIds?: Array<string | number>
}

export interface DiscordChannelStatus extends ChannelRuntimeStatus {
  text_enabled?: boolean
  textEnabled?: boolean
  voice_enabled?: boolean
  voiceEnabled?: boolean
  bot_token_configured?: boolean
  botTokenConfigured?: boolean
  allowed_guild_ids?: Array<string | number>
  allowedGuildIds?: Array<string | number>
  allowed_voice_channel_ids?: Array<string | number>
  allowedVoiceChannelIds?: Array<string | number>
  message_mode?: string
  messageMode?: string
  auto_join_policy?: string
  autoJoinPolicy?: string
  voice_ingress?: Record<string, unknown>
  voiceIngress?: Record<string, unknown>
  voice_runtime?: Record<string, unknown>
  voiceRuntime?: Record<string, unknown>
}

export interface DiscordVoiceRoomSummary {
  guildId: string | null
  channelId: string | null
  sessionId: string | null
  participantCount: number | null
  joinedAt: string | null
  playbackState: string | null
  speakingState: string | null
  error: string | null
}

export interface DiscordChannelSummary {
  enabled: boolean | null
  registered: boolean | null
  running: boolean | null
  textEnabled: boolean | null
  voiceEnabled: boolean | null
  botTokenConfigured: boolean | null
  allowedChannelIds: string[]
  allowedGuildIds: string[]
  allowedUserIds: string[]
  allowedVoiceChannelIds: string[]
  messageMode: string | null
  autoJoinPolicy: string | null
  activeVoiceRoomCount: number | null
  reconnectCount: number | null
  voiceIngressEnabled: boolean | null
  voiceIngressAvailable: boolean | null
  voiceIngressGuildIds: string[]
  voiceIngressError: string | null
  voiceRuntimePhase: string | null
  voiceRuntimeError: string | null
  playbackState: string | null
  speakingState: string | null
  activeRooms: DiscordVoiceRoomSummary[]
}

interface BackendChannelsStatus {
  type?: unknown
  phase?: unknown
  supported_channels?: unknown
  channels?: unknown
}

interface BackendChannelsControlResponse {
  type?: unknown
  action?: unknown
  scope?: unknown
  channel?: unknown
  running?: unknown
  running_channels?: unknown
}

export interface ChannelsStatus {
  type: string
  phase: string | null
  supportedChannels: string[]
  channels: Record<string, ChannelRuntimeStatus>
}

export interface ChannelsControlResult {
  type: string
  action: string | null
  scope: string | null
  channel: string | null
  running: boolean | null
  runningChannels: string[]
}

function getRecordMap(value: unknown): Record<string, ChannelRuntimeStatus> {
  if (!isRecord(value)) {
    return {}
  }

  const entries = Object.entries(value)
  const map: Record<string, ChannelRuntimeStatus> = {}

  for (const [key, item] of entries) {
    if (isRecord(item)) {
      map[key] = item
    }
  }

  return map
}

function getBooleanField(record: Record<string, unknown>, keys: string[]): boolean | null {
  for (const key of keys) {
    const value = getBoolean(record[key])
    if (value !== null) {
      return value
    }
  }

  return null
}

function getNumberField(record: Record<string, unknown>, keys: string[]): number | null {
  for (const key of keys) {
    const value = getNumber(record[key])
    if (value !== null) {
      return value
    }
  }

  return null
}

function getStringField(record: Record<string, unknown>, keys: string[]): string | null {
  for (const key of keys) {
    const value = getNonEmptyString(record[key])
    if (value !== null) {
      return value
    }
  }

  return null
}

function getRecordArrayField(record: Record<string, unknown>, keys: string[]): Record<string, unknown>[] {
  for (const key of keys) {
    const value = record[key]
    if (!Array.isArray(value)) {
      continue
    }
    return value.filter(isRecord)
  }

  return []
}

function normalizeDiscordVoiceRoom(room: Record<string, unknown>): DiscordVoiceRoomSummary {
  return {
    guildId: getStringField(room, ['guild_id', 'guildId']),
    channelId: getStringField(room, ['channel_id', 'channelId']),
    sessionId: getStringField(room, ['session_id', 'sessionId']),
    participantCount: getNumberField(room, ['participant_count', 'participantCount']),
    joinedAt: getStringField(room, ['joined_at', 'joinedAt']),
    playbackState:
      getStringField(room, ['playback_state', 'playbackState', 'playback_status', 'playbackStatus']) ??
      (getBooleanField(room, ['is_playing', 'playing']) === true ? 'playing' : null),
    speakingState:
      getStringField(room, ['speaking_state', 'speakingState', 'speaking_status', 'speakingStatus']) ??
      (getBooleanField(room, ['is_speaking', 'speaking']) === true ? 'speaking' : null),
    error: getStringField(room, ['last_error', 'lastError', 'error']),
  }
}

function getIdList(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return []
  }

  return value
    .map((entry) => {
      if (typeof entry === 'string') {
        return entry.trim()
      }
      if (typeof entry === 'number' && Number.isFinite(entry)) {
        return String(entry)
      }
      return ''
    })
    .filter((entry) => entry.length > 0)
}

function mergeIdLists(...sources: unknown[]): string[] {
  return Array.from(new Set(sources.flatMap((source) => getIdList(source))))
}

export function normalizeDiscordChannelStatus(
  configured: unknown,
  runtime?: unknown
): DiscordChannelSummary {
  const configuredRecord = isRecord(configured) ? configured : {}
  const runtimeRecord = isRecord(runtime) ? runtime : {}
  const voiceIngress =
    (isRecord(runtimeRecord.voice_ingress) ? runtimeRecord.voice_ingress : null) ??
    (isRecord(runtimeRecord.voiceIngress) ? runtimeRecord.voiceIngress : null)
  const voiceRuntime =
    (isRecord(runtimeRecord.voice_runtime) ? runtimeRecord.voice_runtime : null) ??
    (isRecord(runtimeRecord.voiceRuntime) ? runtimeRecord.voiceRuntime : null)
  const activeRooms = getRecordArrayField(voiceRuntime ?? {}, ['active_voice_rooms', 'activeVoiceRooms'])
    .map(normalizeDiscordVoiceRoom)

  const textEnabled =
    getBooleanField(runtimeRecord, ['text_enabled', 'textEnabled']) ??
    getBooleanField(configuredRecord, ['text_enabled', 'textEnabled'])
  const voiceEnabled =
    getBooleanField(runtimeRecord, ['voice_enabled', 'voiceEnabled']) ??
    getBooleanField(configuredRecord, ['voice_enabled', 'voiceEnabled'])

  let enabled =
    getBooleanField(runtimeRecord, ['enabled']) ??
    getBooleanField(configuredRecord, ['enabled'])

  if (enabled === null) {
    if (textEnabled === true || voiceEnabled === true) {
      enabled = true
    } else if (textEnabled === false && voiceEnabled === false) {
      enabled = false
    }
  }

  const botTokenConfigured =
    getBooleanField(runtimeRecord, [
      'bot_token_configured',
      'botTokenConfigured',
      'token_configured',
      'tokenConfigured',
    ]) ??
    getBooleanField(configuredRecord, [
      'bot_token_configured',
      'botTokenConfigured',
      'token_configured',
      'tokenConfigured',
      'has_token',
    ]) ??
    (typeof configuredRecord.token === 'string' && configuredRecord.token.trim().length > 0)

  return {
    enabled,
    registered: getBooleanField(runtimeRecord, ['registered']),
    running: getBooleanField(runtimeRecord, ['running']),
    textEnabled,
    voiceEnabled,
    botTokenConfigured,
    allowedChannelIds: mergeIdLists(
      runtimeRecord.allowed_channel_ids,
      runtimeRecord.allowedChannelIds,
      configuredRecord.allowed_channel_ids,
      configuredRecord.allowedChannelIds
    ),
    allowedGuildIds: mergeIdLists(
      runtimeRecord.allowed_guild_ids,
      runtimeRecord.allowedGuildIds,
      configuredRecord.allowed_guild_ids,
      configuredRecord.allowedGuildIds
    ),
    allowedUserIds: mergeIdLists(
      runtimeRecord.allowed_user_ids,
      runtimeRecord.allowedUserIds,
      configuredRecord.allowed_user_ids,
      configuredRecord.allowedUserIds
    ),
    allowedVoiceChannelIds: mergeIdLists(
      runtimeRecord.allowed_voice_channel_ids,
      runtimeRecord.allowedVoiceChannelIds,
      configuredRecord.allowed_voice_channel_ids,
      configuredRecord.allowedVoiceChannelIds
    ),
    messageMode:
      getStringField(runtimeRecord, ['message_mode', 'messageMode']) ??
      getStringField(configuredRecord, ['message_mode', 'messageMode']),
    autoJoinPolicy:
      getStringField(runtimeRecord, ['auto_join_policy', 'autoJoinPolicy']) ??
      getStringField(configuredRecord, ['auto_join_policy', 'autoJoinPolicy']),
    activeVoiceRoomCount:
      getNumberField(voiceRuntime ?? {}, ['active_voice_room_count', 'activeVoiceRoomCount']) ??
      activeRooms.length,
    reconnectCount: getNumberField(voiceRuntime ?? {}, ['reconnect_count', 'reconnectCount']),
    voiceIngressEnabled: getBooleanField(voiceIngress ?? {}, ['enabled']),
    voiceIngressAvailable: getBooleanField(voiceIngress ?? {}, ['extension_available', 'extensionAvailable']),
    voiceIngressGuildIds: mergeIdLists(
      voiceIngress?.active_guild_ids,
      voiceIngress?.activeGuildIds
    ),
    voiceIngressError: getStringField(voiceIngress ?? {}, ['last_error', 'lastError', 'error']),
    voiceRuntimePhase: getStringField(voiceRuntime ?? {}, ['phase']),
    voiceRuntimeError: getStringField(voiceRuntime ?? {}, ['last_error', 'lastError', 'error']),
    playbackState:
      getStringField(voiceRuntime ?? {}, ['playback_state', 'playbackState', 'playback_status', 'playbackStatus']) ??
      (getBooleanField(voiceRuntime ?? {}, ['is_playing', 'playing']) === true ? 'playing' : null),
    speakingState:
      getStringField(voiceRuntime ?? {}, ['speaking_state', 'speakingState', 'speaking_status', 'speakingStatus']) ??
      (getBooleanField(voiceRuntime ?? {}, ['is_speaking', 'speaking']) === true ? 'speaking' : null),
    activeRooms,
  }
}

export async function fetchSettings(): Promise<Settings> {
  const payload = await requestJson<BackendSettings>('/settings')
  return {
    type: payload.type,
    model: payload.model,
    model_config: isRecord(payload.model_config)
      ? payload.model_config as Record<string, ApiValue>
      : undefined,
    model_setup: isRecord(payload.model_setup)
      ? payload.model_setup as Record<string, ApiValue>
      : undefined,
    voice: payload.voice,
    memory: payload.memory,
    learning: payload.learning,
    channels: payload.channels,
    web: payload.web,
    paths: isRecord(payload.paths) ? payload.paths as Record<string, ApiValue> : undefined,
    update: isRecord(payload.update) ? payload.update as Record<string, ApiValue> : undefined,
  }
}

export async function fetchChannelsStatus(): Promise<ChannelsStatus> {
  const payload = await requestJson<BackendChannelsStatus>('/channels')
  return {
    type: getString(payload.type) ?? 'channels_status',
    phase: getString(payload.phase),
    supportedChannels: getStringArray(payload.supported_channels),
    channels: getRecordMap(payload.channels),
  }
}

export async function startChannel(name: string): Promise<ChannelsControlResult> {
  const payload = await requestJson<BackendChannelsControlResponse>(`/channels/${name}/start`, {
    method: 'POST',
  })
  return {
    type: getString(payload.type) ?? 'channels_control',
    action: getString(payload.action),
    scope: getString(payload.scope),
    channel: getString(payload.channel),
    running: getBoolean(payload.running),
    runningChannels: getStringArray(payload.running_channels),
  }
}

export async function stopChannel(name: string): Promise<ChannelsControlResult> {
  const payload = await requestJson<BackendChannelsControlResponse>(`/channels/${name}/stop`, {
    method: 'POST',
  })
  return {
    type: getString(payload.type) ?? 'channels_control',
    action: getString(payload.action),
    scope: getString(payload.scope),
    channel: getString(payload.channel),
    running: getBoolean(payload.running),
    runningChannels: getStringArray(payload.running_channels),
  }
}

export interface VoiceSettingsUpdate {
  enabled?: boolean
  stt_backend?: string
  stt_model?: string
  stt_language?: string
  stt_device?: string
  stt_model_cache_dir?: string
  stt_model_path?: string
  tts_backend?: string
  tts_model?: string | null
  tts_voice?: string
  tts_language?: string | null
  tts_speed?: number
  tts_use_gpu?: boolean
  tts_kokoro_lang_code?: string
  tts_openai_base_url?: string | null
  tts_openai_response_format?: 'pcm' | 'wav'
}

export interface MemorySettingsUpdate {
  db_path?: string
  max_short_term_messages?: number
  fts_top_k?: number
}

export interface LearningSettingsUpdate {
  enabled?: boolean
  auto_extract_skills?: boolean
  auto_sync_filesystem_skills?: boolean
  min_steps_for_extraction?: number
  trajectory_retention_days?: number
  skill_improvement_threshold?: number
  max_skills?: number
}

export interface PathSettingsUpdate {
  workspace_dir?: string
  sessions_dir?: string
  skills_dir?: string
  plugins_dir?: string
}

export interface UpdateSettingsInput {
  voice?: VoiceSettingsUpdate
  memory?: MemorySettingsUpdate
  learning?: LearningSettingsUpdate
  paths?: PathSettingsUpdate
  download_missing_models?: boolean
  reload_voice?: boolean
  persist?: boolean
}

export interface DiscordSetupInput {
  bot_token: string
  enabled?: boolean
  text_enabled?: boolean
  voice_enabled?: boolean
  allowed_guild_ids?: number[]
  allowed_channel_ids?: number[]
  allowed_voice_channel_ids?: number[]
  allowed_user_ids?: number[]
  rate_limit_per_user?: number
  message_mode?: 'all_messages' | 'mentions_only' | 'slash_only'
  auto_join_policy?: 'manual_only'
  voice_auto_reply?: boolean
  voice_stt_enabled?: boolean
  voice_tts_enabled?: boolean
  persist?: boolean
  reload_voice?: boolean
}

export async function updateSettings(input: UpdateSettingsInput): Promise<Settings> {
  const payload = await requestJson<BackendSettings>('/settings', {
    method: 'PATCH',
    body: JSON.stringify(input),
  })

  return {
    type: payload.type,
    model: payload.model,
    model_config: isRecord(payload.model_config)
      ? payload.model_config as Record<string, ApiValue>
      : undefined,
    voice: payload.voice,
    memory: payload.memory,
    learning: payload.learning,
    channels: payload.channels,
    web: payload.web,
    paths: isRecord(payload.paths) ? payload.paths as Record<string, ApiValue> : undefined,
    update: isRecord(payload.update) ? payload.update as Record<string, ApiValue> : undefined,
  }
}

export async function setupDiscord(input: DiscordSetupInput): Promise<Settings> {
  const payload = await requestJson<BackendSettings>('/setup/discord', {
    method: 'POST',
    body: JSON.stringify(input),
  })

  return {
    type: payload.type,
    model: payload.model,
    model_config: isRecord(payload.model_config)
      ? payload.model_config as Record<string, ApiValue>
      : undefined,
    voice: payload.voice,
    memory: payload.memory,
    learning: payload.learning,
    channels: payload.channels,
    web: payload.web,
    paths: isRecord(payload.paths) ? payload.paths as Record<string, ApiValue> : undefined,
    update: isRecord(payload.update) ? payload.update as Record<string, ApiValue> : undefined,
  }
}
