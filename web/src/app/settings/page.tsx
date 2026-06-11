'use client'

import * as React from 'react'
import Link from 'next/link'
import { useRouter, useSearchParams } from 'next/navigation'
import {
  AlertCircle,
  Bot,
  BookOpen,
  BrainCircuit,
  CheckCircle2,
  Cpu,
  Database,
  Download,
  Globe,
  KeyRound,
  Mic,
  Network,
  Pencil,
  PlugZap,
  RefreshCw,
  Save,
  Send,
  Shield,
  Sparkles,
  Terminal,
  Trash2,
  Upload,
} from 'lucide-react'
import { Tabs, TabsContent } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { PathPicker } from '@/components/settings/PathPicker'
import { CodeThemePreview } from '@/components/settings/CodeThemePreview'
import {
  DEFAULT_SETTINGS_TAB,
  SettingsNav,
  isSettingsTab,
  settingsTabHref,
  type SettingsTab,
} from '@/components/settings/SettingsNav'
import { Switch } from '@/components/ui/switch'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import * as api from '@/lib/api'
import { InferenceControls } from '@/components/chat/InferenceControls'
import {
  AUTO_LANGUAGE,
  AUTO_TIMEZONE,
  SYSTEM_APPEARANCE,
  type UIAppearanceMode,
  type UIFontSize,
  type UILanguageMode,
  useI18n,
} from '@/lib/i18n'
import { CODE_THEME_OPTIONS, type UICodeTheme } from '@/lib/code-theme'
import {
  getActivePreset,
  inferencePresetToParams,
  resolveEffectiveInferenceParams,
  type InferenceParams,
} from '@/lib/stores/inference-store'
import {
  normalizeReplyModelModeForApi,
  normalizeReplyModelModeFromApi,
  normalizeSessionModeForApi,
  normalizeSessionModeFromApi,
} from '@/lib/voice-settings'
import {
  buildContextLengthSettingsUpdate,
  resolveContextLengthSettingsTarget,
} from '@/lib/model-context-settings'

type SummaryItem = {
  label: string
  value: string
}

type Translator = (key: string, values?: Record<string, string | number | boolean | null | undefined>) => string

type ApiModule = typeof api & {
  fetchSettings?: () => Promise<unknown>
  fetchModels?: () => Promise<unknown[]>
  fetchChannelsStatus?: () => Promise<api.ChannelsStatus>
  configureModel?: (input: api.ConfigureModelInput) => Promise<api.ConfigureModelResult>
  fetchOpenAICodexAuthStatus?: () => Promise<api.OpenAICodexAuthStatus>
  importOpenAICodexCliLogin?: () => Promise<api.OpenAICodexImportResult>
  startOpenAICodexBrowserLogin?: (frontendOrigin?: string) => Promise<api.OpenAICodexLoginStartResult>
  completeOpenAICodexBrowserLogin?: (input: api.OpenAICodexLoginCompleteInput) => Promise<api.OpenAICodexImportResult>
  refreshOpenAICodexAuth?: () => Promise<api.OpenAICodexImportResult>
  logoutOpenAICodexAuth?: () => Promise<api.OpenAICodexLogoutResult>
  fetchOllamaModels?: (baseUrl: string) => Promise<api.OllamaModelsResult>
  fetchLocalModels?: (root: string) => Promise<api.LocalModelsResult>
  fetchLocalModelCapabilities?: (model: string) => Promise<api.LocalModelCapabilitiesResult>
  fetchActiveLocalModelRuntimeStatus?: () => Promise<api.LocalActiveModelRuntimeStatus>
  unloadActiveLocalModelRuntime?: () => Promise<api.LocalActiveModelRuntimeUnloadResult>
  convertLocalModel?: (input: api.LocalModelConvertInput) => Promise<api.LocalModelConvertResult>
  updateModelEntry?: (input: api.UpdateModelEntryInput) => Promise<api.UpdateModelEntryResult>
  deleteModelEntry?: (modelId: string, persist?: boolean) => Promise<api.DeleteModelEntryResult>
  importFilesystemFiles?: (input: api.FilesystemImportInput) => Promise<api.FilesystemImportResult>
  fetchVoiceStatus?: () => Promise<api.VoiceRuntimeStatus>
  fetchVoiceCatalog?: () => Promise<api.VoiceCatalog>
  uploadVoicePack?: (file: File) => Promise<api.VoiceCatalog>
  registerVoicePackPath?: (path: string) => Promise<api.VoiceCatalog>
  deleteVoice?: (voiceId: string) => Promise<api.VoiceCatalog>
  updateSettings?: (input: api.UpdateSettingsInput) => Promise<api.Settings>
  setupDiscord?: (input: api.DiscordSetupInput) => Promise<api.Settings>
  startChannel?: (name: string) => Promise<api.ChannelsControlResult>
  stopChannel?: (name: string) => Promise<api.ChannelsControlResult>
}

const settingsApi = api as ApiModule
const SENSITIVE_KEY_PATTERN = /(token|secret|password|api[_-]?key|credential|authorization)/i
const MODELS_UPDATED_EVENT = 'mochi:models-updated'
const VOICE_ROUTE_UNAVAILABLE_STATUSES = new Set([404, 405])

function isVoiceRouteUnavailable(error: unknown): boolean {
  return error instanceof api.ApiError && VOICE_ROUTE_UNAVAILABLE_STATUSES.has(error.status)
}

function getSettingsTabFromSearch(searchParams: { get: (name: string) => string | null }): SettingsTab {
  const requestedTab = searchParams.get('tab')
  return isSettingsTab(requestedTab) ? requestedTab : DEFAULT_SETTINGS_TAB
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null
}

function toTitleCase(input: string): string {
  return input
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/\b\w/g, (char) => char.toUpperCase())
}

function formatScalar(value: unknown, t?: Translator): string | null {
  if (typeof value === 'string') {
    return value || (t ? t('common.notSet') : 'Not set')
  }

  if (typeof value === 'number') {
    return Number.isFinite(value) ? String(value) : null
  }

  if (typeof value === 'boolean') {
    return value
      ? (t ? t('common.enabled') : 'Enabled')
      : (t ? t('common.disabled') : 'Disabled')
  }

  return null
}

function getStringSetting(section: Record<string, unknown>, key: string, fallback = ''): string {
  const value = section[key]
  return typeof value === 'string' ? value : fallback
}

function getNumberSetting(section: Record<string, unknown>, key: string, fallback: number): number {
  const value = section[key]
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

function getBooleanSetting(section: Record<string, unknown>, key: string, fallback = false): boolean {
  const value = section[key]
  return typeof value === 'boolean' ? value : fallback
}

function getStringOptions(section: Record<string, unknown>, key: string, fallback: string[]): string[] {
  const value = section[key]
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : fallback
}

function getIdSetting(section: Record<string, unknown>, key: string): string[] {
  const value = section[key]
  if (!Array.isArray(value)) {
    return []
  }
  return value
    .map((item) => {
      if (typeof item === 'string') {
        return item.trim()
      }
      if (typeof item === 'number' && Number.isFinite(item)) {
        return String(item)
      }
      return ''
    })
    .filter((item) => item.length > 0)
}

function getOptionsByBackend(
  section: Record<string, unknown>,
  key: string,
  fallback: Record<string, string[]>
): Record<string, string[]> {
  const raw = asRecord(section[key])
  if (!raw) {
    return fallback
  }

  const result: Record<string, string[]> = {}
  for (const [backend, options] of Object.entries(raw)) {
    if (!Array.isArray(options)) {
      continue
    }
    result[backend] = options.filter((item): item is string => typeof item === 'string' && item.length > 0)
  }
  return Object.keys(result).length > 0 ? result : fallback
}

function withCurrentOption(options: string[], current: string): string[] {
  if (!current || options.includes(current)) {
    return options
  }
  return [current, ...options]
}

function summarizeValue(value: unknown, t?: Translator): string | null {
  const scalar = formatScalar(value, t)
  if (scalar !== null) {
    return scalar
  }

  if (Array.isArray(value)) {
    const items = value.map((item) => formatScalar(item, t)).filter((item): item is string => item !== null)
    if (items.length > 0) {
      return items.slice(0, 3).join(', ')
    }

    return t ? t('common.items', { count: value.length }) : `${value.length} items`
  }

  const record = asRecord(value)
  if (!record) {
    return null
  }

  if (typeof record.enabled === 'boolean') {
    return record.enabled
      ? (t ? t('common.enabled') : 'Enabled')
      : (t ? t('common.disabled') : 'Disabled')
  }

  const preferredKeys = ['name', 'model', 'provider', 'backend', 'host', 'url', 'path', 'status']
  for (const key of preferredKeys) {
    const preferredValue = formatScalar(record[key], t)
    if (preferredValue !== null) {
      return preferredValue
    }
  }

  const visibleKeys = Object.keys(record).filter((key) => !SENSITIVE_KEY_PATTERN.test(key))
  if (visibleKeys.length > 0) {
    return t ? t('common.fields', { count: visibleKeys.length }) : `${visibleKeys.length} fields`
  }

  return null
}

function extractSection(settings: unknown, key: string): Record<string, unknown> {
  const root = asRecord(settings)
  const section = root ? asRecord(root[key]) : null
  return section ?? {}
}

function collectSummary(section: Record<string, unknown>, preferredKeys: string[], limit = 6, t?: Translator): SummaryItem[] {
  const seen = new Set<string>()
  const items: SummaryItem[] = []

  const append = (key: string) => {
    if (seen.has(key) || SENSITIVE_KEY_PATTERN.test(key)) {
      return
    }

    const summary = summarizeValue(section[key], t)
    if (!summary) {
      return
    }

    seen.add(key)
    items.push({
      label: toTitleCase(key),
      value: summary,
    })
  }

  preferredKeys.forEach(append)

  Object.keys(section)
    .sort()
    .forEach(append)

  return items.slice(0, limit)
}

function getBooleanValue(value: unknown): boolean | null {
  return typeof value === 'boolean' ? value : null
}

function pickBoolean(...values: unknown[]): boolean | null {
  for (const value of values) {
    const boolValue = getBooleanValue(value)
    if (boolValue !== null) {
      return boolValue
    }
  }
  return null
}

function getIdArray(value: unknown): string[] {
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

function mergeIds(...sources: unknown[]): string[] {
  return Array.from(new Set(sources.flatMap((source) => getIdArray(source))))
}

function summarizeIds(ids: string[], t: Translator): string {
  if (ids.length === 0) {
    return t('common.notConfigured')
  }
  if (ids.length <= 3) {
    return ids.join(', ')
  }
  return t('common.listSummary', { items: ids.slice(0, 3).join(', '), count: ids.length })
}

function runtimeInstallLabel(t: Translator, hardware: api.LocalModelHardwareSummary | null): string {
  const suffix = hardware?.recommendedRuntimeLabel?.trim()
  if (!suffix) {
    return t('settings.runtime.localRuntimeInstallAction')
  }
  return `${t('settings.runtime.localRuntimeInstallAction')} (${suffix})`
}

type ChannelPanelState = {
  enabled: boolean | null
  registered: boolean | null
  running: boolean | null
  tokenConfigured: boolean | null
  allowedTargetSummary: string
  allowedUserSummary: string
  details?: SummaryItem[]
}

function resolveChannelPanelState(
  name: 'discord' | 'telegram',
  channelsSection: Record<string, unknown>,
  channelsStatus: api.ChannelsStatus | null,
  t: Translator
): ChannelPanelState {
  const configured = asRecord(channelsSection[name]) ?? {}
  const runtime = channelsStatus?.channels[name]

  if (name === 'discord') {
    const summary = api.normalizeDiscordChannelStatus(configured, runtime)
    const primaryRoom = summary.activeRooms[0] ?? null
    const voiceRuntimeDetails: SummaryItem[] = [
      { label: t('settings.channel.discord.textEnabled'), value: boolText(summary.textEnabled, t('common.yes'), t('common.no'), t('common.notReported')) },
      { label: t('settings.channel.discord.voiceEnabled'), value: boolText(summary.voiceEnabled, t('common.yes'), t('common.no'), t('common.notReported')) },
      {
        label: t('settings.channel.discord.voiceIngressEnabled'),
        value: boolText(summary.voiceIngressEnabled, t('common.yes'), t('common.no'), t('common.notReported')),
      },
      {
        label: t('settings.channel.discord.voiceReceiveExtension'),
        value: boolText(summary.voiceIngressAvailable, t('common.available'), t('common.unavailable'), t('common.notReported')),
      },
      { label: t('settings.channel.discord.allowedGuildIds'), value: summarizeIds(summary.allowedGuildIds, t) },
      { label: t('settings.channel.discord.allowedVoiceChannelIds'), value: summarizeIds(summary.allowedVoiceChannelIds, t) },
      { label: t('settings.channel.discord.messageMode'), value: localizeDiscordValue(summary.messageMode, 'discordSetup.messageMode', t) },
      { label: t('settings.channel.discord.autoJoinPolicy'), value: localizeDiscordValue(summary.autoJoinPolicy, 'discordSetup.autoJoinPolicy', t) },
      { label: t('settings.channel.discord.ingressGuildIds'), value: summarizeIds(summary.voiceIngressGuildIds, t) },
      {
        label: t('settings.channel.discord.activeVoiceRooms'),
        value: summary.activeVoiceRoomCount !== null ? String(summary.activeVoiceRoomCount) : t('common.notReported'),
      },
      {
        label: t('settings.channel.discord.voiceRuntimePhase'),
        value: localizeDiscordValue(summary.voiceRuntimePhase, 'settings.channel.discord.phase', t),
      },
      {
        label: t('settings.channel.discord.playback'),
        value: localizeDiscordValue(summary.playbackState ?? primaryRoom?.playbackState, 'settings.channel.discord.playbackState', t),
      },
      {
        label: t('settings.channel.discord.speaking'),
        value: localizeDiscordValue(summary.speakingState ?? primaryRoom?.speakingState, 'settings.channel.discord.speakingState', t),
      },
      {
        label: t('settings.channel.discord.voiceError'),
        value: summary.voiceRuntimeError ?? primaryRoom?.error ?? t('common.notReported'),
      },
      {
        label: t('settings.channel.discord.voiceIngressError'),
        value: summary.voiceIngressError ?? t('common.notReported'),
      },
    ]

    if (primaryRoom) {
      voiceRuntimeDetails.push(
        { label: t('settings.channel.discord.activeGuildId'), value: primaryRoom.guildId ?? t('common.notReported') },
        { label: t('settings.channel.discord.activeVoiceChannelId'), value: primaryRoom.channelId ?? t('common.notReported') },
        { label: t('settings.channel.discord.activeSessionId'), value: primaryRoom.sessionId ?? t('common.notReported') },
        { label: t('settings.channel.discord.joinedAt'), value: primaryRoom.joinedAt ?? t('common.notReported') },
        {
          label: t('settings.channel.discord.participants'),
          value: primaryRoom.participantCount !== null ? String(primaryRoom.participantCount) : t('common.notReported'),
        },
      )
    }

    return {
      enabled: summary.enabled,
      registered: summary.registered,
      running: summary.running,
      tokenConfigured: summary.botTokenConfigured,
      allowedTargetSummary: summarizeIds(summary.allowedChannelIds, t),
      allowedUserSummary: summarizeIds(summary.allowedUserIds, t),
      details: voiceRuntimeDetails,
    }
  }

  const runtimeRecord = asRecord(runtime) ?? {}

  const enabled = pickBoolean(runtimeRecord.enabled, configured.enabled)
  const registered = pickBoolean(runtimeRecord.registered)
  const running = pickBoolean(runtimeRecord.running)
  const tokenConfigured = pickBoolean(
    runtimeRecord.token_configured,
    runtimeRecord.tokenConfigured,
    configured.token_configured,
    configured.tokenConfigured,
    configured.has_token
  ) ?? (
    typeof configured.token === 'string' && configured.token.trim().length > 0
  )

  const allowedTargets = mergeIds(
    runtimeRecord.allowed_chat_ids,
    configured.allowed_chat_ids,
    runtimeRecord.allowed_channel_ids,
    configured.allowed_channel_ids
  )
  const allowedUsers = mergeIds(runtimeRecord.allowed_user_ids, configured.allowed_user_ids)

  return {
    enabled,
    registered,
    running,
    tokenConfigured,
    allowedTargetSummary: summarizeIds(allowedTargets, t),
    allowedUserSummary: summarizeIds(allowedUsers, t),
  }
}

function boolText(value: boolean | null, trueText: string, falseText: string, unknownText: string): string {
  if (value === true) {
    return trueText
  }
  if (value === false) {
    return falseText
  }
  return unknownText
}

function localizeDiscordValue(value: string | null, namespace: string, t: Translator): string {
  if (!value) {
    return t('common.notReported')
  }

  const normalized = value.trim()
  const key = `${namespace}.${normalized.replace(/[^a-zA-Z0-9]+/g, '_')}`
  const translated = t(key)
  return translated === key ? normalized : translated
}

function messageWithDetail(label: string, error: unknown): string {
  if (error instanceof Error && error.message.trim()) {
    return `${label}: ${error.message}`
  }
  return label
}

function getConnectedModelCount(models: unknown[]): number {
  return models.filter((entry) => {
    const record = asRecord(entry)
    if (!record) {
      return false
    }

    const status = typeof record.status === 'string' ? record.status.toLowerCase() : ''
    const healthy = typeof record.healthy === 'boolean' ? record.healthy : false
    const available = typeof record.available === 'boolean' ? record.available : false
    const hasName = typeof record.name === 'string' && record.name.length > 0

    return status === 'connected' || status === 'ready' || healthy || available || hasName
  }).length
}

function isUrlLike(value: string): boolean {
  return /^https?:\/\//i.test(value)
}

function looksLikeLocalModelPath(value: string): boolean {
  if (!value) {
    return false
  }
  if (isUrlLike(value)) {
    return false
  }
  return value.startsWith('/') || /^[A-Za-z]:[\\/]/.test(value) || value.startsWith('~/' )
}

function localModelDisplayName(value: string): string {
  const normalized = value.replace(/[\\]+/g, '/').replace(/\/+$/, '')
  const lastSegment = normalized.split('/').filter((segment) => segment.length > 0).pop()
  return lastSegment || value
}

function localModelRootFromModelInfo(modelInfo: api.ModelInfo): string {
  const metadataPath = modelInfo.metadata.path
  if (typeof metadataPath === 'string' && metadataPath.trim()) {
    const normalized = metadataPath.trim()
    return isGgufModelPath(normalized)
      ? normalized.replace(/[\\]+/g, '/').replace(/\/[^/]+$/, '')
      : normalized
  }

  const modelSpec = modelInfo.modelSpec ?? modelInfo.name
  if (!modelSpec) {
    return ''
  }
  const normalized = modelSpec.trim().replace(/[\\]+/g, '/')
  return isGgufModelPath(normalized)
    ? normalized.replace(/\/[^/]+$/, '')
    : normalized.replace(/\/+$/, '')
}

function providerLabel(provider: string | null | undefined): string | null {
  if (!provider) {
    return null
  }
  if (provider === 'vllm') {
    return 'vLLM'
  }
  return provider
}

function inferProviderChoice(value: string | null | undefined): ProviderChoice | null {
  if (!value) {
    return null
  }
  const prefix = value.split(':', 1)[0]
  return isProviderChoice(prefix) ? prefix : null
}

function toolCallModeLabel(mode: string | null | undefined): string {
  if (mode === 'simulated_fallback') {
    return 'Simulated Fallback'
  }
  if (mode === 'native') {
    return 'Native'
  }
  return 'Unknown'
}

function modelToolCallingMetadata(modelInfo: api.ModelInfo | null | undefined): {
  mode: string | null
  status: string | null
  message: string | null
  checkedAt: string | null
} {
  const metadata = modelInfo?.metadata
  return {
    mode: typeof metadata?.tool_call_mode === 'string' ? metadata.tool_call_mode : null,
    status: typeof metadata?.native_tool_calling_status === 'string' ? metadata.native_tool_calling_status : null,
    message: typeof metadata?.native_tool_calling_message === 'string' ? metadata.native_tool_calling_message : null,
    checkedAt: typeof metadata?.native_tool_calling_checked_at === 'string' ? metadata.native_tool_calling_checked_at : null,
  }
}

function modelReasoningMetadata(modelInfo: api.ModelInfo | null | undefined): {
  transportPreference: string | null
  continuityMode: string | null
  summaryRequested: boolean | null
  summarySupported: boolean | null
  summaryReceived: boolean | null
  replayedItems: number | null
} {
  const metadata = modelInfo?.metadata
  return {
    transportPreference:
      typeof metadata?.reasoning_transport_preference === 'string'
        ? metadata.reasoning_transport_preference
        : null,
    continuityMode:
      typeof metadata?.responses_continuity_mode === 'string'
        ? metadata.responses_continuity_mode
        : null,
    summaryRequested:
      typeof metadata?.reasoning_summary_requested === 'boolean'
        ? metadata.reasoning_summary_requested
        : null,
    summarySupported:
      typeof metadata?.reasoning_summary_supported === 'boolean'
        ? metadata.reasoning_summary_supported
        : null,
    summaryReceived:
      typeof metadata?.reasoning_summary_received === 'boolean'
        ? metadata.reasoning_summary_received
        : null,
    replayedItems:
      typeof metadata?.reasoning_items_replayed === 'number'
        ? metadata.reasoning_items_replayed
        : null,
  }
}

function modelInfoId(modelInfo: api.ModelInfo): string {
  return modelInfo.id || modelInfo.modelSpec || modelInfo.name
}

function modelInfoLabel(modelInfo: api.ModelInfo): string {
  const provider = providerLabel(modelInfo.provider)
  if (modelInfo.label && !isUrlLike(modelInfo.label)) {
    return modelInfo.label
  }
  if (modelInfo.name && !isUrlLike(modelInfo.name)) {
    return provider ? `${modelInfo.name} (${provider})` : modelInfo.name
  }
  return modelInfo.label || modelInfo.name || modelInfoId(modelInfo)
}

function activeConfiguredModel(settings: unknown, models: api.ModelInfo[]): api.ModelInfo | null {
  const root = asRecord(settings)
  const rootModel = typeof root?.model === 'string' ? root.model : ''
  const modelConfig = extractSection(settings, 'model_config')
  const provider = stringField(modelConfig, 'openai_compat_provider') ?? stringField(modelConfig, 'provider')
  const remoteModel = stringField(modelConfig, 'openai_compat_model')
  const remoteBaseUrl = stringField(modelConfig, 'openai_compat_base_url')
  const ollamaModel = stringField(modelConfig, 'ollama_model')

  for (const modelInfo of models) {
    const id = modelInfoId(modelInfo)
    if (
      id === rootModel ||
      modelInfo.modelSpec === rootModel ||
      modelInfo.baseUrl === rootModel ||
      (remoteModel && modelInfo.name === remoteModel && (!remoteBaseUrl || modelInfo.baseUrl === remoteBaseUrl)) ||
      (ollamaModel && modelInfo.name === ollamaModel)
    ) {
      return modelInfo
    }
  }

  if (remoteModel) {
    return {
      id: `${provider ?? 'openai_compat'}:${remoteBaseUrl ?? rootModel}:${remoteModel}`,
      name: remoteModel,
      label: provider ? `${remoteModel} (${provider})` : remoteModel,
      provider,
      modelSpec: rootModel || remoteBaseUrl,
      baseUrl: remoteBaseUrl ?? (isUrlLike(rootModel) ? rootModel : null),
      backendType: provider === 'openai_codex' ? 'openai_codex' : 'openai_compat',
      authProfileId: null,
      authMode: null,
      contextLength: null,
      supportsToolCalling: null,
      metadata: {},
    }
  }

  return null
}

function getPrimaryModel(settings: unknown, models: unknown[], t?: Translator): string {
  const typedModels = models.filter((entry): entry is api.ModelInfo => Boolean(asRecord(entry)))
  const activeModel = activeConfiguredModel(settings, typedModels)
  if (activeModel) {
    return modelInfoLabel(activeModel)
  }

  const root = asRecord(settings)
  const rootModel = root ? formatScalar(root.model, t) : null
  if (rootModel !== null) {
    if (looksLikeLocalModelPath(rootModel)) {
      return localModelDisplayName(rootModel)
    }
    return rootModel
  }

  const modelSection = extractSection(settings, 'model')
  const modelKeys = ['model', 'name', 'selected_model', 'default_model', 'backend']

  for (const key of modelKeys) {
    const value = formatScalar(modelSection[key], t)
    if (value !== null) {
      return value
    }
  }

  for (const entry of models) {
    const record = asRecord(entry)
    if (!record) {
      continue
    }

    const value = formatScalar(record.name, t) ?? formatScalar(record.id, t) ?? formatScalar(record.model, t)
    if (value !== null) {
      return value
    }
  }

  return t ? t('common.unknown') : 'Unknown'
}

function stringField(record: Record<string, unknown>, key: string): string | null {
  const value = record[key]
  return typeof value === 'string' && value.length > 0 ? value : null
}

function modelInfoFromRecord(record: Record<string, unknown>): api.ModelInfo | null {
  const id =
    stringField(record, 'id') ??
    stringField(record, 'model_spec') ??
    stringField(record, 'name') ??
    stringField(record, 'model')
  if (!id) {
    return null
  }

  const name =
    stringField(record, 'name') ??
    stringField(record, 'model') ??
    stringField(record, 'model_spec') ??
    id

  return {
    id,
    name,
    label: stringField(record, 'label') ?? name,
    provider: stringField(record, 'provider') ?? inferProviderChoice(id),
    modelSpec: stringField(record, 'model_spec'),
    baseUrl: stringField(record, 'base_url'),
    backendType: stringField(record, 'backend_type') ?? '',
    authProfileId: stringField(record, 'auth_profile_id'),
    authMode: stringField(record, 'auth_mode'),
    contextLength: typeof record.context_length === 'number' ? record.context_length : null,
    supportsToolCalling: typeof record.supports_tool_calling === 'boolean' ? record.supports_tool_calling : null,
    metadata: {},
  }
}

function configuredModelsFromSettings(settings: unknown): api.ModelInfo[] {
  const modelSetup = extractSection(settings, 'model_setup')
  const configuredModels = modelSetup.configured_models
  if (!Array.isArray(configuredModels)) {
    return []
  }

  return configuredModels
    .map((entry) => {
      const record = asRecord(entry)
      return record ? modelInfoFromRecord(record) : null
    })
    .filter((entry): entry is api.ModelInfo => entry !== null)
}

function mergeModelInfos(primary: api.ModelInfo[], secondary: api.ModelInfo[]): api.ModelInfo[] {
  const seen = new Set<string>()
  const merged: api.ModelInfo[] = []
  for (const modelInfo of [...primary, ...secondary]) {
    const id = modelInfoId(modelInfo)
    if (!id || seen.has(id)) {
      continue
    }
    seen.add(id)
    merged.push(modelInfo)
  }
  return merged
}

function configuredModelRecordFromModelInfo(modelInfo: api.ModelInfo): Record<string, string | null> {
  return {
    id: modelInfoId(modelInfo),
    label: modelInfoLabel(modelInfo),
    provider: modelInfo.provider ?? null,
    model: modelInfo.name || null,
    model_spec: modelInfo.modelSpec || null,
    base_url: modelInfo.baseUrl,
    backend_type: modelInfo.backendType ?? null,
    auth_profile_id: modelInfo.authProfileId,
    auth_mode: modelInfo.authMode,
  }
}

function updateSavedModelsInSettings(
  current: api.Settings | null,
  savedModels: api.ModelInfo[],
  configuredModel: string
): api.Settings | null {
  if (!current) {
    return current
  }

  return {
    ...current,
    model: configuredModel,
    model_setup: {
      ...(current.model_setup ?? {}),
      configured_models: savedModels.map(configuredModelRecordFromModelInfo),
    },
  }
}

function omitConfiguredModels(section: Record<string, unknown>): Record<string, unknown> {
  const next = { ...section }
  delete next.configured_models
  return next
}

function withNullableString(value: string | null | undefined): string | null {
  return value ?? null
}

function notifyModelsUpdated() {
  window.dispatchEvent(new Event(MODELS_UPDATED_EVENT))
}

function baseUrlFromModelInfo(modelInfo: api.ModelInfo): string | null {
  if (modelInfo.baseUrl) {
    return modelInfo.baseUrl
  }
  const baseUrl = modelInfo.metadata.base_url
  return typeof baseUrl === 'string' && baseUrl.length > 0 ? baseUrl : null
}

function SurfaceSection({
  title,
  description,
  items,
}: {
  title: string
  description?: string
  items: SummaryItem[]
}) {
  const { t } = useI18n()

  return (
    <section className="rounded-lg border border-border bg-surface-layer">
      <div className="border-b border-border px-4 py-3">
        <h3 className="text-sm font-semibold text-foreground">{title}</h3>
        {description ? <p className="mt-0.5 text-xs text-muted-foreground">{description}</p> : null}
      </div>
      <div className="divide-y divide-border">
        {items.length > 0 ? (
          items.map((item) => (
            <div key={item.label} className="flex items-start justify-between gap-4 px-4 py-3">
              <p className="text-sm text-muted-foreground">{item.label}</p>
              <p className="max-w-[60%] text-right text-sm font-medium text-foreground">{item.value}</p>
            </div>
          ))
        ) : (
          <div className="px-4 py-6 text-sm text-muted-foreground">{t('settings.noSummary')}</div>
        )}
      </div>
    </section>
  )
}

function StatTile({
  title,
  value,
  icon: Icon,
}: {
  title: string
  value: string
  icon: React.ComponentType<{ className?: string }>
}) {
  return (
    <div className="rounded-lg border border-border bg-surface-layer px-4 py-3">
      <div className="flex items-center justify-between gap-3">
        <p className="text-[11px] uppercase tracking-wide text-muted-foreground">{title}</p>
        <Icon className="h-4 w-4 text-muted-foreground" />
      </div>
      <p className="mt-2 text-lg font-semibold text-foreground">{value}</p>
    </div>
  )
}

function OverviewBadge({ ok }: { ok: boolean }) {
  const { t } = useI18n()

  return ok ? (
    <Badge variant="success">
      <CheckCircle2 className="h-3 w-3" />
      {t('settings.badge.connected')}
    </Badge>
  ) : (
    <Badge variant="warning">
      <AlertCircle className="h-3 w-3" />
      {t('settings.badge.partial')}
    </Badge>
  )
}

function ChannelPanel({
  channelName,
  title,
  icon: Icon,
  state,
  targetLabel,
  loading,
  onRefresh,
  onControlSuccess,
}: {
  channelName: 'discord' | 'telegram'
  title: string
  icon: React.ComponentType<{ className?: string }>
  state: ChannelPanelState
  targetLabel: string
  loading: boolean
  onRefresh: () => void
  onControlSuccess?: () => void
}) {
  const { t } = useI18n()
  const hasDetails = (state.details?.length ?? 0) > 0
  const [controlLoading, setControlLoading] = React.useState(false)
  const [controlMessage, setControlMessage] = React.useState<FormMessage>(null)

  const canStart = state.enabled === true && state.tokenConfigured === true && state.running !== true
  const canStop = state.running === true

  const handleStart = async () => {
    setControlLoading(true)
    setControlMessage(null)
    try {
      if (typeof settingsApi.startChannel !== 'function') {
        throw new Error(t('channelControl.startApiUnavailable'))
      }
      await settingsApi.startChannel(channelName)
      onControlSuccess?.()
      setControlMessage({ type: 'success', text: t('channelControl.started', { name: title }) })
    } catch (controlError) {
      setControlMessage({
        type: 'error',
        text: messageWithDetail(t('channelControl.errorStart', { name: title }), controlError),
      })
    } finally {
      setControlLoading(false)
    }
  }

  const handleStop = async () => {
    setControlLoading(true)
    setControlMessage(null)
    try {
      if (typeof settingsApi.stopChannel !== 'function') {
        throw new Error(t('channelControl.stopApiUnavailable'))
      }
      await settingsApi.stopChannel(channelName)
      onControlSuccess?.()
      setControlMessage({ type: 'success', text: t('channelControl.stopped', { name: title }) })
    } catch (controlError) {
      setControlMessage({
        type: 'error',
        text: messageWithDetail(t('channelControl.errorStop', { name: title }), controlError),
      })
    } finally {
      setControlLoading(false)
    }
  }

  return (
    <section className="rounded-lg border border-border bg-surface-layer">
      <div className="flex items-center justify-between border-b border-border px-4 py-3">
        <div className="flex items-center gap-2">
          <Icon className="h-4 w-4 text-muted-foreground" />
          <h3 className="text-base font-semibold text-foreground">{title}</h3>
        </div>
        <div className="flex items-center gap-2">
          <Button type="button" variant="secondary" size="sm" loading={loading} onClick={onRefresh}>
            <RefreshCw className="h-3.5 w-3.5" />
            {t('common.refresh')}
          </Button>
          <Button
            type="button"
            variant="secondary"
            size="sm"
            disabled={!canStart || controlLoading}
            loading={controlLoading && canStart}
            onClick={() => void handleStart()}
          >
            {t('channelControl.start')}
          </Button>
          <Button
            type="button"
            variant="secondary"
            size="sm"
            disabled={!canStop || controlLoading}
            loading={controlLoading && canStop}
            onClick={() => void handleStop()}
          >
            {t('channelControl.stop')}
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-x-6 gap-y-2 px-4 py-4 text-sm md:grid-cols-2">
        <div className="flex items-center justify-between gap-4">
          <span className="text-muted-foreground">{t('settings.channel.enabledState')}</span>
          <span className="font-medium text-foreground">
            {boolText(state.enabled, t('settings.boolean.enabled'), t('settings.boolean.notEnabled'), t('common.notReported'))}
          </span>
        </div>
        <div className="flex items-center justify-between gap-4">
          <span className="text-muted-foreground">{t('settings.channel.registeredManager')}</span>
          <span className="font-medium text-foreground">{boolText(state.registered, t('common.yes'), t('common.no'), t('common.notReported'))}</span>
        </div>
        <div className="flex items-center justify-between gap-4">
          <span className="text-muted-foreground">{t('settings.channel.running')}</span>
          <span className="font-medium text-foreground">{boolText(state.running, t('common.yes'), t('common.no'), t('common.notReported'))}</span>
        </div>
        <div className="flex items-center justify-between gap-4">
          <span className="text-muted-foreground">{t('settings.channel.token')}</span>
          <span className="font-medium text-foreground">
            {boolText(state.tokenConfigured, t('common.configured'), t('common.notConfigured'), t('common.notReported'))}
          </span>
        </div>
        <div className="flex items-center justify-between gap-4">
          <span className="text-muted-foreground">{targetLabel}</span>
          <span className="text-right font-medium text-foreground">{state.allowedTargetSummary}</span>
        </div>
        <div className="flex items-center justify-between gap-4">
          <span className="text-muted-foreground">{t('settings.channel.allowedUserIds')}</span>
          <span className="text-right font-medium text-foreground">{state.allowedUserSummary}</span>
        </div>
      </div>

      <div className="px-4 pb-4">
        <SettingMessage message={controlMessage} />
      </div>

      {hasDetails ? (
        <div className="border-t border-border px-4 py-4">
          <div className="grid grid-cols-1 gap-x-6 gap-y-2 text-sm md:grid-cols-2">
            {state.details?.map((item) => (
              <div key={item.label} className="flex items-center justify-between gap-4">
                <span className="text-muted-foreground">{item.label}</span>
                <span className="text-right font-medium text-foreground">{item.value}</span>
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </section>
  )
}

function parseIdCsv(input: string): number[] {
  return input
    .split(',')
    .map((item) => item.trim())
    .filter((item) => item.length > 0)
    .map((item) => Number.parseInt(item, 10))
    .filter((item) => Number.isFinite(item) && item > 0)
}

function DiscordSetupForm({
  channels,
  onUpdated,
}: {
  channels: Record<string, unknown>
  onUpdated: (settings: api.Settings) => void
}) {
  const { t } = useI18n()
  const discord = asRecord(channels.discord) ?? {}
  const [botToken, setBotToken] = React.useState('')
  const [enabled, setEnabled] = React.useState(getBooleanSetting(discord, 'enabled', false))
  const [textEnabled, setTextEnabled] = React.useState(getBooleanSetting(discord, 'text_enabled', true))
  const [voiceEnabled, setVoiceEnabled] = React.useState(getBooleanSetting(discord, 'voice_enabled', true))
  const [voiceAutoReply, setVoiceAutoReply] = React.useState(getBooleanSetting(discord, 'voice_auto_reply', true))
  const [voiceSttEnabled, setVoiceSttEnabled] = React.useState(getBooleanSetting(discord, 'voice_stt_enabled', true))
  const [voiceTtsEnabled, setVoiceTtsEnabled] = React.useState(getBooleanSetting(discord, 'voice_tts_enabled', true))
  const [messageMode, setMessageMode] = React.useState(getStringSetting(discord, 'message_mode', 'mentions_only'))
  const [allowedGuildIds, setAllowedGuildIds] = React.useState(
    getIdSetting(discord, 'allowed_guild_ids').join(', ')
  )
  const [allowedChannelIds, setAllowedChannelIds] = React.useState(
    getIdSetting(discord, 'allowed_channel_ids').join(', ')
  )
  const [allowedVoiceChannelIds, setAllowedVoiceChannelIds] = React.useState(
    getIdSetting(discord, 'allowed_voice_channel_ids').join(', ')
  )
  const [allowedUserIds, setAllowedUserIds] = React.useState(
    getIdSetting(discord, 'allowed_user_ids').join(', ')
  )
  const [submitting, setSubmitting] = React.useState(false)
  const [message, setMessage] = React.useState<FormMessage>(null)

  React.useEffect(() => {
    const latestDiscord = asRecord(channels.discord) ?? {}
    setEnabled(getBooleanSetting(latestDiscord, 'enabled', false))
    setTextEnabled(getBooleanSetting(latestDiscord, 'text_enabled', true))
    setVoiceEnabled(getBooleanSetting(latestDiscord, 'voice_enabled', true))
    setVoiceAutoReply(getBooleanSetting(latestDiscord, 'voice_auto_reply', true))
    setVoiceSttEnabled(getBooleanSetting(latestDiscord, 'voice_stt_enabled', true))
    setVoiceTtsEnabled(getBooleanSetting(latestDiscord, 'voice_tts_enabled', true))
    setMessageMode(getStringSetting(latestDiscord, 'message_mode', 'mentions_only'))
    setAllowedGuildIds(getIdSetting(latestDiscord, 'allowed_guild_ids').join(', '))
    setAllowedChannelIds(getIdSetting(latestDiscord, 'allowed_channel_ids').join(', '))
    setAllowedVoiceChannelIds(getIdSetting(latestDiscord, 'allowed_voice_channel_ids').join(', '))
    setAllowedUserIds(getIdSetting(latestDiscord, 'allowed_user_ids').join(', '))
  }, [channels])

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setSubmitting(true)
    setMessage(null)

    try {
      if (typeof settingsApi.setupDiscord !== 'function') {
        throw new Error(t('discordSetup.errorApiUnavailable'))
      }
      if (botToken.trim().length === 0) {
        throw new Error(t('discordSetup.errorTokenRequired'))
      }

      const settings = await settingsApi.setupDiscord({
        bot_token: botToken.trim(),
        enabled,
        text_enabled: textEnabled,
        voice_enabled: voiceEnabled,
        allowed_guild_ids: parseIdCsv(allowedGuildIds),
        allowed_channel_ids: parseIdCsv(allowedChannelIds),
        allowed_voice_channel_ids: parseIdCsv(allowedVoiceChannelIds),
        allowed_user_ids: parseIdCsv(allowedUserIds),
        message_mode:
          messageMode === 'all_messages' || messageMode === 'slash_only'
            ? messageMode
            : 'mentions_only',
        auto_join_policy: 'manual_only',
        voice_auto_reply: voiceAutoReply,
        voice_stt_enabled: voiceSttEnabled,
        voice_tts_enabled: voiceTtsEnabled,
      })
      setBotToken('')
      onUpdated(settings)
      setMessage({
        type: 'success',
        text: t('discordSetup.successSaved'),
      })
    } catch (setupError) {
      setMessage({
        type: 'error',
        text: messageWithDetail(t('discordSetup.errorSave'), setupError),
      })
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <section className="rounded-lg border border-border bg-surface-layer">
      <div className="border-b border-border px-4 py-3">
        <div className="flex items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold text-foreground">{t('discordSetup.title')}</h3>
            <p className="mt-0.5 text-xs text-muted-foreground">
              {t('discordSetup.description')}
            </p>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            <Button asChild type="button" variant="secondary" size="sm">
              <Link href="/settings/discord-guide">
                <BookOpen className="h-3.5 w-3.5" />
                {t('discordGuide.openPage')}
              </Link>
            </Button>
            <KeyRound className="h-4 w-4 text-muted-foreground" />
          </div>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4 px-4 py-4">
        <label className="space-y-1.5">
          <SettingLabel>{t('discordSetup.botToken')}</SettingLabel>
          <p className="text-xs text-muted-foreground">
            {t('discordSetup.botTokenHelp')}
          </p>
          <Input
            type="password"
            value={botToken}
            onChange={(event) => setBotToken(event.target.value)}
            placeholder={t('discordSetup.botTokenPlaceholder')}
            autoComplete="off"
            className="min-w-0 font-mono text-xs"
          />
        </label>

        <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
          <div className="flex items-center justify-between gap-3 rounded-md border border-border bg-canvas px-3 py-2">
            <span className="text-sm text-foreground">{t('discordSetup.enableChannel')}</span>
            <Switch checked={enabled} onCheckedChange={setEnabled} />
          </div>
          <div className="flex items-center justify-between gap-3 rounded-md border border-border bg-canvas px-3 py-2">
            <span className="text-sm text-foreground">{t('discordSetup.enableText')}</span>
            <Switch checked={textEnabled} onCheckedChange={setTextEnabled} />
          </div>
          <div className="flex items-center justify-between gap-3 rounded-md border border-border bg-canvas px-3 py-2">
            <span className="text-sm text-foreground">{t('discordSetup.enableVoice')}</span>
            <Switch checked={voiceEnabled} onCheckedChange={setVoiceEnabled} />
          </div>
          <div className="flex items-center justify-between gap-3 rounded-md border border-border bg-canvas px-3 py-2">
            <span className="text-sm text-foreground">{t('discordSetup.autoReply')}</span>
            <Switch checked={voiceAutoReply} onCheckedChange={setVoiceAutoReply} />
          </div>
          <div className="flex items-center justify-between gap-3 rounded-md border border-border bg-canvas px-3 py-2">
            <span className="text-sm text-foreground">{t('discordSetup.voiceStt')}</span>
            <Switch checked={voiceSttEnabled} onCheckedChange={setVoiceSttEnabled} />
          </div>
          <div className="flex items-center justify-between gap-3 rounded-md border border-border bg-canvas px-3 py-2">
            <span className="text-sm text-foreground">{t('discordSetup.voiceTts')}</span>
            <Switch checked={voiceTtsEnabled} onCheckedChange={setVoiceTtsEnabled} />
          </div>
        </div>

        <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
          <label className="space-y-1.5">
            <SettingLabel>{t('discordSetup.messageMode')}</SettingLabel>
            <SelectSetting
              value={messageMode}
              onValueChange={setMessageMode}
              options={['mentions_only', 'all_messages', 'slash_only']}
              getOptionLabel={(option) => {
                if (option === 'all_messages') {
                  return t('discordSetup.messageMode.allMessages')
                }
                if (option === 'slash_only') {
                  return t('discordSetup.messageMode.slashOnly')
                }
                return t('discordSetup.messageMode.mentionsOnly')
              }}
            />
          </label>
          <label className="space-y-1.5">
            <SettingLabel>{t('discordSetup.autoJoinPolicy')}</SettingLabel>
            <Input value={t('discordSetup.autoJoinPolicy.manualOnly')} disabled className="text-xs" />
          </label>
        </div>

        <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
          <label className="space-y-1.5">
            <SettingLabel>{t('discordSetup.allowedGuildIds')}</SettingLabel>
            <Input
              value={allowedGuildIds}
              onChange={(event) => setAllowedGuildIds(event.target.value)}
              placeholder={t('discordSetup.idsPlaceholder')}
              className="font-mono text-xs"
            />
          </label>
          <label className="space-y-1.5">
            <SettingLabel>{t('discordSetup.allowedTextChannelIds')}</SettingLabel>
            <Input
              value={allowedChannelIds}
              onChange={(event) => setAllowedChannelIds(event.target.value)}
              placeholder={t('discordSetup.idsPlaceholder')}
              className="font-mono text-xs"
            />
          </label>
          <label className="space-y-1.5">
            <SettingLabel>{t('discordSetup.allowedVoiceChannelIds')}</SettingLabel>
            <Input
              value={allowedVoiceChannelIds}
              onChange={(event) => setAllowedVoiceChannelIds(event.target.value)}
              placeholder={t('discordSetup.idsPlaceholder')}
              className="font-mono text-xs"
            />
          </label>
          <label className="space-y-1.5">
            <SettingLabel>{t('discordSetup.allowedUserIds')}</SettingLabel>
            <Input
              value={allowedUserIds}
              onChange={(event) => setAllowedUserIds(event.target.value)}
              placeholder={t('discordSetup.idsPlaceholder')}
              className="font-mono text-xs"
            />
          </label>
        </div>

        <SettingMessage message={message} />

        <div className="flex justify-end">
          <Button type="submit" variant="primary" size="sm" loading={submitting}>
            <Save className="h-3.5 w-3.5" />
            {t('discordSetup.save')}
          </Button>
        </div>
      </form>
    </section>
  )
}

function TerminalLocalPanel({
  channelsRunning,
  enabledChannels,
}: {
  channelsRunning: boolean
  enabledChannels: string[]
}) {
  const { t } = useI18n()

  return (
    <section className="rounded-lg border border-border bg-surface-layer">
      <div className="flex items-center gap-2 border-b border-border px-4 py-3">
        <Terminal className="h-4 w-4 text-muted-foreground" />
        <h3 className="text-base font-semibold text-foreground">Terminal / Local</h3>
      </div>

      <div className="space-y-3 px-4 py-4 text-sm">
        <div className="flex items-center justify-between gap-4">
          <span className="text-muted-foreground">{t('settings.channel.localCli')}</span>
          <span className="font-medium text-foreground">{t('common.available')}</span>
        </div>
        <div className="flex items-center justify-between gap-4">
          <span className="text-muted-foreground">{t('settings.channel.channelsRunner')}</span>
          <span className="font-medium text-foreground">
            {channelsRunning ? t('settings.channel.running') : t('settings.boolean.notEnabled')}
          </span>
        </div>
        <div className="flex items-center justify-between gap-4">
          <span className="text-muted-foreground">{t('settings.channel.enabledExternalChannels')}</span>
          <span className="text-right font-medium text-foreground">
            {enabledChannels.length > 0 ? enabledChannels.join(', ') : t('common.none')}
          </span>
        </div>

        <div className="border-t border-border pt-3">
          <p className="text-sm font-medium text-foreground">{t('settings.channel.commands')}</p>
          <pre className="mt-2 overflow-x-auto rounded-md bg-canvas px-3 py-2 text-sm text-foreground">
{`uv run mochi channels run
uv run mochi chat
uv run mochi --help`}
          </pre>
        </div>
      </div>
    </section>
  )
}

type ProviderChoice = api.ModelProvider

const providerOptions: Array<{
  value: ProviderChoice
  label: string
  defaultBaseUrl: string
  defaultModel: string
  needsApiKey: boolean
}> = [
  {
    value: 'ollama',
    label: 'Ollama',
    defaultBaseUrl: 'http://localhost:11434',
    defaultModel: 'llama3.2',
    needsApiKey: false,
  },
  {
    value: 'openai_compat',
    label: 'OpenAI-compatible',
    defaultBaseUrl: 'https://api.openai.com/v1',
    defaultModel: 'gpt-4o-mini',
    needsApiKey: true,
  },
  {
    value: 'vllm',
    label: 'vLLM',
    defaultBaseUrl: 'http://localhost:8000/v1',
    defaultModel: 'Qwen/Qwen2.5-7B-Instruct',
    needsApiKey: false,
  },
  {
    value: 'openai_codex',
    label: 'OpenAI Codex',
    defaultBaseUrl: 'https://chatgpt.com/backend-api',
    defaultModel: 'gpt-5.4',
    needsApiKey: false,
  },
  {
    value: 'gemini',
    label: 'Gemini',
    defaultBaseUrl: 'https://generativelanguage.googleapis.com/v1beta/openai',
    defaultModel: 'gemini-3-flash-preview',
    needsApiKey: true,
  },
  {
    value: 'anthropic',
    label: 'Anthropic',
    defaultBaseUrl: 'https://api.anthropic.com/v1',
    defaultModel: 'claude-sonnet-4-6',
    needsApiKey: true,
  },
  {
    value: 'local',
    label: 'Local model',
    defaultBaseUrl: '',
    defaultModel: '',
    needsApiKey: false,
  },
]

const defaultSttBackends = [
  'auto',
  'faster-whisper',
  'external-api',
  'openai-api',
  'openai-whisper',
  'qwen-asr',
  'vosk',
  'whisper-cpp',
  'whisperlivekit',
]

const defaultTtsBackends = [
  'kokoro-tts',
  'piper',
  'coqui-tts',
  'external-api',
  'openai-tts',
  'edge-tts',
  'auto',
]

const defaultSttModelsByBackend: Record<string, string[]> = {
  auto: ['tiny', 'base', 'small', 'medium', 'large-v3', 'turbo', 'distil-large-v3'],
  'faster-whisper': ['tiny', 'base', 'small', 'medium', 'large-v3', 'turbo', 'distil-large-v3'],
  'openai-whisper': ['tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large-v3', 'turbo'],
  'external-api': ['whisper-1'],
  'openai-api': ['whisper-1'],
  'qwen-asr': ['qwen3-asr-0.6b', 'qwen3-asr-1.7b'],
  vosk: ['vosk-model-small-cn-0.22', 'vosk-model-cn-0.22', 'vosk-model-small-en-us-0.15'],
  'whisper-cpp': ['tiny', 'base', 'small', 'medium', 'large-v3'],
  whisperlivekit: ['tiny', 'base', 'small', 'medium', 'large-v3', 'turbo'],
}

const defaultTtsModelsByBackend: Record<string, string[]> = {
  auto: ['none'],
  'edge-tts': ['none'],
  'external-api': ['gpt-4o-mini-tts', 'tts-1', 'tts-1-hd'],
  'openai-tts': ['gpt-4o-mini-tts', 'tts-1', 'tts-1-hd'],
  piper: ['none'],
  'coqui-tts': [
    'tts_models/en/ljspeech/tacotron2-DDC',
    'tts_models/en/ljspeech/glow-tts',
    'tts_models/multilingual/multi-dataset/xtts_v2',
  ],
  'kokoro-tts': ['none'],
}

const defaultTtsVoice = 'af_heart'

const defaultTtsVoicesByBackend: Record<string, string[]> = {
  auto: [defaultTtsVoice, 'af_bella', 'bf_emma', 'am_adam', 'bm_george'],
  'edge-tts': ['en-US-AriaNeural', 'zh-CN-XiaoxiaoNeural', 'zh-TW-HsiaoChenNeural'],
  'external-api': ['alloy', 'verse', 'aria', 'coral', 'sage', 'nova', 'shimmer'],
  'openai-tts': ['alloy', 'verse', 'aria', 'coral', 'sage', 'nova', 'shimmer'],
  piper: ['zh_CN-huayan-medium', 'en_US-lessac-medium'],
  'coqui-tts': ['default'],
  'kokoro-tts': ['af_heart', 'af_bella', 'bf_emma', 'am_adam', 'bm_george'],
}

const sttLanguageOptions = ['auto', 'zh', 'en', 'ja', 'ko', 'fr', 'de', 'es']
const sttDeviceOptions = ['auto', 'cpu', 'cuda']
const ttsLanguageOptions = ['none', 'zh', 'en', 'ja', 'ko', 'fr', 'de', 'es']
const ttsSpeedOptions = ['0.75', '0.9', '1', '1.1', '1.25', '1.5']
const replyModelModeOptions = ['global', 'fixed']
const sessionModeOptions = ['shared', 'isolated']

function isExternalSttBackend(value: string): boolean {
  return value === 'external-api' || value === 'openai-api'
}

function isExternalTtsBackend(value: string): boolean {
  return value === 'external-api' || value === 'openai-tts'
}

function voiceBackendOptionLabel(value: string): string {
  if (value === 'external-api') {
    return 'External API'
  }
  if (value === 'openai-api') {
    return 'OpenAI-compatible (legacy)'
  }
  if (value === 'openai-tts') {
    return 'OpenAI-compatible (legacy)'
  }
  return value
}

function dedupeVoiceBackendOptions(options: string[], kind: 'stt' | 'tts', current: string): string[] {
  const filtered = options.filter((option) => {
    if (option === current) {
      return true
    }
    if (kind === 'stt' && option === 'openai-api' && options.includes('external-api')) {
      return false
    }
    if (kind === 'tts' && option === 'openai-tts' && options.includes('external-api')) {
      return false
    }
    return true
  })
  const priority =
    kind === 'tts'
      ? ['kokoro-tts', 'piper', 'coqui-tts', 'external-api', 'openai-tts', 'edge-tts', 'auto']
      : ['faster-whisper', 'whisper-cpp', 'vosk', 'qwen-asr', 'external-api', 'openai-api', 'openai-whisper', 'whisperlivekit', 'auto']
  const ordered = (filtered.length > 0 ? filtered : options).slice().sort((a, b) => {
    const indexA = priority.indexOf(a)
    const indexB = priority.indexOf(b)
    const scoreA = indexA === -1 ? Number.MAX_SAFE_INTEGER : indexA
    const scoreB = indexB === -1 ? Number.MAX_SAFE_INTEGER : indexB
    return scoreA - scoreB || a.localeCompare(b)
  })
  return ordered
}

type FormMessage = { type: 'success' | 'error'; text: string } | null

function InferenceSettingsForm({
  agent,
  onUpdated,
}: {
  agent: api.AgentSettings | undefined
  onUpdated: (settings: api.Settings) => void
}) {
  const { t } = useI18n()
  const [params, setParams] = React.useState<InferenceParams>(resolveEffectiveInferenceParams(undefined, agent))
  const [presets, setPresets] = React.useState<api.InferencePreset[]>(agent?.presets ?? [])
  const [activePresetName, setActivePresetName] = React.useState(getActivePreset(agent)?.name ?? 'default')
  const [selectedPresetName, setSelectedPresetName] = React.useState(getActivePreset(agent)?.name ?? 'default')
  const [dialogMode, setDialogMode] = React.useState<'save' | 'rename' | 'duplicate' | null>(null)
  const [presetNameDraft, setPresetNameDraft] = React.useState('')
  const [submitting, setSubmitting] = React.useState(false)
  const [message, setMessage] = React.useState<FormMessage>(null)

  React.useEffect(() => {
    setParams(resolveEffectiveInferenceParams(undefined, agent))
    setPresets(agent?.presets ?? [])
    const nextPreset = getActivePreset(agent)?.name ?? 'default'
    setActivePresetName(nextPreset)
    setSelectedPresetName(nextPreset)
  }, [agent])

  const setParam = React.useCallback(<K extends keyof InferenceParams>(key: K, value: InferenceParams[K]) => {
    setParams((current) => ({ ...current, [key]: value }))
  }, [])

  const selectedPreset = React.useMemo(
    () => presets.find((preset) => preset.name === selectedPresetName) ?? presets[0] ?? null,
    [presets, selectedPresetName]
  )

  const buildPresetFromParams = React.useCallback((name: string): api.InferencePreset => ({
    name,
    system_prompt: params.systemPrompt,
    temperature: params.temperature,
    max_tokens: params.maxTokens,
    top_p: params.topP,
    min_p: params.minP,
    top_k: params.topK,
    frequency_penalty: params.frequencyPenalty,
    presence_penalty: params.presencePenalty,
    repeat_penalty: params.repeatPenalty,
    reasoning_effort: params.reasoningEffort,
  }), [params])

  const persistAgent = React.useCallback(async (nextAgent: api.AgentSettingsUpdate) => {
    if (typeof settingsApi.updateSettings !== 'function') {
      throw new Error('Settings update API client is unavailable.')
    }
    const settings = await settingsApi.updateSettings({ agent: nextAgent })
    onUpdated(settings)
    window.dispatchEvent(new Event('mochi:settings-updated'))
    return settings
  }, [onUpdated])

  const handleSaveSettings = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setSubmitting(true)
    setMessage(null)

    try {
      const targetPresetName =
        selectedPresetName ||
        getActivePreset(agent)?.name ||
        presets[0]?.name ||
        'default'
      const nextPresets = presets.some((preset) => preset.name === targetPresetName)
        ? presets.map((preset) => (
            preset.name === targetPresetName ? buildPresetFromParams(targetPresetName) : preset
          ))
        : [...presets, buildPresetFromParams(targetPresetName)]
      const settings = await persistAgent({
        system_prompt: params.systemPrompt,
        temperature: params.temperature,
        max_tokens: params.maxTokens,
        top_p: params.topP,
        min_p: params.minP,
        top_k: params.topK,
        frequency_penalty: params.frequencyPenalty,
        presence_penalty: params.presencePenalty,
        repeat_penalty: params.repeatPenalty,
        reasoning_effort: params.reasoningEffort,
        show_token_stats: params.showTokenStats,
        presets: nextPresets,
        active_preset: activePresetName,
      })
      setPresets(settings.agent?.presets ?? [])
      setActivePresetName(settings.agent?.active_preset ?? activePresetName)
      setSelectedPresetName(targetPresetName)
      setMessage({ type: 'success', text: 'Inference settings saved.' })
    } catch (updateError) {
      setMessage({ type: 'error', text: messageWithDetail('Failed to save inference settings', updateError) })
    } finally {
      setSubmitting(false)
    }
  }

  const handleApplyPreset = React.useCallback((preset: api.InferencePreset | null) => {
    if (!preset) {
      return
    }
    setParams((current) => ({
      ...current,
      ...inferencePresetToParams(preset),
      showTokenStats: current.showTokenStats,
    }))
  }, [])

  const handlePresetMutation = async (nextPresets: api.InferencePreset[], nextActivePreset: string) => {
    const settings = await persistAgent({
      presets: nextPresets,
      active_preset: nextActivePreset,
    })
    setPresets(settings.agent?.presets ?? [])
    setActivePresetName(settings.agent?.active_preset ?? nextActivePreset)
    setSelectedPresetName(settings.agent?.active_preset ?? nextActivePreset)
  }

  const handleDialogConfirm = async () => {
    const nextName = presetNameDraft.trim()
    if (!nextName) {
      return
    }
    setSubmitting(true)
    setMessage(null)
    try {
      if (dialogMode === 'save') {
        const nextPresets = [...presets.filter((preset) => preset.name !== nextName), buildPresetFromParams(nextName)]
        await handlePresetMutation(nextPresets, nextName)
      } else if (dialogMode === 'rename' && selectedPreset) {
        const nextPresets = presets.map((preset) =>
          preset.name === selectedPreset.name ? { ...preset, name: nextName } : preset
        )
        const nextActivePreset = activePresetName === selectedPreset.name ? nextName : activePresetName
        await handlePresetMutation(nextPresets, nextActivePreset)
      } else if (dialogMode === 'duplicate' && selectedPreset) {
        const nextPresets = [...presets, { ...selectedPreset, name: nextName }]
        await handlePresetMutation(nextPresets, activePresetName)
      }
      setDialogMode(null)
      setPresetNameDraft('')
      setMessage({ type: 'success', text: 'Preset updated.' })
    } catch (updateError) {
      setMessage({ type: 'error', text: messageWithDetail('Failed to update preset', updateError) })
    } finally {
      setSubmitting(false)
    }
  }

  const handleDeletePreset = async () => {
    if (!selectedPreset || presets.length <= 1) {
      return
    }
    setSubmitting(true)
    setMessage(null)
    try {
      const nextPresets = presets.filter((preset) => preset.name !== selectedPreset.name)
      const nextActivePreset = activePresetName === selectedPreset.name
        ? (nextPresets[0]?.name ?? 'default')
        : activePresetName
      await handlePresetMutation(nextPresets, nextActivePreset)
      setMessage({ type: 'success', text: 'Preset deleted.' })
    } catch (updateError) {
      setMessage({ type: 'error', text: messageWithDetail('Failed to delete preset', updateError) })
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <section className="rounded-lg border border-border bg-surface-layer">
      <div className="border-b border-border px-4 py-3">
        <div className="flex items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold text-foreground">Inference Parameters</h3>
            <p className="mt-0.5 text-xs text-muted-foreground">Manage global defaults and reusable presets.</p>
          </div>
        </div>
      </div>

      <form onSubmit={handleSaveSettings} className="space-y-4 px-4 py-4">
        <div className="flex flex-wrap items-end gap-2">
          <label className="min-w-[220px] flex-1 space-y-1.5">
            <SettingLabel>Preset</SettingLabel>
            <SelectSetting
              value={selectedPresetName}
              onValueChange={(value) => {
                setSelectedPresetName(value)
                handleApplyPreset(presets.find((preset) => preset.name === value) ?? null)
              }}
              options={presets.map((preset) => preset.name)}
              getOptionLabel={(value) => value === activePresetName ? `${value} (active)` : value}
            />
          </label>
          <Button type="button" variant="secondary" size="sm" onClick={() => { setDialogMode('save'); setPresetNameDraft(selectedPresetName) }}>
            Save As
          </Button>
          <Button type="button" variant="secondary" size="sm" disabled={!selectedPreset} onClick={() => { setDialogMode('rename'); setPresetNameDraft(selectedPreset?.name ?? '') }}>
            Rename
          </Button>
          <Button type="button" variant="secondary" size="sm" disabled={!selectedPreset} onClick={() => { setDialogMode('duplicate'); setPresetNameDraft(`${selectedPreset?.name ?? 'preset'}-copy`) }}>
            Duplicate
          </Button>
          <Button type="button" variant="ghost" size="sm" disabled={presets.length <= 1} onClick={() => void handleDeletePreset()}>
            Delete
          </Button>
        </div>

        <InferenceControls
          value={params}
          onChange={setParam}
          supportsReasoningEffort
          reasoningEffortOptions={['none', 'minimal', 'low', 'medium', 'high', 'xhigh']}
        />

        <SettingMessage message={message} />

        <div className="flex justify-end">
          <Button type="submit" variant="primary" size="sm" loading={submitting}>
            <Save className="h-3.5 w-3.5" />
            Save Inference
          </Button>
        </div>
      </form>

      <Dialog open={dialogMode !== null} onOpenChange={(open) => { if (!open) { setDialogMode(null); setPresetNameDraft('') } }}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              {dialogMode === 'rename' ? 'Rename Preset' : dialogMode === 'duplicate' ? 'Duplicate Preset' : 'Save Preset'}
            </DialogTitle>
            <DialogDescription>Choose a preset name.</DialogDescription>
          </DialogHeader>
          <Input value={presetNameDraft} onChange={(event) => setPresetNameDraft(event.target.value)} />
          <DialogFooter>
            <Button type="button" variant="ghost" onClick={() => { setDialogMode(null); setPresetNameDraft('') }}>
              {t('common.cancel')}
            </Button>
            <Button type="button" variant="primary" loading={submitting} onClick={() => void handleDialogConfirm()}>
              Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </section>
  )
}

function isProviderChoice(value: unknown): value is ProviderChoice {
  return (
    value === 'ollama' ||
    value === 'openai_compat' ||
    value === 'openai_codex' ||
    value === 'gemini' ||
    value === 'anthropic' ||
    value === 'local'
  )
}

function providerOption(provider: ProviderChoice) {
  return providerOptions.find((item) => item.value === provider) ?? providerOptions[0]
}

function providerDescription(provider: ProviderChoice, t: Translator): string {
  if (provider === 'vllm') {
    return 'Dedicated vLLM endpoint with explicit served model selection.'
  }
  const keys: Record<ProviderChoice, string> = {
    ollama: 'settings.provider.ollama.description',
    openai_compat: 'settings.provider.openaiCompat.description',
    openai_codex: 'settings.provider.openaiCodex.description',
    gemini: 'settings.provider.gemini.description',
    anthropic: 'settings.provider.anthropic.description',
    vllm: 'settings.provider.openaiCompat.description',
    local: 'settings.provider.local.description',
  }
  return t(keys[provider])
}

function providerNote(provider: ProviderChoice, t: Translator): string {
  if (provider === 'vllm') {
    return 'Use the exact served model id returned by the remote /v1/models endpoint.'
  }
  const keys: Record<ProviderChoice, string> = {
    ollama: 'settings.provider.ollama.note',
    openai_compat: 'settings.provider.openaiCompat.note',
    openai_codex: 'settings.provider.openaiCodex.note',
    gemini: 'settings.provider.gemini.note',
    anthropic: 'settings.provider.anthropic.note',
    vllm: 'settings.provider.openaiCompat.note',
    local: 'settings.provider.local.note',
  }
  return t(keys[provider])
}

function openAICodexStatusVariant(status: string | null | undefined, configured: boolean | null | undefined): 'success' | 'warning' | 'error' | 'neutral' {
  if (!configured) {
    return 'neutral'
  }
  if (status === 'refresh_failed' || status === 'expired') {
    return 'error'
  }
  if (status === 'expiring') {
    return 'warning'
  }
  return 'success'
}

function openAICodexStatusLabel(status: string | null | undefined, configured: boolean | null | undefined): string {
  if (!configured) {
    return 'Not connected'
  }
  if (status === 'refresh_failed') {
    return 'Refresh failed'
  }
  if (status === 'expired') {
    return 'Expired'
  }
  if (status === 'expiring') {
    return 'Expiring soon'
  }
  return 'Connected'
}

function openAICodexCliAuthVariant(state: string | null | undefined): 'success' | 'warning' | 'error' | 'neutral' {
  if (state === 'ready') {
    return 'success'
  }
  if (state === 'apikey' || state === 'unsupported_auth_mode' || state === 'missing_tokens') {
    return 'warning'
  }
  if (state === 'invalid_json' || state === 'invalid_payload') {
    return 'error'
  }
  return 'neutral'
}

function openAICodexCliAuthLabel(state: string | null | undefined): string {
  if (state === 'ready') {
    return 'Ready to import'
  }
  if (state === 'apikey') {
    return 'API key mode'
  }
  if (state === 'unsupported_auth_mode') {
    return 'Unsupported auth mode'
  }
  if (state === 'missing_tokens') {
    return 'Missing OAuth tokens'
  }
  if (state === 'invalid_json') {
    return 'Invalid JSON'
  }
  if (state === 'invalid_payload') {
    return 'Invalid auth payload'
  }
  return 'No CLI login found'
}

function configuredProvider(modelConfig: Record<string, unknown>, configuredModel: string | null): ProviderChoice {
  const provider = modelConfig.provider
  const ollamaModel = getStringSetting(modelConfig, 'ollama_model').trim()
  if (ollamaModel.length > 0) {
    return 'ollama'
  }
  if (configuredModel && looksLikeLocalModelPath(configuredModel)) {
    return 'local'
  }
  if (configuredModel?.startsWith('http://') || configuredModel?.startsWith('https://')) {
    const codexBaseUrl = getStringSetting(modelConfig, 'openai_codex_base_url').trim()
    if (codexBaseUrl && configuredModel === codexBaseUrl) {
      return 'openai_codex'
    }
    const remoteProvider = modelConfig.openai_compat_provider
    if (isProviderChoice(remoteProvider) && remoteProvider !== 'ollama') {
      return remoteProvider
    }
    if (isProviderChoice(provider) && provider !== 'ollama' && provider !== 'local') {
      return provider
    }
    const inferred = inferProviderChoice(getStringSetting(modelConfig, 'openai_compat_model'))
    if (inferred && inferred !== 'ollama' && inferred !== 'local') {
      return inferred
    }
    return 'openai_compat'
  }
  if (isProviderChoice(provider)) {
    return provider
  }
  return 'ollama'
}

function configuredBaseUrl(
  provider: ProviderChoice,
  modelConfig: Record<string, unknown>
): string {
  if (provider === 'local') {
    return getStringSetting(modelConfig, 'local_model_root')
  }
  if (provider === 'ollama') {
    return getStringSetting(modelConfig, 'ollama_base_url', providerOption(provider).defaultBaseUrl)
  }
  if (provider === 'openai_codex') {
    return getStringSetting(modelConfig, 'openai_codex_base_url', providerOption(provider).defaultBaseUrl)
  }
  return getStringSetting(modelConfig, 'openai_compat_base_url', providerOption(provider).defaultBaseUrl)
}

function configuredModelName(
  provider: ProviderChoice,
  configuredModel: string | null,
  modelConfig: Record<string, unknown>
): string {
  if (provider === 'local') {
    return getStringSetting(modelConfig, 'local_model_path') || configuredModel || ''
  }
  if (provider === 'ollama') {
    return (
      getStringSetting(modelConfig, 'ollama_model') ||
      configuredModel?.replace(/^ollama:/, '') ||
      providerOption(provider).defaultModel
    )
  }
  if (provider === 'openai_codex') {
    return getStringSetting(modelConfig, 'openai_codex_model', providerOption(provider).defaultModel)
  }
  return getStringSetting(modelConfig, 'openai_compat_model', providerOption(provider).defaultModel)
}

function SettingLabel({ children }: { children: React.ReactNode }) {
  return <span className="text-xs font-medium text-muted-foreground">{children}</span>
}

function SettingMessage({ message }: { message: FormMessage }) {
  if (!message) {
    return null
  }

  return (
    <div
      className={[
        'rounded-md border px-3 py-2 text-xs',
        message.type === 'success'
          ? 'border-success/30 bg-success/10 text-success'
          : 'border-destructive/30 bg-destructive/10 text-destructive',
      ].join(' ')}
    >
      {message.text}
    </div>
  )
}

function SelectSetting({
  value,
  onValueChange,
  options,
  getOptionLabel,
  className,
}: {
  value: string
  onValueChange: (value: string) => void
  options: string[]
  getOptionLabel?: (value: string) => string
  className?: string
}) {
  return (
    <Select value={value} onValueChange={onValueChange}>
      <SelectTrigger className={className}>
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {options.map((option) => (
          <SelectItem key={option} value={option}>
            {getOptionLabel ? getOptionLabel(option) : option}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

function isGgufModelPath(value: string): boolean {
  return value.trim().toLowerCase().endsWith('.gguf')
}

function LocalRuntimeCard({
  runtimeStatus,
  runtimeLoading,
  runtimeMessage,
  runtimePath,
  runtimeBusy,
  onRuntimePathChange,
  onPrepareManagedRuntime,
  onRegisterExistingRuntime,
  hardware,
}: {
  runtimeStatus: api.LocalModelRuntimeStatus | null
  runtimeLoading: boolean
  runtimeMessage: FormMessage
  runtimePath: string
  runtimeBusy: boolean
  onRuntimePathChange: (value: string) => void
  onPrepareManagedRuntime: () => void
  onRegisterExistingRuntime: () => void
  hardware: api.LocalModelHardwareSummary | null
}) {
  const { t } = useI18n()
  const readiness = runtimeStatus?.readiness ?? 'unknown'
  const runtimeReady = readiness === 'ready'
  const runtimeAvailableActions = new Set(runtimeStatus?.actions ?? [])

  return (
    <section className="rounded-md border border-border bg-surface-layer px-3 py-3">
      <div className="rounded-md border border-border bg-canvas px-2.5 py-2 text-xs">
        <div className="flex items-start justify-between gap-3">
          <div>
            <p className="font-medium text-foreground">{t('settings.runtime.localRuntimeTitle')}</p>
            <p className="mt-0.5 text-muted-foreground">{t('settings.runtime.localRuntimeDescription')}</p>
          </div>
          <Badge variant={runtimeReady ? 'success' : readiness === 'degraded' ? 'warning' : 'neutral'}>
            {t(`settings.runtime.localRuntimeState.${readiness}`)}
          </Badge>
        </div>

        {runtimeLoading ? (
          <p className="mt-2 text-muted-foreground">{t('settings.runtime.localRuntimeLoading')}</p>
        ) : null}

        {runtimeStatus ? (
          <div className="mt-2 grid grid-cols-1 gap-1 text-muted-foreground sm:grid-cols-2">
            <p>
              {t('settings.runtime.localRuntimeSource')}:{' '}
              <span className="font-medium text-foreground">{runtimeStatus.source || t('common.notReported')}</span>
            </p>
            <p>
              {t('settings.runtime.localRuntimeVersion')}:{' '}
              <span className="font-medium text-foreground">{runtimeStatus.version ?? t('common.notSet')}</span>
            </p>
            <p>
              Runtime variant:{' '}
              <span className="font-medium text-foreground">{runtimeStatus.platform ?? t('common.notSet')}</span>
            </p>
            <p className="sm:col-span-2">
              Release asset:{' '}
              <span className="font-mono text-foreground">{runtimeStatus.binaryAsset ?? t('common.notSet')}</span>
            </p>
            <p className="sm:col-span-2">
              {t('settings.runtime.localRuntimeRoot')}:{' '}
              <span className="font-mono text-foreground">{runtimeStatus.rootDir ?? runtimeStatus.installDir ?? t('common.notSet')}</span>
            </p>
            <p className="sm:col-span-2">
              {t('settings.runtime.localRuntimeMissing')}:{' '}
              <span className="font-medium text-foreground">
                {runtimeStatus.missingComponents.length > 0
                  ? runtimeStatus.missingComponents.join(', ')
                  : t('common.none')}
              </span>
            </p>
            <p>
              GPU vendor:{' '}
              <span className="font-medium text-foreground">{hardware?.gpuVendor ?? t('common.notReported')}</span>
            </p>
            <p>
              Recommended llama.cpp backend:{' '}
              <span className="font-medium text-foreground">{hardware?.recommendedRuntimeLabel ?? t('common.notReported')}</span>
            </p>
          </div>
        ) : null}

        {runtimeStatus?.warnings.length ? (
          <div className="mt-2 text-warning">
            <p className="font-medium">{t('settings.quantization.warningsLabel')}</p>
            <ul className="mt-1 list-disc space-y-0.5 pl-4">
              {runtimeStatus.warnings.map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          </div>
        ) : null}

        {!runtimeReady ? (
          <div className="mt-2 space-y-2">
            <div className="flex flex-wrap items-center gap-2">
              <Button
                type="button"
                size="sm"
                variant="secondary"
                loading={runtimeBusy}
                disabled={!runtimeAvailableActions.has('prepare_managed_runtime')}
                onClick={onPrepareManagedRuntime}
              >
                {runtimeInstallLabel(t, hardware)}
              </Button>
            </div>

            <label className="block space-y-1.5">
              <span className="text-xs font-medium text-muted-foreground">
                {t('settings.runtime.localRuntimeExistingPath')}
              </span>
              <div className="flex min-w-0 gap-2">
                <Input
                  value={runtimePath}
                  onChange={(event) => onRuntimePathChange(event.target.value)}
                  placeholder={t('settings.runtime.localRuntimeExistingPathPlaceholder')}
                  className="min-w-0 font-mono text-xs"
                />
                <Button
                  type="button"
                  size="sm"
                  variant="secondary"
                  loading={runtimeBusy}
                  disabled={!runtimeAvailableActions.has('register_existing_path')}
                  onClick={onRegisterExistingRuntime}
                >
                  {t('settings.runtime.localRuntimeRegisterAction')}
                </Button>
              </div>
            </label>
          </div>
        ) : null}

        <SettingMessage message={runtimeMessage} />
      </div>
    </section>
  )
}

function LocalQuantizationCapabilities({
  modelPath,
  showGgufHint,
  loading,
  error,
  capabilities,
  runtimeReady,
  selectedQuantization,
  onSelectQuantization,
  convertLoading,
  convertMessage,
  convertOutputPath,
  onConvertGguf,
}: {
  modelPath: string
  showGgufHint: boolean
  loading: boolean
  error: string | null
  capabilities: api.LocalModelCapabilitiesResult | null
  runtimeReady: boolean
  selectedQuantization: string
  onSelectQuantization: (value: string) => void
  convertLoading: boolean
  convertMessage: FormMessage
  convertOutputPath: string | null
  onConvertGguf: () => void
}) {
  const { t } = useI18n()
  const ggufFormat = capabilities?.formats.find((format) => format.formatId.toLowerCase() === 'gguf') ?? null
  const runtimeHardware = capabilities?.hardware ?? null

  return (
    <section className="rounded-md border border-border bg-surface-layer px-3 py-3">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-xs font-semibold text-foreground">{t('settings.quantization.title')}</p>
          <p className="mt-0.5 text-xs text-muted-foreground">{t('settings.quantization.description')}</p>
        </div>
        <BrainCircuit className="mt-0.5 h-4 w-4 text-muted-foreground" />
      </div>

      <p className="mt-2 rounded-md border border-border bg-canvas px-2.5 py-2 font-mono text-[11px] text-muted-foreground">
        {modelPath}
      </p>

      {showGgufHint ? (
        <div className="mt-2 rounded-md border border-warning/30 bg-warning/10 px-2.5 py-2 text-xs text-warning">
          <p className="font-medium">{t('settings.quantization.ggufHintTitle')}</p>
          <p className="mt-1">{t('settings.quantization.ggufHintBody')}</p>
          <p className="mt-1 text-muted-foreground">{t('settings.quantization.ggufHintDetail')}</p>
        </div>
      ) : null}

      {loading ? (
        <p className="mt-2 text-xs text-muted-foreground">{t('settings.quantization.loading')}</p>
      ) : null}

      {error ? (
        <div className="mt-2 rounded-md border border-destructive/30 bg-destructive/10 px-2.5 py-2 text-xs text-destructive">
          {error}
        </div>
      ) : null}

      {!loading && !error && capabilities ? (
        <>
          {capabilities.warnings.length > 0 ? (
            <div className="mt-2 rounded-md border border-warning/30 bg-warning/10 px-2.5 py-2 text-xs text-warning">
              <p className="font-medium">{t('settings.quantization.warningsLabel')}</p>
              <ul className="mt-1 list-disc space-y-0.5 pl-4">
                {capabilities.warnings.map((warning) => (
                  <li key={warning}>{warning}</li>
                ))}
              </ul>
            </div>
          ) : null}

          {ggufFormat?.supported ? (
            <div className="mt-2 rounded-md border border-border bg-canvas px-2.5 py-2 text-xs">
              <p className="font-medium text-foreground">{t('settings.quantization.optionsLabel')}</p>
              {ggufFormat.quantizationOptions.length > 0 ? (
                <SelectSetting
                  value={selectedQuantization}
                  onValueChange={onSelectQuantization}
                  options={ggufFormat.quantizationOptions.map((option) => option.id)}
                  getOptionLabel={(value) => {
                    const option = ggufFormat.quantizationOptions.find((item) => item.id === value)
                    if (!option) {
                      return value
                    }
                    return option.bits ? `${option.name} (${option.bits})` : option.name
                  }}
                  className="mt-1 font-mono text-xs"
                />
              ) : (
                <p className="mt-1 text-muted-foreground">{t('common.none')}</p>
              )}
              <p className="mt-1 text-muted-foreground">
                {t('settings.quantization.suggestedDefaultLabel')}:{' '}
                <span className="font-medium text-foreground">
                  {ggufFormat.suggestedDefaultQuantization ?? t('common.notSet')}
                </span>
              </p>
              <div className="mt-2 flex items-center gap-2">
                <Button
                  type="button"
                  size="sm"
                  variant="secondary"
                  loading={convertLoading}
                  disabled={ggufFormat.quantizationOptions.length === 0 || !selectedQuantization || !runtimeReady}
                  onClick={onConvertGguf}
                >
                  {t('settings.quantization.convertAction')}
                </Button>
                {!runtimeReady ? (
                  <span className="text-muted-foreground">{t('settings.runtime.localRuntimeBlocked')}</span>
                ) : null}
              </div>

              {convertOutputPath ? (
                <p className="mt-2 rounded-md border border-success/30 bg-success/10 px-2 py-1.5 font-mono text-[11px] text-success">
                  {t('settings.quantization.outputPathLabel')}: {convertOutputPath}
                </p>
              ) : null}

              <SettingMessage message={convertMessage} />
            </div>
          ) : null}

          {runtimeHardware ? (
            <div className="mt-2 rounded-md border border-border bg-canvas px-2.5 py-2 text-xs">
              <p className="font-medium text-foreground">{t('settings.quantization.hardwareTitle')}</p>
              <div className="mt-1 grid grid-cols-1 gap-1 text-muted-foreground sm:grid-cols-2">
                <p>
                  {t('settings.quantization.hardwareProvider')}:{' '}
                  <span className="font-medium text-foreground">{runtimeHardware.provider ?? t('common.notSet')}</span>
                </p>
                <p>
                  GPU vendor:{' '}
                  <span className="font-medium text-foreground">{runtimeHardware.gpuVendor ?? t('common.notReported')}</span>
                </p>
                <p>
                  {t('settings.quantization.hardwareCuda')}:{' '}
                  <span className="font-medium text-foreground">
                    {runtimeHardware.cudaAvailable === true
                      ? t('common.yes')
                      : runtimeHardware.cudaAvailable === false
                        ? t('common.no')
                        : t('common.notReported')}
                  </span>
                </p>
                <p>
                  {t('settings.quantization.hardwareGpuCount')}:{' '}
                  <span className="font-medium text-foreground">
                    {runtimeHardware.gpuCount ?? t('common.notReported')}
                  </span>
                </p>
                <p>
                  {t('settings.quantization.hardwareVram')}:{' '}
                  <span className="font-medium text-foreground">
                    {typeof runtimeHardware.totalVramGb === 'number'
                      ? `${runtimeHardware.totalVramGb} GB`
                      : t('common.notReported')}
                  </span>
                </p>
                <p className="sm:col-span-2">
                  {t('settings.quantization.hardwarePrimaryGpu')}:{' '}
                  <span className="font-medium text-foreground">
                    {runtimeHardware.primaryGpuName ?? t('common.notReported')}
                  </span>
                </p>
                <p className="sm:col-span-2">
                  Recommended llama.cpp backend:{' '}
                  <span className="font-medium text-foreground">
                    {runtimeHardware.recommendedRuntimeLabel ?? t('common.notReported')}
                  </span>
                </p>
              </div>
              {runtimeHardware.warnings.length > 0 ? (
                <div className="mt-2 text-warning">
                  <p className="font-medium">{t('settings.quantization.hardwareWarningsLabel')}</p>
                  <ul className="mt-1 list-disc space-y-0.5 pl-4">
                    {runtimeHardware.warnings.map((warning) => (
                      <li key={warning}>{warning}</li>
                    ))}
                  </ul>
                </div>
              ) : null}
            </div>
          ) : null}
        </>
      ) : null}
    </section>
  )
}

function ModelConnectionForm({
  settings,
  configuredModel,
  modelConfig,
  models,
  setModels,
  setSettings,
  onConfigured,
}: {
  settings: api.Settings | null
  configuredModel: string | null
  modelConfig: Record<string, unknown>
  models: api.ModelInfo[]
  setModels: React.Dispatch<React.SetStateAction<api.ModelInfo[]>>
  setSettings: React.Dispatch<React.SetStateAction<api.Settings | null>>
  onConfigured: (result: api.ConfigureModelResult) => void
}) {
  const { t } = useI18n()
  const initialProvider = configuredProvider(modelConfig, configuredModel)
  const [provider, setProvider] = React.useState<ProviderChoice>(initialProvider)
  const currentProvider = providerOption(provider)
  const activeModelInfo = React.useMemo(
    () => activeConfiguredModel(settings, models),
    [settings, models]
  )
  const activeToolCallingInfo = React.useMemo(
    () => modelToolCallingMetadata(activeModelInfo),
    [activeModelInfo]
  )
  const activeReasoningInfo = React.useMemo(
    () => modelReasoningMetadata(activeModelInfo),
    [activeModelInfo]
  )
  const contextLengthTarget = React.useMemo(
    () => resolveContextLengthSettingsTarget(settings),
    [settings]
  )
  const [baseUrl, setBaseUrl] = React.useState(configuredBaseUrl(initialProvider, modelConfig))
  const [model, setModel] = React.useState(configuredModelName(initialProvider, configuredModel, modelConfig))
  const [apiKey, setApiKey] = React.useState('')
  const [contextLengthInput, setContextLengthInput] = React.useState(
    contextLengthTarget.value === null ? '' : String(contextLengthTarget.value)
  )
  const [contextSettingsBusy, setContextSettingsBusy] = React.useState(false)
  const [contextSettingsMessage, setContextSettingsMessage] = React.useState<FormMessage>(null)
  const [ollamaModels, setOllamaModels] = React.useState<string[]>([])
  const [localModels, setLocalModels] = React.useState<api.ModelInfo[]>([])
  const [discovering, setDiscovering] = React.useState(false)
  const [discoverMessage, setDiscoverMessage] = React.useState<FormMessage>(null)
  const [capabilitiesLoading, setCapabilitiesLoading] = React.useState(false)
  const [capabilitiesError, setCapabilitiesError] = React.useState<string | null>(null)
  const [localCapabilities, setLocalCapabilities] = React.useState<api.LocalModelCapabilitiesResult | null>(null)
  const [localRuntimeLoading, setLocalRuntimeLoading] = React.useState(false)
  const [localRuntimeStatus, setLocalRuntimeStatus] = React.useState<api.LocalModelRuntimeStatus | null>(null)
  const [localRuntimeMessage, setLocalRuntimeMessage] = React.useState<FormMessage>(null)
  const [localRuntimeBusy, setLocalRuntimeBusy] = React.useState(false)
  const [localRuntimePath, setLocalRuntimePath] = React.useState('')
  const [localIdleUnloadEnabled, setLocalIdleUnloadEnabled] = React.useState(
    settings?.local_models?.idle_unload_enabled ?? false
  )
  const [localIdleUnloadSeconds, setLocalIdleUnloadSeconds] = React.useState(
    String(settings?.local_models?.idle_unload_seconds ?? 300)
  )
  const [localMountBusy, setLocalMountBusy] = React.useState(false)
  const [localMountMessage, setLocalMountMessage] = React.useState<FormMessage>(null)
  const [activeLocalRuntimeStatus, setActiveLocalRuntimeStatus] = React.useState<api.LocalActiveModelRuntimeStatus | null>(null)
  const [selectedGgufQuantization, setSelectedGgufQuantization] = React.useState('')
  const [convertingLocalModel, setConvertingLocalModel] = React.useState(false)
  const [convertMessage, setConvertMessage] = React.useState<FormMessage>(null)
  const [convertedOutputPath, setConvertedOutputPath] = React.useState<string | null>(null)
  const [openAICodexStatus, setOpenAICodexStatus] = React.useState<api.OpenAICodexAuthStatus | null>(null)
  const [openAICodexAuthBusy, setOpenAICodexAuthBusy] = React.useState(false)
  const [openAICodexAuthMessage, setOpenAICodexAuthMessage] = React.useState<FormMessage>(null)
  const [openAICodexLoginStart, setOpenAICodexLoginStart] = React.useState<api.OpenAICodexLoginStartResult | null>(null)
  const [openAICodexManualCallbackUrl, setOpenAICodexManualCallbackUrl] = React.useState('')
  const [submitting, setSubmitting] = React.useState(false)
  const [message, setMessage] = React.useState<{ type: 'success' | 'error'; text: string } | null>(null)
  const [editingModelId, setEditingModelId] = React.useState<string | null>(null)
  const [editingProvider, setEditingProvider] = React.useState<ProviderChoice>('ollama')
  const [editingModelName, setEditingModelName] = React.useState('')
  const [editingModelSpec, setEditingModelSpec] = React.useState('')
  const [editingBaseUrl, setEditingBaseUrl] = React.useState('')
  const [editingApiKey, setEditingApiKey] = React.useState('')
  const [entrySubmitting, setEntrySubmitting] = React.useState(false)
  const [entryMessage, setEntryMessage] = React.useState<FormMessage>(null)
  const [toolProbeBusy, setToolProbeBusy] = React.useState(false)
  const [toolProbeMessage, setToolProbeMessage] = React.useState<FormMessage>(null)
  const [toolProbeResult, setToolProbeResult] = React.useState<Record<string, unknown> | null>(null)
  const savedModels = React.useMemo(() => configuredModelsFromSettings(settings), [settings])
  const discoveryKeyRef = React.useRef(`${initialProvider}:${baseUrl}`)
  const openAICodexPopupRef = React.useRef<Window | null>(null)
  const localModelOptions = React.useMemo(
    () => localModels.map((entry) => modelInfoId(entry)).filter((id) => id.length > 0),
    [localModels]
  )
  const localModelLabelById = React.useMemo(() => {
    const labels = new Map<string, string>()
    for (const entry of localModels) {
      const id = modelInfoId(entry)
      if (id) {
        const backendSuffix = entry.backendType ? ` (${entry.backendType})` : ''
        labels.set(id, `${modelInfoLabel(entry)}${backendSuffix}`)
      }
    }
    return labels
  }, [localModels])
  const selectedLocalModel = React.useMemo(
    () => localModels.find((entry) => modelInfoId(entry) === model) ?? null,
    [localModels, model]
  )
  const normalizedModelPath = model.trim()
  const localModelInputLooksLikeDirectory = normalizedModelPath.length > 0 && !isGgufModelPath(normalizedModelPath)
  const localModelInputLooksLikeGguf = isGgufModelPath(normalizedModelPath)
  const selectedLocalModelIsHfDir = selectedLocalModel?.backendType === 'safetensors'
  const selectedLocalModelIsGguf = selectedLocalModel?.backendType === 'gguf'
  const shouldShowGgufHint = provider === 'local' && localModelInputLooksLikeDirectory && !selectedLocalModelIsGguf
  const shouldShowQuantization = (
    provider === 'local' &&
    localModelInputLooksLikeDirectory &&
    (selectedLocalModelIsHfDir || !selectedLocalModelIsGguf)
  )

  React.useEffect(() => {
    const nextProvider = configuredProvider(modelConfig, configuredModel)
    const nextBaseUrl = configuredBaseUrl(nextProvider, modelConfig)
    const nextDiscoveryKey = `${nextProvider}:${nextBaseUrl}`
    setProvider(nextProvider)
    setBaseUrl(nextBaseUrl)
    setModel(configuredModelName(nextProvider, configuredModel, modelConfig))
    setLocalIdleUnloadEnabled(settings?.local_models?.idle_unload_enabled ?? false)
    setLocalIdleUnloadSeconds(String(settings?.local_models?.idle_unload_seconds ?? 300))
    if (discoveryKeyRef.current !== nextDiscoveryKey) {
      discoveryKeyRef.current = nextDiscoveryKey
      setOllamaModels([])
      setLocalModels([])
      setDiscoverMessage(null)
      setCapabilitiesLoading(false)
      setCapabilitiesError(null)
      setLocalCapabilities(null)
      setLocalRuntimeLoading(false)
      setLocalRuntimeStatus(null)
      setLocalRuntimeMessage(null)
      setLocalRuntimeBusy(false)
      setLocalRuntimePath('')
      setLocalMountBusy(false)
      setLocalMountMessage(null)
      setActiveLocalRuntimeStatus(null)
      setSelectedGgufQuantization('')
      setConvertingLocalModel(false)
      setConvertMessage(null)
      setConvertedOutputPath(null)
    }
  }, [configuredModel, modelConfig, settings])

  React.useEffect(() => {
    setContextLengthInput(contextLengthTarget.value === null ? '' : String(contextLengthTarget.value))
    setContextSettingsMessage(null)
  }, [contextLengthTarget.kind, contextLengthTarget.value])

  const handleProviderChange = (nextProvider: ProviderChoice) => {
    const next = providerOption(nextProvider)
    setProvider(nextProvider)
    setBaseUrl(next.defaultBaseUrl)
    setModel(next.defaultModel)
    setApiKey('')
    setOllamaModels([])
    setLocalModels([])
    setDiscoverMessage(null)
    setCapabilitiesLoading(false)
    setCapabilitiesError(null)
    setLocalCapabilities(null)
    setLocalRuntimeLoading(false)
    setLocalRuntimeStatus(null)
    setLocalRuntimeMessage(null)
    setLocalRuntimeBusy(false)
    setLocalRuntimePath('')
    setSelectedGgufQuantization('')
    setConvertingLocalModel(false)
    setConvertMessage(null)
    setConvertedOutputPath(null)
    setOpenAICodexAuthMessage(null)
    setMessage(null)
  }

  const discoverOllamaModels = React.useCallback(async () => {
    const normalizedBaseUrl = baseUrl.trim()
    if (provider !== 'ollama' || !normalizedBaseUrl) {
      return
    }

    setDiscovering(true)
    setDiscoverMessage(null)
    try {
      if (typeof settingsApi.fetchOllamaModels !== 'function') {
        throw new Error('Ollama model discovery API client is unavailable.')
      }
      const result = await settingsApi.fetchOllamaModels(normalizedBaseUrl)
      setOllamaModels(result.models)
      if (result.models.length > 0 && !result.models.includes(model)) {
        setModel(result.models[0])
      }
      setDiscoverMessage({
        type: 'success',
        text: result.models.length > 0
          ? t('settings.modelConnection.successDiscovered', { count: result.models.length })
          : t('settings.modelConnection.successNoModels'),
      })
    } catch (discoverError) {
      setOllamaModels([])
      setDiscoverMessage({
        type: 'error',
        text: messageWithDetail(t('settings.modelConnection.errorDiscover'), discoverError),
      })
    } finally {
      setDiscovering(false)
    }
  }, [baseUrl, model, provider, t])

  const discoverLocalModels = React.useCallback(async () => {
    const root = baseUrl.trim()
    if (provider !== 'local' || !root) {
      return
    }
    if (isGgufModelPath(root)) {
      setLocalModels([])
      setDiscoverMessage(null)
      return
    }

    setDiscovering(true)
    setDiscoverMessage(null)
    try {
      if (typeof settingsApi.fetchLocalModels !== 'function') {
        throw new Error('Local model discovery API client is unavailable.')
      }
      const result = await settingsApi.fetchLocalModels(root)
      setLocalModels(result.models)
      const firstModel = result.models[0]
      const firstModelId = firstModel ? modelInfoId(firstModel) : ''
      if (firstModelId) {
        setModel((currentModel) => (
          result.models.some((entry) => modelInfoId(entry) === currentModel) ? currentModel : firstModelId
        ))
      }
      const warningText = result.warnings.length > 0 ? ` ${result.warnings.join(' ')}` : ''
      setDiscoverMessage({
        type: 'success',
        text: result.models.length > 0
          ? `${t('settings.modelConnection.successLocalDiscovered', { count: result.models.length })}${warningText}`
          : `${t('settings.modelConnection.successNoLocalModels')}${warningText}`,
      })
    } catch (discoverError) {
      setLocalModels([])
      setDiscoverMessage({
        type: 'error',
        text: messageWithDetail(t('settings.modelConnection.errorDiscoverLocal'), discoverError),
      })
    } finally {
      setDiscovering(false)
    }
  }, [baseUrl, provider, t])

  React.useEffect(() => {
    if (provider !== 'openai_codex') {
      openAICodexPopupRef.current?.close()
      openAICodexPopupRef.current = null
      setOpenAICodexAuthBusy(false)
      setOpenAICodexAuthMessage(null)
      setOpenAICodexLoginStart(null)
      setOpenAICodexManualCallbackUrl('')
      return
    }

    const timer = window.setTimeout(() => {
      const loadStatus = async () => {
        setOpenAICodexAuthBusy(true)
        setOpenAICodexAuthMessage(null)
        try {
          if (typeof settingsApi.fetchOpenAICodexAuthStatus !== 'function') {
            throw new Error('OpenAI Codex auth status API client is unavailable.')
          }
          const result = await settingsApi.fetchOpenAICodexAuthStatus()
          setOpenAICodexStatus(result)
        } catch (statusError) {
          setOpenAICodexStatus(null)
          setOpenAICodexAuthMessage({
            type: 'error',
            text: messageWithDetail('Failed to load OpenAI Codex auth status', statusError),
          })
        } finally {
          setOpenAICodexAuthBusy(false)
        }
      }

      void loadStatus()
    }, 150)

    return () => window.clearTimeout(timer)
  }, [provider])

  React.useEffect(() => {
    const handleMessage = (event: MessageEvent) => {
      if (!openAICodexLoginStart) {
        return
      }
      const expectedOrigin = (() => {
        try {
          return new URL(openAICodexLoginStart.callbackUrl).origin
        } catch {
          return null
        }
      })()
      if (!expectedOrigin || event.origin !== expectedOrigin) {
        return
      }
      if (openAICodexPopupRef.current && event.source !== openAICodexPopupRef.current) {
        return
      }
      const payload = asRecord(event.data)
      if (!payload || payload.type !== 'mochi-openai-codex-auth-callback') {
        return
      }

      const status = typeof payload.status === 'string' ? payload.status : 'error'
      const callbackMessage = typeof payload.message === 'string' && payload.message.trim()
        ? payload.message
        : 'OpenAI Codex login callback returned without a message.'

      openAICodexPopupRef.current?.close()
      openAICodexPopupRef.current = null

      if (status !== 'success') {
        setOpenAICodexAuthBusy(false)
        setOpenAICodexAuthMessage({
          type: 'error',
          text: callbackMessage,
        })
        return
      }

      const refreshStatus = async () => {
        setOpenAICodexAuthBusy(true)
        try {
          if (typeof settingsApi.fetchOpenAICodexAuthStatus !== 'function') {
            throw new Error('OpenAI Codex auth status API client is unavailable.')
          }
          const statusResult = await settingsApi.fetchOpenAICodexAuthStatus()
          setOpenAICodexStatus(statusResult)
          setOpenAICodexLoginStart(null)
          setOpenAICodexManualCallbackUrl('')
          setOpenAICodexAuthMessage({
            type: 'success',
            text: callbackMessage,
          })
        } catch (statusError) {
          setOpenAICodexAuthMessage({
            type: 'error',
            text: messageWithDetail('OpenAI Codex login completed but status refresh failed', statusError),
          })
        } finally {
          setOpenAICodexAuthBusy(false)
        }
      }

      void refreshStatus()
    }

    window.addEventListener('message', handleMessage)
    return () => window.removeEventListener('message', handleMessage)
  }, [openAICodexLoginStart])

  React.useEffect(() => {
    if (provider !== 'ollama') {
      return
    }
    const timer = window.setTimeout(() => {
      void discoverOllamaModels()
    }, 600)
    return () => window.clearTimeout(timer)
  }, [discoverOllamaModels, provider])

  React.useEffect(() => {
    if (provider !== 'local') {
      return
    }
    const timer = window.setTimeout(() => {
      void discoverLocalModels()
    }, 600)
    return () => window.clearTimeout(timer)
  }, [discoverLocalModels, provider])

  const shouldLoadLocalRuntimeStatus = provider === 'local'

  React.useEffect(() => {
    if (!shouldLoadLocalRuntimeStatus) {
      setLocalRuntimeLoading(false)
      setLocalRuntimeStatus(null)
      setLocalRuntimeMessage(null)
      setLocalRuntimeBusy(false)
      setLocalRuntimePath('')
      return
    }

    const timer = window.setTimeout(() => {
      const loadRuntime = async () => {
        setLocalRuntimeLoading(true)
        try {
          if (typeof settingsApi.fetchLocalModelRuntimeStatus !== 'function') {
            throw new Error('Local model runtime status API client is unavailable.')
          }
          const result = await settingsApi.fetchLocalModelRuntimeStatus()
          setLocalRuntimeStatus(result)
          setLocalRuntimePath((current) => current || result.rootDir || '')
        } catch (runtimeError) {
          setLocalRuntimeStatus(null)
          setLocalRuntimeMessage({
            type: 'error',
            text: messageWithDetail(t('settings.runtime.localRuntimeErrorLoad'), runtimeError),
          })
        } finally {
          setLocalRuntimeLoading(false)
        }
      }

      void loadRuntime()
    }, 200)

    return () => window.clearTimeout(timer)
  }, [shouldLoadLocalRuntimeStatus, t])

  React.useEffect(() => {
    if (provider !== 'local') {
      setActiveLocalRuntimeStatus(null)
      setLocalMountMessage(null)
      setLocalMountBusy(false)
      return
    }

    const timer = window.setTimeout(() => {
      const loadActiveRuntime = async () => {
        try {
          if (typeof settingsApi.fetchActiveLocalModelRuntimeStatus !== 'function') {
            throw new Error('Active local model runtime status API client is unavailable.')
          }
          const result = await settingsApi.fetchActiveLocalModelRuntimeStatus()
          setActiveLocalRuntimeStatus(result)
        } catch (runtimeError) {
          setActiveLocalRuntimeStatus(null)
          setLocalMountMessage({
            type: 'error',
            text: messageWithDetail('Failed to load active local model runtime status', runtimeError),
          })
        }
      }

      void loadActiveRuntime()
    }, 150)

    return () => window.clearTimeout(timer)
  }, [provider, configuredModel])

  React.useEffect(() => {
    if (!shouldShowQuantization) {
      setCapabilitiesLoading(false)
      setCapabilitiesError(null)
      setLocalCapabilities(null)
      setSelectedGgufQuantization('')
      setConvertingLocalModel(false)
      setConvertMessage(null)
      setConvertedOutputPath(null)
      return
    }

    const timer = window.setTimeout(() => {
      const loadCapabilities = async () => {
        setCapabilitiesLoading(true)
        setCapabilitiesError(null)
        try {
          if (typeof settingsApi.fetchLocalModelCapabilities !== 'function') {
            throw new Error('Local model capabilities API client is unavailable.')
          }
          const result = await settingsApi.fetchLocalModelCapabilities(normalizedModelPath)
          setLocalCapabilities(result)
        } catch (capabilityError) {
          setLocalCapabilities(null)
          setCapabilitiesError(messageWithDetail(t('settings.quantization.errorLoad'), capabilityError))
        } finally {
          setCapabilitiesLoading(false)
        }
      }

      void loadCapabilities()
    }, 350)

    return () => window.clearTimeout(timer)
  }, [normalizedModelPath, shouldShowQuantization, t])

  React.useEffect(() => {
    const ggufFormat = localCapabilities?.formats.find((format) => format.formatId.toLowerCase() === 'gguf')
    if (!ggufFormat?.supported) {
      setSelectedGgufQuantization('')
      return
    }
    const options = ggufFormat.quantizationOptions
    if (options.length === 0) {
      setSelectedGgufQuantization('')
      return
    }
    const suggested = ggufFormat.suggestedDefaultQuantization
    const fallback = options[0]?.id ?? ''
    const preferred = suggested && options.some((option) => option.id === suggested) ? suggested : fallback
    setSelectedGgufQuantization((current) => (
      current && options.some((option) => option.id === current) ? current : preferred
    ))
  }, [localCapabilities])

  const discoverCurrentProviderModels = () => {
    if (provider === 'ollama') {
      void discoverOllamaModels()
      return
    }
    if (provider === 'local') {
      void discoverLocalModels()
    }
  }

  const importOpenAICodexCliLogin = async () => {
    setOpenAICodexAuthBusy(true)
    setOpenAICodexAuthMessage(null)
    try {
      if (typeof settingsApi.importOpenAICodexCliLogin !== 'function' || typeof settingsApi.fetchOpenAICodexAuthStatus !== 'function') {
        throw new Error('OpenAI Codex auth API client is unavailable.')
      }
      await settingsApi.importOpenAICodexCliLogin()
      const status = await settingsApi.fetchOpenAICodexAuthStatus()
      setOpenAICodexStatus(status)
      setOpenAICodexAuthMessage({
        type: 'success',
        text: 'Imported Codex CLI login.',
      })
    } catch (importError) {
      setOpenAICodexAuthMessage({
        type: 'error',
        text: messageWithDetail('Failed to import Codex CLI login', importError),
      })
    } finally {
      setOpenAICodexAuthBusy(false)
    }
  }

  const connectOpenAICodexLogin = async () => {
    setOpenAICodexAuthBusy(true)
    setOpenAICodexAuthMessage(null)
    const popup = typeof window !== 'undefined'
      ? window.open('', 'mochi-openai-codex-login', 'popup=yes,width=560,height=760')
      : null

    try {
      if (
        typeof settingsApi.startOpenAICodexBrowserLogin !== 'function' ||
        typeof settingsApi.fetchOpenAICodexAuthStatus !== 'function'
      ) {
        throw new Error('OpenAI Codex browser login API client is unavailable.')
      }
      const login = await settingsApi.startOpenAICodexBrowserLogin(
        typeof window !== 'undefined' ? window.location.origin : undefined
      )
      setOpenAICodexLoginStart(login)
      setOpenAICodexManualCallbackUrl('')

      if (popup) {
        popup.location.href = login.authUrl
        openAICodexPopupRef.current = popup
      }

      setOpenAICodexAuthMessage({
        type: 'success',
        text: login.callbackReady
          ? (popup
            ? 'Browser login window opened. Complete sign-in there.'
            : 'Browser login link prepared. Open the sign-in URL below.')
          : 'Browser login started, but local callback binding is unavailable. After sign-in, paste the full callback URL below.',
      })
    } catch (loginError) {
      popup?.close()
      openAICodexPopupRef.current = null
      setOpenAICodexAuthMessage({
        type: 'error',
        text: messageWithDetail('Failed to start OpenAI Codex browser login', loginError),
      })
    } finally {
      setOpenAICodexAuthBusy(false)
    }
  }

  const completeOpenAICodexLogin = async () => {
    const callbackUrl = openAICodexManualCallbackUrl.trim()
    if (!callbackUrl) {
      setOpenAICodexAuthMessage({
        type: 'error',
        text: 'Paste the full callback URL before completing OpenAI Codex login.',
      })
      return
    }

    setOpenAICodexAuthBusy(true)
    setOpenAICodexAuthMessage(null)
    try {
      if (
        typeof settingsApi.completeOpenAICodexBrowserLogin !== 'function' ||
        typeof settingsApi.fetchOpenAICodexAuthStatus !== 'function'
      ) {
        throw new Error('OpenAI Codex browser login API client is unavailable.')
      }
      await settingsApi.completeOpenAICodexBrowserLogin({ callbackUrl })
      const status = await settingsApi.fetchOpenAICodexAuthStatus()
      setOpenAICodexStatus(status)
      setOpenAICodexLoginStart(null)
      setOpenAICodexManualCallbackUrl('')
      setOpenAICodexAuthMessage({
        type: 'success',
        text: 'OpenAI Codex browser login saved.',
      })
    } catch (completeError) {
      setOpenAICodexAuthMessage({
        type: 'error',
        text: messageWithDetail('Failed to complete OpenAI Codex browser login', completeError),
      })
    } finally {
      setOpenAICodexAuthBusy(false)
    }
  }

  const refreshOpenAICodexLogin = async () => {
    setOpenAICodexAuthBusy(true)
    setOpenAICodexAuthMessage(null)
    try {
      if (typeof settingsApi.refreshOpenAICodexAuth !== 'function' || typeof settingsApi.fetchOpenAICodexAuthStatus !== 'function') {
        throw new Error('OpenAI Codex auth API client is unavailable.')
      }
      await settingsApi.refreshOpenAICodexAuth()
      const status = await settingsApi.fetchOpenAICodexAuthStatus()
      setOpenAICodexStatus(status)
      setOpenAICodexAuthMessage({
        type: 'success',
        text: 'Refreshed Codex CLI login import.',
      })
    } catch (refreshError) {
      setOpenAICodexAuthMessage({
        type: 'error',
        text: messageWithDetail('Failed to refresh Codex CLI login import', refreshError),
      })
    } finally {
      setOpenAICodexAuthBusy(false)
    }
  }

  const logoutOpenAICodexLogin = async () => {
    setOpenAICodexAuthBusy(true)
    setOpenAICodexAuthMessage(null)
    try {
      if (typeof settingsApi.logoutOpenAICodexAuth !== 'function' || typeof settingsApi.fetchOpenAICodexAuthStatus !== 'function') {
        throw new Error('OpenAI Codex auth API client is unavailable.')
      }
      await settingsApi.logoutOpenAICodexAuth()
      const status = await settingsApi.fetchOpenAICodexAuthStatus()
      setOpenAICodexStatus(status)
      setOpenAICodexLoginStart(null)
      setOpenAICodexManualCallbackUrl('')
      setOpenAICodexAuthMessage({
        type: 'success',
        text: 'Removed stored OpenAI Codex login.',
      })
    } catch (logoutError) {
      setOpenAICodexAuthMessage({
        type: 'error',
        text: messageWithDetail('Failed to remove stored OpenAI Codex login', logoutError),
      })
    } finally {
      setOpenAICodexAuthBusy(false)
    }
  }

  const handlePrepareManagedRuntime = async () => {
    setLocalRuntimeBusy(true)
    setLocalRuntimeMessage(null)
    try {
      if (typeof settingsApi.installLocalModelRuntime !== 'function') {
        throw new Error('Local model runtime install API client is unavailable.')
      }
      const result = await settingsApi.installLocalModelRuntime({
        action: 'prepare_managed',
        persist: true,
      })
      setLocalRuntimeStatus(result.runtimeStatus)
      setLocalRuntimePath(result.runtimeStatus?.rootDir ?? result.installDir ?? '')
      setLocalRuntimeMessage({
        type: 'success',
        text: result.message ?? t('settings.runtime.localRuntimeInstallPrepared'),
      })
    } catch (runtimeError) {
      setLocalRuntimeMessage({
        type: 'error',
        text: messageWithDetail(t('settings.runtime.localRuntimeInstallError'), runtimeError),
      })
    } finally {
      setLocalRuntimeBusy(false)
    }
  }

  const handleRegisterExistingRuntime = async () => {
    const existingPath = localRuntimePath.trim()
    if (!existingPath) {
      setLocalRuntimeMessage({
        type: 'error',
        text: t('settings.runtime.localRuntimeExistingPathRequired'),
      })
      return
    }

    setLocalRuntimeBusy(true)
    setLocalRuntimeMessage(null)
    try {
      if (typeof settingsApi.installLocalModelRuntime !== 'function') {
        throw new Error('Local model runtime install API client is unavailable.')
      }
      const result = await settingsApi.installLocalModelRuntime({
        action: 'register_existing_path',
        existingPath,
        persist: true,
      })
      setLocalRuntimeStatus(result.runtimeStatus)
      setLocalRuntimePath(result.runtimeStatus?.rootDir ?? existingPath)
      setLocalRuntimeMessage({
        type: 'success',
        text: result.message ?? t('settings.runtime.localRuntimeRegistered'),
      })
    } catch (runtimeError) {
      setLocalRuntimeMessage({
        type: 'error',
        text: messageWithDetail(t('settings.runtime.localRuntimeRegisterError'), runtimeError),
      })
    } finally {
      setLocalRuntimeBusy(false)
    }
  }

  const handleSaveContextLengthSettings = async () => {
    if (!contextLengthTarget.kind) {
      return
    }

    const trimmed = contextLengthInput.trim()
    let parsedValue: number | null = null

    if (contextLengthTarget.kind === 'gguf') {
      const parsed = Number.parseInt(trimmed, 10)
      if (!Number.isInteger(parsed) || parsed <= 0) {
        setContextSettingsMessage({
          type: 'error',
          text: 'GGUF n_ctx must be a positive integer.',
        })
        return
      }
      parsedValue = parsed
    } else if (trimmed.length > 0) {
      const parsed = Number.parseInt(trimmed, 10)
      if (!Number.isInteger(parsed) || parsed <= 0) {
        setContextSettingsMessage({
          type: 'error',
          text: 'vLLM max model length must be a positive integer or left blank for auto.',
        })
        return
      }
      parsedValue = parsed
    }

    setContextSettingsBusy(true)
    setContextSettingsMessage(null)
    try {
      if (typeof settingsApi.updateSettings !== 'function') {
        throw new Error('Settings update API client is unavailable.')
      }
      const nextSettings = await settingsApi.updateSettings({
        ...buildContextLengthSettingsUpdate(contextLengthTarget.kind, parsedValue),
      })
      setSettings(nextSettings)
      setContextSettingsMessage({
        type: 'success',
        text: contextLengthTarget.kind === 'gguf'
          ? 'Saved GGUF context window.'
          : 'Saved vLLM max model length.',
      })
    } catch (updateError) {
      setContextSettingsMessage({
        type: 'error',
        text: messageWithDetail('Failed to save model context settings', updateError),
      })
    } finally {
      setContextSettingsBusy(false)
    }
  }

  const handleProbeToolCalling = async () => {
    setToolProbeBusy(true)
    setToolProbeMessage(null)
    try {
      if (typeof api.probeToolCalling !== 'function') {
        throw new Error('Tool-calling probe API client is unavailable.')
      }
      const result = await api.probeToolCalling()
      setToolProbeResult(asRecord(result.probe))
      const probedActiveModel = result.activeModel
      if (probedActiveModel) {
        setModels((current) => {
          const next = [...current]
          const activeId = modelInfoId(probedActiveModel)
          const index = next.findIndex((entry) => modelInfoId(entry) === activeId)
          if (index >= 0) {
            next[index] = probedActiveModel
            return next
          }
          return [probedActiveModel, ...next]
        })
      }

      const status = typeof result.probe?.status === 'string' ? result.probe.status : 'unknown'
      const detail = typeof result.probe?.message === 'string' ? result.probe.message : 'Probe finished.'
      setToolProbeMessage({
        type: status === 'supported' ? 'success' : 'error',
        text: `${status}: ${detail}`,
      })
    } catch (probeError) {
      setToolProbeResult(null)
      setToolProbeMessage({
        type: 'error',
        text: messageWithDetail('Failed to probe native tool calling', probeError),
      })
    } finally {
      setToolProbeBusy(false)
    }
  }

  const handleSaveLocalMountPolicy = async () => {
    setLocalMountBusy(true)
    setLocalMountMessage(null)
    try {
      if (typeof settingsApi.updateSettings !== 'function') {
        throw new Error('Settings update API client is unavailable.')
      }
      const parsedSeconds = Number(localIdleUnloadSeconds.trim())
      const idleSeconds = Number.isFinite(parsedSeconds) ? parsedSeconds : 300
      const nextSettings = await settingsApi.updateSettings({
        local_models: {
          idle_unload_enabled: localIdleUnloadEnabled,
          idle_unload_seconds: idleSeconds,
        },
      })
      setSettings(nextSettings)
      setLocalIdleUnloadSeconds(String(nextSettings.local_models?.idle_unload_seconds ?? idleSeconds))
      setLocalMountMessage({
        type: 'success',
        text: 'Local model mount policy saved.',
      })
    } catch (updateError) {
      setLocalMountMessage({
        type: 'error',
        text: messageWithDetail('Failed to save local model mount policy', updateError),
      })
    } finally {
      setLocalMountBusy(false)
    }
  }

  const handleUnloadActiveLocalModel = async () => {
    setLocalMountBusy(true)
    setLocalMountMessage(null)
    try {
      if (typeof settingsApi.unloadActiveLocalModelRuntime !== 'function') {
        throw new Error('Active local model runtime unload API client is unavailable.')
      }
      const result = await settingsApi.unloadActiveLocalModelRuntime()
      setActiveLocalRuntimeStatus(result.activeRuntime)
      setLocalMountMessage({
        type: 'success',
        text: result.unloaded ? 'Current local model unloaded.' : 'No active local model to unload.',
      })
      notifyModelsUpdated()
    } catch (unloadError) {
      setLocalMountMessage({
        type: 'error',
        text: messageWithDetail('Failed to unload current local model', unloadError),
      })
    } finally {
      setLocalMountBusy(false)
    }
  }

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setSubmitting(true)
    setMessage(null)

    try {
      if (typeof settingsApi.configureModel !== 'function') {
        throw new Error('Model configure API client is unavailable.')
      }

      if (provider === 'local' && localModelInputLooksLikeGguf) {
        const ggufRuntimeReady = (
          localRuntimeStatus?.readiness === 'ready' ||
          localRuntimeStatus?.readiness === 'degraded'
        )
        if (!ggufRuntimeReady) {
          throw new Error(t('settings.modelConnection.ggufRuntimeMissing'))
        }
      }
      if (provider === 'openai_codex' && !openAICodexStatus?.activeProfileId) {
        throw new Error('Connect ChatGPT or import a Codex CLI login before connecting OpenAI Codex.')
      }

      const result = await settingsApi.configureModel({
        provider,
        model,
        baseUrl: provider === 'local' ? undefined : baseUrl,
        apiKey: provider === 'local' || provider === 'openai_codex' ? '' : apiKey,
        authProfileId: provider === 'openai_codex' ? openAICodexStatus?.activeProfileId ?? undefined : undefined,
        persist: true,
      })
      onConfigured(result)
      setApiKey('')
      setMessage({
        type: 'success',
        text: result.persisted
          ? t('settings.modelConnection.successSwitchedPersisted', { model: modelInfoLabel(result.activeModel) || model })
          : t('settings.modelConnection.successSwitched', { model: modelInfoLabel(result.activeModel) || model }),
      })
    } catch (configureError) {
      setMessage({
        type: 'error',
        text: messageWithDetail(t('settings.modelConnection.errorConfigure'), configureError),
      })
    } finally {
      setSubmitting(false)
    }
  }

  const handleConvertToGguf = async () => {
    if (!selectedGgufQuantization) {
      return
    }

    setConvertingLocalModel(true)
    setConvertMessage(null)
    try {
      if (typeof settingsApi.convertLocalModel !== 'function') {
        throw new Error('Local model convert API client is unavailable.')
      }
      const result = await settingsApi.convertLocalModel({
        sourceModelDir: normalizedModelPath,
        targetFormat: 'gguf',
        quantization: selectedGgufQuantization,
        persist: true,
      })

      const outputPath = result.outputPath?.trim() || null
      setConvertedOutputPath(outputPath)

      if (outputPath) {
        setModel(outputPath)
      }

      const activeModel = result.activeModel ?? (
        outputPath
          ? {
              id: outputPath,
              name: outputPath.split(/[\\/]/).pop() ?? outputPath,
              label: outputPath.split(/[\\/]/).pop() ?? outputPath,
              provider: 'local',
              modelSpec: outputPath,
              baseUrl: null,
              backendType: 'gguf',
              authProfileId: null,
              authMode: null,
              contextLength: null,
              supportsToolCalling: null,
              metadata: {},
            } satisfies api.ModelInfo
          : null
      )
      const availableModels = result.availableModels.length > 0
        ? result.availableModels
        : activeModel
          ? [activeModel]
          : []

      if (activeModel && availableModels.length > 0) {
        onConfigured({
          type: 'model_configure',
          provider: result.provider ?? 'local',
          activeModel,
          availableModels,
          apiKeyConfigured: false,
          persisted: result.persisted,
          configPath: result.configPath,
        })
      }

      setConvertMessage({
        type: 'success',
        text: outputPath
          ? t('settings.quantization.convertSuccessWithPath', { path: outputPath })
          : t('settings.quantization.convertSuccess'),
      })
    } catch (convertError) {
      setConvertMessage({
        type: 'error',
        text: messageWithDetail(t('settings.quantization.convertError'), convertError),
      })
    } finally {
      setConvertingLocalModel(false)
    }
  }

  const startEditSavedModel = (entry: api.ModelInfo) => {
    setEditingModelId(modelInfoId(entry))
    const nextProvider = (entry.provider && isProviderChoice(entry.provider))
      ? entry.provider
      : inferProviderChoice(modelInfoId(entry))
        ?? (entry.backendType === 'openai_codex'
          ? 'openai_codex'
          : (entry.backendType === 'openai_compat' ? 'openai_compat' : 'local'))
    setEditingProvider(nextProvider)
    setEditingModelName(entry.name || '')
    setEditingModelSpec(entry.modelSpec || modelInfoId(entry))
    setEditingBaseUrl(baseUrlFromModelInfo(entry) ?? '')
    setEditingApiKey('')
    setEntryMessage(null)
  }

  const cancelEditSavedModel = () => {
    setEditingModelId(null)
    setEditingApiKey('')
    setEntryMessage(null)
  }

  const submitEditSavedModel = async () => {
    if (!editingModelId) {
      return
    }
    setEntrySubmitting(true)
    setEntryMessage(null)
    try {
      if (typeof settingsApi.updateModelEntry !== 'function') {
        throw new Error('Model entry update API client is unavailable.')
      }
      const payloadModelSpec = editingProvider === 'local'
        ? editingModelSpec.trim()
        : editingProvider === 'ollama'
          ? `ollama:${editingModelName.trim()}`
          : editingBaseUrl.trim()

      const result = await settingsApi.updateModelEntry({
        modelId: editingModelId,
        provider: editingProvider,
        model: editingModelName.trim(),
        modelSpec: payloadModelSpec,
        baseUrl: editingProvider === 'local' ? null : editingBaseUrl.trim(),
        apiKey: editingProvider === 'openai_codex' ? null : (editingApiKey.trim() || null),
        authProfileId: editingProvider === 'openai_codex' ? openAICodexStatus?.activeProfileId ?? null : null,
        persist: true,
      })
      const availableModels = result.availableModels
      setModels(availableModels)
      setSettings((current) => updateSavedModelsInSettings(current, availableModels, result.configuredModel))
      notifyModelsUpdated()
      setEntryMessage({ type: 'success', text: t('settings.savedModels.successUpdate') })
      setEditingModelId(null)
      setEditingApiKey('')
    } catch (updateError) {
      setEntryMessage({
        type: 'error',
        text: messageWithDetail(t('settings.savedModels.errorUpdate'), updateError),
      })
    } finally {
      setEntrySubmitting(false)
    }
  }

  const deleteSavedModel = async (modelId: string) => {
    setEntrySubmitting(true)
    setEntryMessage(null)
    try {
      if (typeof settingsApi.deleteModelEntry !== 'function') {
        throw new Error('Model entry delete API client is unavailable.')
      }
      const result = await settingsApi.deleteModelEntry(modelId, true)
      const availableModels = result.availableModels
      setModels(availableModels)
      setSettings((current) => updateSavedModelsInSettings(current, availableModels, result.configuredModel))
      notifyModelsUpdated()
      if (editingModelId === modelId) {
        setEditingModelId(null)
      }
      setEntryMessage({ type: 'success', text: t('settings.savedModels.successDelete') })
    } catch (deleteError) {
      setEntryMessage({
        type: 'error',
        text: messageWithDetail(t('settings.savedModels.errorDelete'), deleteError),
      })
    } finally {
      setEntrySubmitting(false)
    }
  }

  return (
    <section className="rounded-lg border border-border bg-surface-layer">
      <div className="border-b border-border px-4 py-3">
        <div className="flex items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold text-foreground">{t('settings.modelConnection.title')}</h3>
            <p className="mt-0.5 text-xs text-muted-foreground">
              {t('settings.modelConnection.description')}
            </p>
          </div>
          <PlugZap className="h-4 w-4 text-muted-foreground" />
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4 px-4 py-4">
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
          {providerOptions.map((option) => (
            <button
              key={option.value}
              type="button"
              onClick={() => handleProviderChange(option.value)}
              className={[
                'min-w-0 rounded-md border px-3 py-3 text-left transition-colors',
                option.value === provider
                  ? 'border-primary-500 bg-primary-500/10 text-foreground'
                  : 'border-border bg-canvas text-muted-foreground hover:text-foreground',
              ].join(' ')}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="truncate text-sm font-semibold">{option.label}</span>
                {option.value === provider ? <CheckCircle2 className="h-3.5 w-3.5 text-primary-400" /> : null}
              </div>
              <p className="mt-1 line-clamp-2 text-xs">{providerDescription(option.value, t)}</p>
            </button>
          ))}
        </div>

        <div className="grid grid-cols-1 gap-3 xl:grid-cols-[minmax(0,1fr)_minmax(220px,320px)]">
          {provider === 'openai_codex' ? (
            <div className="min-w-0 space-y-1.5">
              <span className="text-xs font-medium text-muted-foreground">Backend URL</span>
              <div className="rounded-md border border-border bg-surface-layer px-3 py-2 font-mono text-xs text-muted-foreground">
                {currentProvider.defaultBaseUrl}
              </div>
            </div>
          ) : (
            <label className="min-w-0 space-y-1.5">
              <span className="text-xs font-medium text-muted-foreground">
                {provider === 'local' ? t('settings.modelConnection.localRootPath') : t('settings.form.apiUrl')}
              </span>
              <div className="flex min-w-0 gap-2">
                <Input
                  value={baseUrl}
                  onChange={(event) => {
                    setBaseUrl(event.target.value)
                    setDiscoverMessage(null)
                  }}
                  placeholder={provider === 'local' ? t('settings.modelConnection.localRootPlaceholder') : currentProvider.defaultBaseUrl}
                  className="min-w-0 font-mono text-xs"
                />
                {provider === 'ollama' || provider === 'local' ? (
                  <Button
                    type="button"
                    variant="secondary"
                    size="icon-sm"
                    loading={discovering}
                    aria-label={provider === 'local' ? t('settings.modelConnection.scanLocal') : t('settings.modelConnection.refreshOllama')}
                    title={provider === 'local' ? t('settings.modelConnection.scanLocal') : t('settings.modelConnection.refreshOllama')}
                    onClick={discoverCurrentProviderModels}
                  >
                    <RefreshCw className="h-3.5 w-3.5" />
                  </Button>
                ) : null}
              </div>
            </label>
          )}

          <label className="min-w-0 space-y-1.5">
            <span className="text-xs font-medium text-muted-foreground">
              {provider === 'local' ? t('settings.modelConnection.localModelPath') : t('settings.form.modelName')}
            </span>
            {provider === 'ollama' && ollamaModels.length > 0 ? (
              <SelectSetting
                value={model}
                onValueChange={setModel}
                options={withCurrentOption(ollamaModels, model)}
                className="font-mono text-xs"
              />
            ) : provider === 'local' && localModelOptions.length > 0 ? (
              <SelectSetting
                value={model}
                onValueChange={setModel}
                options={withCurrentOption(localModelOptions, model)}
                getOptionLabel={(value) => localModelLabelById.get(value) ?? value}
                className="font-mono text-xs"
              />
            ) : (
              <Input
                value={model}
                onChange={(event) => setModel(event.target.value)}
                placeholder={provider === 'local' ? t('settings.modelConnection.localModelPlaceholder') : currentProvider.defaultModel}
                className="min-w-0 font-mono text-xs"
              />
            )}
          </label>
        </div>

        {provider === 'openai_codex' ? (
          <div className="space-y-3 rounded-md border border-border bg-canvas px-3 py-3">
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="text-xs font-semibold text-foreground">OpenAI Codex Login</p>
                <p className="mt-0.5 text-[11px] text-muted-foreground">
                  Connect ChatGPT in Mochi, or import an existing Codex CLI ChatGPT OAuth login. API key mode in `.codex/auth.json` cannot be imported.
                </p>
              </div>
              <Badge variant={openAICodexStatusVariant(openAICodexStatus?.status, openAICodexStatus?.configured)}>
                {openAICodexStatusLabel(openAICodexStatus?.status, openAICodexStatus?.configured)}
              </Badge>
            </div>

            <div className="rounded-md border border-border bg-surface-layer px-3 py-2 text-xs text-muted-foreground">
              {openAICodexStatus?.activeProfileId
                ? `Active profile: ${openAICodexStatus.activeProfileId}`
                : 'No saved OpenAI Codex login in Mochi'}
            </div>

            {openAICodexStatus?.profiles?.[0]?.email ? (
              <div className="rounded-md border border-border bg-surface-layer px-3 py-2 text-xs text-muted-foreground">
                Account: {openAICodexStatus.profiles[0].email}
                {openAICodexStatus.profiles[0].expiresAt
                  ? ` | Expires: ${new Date(openAICodexStatus.profiles[0].expiresAt * 1000).toLocaleString()}`
                  : ''}
              </div>
            ) : null}

            {openAICodexStatus?.lastRefreshError ? (
              <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                Refresh error: {openAICodexStatus.lastRefreshError}
              </div>
            ) : null}

            {openAICodexStatus?.cliAuthMessage ? (
              <div
                className={[
                  'rounded-md border px-3 py-2 text-xs',
                  openAICodexCliAuthVariant(openAICodexStatus.cliAuthState) === 'success'
                    ? 'border-success/30 bg-success/10 text-success'
                    : openAICodexCliAuthVariant(openAICodexStatus.cliAuthState) === 'warning'
                      ? 'border-warning/30 bg-warning/10 text-warning'
                      : openAICodexCliAuthVariant(openAICodexStatus.cliAuthState) === 'error'
                        ? 'border-destructive/30 bg-destructive/10 text-destructive'
                        : 'border-border bg-surface-layer text-muted-foreground',
                ].join(' ')}
              >
                <p className="font-medium">
                  Local Codex CLI: {openAICodexCliAuthLabel(openAICodexStatus.cliAuthState)}
                  {openAICodexStatus.cliAuthMode ? ` (${openAICodexStatus.cliAuthMode})` : ''}
                </p>
                <p className="mt-1">{openAICodexStatus.cliAuthMessage}</p>
              </div>
            ) : null}

            <div className="flex flex-wrap gap-2">
              <Button type="button" variant="secondary" loading={openAICodexAuthBusy} onClick={() => void connectOpenAICodexLogin()}>
                <Globe className="h-3.5 w-3.5" />
                Connect ChatGPT
              </Button>
              <Button type="button" variant="outline" loading={openAICodexAuthBusy} onClick={() => void importOpenAICodexCliLogin()}>
                <Download className="h-3.5 w-3.5" />
                Import Existing Codex CLI OAuth
              </Button>
              <Button type="button" variant="secondary" loading={openAICodexAuthBusy} onClick={() => void refreshOpenAICodexLogin()}>
                <RefreshCw className="h-3.5 w-3.5" />
                Refresh Login
              </Button>
              <Button type="button" variant="ghost" loading={openAICodexAuthBusy} onClick={() => void logoutOpenAICodexLogin()}>
                <Trash2 className="h-3.5 w-3.5" />
                Logout
              </Button>
            </div>

            {openAICodexLoginStart ? (
              <div className="space-y-3 rounded-md border border-border bg-surface-layer px-3 py-3">
                <div className="space-y-1">
                  <p className="text-xs font-semibold text-foreground">Browser OAuth in progress</p>
                  <p className="text-[11px] text-muted-foreground">
                    Callback expires at {new Date(openAICodexLoginStart.expiresAt * 1000).toLocaleString()}.
                  </p>
                </div>

                <div className="rounded-md border border-border bg-canvas px-3 py-2 text-xs text-muted-foreground">
                  Sign-in URL:{' '}
                  <a
                    href={openAICodexLoginStart.authUrl}
                    target="_blank"
                    rel="noreferrer"
                    className="break-all text-primary-400 underline underline-offset-2"
                  >
                    {openAICodexLoginStart.authUrl}
                  </a>
                </div>

                <div className="rounded-md border border-border bg-canvas px-3 py-2 text-xs text-muted-foreground">
                  Callback URL: <span className="break-all font-mono">{openAICodexLoginStart.callbackUrl}</span>
                </div>

                <label className="block space-y-1.5">
                  <span className="text-xs font-medium text-muted-foreground">Manual callback URL</span>
                  <Input
                    value={openAICodexManualCallbackUrl}
                    onChange={(event) => setOpenAICodexManualCallbackUrl(event.target.value)}
                    placeholder="Paste the full callback URL here if the popup cannot finish automatically"
                    className="font-mono text-xs"
                  />
                </label>

                <div className="flex flex-wrap gap-2">
                  <Button type="button" variant="secondary" loading={openAICodexAuthBusy} onClick={() => void completeOpenAICodexLogin()}>
                    Save Browser Login
                  </Button>
                </div>

                {openAICodexLoginStart.guidance.length > 0 ? (
                  <div className="rounded-md border border-warning/30 bg-warning/10 px-3 py-2 text-xs text-warning">
                    {openAICodexLoginStart.guidance.map((item, index) => (
                      <p key={`${item}-${index}`} className={index === 0 ? '' : 'mt-1'}>
                        {item}
                      </p>
                    ))}
                  </div>
                ) : null}
              </div>
            ) : null}

            <SettingMessage message={openAICodexAuthMessage} />
          </div>
        ) : null}

        {provider !== 'local' && provider !== 'openai_codex' ? (
          <label className="block space-y-1.5">
            <span className="text-xs font-medium text-muted-foreground">{t('settings.form.apiKey')}</span>
            <div className="relative">
              <Input
                type="password"
                value={apiKey}
                onChange={(event) => setApiKey(event.target.value)}
                placeholder={currentProvider.needsApiKey ? 'sk-...' : t('settings.modelConnection.apiKeyPlaceholderNoKey')}
                leftIcon={<KeyRound className="h-3.5 w-3.5" />}
                className="pl-8 font-mono text-xs"
              />
            </div>
          </label>
        ) : null}

        <div className="rounded-md border border-border bg-canvas px-3 py-2 text-xs text-muted-foreground">
          {providerNote(currentProvider.value, t)}
        </div>

        {activeModelInfo && activeModelInfo.backendType === 'openai_compat' ? (
          <div className="space-y-3 rounded-md border border-border bg-canvas px-3 py-3">
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="text-xs font-semibold text-foreground">Tool Calling</p>
                <p className="mt-0.5 text-[11px] text-muted-foreground">
                  Diagnose whether the active remote endpoint is using native structured tools or Mochi fallback simulation.
                </p>
              </div>
              <Badge variant={activeToolCallingInfo.mode === 'native' ? 'success' : 'neutral'}>
                {toolCallModeLabel(activeToolCallingInfo.mode)}
              </Badge>
            </div>

            <div className="space-y-1 text-xs text-muted-foreground">
              <p>
                <span className="font-medium text-foreground">Native status:</span>{' '}
                {typeof toolProbeResult?.status === 'string'
                  ? toolProbeResult.status
                  : (activeToolCallingInfo.status ?? 'unknown')}
              </p>
              {(typeof toolProbeResult?.message === 'string' || activeToolCallingInfo.message) ? (
                <p>
                  {typeof toolProbeResult?.message === 'string'
                    ? toolProbeResult.message
                    : activeToolCallingInfo.message}
                </p>
              ) : null}
              {(typeof toolProbeResult?.checked_at === 'string' || activeToolCallingInfo.checkedAt) ? (
                <p className="font-mono text-[11px]">
                  checked: {typeof toolProbeResult?.checked_at === 'string'
                    ? toolProbeResult.checked_at
                    : activeToolCallingInfo.checkedAt}
                </p>
              ) : null}
            </div>

            <div className="flex justify-end">
              <Button
                type="button"
                variant="secondary"
                size="sm"
                loading={toolProbeBusy}
                onClick={() => void handleProbeToolCalling()}
              >
                Probe Native Tool Calling
              </Button>
            </div>

            <SettingMessage message={toolProbeMessage} />
          </div>
        ) : null}

        {activeModelInfo && activeModelInfo.backendType === 'openai_compat' ? (
          <div className="space-y-3 rounded-md border border-border bg-canvas px-3 py-3">
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="text-xs font-semibold text-foreground">Reasoning Continuity</p>
                <p className="mt-0.5 text-[11px] text-muted-foreground">
                  Surface Responses transport preference, summary support, and replay continuity diagnostics.
                </p>
              </div>
              <Badge variant={activeReasoningInfo.continuityMode === 'previous_response_id' ? 'success' : 'neutral'}>
                {activeReasoningInfo.continuityMode ?? 'unknown'}
              </Badge>
            </div>

            <div className="space-y-1 text-xs text-muted-foreground">
              <p>
                <span className="font-medium text-foreground">Transport preference:</span>{' '}
                {activeReasoningInfo.transportPreference ?? 'unknown'}
              </p>
              <p>
                <span className="font-medium text-foreground">Reasoning summary:</span>{' '}
                requested={String(activeReasoningInfo.summaryRequested ?? false)} supported=
                {String(activeReasoningInfo.summarySupported ?? 'unknown')} received=
                {String(activeReasoningInfo.summaryReceived ?? false)}
              </p>
              <p>
                <span className="font-medium text-foreground">Replay items:</span>{' '}
                {activeReasoningInfo.replayedItems ?? 0}
              </p>
            </div>
          </div>
        ) : null}

        {contextLengthTarget.kind ? (
          <div className="space-y-3 rounded-md border border-border bg-canvas px-3 py-3">
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="text-xs font-semibold text-foreground">
                  {contextLengthTarget.kind === 'gguf' ? 'GGUF Context Window' : 'vLLM Max Model Length'}
                </p>
                <p className="mt-0.5 text-[11px] text-muted-foreground">
                  {contextLengthTarget.kind === 'gguf'
                    ? 'Adjusts `gguf.n_ctx` for the active GGUF model.'
                    : 'Sets the managed vLLM startup override for `vllm.max_model_len`. Leave blank to use vLLM auto sizing.'}
                </p>
              </div>
              <Badge variant="neutral">
                {contextLengthTarget.kind === 'gguf' ? 'gguf.n_ctx' : 'vllm.max_model_len'}
              </Badge>
            </div>

            <label className="block space-y-1.5">
              <span className="text-xs font-medium text-muted-foreground">
                {contextLengthTarget.kind === 'gguf' ? 'Context length' : 'Max model length'}
              </span>
              <Input
                value={contextLengthInput}
                onChange={(event) => setContextLengthInput(event.target.value)}
                inputMode="numeric"
                placeholder={contextLengthTarget.kind === 'gguf' ? '4096' : 'auto'}
                className="font-mono text-xs"
              />
            </label>

            <div className="flex flex-wrap justify-end gap-2">
              <Button
                type="button"
                variant="primary"
                size="sm"
                loading={contextSettingsBusy}
                onClick={() => void handleSaveContextLengthSettings()}
              >
                <Save className="h-3.5 w-3.5" />
                Save Context Setting
              </Button>
            </div>

            <SettingMessage message={contextSettingsMessage} />
          </div>
        ) : null}

        {provider === 'local' ? (
          <div className="space-y-3 rounded-md border border-border bg-canvas px-3 py-3">
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="text-xs font-semibold text-foreground">Local Model Mount Policy</p>
                <p className="mt-0.5 text-[11px] text-muted-foreground">
                  Control whether local inference models stay mounted after loading.
                </p>
              </div>
              <Badge variant={activeLocalRuntimeStatus?.loaded ? 'success' : 'neutral'}>
                {activeLocalRuntimeStatus?.hasActiveLocalModel
                  ? (activeLocalRuntimeStatus.loaded ? 'Mounted' : 'Not mounted')
                  : 'No active local model'}
              </Badge>
            </div>

            <div className="flex items-center justify-between gap-3 rounded-md border border-border bg-surface-layer px-3 py-2">
              <div>
                <p className="text-xs font-medium text-foreground">Enable idle timeout unload</p>
                <p className="text-[11px] text-muted-foreground">
                  Disabled means the model stays mounted until switched or manually unloaded.
                </p>
              </div>
              <Switch checked={localIdleUnloadEnabled} onCheckedChange={setLocalIdleUnloadEnabled} />
            </div>

            <label className="block space-y-1.5">
              <span className="text-xs font-medium text-muted-foreground">Idle unload seconds</span>
              <Input
                value={localIdleUnloadSeconds}
                onChange={(event) => setLocalIdleUnloadSeconds(event.target.value)}
                inputMode="numeric"
                disabled={!localIdleUnloadEnabled}
                className="font-mono text-xs"
              />
            </label>

            <div className="rounded-md border border-border bg-surface-layer px-3 py-2 text-xs text-muted-foreground">
              {activeLocalRuntimeStatus?.hasActiveLocalModel
                ? `Current model: ${activeLocalRuntimeStatus.modelSpec ?? t('common.notReported')}`
                : 'Current model: no active local model'}
            </div>

            <div className="flex flex-wrap justify-end gap-2">
              <Button
                type="button"
                variant="secondary"
                size="sm"
                loading={localMountBusy}
                disabled={!activeLocalRuntimeStatus?.canUnload}
                onClick={() => void handleUnloadActiveLocalModel()}
              >
                Unload Current Model
              </Button>
              <Button
                type="button"
                variant="primary"
                size="sm"
                loading={localMountBusy}
                onClick={() => void handleSaveLocalMountPolicy()}
              >
                <Save className="h-3.5 w-3.5" />
                Save Mount Policy
              </Button>
            </div>

            <SettingMessage message={localMountMessage} />
          </div>
        ) : null}

        {provider === 'local' ? (
          <LocalRuntimeCard
            runtimeStatus={localRuntimeStatus}
            runtimeLoading={localRuntimeLoading}
            runtimeMessage={localRuntimeMessage}
            runtimePath={localRuntimePath}
            runtimeBusy={localRuntimeBusy}
            onRuntimePathChange={setLocalRuntimePath}
            onPrepareManagedRuntime={handlePrepareManagedRuntime}
            onRegisterExistingRuntime={handleRegisterExistingRuntime}
            hardware={localRuntimeStatus?.hardware ?? localCapabilities?.hardware ?? null}
          />
        ) : null}

        {shouldShowQuantization ? (
          <LocalQuantizationCapabilities
            modelPath={normalizedModelPath}
            showGgufHint={shouldShowGgufHint}
            loading={capabilitiesLoading}
            error={capabilitiesError}
            capabilities={localCapabilities}
            runtimeReady={localRuntimeStatus?.readiness === 'ready'}
            selectedQuantization={selectedGgufQuantization}
            onSelectQuantization={setSelectedGgufQuantization}
            convertLoading={convertingLocalModel}
            convertMessage={convertMessage}
            convertOutputPath={convertedOutputPath}
            onConvertGguf={handleConvertToGguf}
          />
        ) : null}

        <SettingMessage message={discoverMessage} />
        <SettingMessage message={entryMessage} />

        <div className="space-y-2 rounded-md border border-border bg-canvas px-3 py-3">
          <p className="text-xs font-semibold text-foreground">{t('settings.savedModels.title')}</p>
          {savedModels.length === 0 ? (
            <p className="text-xs text-muted-foreground">{t('settings.savedModels.none')}</p>
          ) : (
            savedModels.map((entry) => {
              const id = modelInfoId(entry)
              const isEditing = editingModelId === id
              const entryProvider = (entry.provider && isProviderChoice(entry.provider))
                ? entry.provider
                : inferProviderChoice(id) ?? 'local'
              return (
                <div key={id} className="rounded-md border border-border bg-surface-layer px-2.5 py-2">
                  {!isEditing ? (
                    <>
                      <div className="flex items-center justify-between gap-2">
                        <p className="truncate text-xs font-medium text-foreground">{modelInfoLabel(entry)}</p>
                        <div className="flex items-center gap-1">
                          <Button
                            type="button"
                            size="icon-sm"
                            variant="ghost"
                            onClick={() => startEditSavedModel(entry)}
                            disabled={entrySubmitting}
                            title={t('settings.savedModels.edit')}
                            aria-label={t('settings.savedModels.edit')}
                          >
                            <Pencil className="h-3.5 w-3.5" />
                          </Button>
                          <Button
                            type="button"
                            size="icon-sm"
                            variant="ghost"
                            onClick={() => void deleteSavedModel(id)}
                            disabled={entrySubmitting}
                            title={t('settings.savedModels.delete')}
                            aria-label={t('settings.savedModels.delete')}
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </Button>
                        </div>
                      </div>
                      <p className="mt-1 truncate font-mono text-[11px] text-muted-foreground">{entry.modelSpec ?? id}</p>
                    </>
                  ) : (
                    <div className="space-y-2">
                      <SelectSetting
                        value={editingProvider}
                        onValueChange={(value) => setEditingProvider(value as ProviderChoice)}
                        options={providerOptions.map((option) => option.value)}
                        getOptionLabel={(value) => providerOption(value as ProviderChoice).label}
                      />
                      <Input
                        value={editingModelName}
                        onChange={(event) => setEditingModelName(event.target.value)}
                        placeholder={t('settings.form.modelName')}
                        className="text-xs"
                      />
                      {editingProvider === 'local' ? (
                        <Input
                          value={editingModelSpec}
                          onChange={(event) => setEditingModelSpec(event.target.value)}
                          placeholder={t('settings.modelConnection.localModelPath')}
                          className="font-mono text-xs"
                        />
                      ) : editingProvider === 'openai_codex' ? (
                        <div className="rounded-md border border-border bg-surface-layer px-3 py-2 font-mono text-xs text-muted-foreground">
                          {editingBaseUrl || 'https://chatgpt.com/backend-api'}
                        </div>
                      ) : (
                        <Input
                          value={editingBaseUrl}
                          onChange={(event) => setEditingBaseUrl(event.target.value)}
                          placeholder={t('settings.form.apiUrl')}
                          className="font-mono text-xs"
                        />
                      )}
                      {editingProvider !== 'local' && editingProvider !== 'ollama' && editingProvider !== 'openai_codex' ? (
                        <>
                          <Input
                            type="password"
                            autoComplete="off"
                            value={editingApiKey}
                            onChange={(event) => setEditingApiKey(event.target.value)}
                            placeholder={t('settings.form.apiKey')}
                            className="font-mono text-xs"
                          />
                          <p className="text-[11px] text-muted-foreground">{t('settings.savedModels.apiKeyHint')}</p>
                        </>
                      ) : null}
                      {editingProvider === 'openai_codex' ? (
                        <p className="text-[11px] text-muted-foreground">
                          OpenAI Codex uses the imported OAuth profile and does not store an API key on the model entry.
                        </p>
                      ) : null}
                      <div className="flex items-center justify-end gap-2">
                        <Button
                          type="button"
                          size="sm"
                          variant="secondary"
                          onClick={cancelEditSavedModel}
                          disabled={entrySubmitting}
                        >
                          {t('settings.savedModels.cancel')}
                        </Button>
                        <Button
                          type="button"
                          size="sm"
                          variant="primary"
                          loading={entrySubmitting}
                          onClick={() => void submitEditSavedModel()}
                        >
                          {t('settings.savedModels.save')}
                        </Button>
                      </div>
                    </div>
                  )}
                  <p className="mt-1 text-[11px] text-muted-foreground">
                    {providerOption(entryProvider).label}{entry.backendType ? ` · ${entry.backendType}` : ''}
                  </p>
                </div>
              )
            })
          )}
        </div>

        {message ? (
          <div
            className={[
              'rounded-md border px-3 py-2 text-xs',
              message.type === 'success'
                ? 'border-success/30 bg-success/10 text-success'
                : 'border-destructive/30 bg-destructive/10 text-destructive',
            ].join(' ')}
          >
            {message.text}
          </div>
        ) : null}

        <div className="flex items-center justify-end gap-2">
          <Button type="submit" variant="primary" size="sm" loading={submitting}>
            {t('settings.action.addApplyTest')}
          </Button>
        </div>
      </form>
    </section>
  )
}

function getImportPackageName(files: File[], fallback: string): string {
  const firstFile = files[0]
  if (!firstFile) {
    return fallback
  }

  const relativePath = firstFile.webkitRelativePath
  if (relativePath) {
    return relativePath.split('/')[0] || fallback
  }

  return firstFile.name || fallback
}

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return '0 B'
  }

  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  const exponent = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1)
  const value = bytes / 1024 ** exponent
  return `${value.toFixed(value >= 10 || exponent === 0 ? 0 : 1)} ${units[exponent]}`
}

function VoicePipelineForm({
  voice,
  onUpdated,
}: {
  voice: api.VoiceSettings
  onUpdated: (settings: api.Settings) => void
}) {
  const { t } = useI18n()
  const sttOptions = dedupeVoiceBackendOptions(
    getStringOptions(voice, 'supported_stt_backends', defaultSttBackends),
    'stt',
    getStringSetting(voice, 'stt_backend', 'faster-whisper')
  )
  const ttsOptions = dedupeVoiceBackendOptions(
    getStringOptions(voice, 'supported_tts_backends', defaultTtsBackends),
    'tts',
    getStringSetting(voice, 'tts_backend', 'kokoro-tts')
  )
  const sttModelsByBackend = getOptionsByBackend(voice, 'supported_stt_models_by_backend', defaultSttModelsByBackend)
  const ttsModelsByBackend = getOptionsByBackend(voice, 'supported_tts_models_by_backend', defaultTtsModelsByBackend)
  const ttsVoicesByBackend = getOptionsByBackend(voice, 'supported_tts_voices_by_backend', defaultTtsVoicesByBackend)
  const [enabled, setEnabled] = React.useState(getBooleanSetting(voice, 'enabled'))
  const [sttBackend, setSttBackend] = React.useState(getStringSetting(voice, 'stt_backend', 'faster-whisper'))
  const [sttModel, setSttModel] = React.useState(getStringSetting(voice, 'stt_model', 'medium'))
  const [sttLanguage, setSttLanguage] = React.useState(getStringSetting(voice, 'stt_language', 'auto'))
  const [sttDevice, setSttDevice] = React.useState(getStringSetting(voice, 'stt_device', 'auto'))
  const [sttCacheDir, setSttCacheDir] = React.useState(getStringSetting(voice, 'stt_model_cache_dir'))
  const [sttModelPath, setSttModelPath] = React.useState(getStringSetting(voice, 'stt_model_path'))
  const [sttApiBaseUrl, setSttApiBaseUrl] = React.useState(getStringSetting(voice, 'stt_openai_base_url'))
  const [sttApiKey, setSttApiKey] = React.useState('')
  const [sttApiTimeout, setSttApiTimeout] = React.useState(String(getNumberSetting(voice, 'stt_openai_timeout', 60)))
  const [ttsBackend, setTtsBackend] = React.useState(getStringSetting(voice, 'tts_backend', 'kokoro-tts'))
  const [ttsModel, setTtsModel] = React.useState(getStringSetting(voice, 'tts_model', 'none') || 'none')
  const [ttsVoice, setTtsVoice] = React.useState(getStringSetting(voice, 'tts_voice', defaultTtsVoice))
  const [ttsLanguage, setTtsLanguage] = React.useState(getStringSetting(voice, 'tts_language', 'none') || 'none')
  const [ttsSpeed, setTtsSpeed] = React.useState(String(getNumberSetting(voice, 'tts_speed', 1)))
  const [ttsApiBaseUrl, setTtsApiBaseUrl] = React.useState(getStringSetting(voice, 'tts_openai_base_url'))
  const [ttsApiKey, setTtsApiKey] = React.useState('')
  const [ttsApiTimeout, setTtsApiTimeout] = React.useState(String(getNumberSetting(voice, 'tts_openai_timeout', 60)))
  const [ttsResponseFormat, setTtsResponseFormat] = React.useState(
    getStringSetting(voice, 'tts_openai_response_format', 'pcm') || 'pcm'
  )
  const [replyModelMode, setReplyModelMode] = React.useState(
    normalizeReplyModelModeFromApi(getStringSetting(voice, 'reply_model_mode', 'inherit_active'))
  )
  const [replyModelId, setReplyModelId] = React.useState(getStringSetting(voice, 'reply_model_id'))
  const [sessionMode, setSessionMode] = React.useState(
    normalizeSessionModeFromApi(getStringSetting(voice, 'session_mode', 'append_current'))
  )
  const [downloadMissing, setDownloadMissing] = React.useState(true)
  const [submitting, setSubmitting] = React.useState(false)
  const [message, setMessage] = React.useState<FormMessage>(null)
  const sttModelOptions = withCurrentOption(sttModelsByBackend[sttBackend] ?? defaultSttModelsByBackend['faster-whisper'], sttModel)
  const ttsModelOptions = withCurrentOption(ttsModelsByBackend[ttsBackend] ?? ['none'], ttsModel)
  const ttsVoiceOptions = withCurrentOption(ttsVoicesByBackend[ttsBackend] ?? defaultTtsVoicesByBackend['kokoro-tts'], ttsVoice)
  const modelFileInputRef = React.useRef<HTMLInputElement>(null)
  const [importingModel, setImportingModel] = React.useState(false)
  const [runtimeStatus, setRuntimeStatus] = React.useState<api.VoiceRuntimeStatus | null>(null)
  const [runtimeStatusLoading, setRuntimeStatusLoading] = React.useState(false)
  const [voiceCatalog, setVoiceCatalog] = React.useState<api.VoiceCatalogVoice[]>([])
  const [voiceCatalogLoading, setVoiceCatalogLoading] = React.useState(false)
  const [voiceCatalogUnsupported, setVoiceCatalogUnsupported] = React.useState(false)
  const [voicePackPath, setVoicePackPath] = React.useState('')
  const [voicePackBusy, setVoicePackBusy] = React.useState(false)
  const [deletingVoiceId, setDeletingVoiceId] = React.useState<string | null>(null)
  const [voicePackMessage, setVoicePackMessage] = React.useState<FormMessage>(null)
  const voicePackInputRef = React.useRef<HTMLInputElement>(null)
  const replyModelModeOptionsWithCurrent = withCurrentOption(replyModelModeOptions, replyModelMode)
  const sessionModeOptionsWithCurrent = withCurrentOption(sessionModeOptions, sessionMode)
  const localTtsRecommendations = voice.recommended_local_tts_backends ?? []
  const externalApiTtsPresets = voice.external_api_tts_presets ?? []
  const hasVoiceRecommendations = localTtsRecommendations.length > 0 || externalApiTtsPresets.length > 0

  React.useEffect(() => {
    setEnabled(getBooleanSetting(voice, 'enabled'))
    setSttBackend(getStringSetting(voice, 'stt_backend', 'faster-whisper'))
    setSttModel(getStringSetting(voice, 'stt_model', 'medium'))
    setSttLanguage(getStringSetting(voice, 'stt_language', 'auto'))
    setSttDevice(getStringSetting(voice, 'stt_device', 'auto'))
    setSttCacheDir(getStringSetting(voice, 'stt_model_cache_dir'))
    setSttModelPath(getStringSetting(voice, 'stt_model_path'))
    setSttApiBaseUrl(getStringSetting(voice, 'stt_openai_base_url'))
    setSttApiKey('')
    setSttApiTimeout(String(getNumberSetting(voice, 'stt_openai_timeout', 60)))
    setTtsBackend(getStringSetting(voice, 'tts_backend', 'kokoro-tts'))
    setTtsModel(getStringSetting(voice, 'tts_model', 'none') || 'none')
    setTtsVoice(getStringSetting(voice, 'tts_voice', defaultTtsVoice))
    setTtsLanguage(getStringSetting(voice, 'tts_language', 'none') || 'none')
    setTtsSpeed(String(getNumberSetting(voice, 'tts_speed', 1)))
    setTtsApiBaseUrl(getStringSetting(voice, 'tts_openai_base_url'))
    setTtsApiKey('')
    setTtsApiTimeout(String(getNumberSetting(voice, 'tts_openai_timeout', 60)))
    setTtsResponseFormat(getStringSetting(voice, 'tts_openai_response_format', 'pcm') || 'pcm')
    setReplyModelMode(
      normalizeReplyModelModeFromApi(getStringSetting(voice, 'reply_model_mode', 'inherit_active'))
    )
    setReplyModelId(getStringSetting(voice, 'reply_model_id'))
    setSessionMode(
      normalizeSessionModeFromApi(getStringSetting(voice, 'session_mode', 'append_current'))
    )
  }, [voice])

  React.useEffect(() => {
    let cancelled = false

    async function loadVoiceRuntime() {
      if (typeof settingsApi.fetchVoiceStatus !== 'function') {
        setRuntimeStatus(null)
        return
      }

      setRuntimeStatusLoading(true)
      try {
        const status = await settingsApi.fetchVoiceStatus()
        if (!cancelled) {
          setRuntimeStatus(status)
        }
      } catch (statusError) {
        if (!cancelled && !isVoiceRouteUnavailable(statusError)) {
          setRuntimeStatus({
            type: 'voice_runtime_status',
            phase: 'error',
            enabled: null,
            loaded: null,
            ready: false,
            error: messageWithDetail('Voice runtime status unavailable', statusError),
            configured: {},
            sessionDiagnostics: {},
            raw: {},
          })
        }
      } finally {
        if (!cancelled) {
          setRuntimeStatusLoading(false)
        }
      }
    }

    void loadVoiceRuntime()
    return () => {
      cancelled = true
    }
  }, [])

  const refreshVoiceCatalog = React.useCallback(async () => {
    if (typeof settingsApi.fetchVoiceCatalog !== 'function') {
      setVoiceCatalogUnsupported(true)
      setVoiceCatalog([])
      return
    }

    setVoiceCatalogLoading(true)
    try {
      const catalog = await settingsApi.fetchVoiceCatalog()
      setVoiceCatalog(catalog.voices)
      setVoiceCatalogUnsupported(false)
    } catch (catalogError) {
      if (isVoiceRouteUnavailable(catalogError)) {
        setVoiceCatalogUnsupported(true)
        setVoiceCatalog([])
      } else {
        setVoiceCatalogUnsupported(false)
        setVoicePackMessage({
          type: 'error',
          text: messageWithDetail('Failed to load voice catalog', catalogError),
        })
      }
    } finally {
      setVoiceCatalogLoading(false)
    }
  }, [])

  React.useEffect(() => {
    void refreshVoiceCatalog()
  }, [refreshVoiceCatalog])

  const handleSttBackendChange = (backend: string) => {
    setSttBackend(backend)
    setSttModel((sttModelsByBackend[backend] ?? defaultSttModelsByBackend['faster-whisper'])[0] ?? 'medium')
    setMessage(null)
  }

  const handleTtsBackendChange = (backend: string) => {
    setTtsBackend(backend)
    setTtsModel((ttsModelsByBackend[backend] ?? ['none'])[0] ?? 'none')
    setTtsVoice((ttsVoicesByBackend[backend] ?? defaultTtsVoicesByBackend['kokoro-tts'])[0] ?? defaultTtsVoice)
    setMessage(null)
  }

  const handleApplyExternalPreset = (preset: api.VoiceExternalApiTtsPreset) => {
    if (preset.apply_supported === false || preset.backend !== 'external-api') {
      return
    }

    setTtsBackend('external-api')
    setTtsModel(preset.model && preset.model.trim().length > 0 ? preset.model : 'none')
    if (preset.voice && preset.voice.trim().length > 0) {
      setTtsVoice(preset.voice)
    }
    setMessage({
      type: 'success',
      text: `Applied ${preset.label}. API URL is left unchanged so you can fill your own endpoint.`,
    })
  }

  const handleModelImport = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.currentTarget.files ?? [])
    event.currentTarget.value = ''
    if (files.length === 0) {
      return
    }

    setImportingModel(true)
    setMessage(null)

    try {
      if (typeof settingsApi.importFilesystemFiles !== 'function') {
        throw new Error('Filesystem import API client is unavailable.')
      }

      const relativePaths = files.map((file) => file.webkitRelativePath || file.name)
      const result = await settingsApi.importFilesystemFiles({
        files,
        relativePaths,
        targetDir: sttCacheDir || undefined,
        packageName: getImportPackageName(files, 'stt-model-file'),
      })

      setSttModelPath(result.importedPath)
      setMessage({
        type: 'success',
        text: t('settings.voice.importSuccess', {
          count: result.fileCount,
          bytes: formatBytes(result.totalBytes),
        }),
      })
    } catch (importError) {
      setMessage({
        type: 'error',
        text: messageWithDetail(t('settings.voice.errorImport'), importError),
      })
    } finally {
      setImportingModel(false)
    }
  }

  const handleVoicePackUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.currentTarget.files?.[0]
    event.currentTarget.value = ''
    if (!file) {
      return
    }

    setVoicePackBusy(true)
    setVoicePackMessage(null)
    try {
      if (typeof settingsApi.uploadVoicePack !== 'function') {
        throw new Error('Voice pack upload API client is unavailable.')
      }
      const result = await settingsApi.uploadVoicePack(file)
      setVoiceCatalog(result.voices)
      setVoiceCatalogUnsupported(false)
      setVoicePackMessage({
        type: 'success',
        text: `Uploaded voice pack: ${file.name}`,
      })
    } catch (uploadError) {
      if (isVoiceRouteUnavailable(uploadError)) {
        setVoiceCatalogUnsupported(true)
      }
      setVoicePackMessage({
        type: 'error',
        text: messageWithDetail('Failed to upload voice pack', uploadError),
      })
    } finally {
      setVoicePackBusy(false)
    }
  }

  const handleRegisterVoicePath = async () => {
    const path = voicePackPath.trim()
    if (!path) {
      setVoicePackMessage({
        type: 'error',
        text: 'Enter a backend voice pack path first.',
      })
      return
    }

    setVoicePackBusy(true)
    setVoicePackMessage(null)
    try {
      if (typeof settingsApi.registerVoicePackPath !== 'function') {
        throw new Error('Voice pack register-path API client is unavailable.')
      }
      const result = await settingsApi.registerVoicePackPath(path)
      setVoiceCatalog(result.voices)
      setVoiceCatalogUnsupported(false)
      setVoicePackMessage({
        type: 'success',
        text: 'Registered voice pack path.',
      })
    } catch (registerError) {
      if (isVoiceRouteUnavailable(registerError)) {
        setVoiceCatalogUnsupported(true)
      }
      setVoicePackMessage({
        type: 'error',
        text: messageWithDetail('Failed to register voice pack path', registerError),
      })
    } finally {
      setVoicePackBusy(false)
    }
  }

  const handleDeleteVoice = async (voiceId: string) => {
    setDeletingVoiceId(voiceId)
    setVoicePackMessage(null)
    try {
      if (typeof settingsApi.deleteVoice !== 'function') {
        throw new Error('Delete voice API client is unavailable.')
      }
      const result = await settingsApi.deleteVoice(voiceId)
      setVoiceCatalog(result.voices)
      setVoicePackMessage({
        type: 'success',
        text: `Deleted voice: ${voiceId}`,
      })
    } catch (deleteError) {
      if (isVoiceRouteUnavailable(deleteError)) {
        setVoiceCatalogUnsupported(true)
      }
      setVoicePackMessage({
        type: 'error',
        text: messageWithDetail('Failed to delete voice', deleteError),
      })
    } finally {
      setDeletingVoiceId(null)
    }
  }

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setSubmitting(true)
    setMessage(null)

    try {
      if (typeof settingsApi.updateSettings !== 'function') {
        throw new Error('Settings update API client is unavailable.')
      }

      const settings = await settingsApi.updateSettings({
        voice: {
          enabled,
          stt_backend: sttBackend,
          stt_model: sttModel,
          stt_language: sttLanguage,
          stt_device: sttDevice,
          stt_model_cache_dir: sttCacheDir,
          stt_model_path: sttModelPath,
          stt_openai_base_url: isExternalSttBackend(sttBackend) ? (sttApiBaseUrl.trim() || null) : null,
          stt_openai_api_key: isExternalSttBackend(sttBackend) ? (sttApiKey.trim() || undefined) : undefined,
          stt_openai_timeout: isExternalSttBackend(sttBackend)
            ? (Number.parseFloat(sttApiTimeout) || 60)
            : undefined,
          tts_backend: ttsBackend,
          tts_model: ttsModel === 'none' ? null : ttsModel,
          tts_voice: ttsVoice,
          tts_language: ttsLanguage === 'none' ? null : ttsLanguage,
          tts_speed: Number.parseFloat(ttsSpeed) || 1,
          tts_openai_base_url: isExternalTtsBackend(ttsBackend) ? (ttsApiBaseUrl.trim() || null) : null,
          tts_openai_api_key: isExternalTtsBackend(ttsBackend) ? (ttsApiKey.trim() || undefined) : undefined,
          tts_openai_timeout: isExternalTtsBackend(ttsBackend)
            ? (Number.parseFloat(ttsApiTimeout) || 60)
            : undefined,
          tts_openai_response_format: isExternalTtsBackend(ttsBackend)
            ? (ttsResponseFormat === 'wav' ? 'wav' : 'pcm')
            : undefined,
          reply_model_mode: normalizeReplyModelModeForApi(replyModelMode),
          reply_model_id: replyModelMode === 'fixed' ? (replyModelId.trim() || null) : null,
          session_mode: normalizeSessionModeForApi(sessionMode),
        },
        download_missing_models: downloadMissing,
        reload_voice: true,
      })
      onUpdated(settings)
      if (typeof settingsApi.fetchVoiceStatus === 'function') {
        try {
          setRuntimeStatus(await settingsApi.fetchVoiceStatus())
        } catch {
          // Keep form-success flow even when status route is unavailable.
        }
      }
      const download = asRecord(settings.update)?.download
      const status = asRecord(download)?.status
      const downloadMessage =
        getStringSetting(asRecord(download) ?? {}, 'message') ||
        getStringSetting(asRecord(asRecord(download)?.tts) ?? {}, 'error') ||
        getStringSetting(asRecord(asRecord(download)?.stt) ?? {}, 'error')
      const needsAttention = String(status ?? '') === 'attention_required'
      setMessage({
        type: needsAttention ? 'error' : 'success',
        text: needsAttention
          ? `Voice settings saved, but model preparation needs attention: ${downloadMessage || String(status)}`
          : status
            ? t('settings.voice.saveSuccessWithStatus', { status: String(status) })
            : t('settings.voice.saveSuccess'),
      })
    } catch (updateError) {
      setMessage({
        type: 'error',
        text: messageWithDetail(t('settings.voice.errorSave'), updateError),
      })
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <section className="rounded-lg border border-border bg-surface-layer">
      <div className="border-b border-border px-4 py-3">
        <div className="flex items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold text-foreground">{t('settings.voice.title')}</h3>
            <p className="mt-0.5 text-xs text-muted-foreground">{t('settings.voice.description')}</p>
          </div>
          <Mic className="h-4 w-4 text-muted-foreground" />
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4 px-4 py-4">
        <input
          ref={modelFileInputRef}
          type="file"
          className="hidden"
          onChange={(event) => void handleModelImport(event)}
        />
        <input
          ref={voicePackInputRef}
          type="file"
          className="hidden"
          onChange={(event) => void handleVoicePackUpload(event)}
        />
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-md border border-border bg-canvas px-3 py-2">
          <div>
            <p className="text-sm font-medium text-foreground">{t('settings.voice.enable')}</p>
            <p className="text-xs text-muted-foreground">{t('settings.voice.enableHelp')}</p>
          </div>
          <Switch checked={enabled} onCheckedChange={setEnabled} />
        </div>

        <div className="rounded-md border border-border bg-canvas p-3">
          <div className="mb-3 flex items-center justify-between gap-3">
            <p className="text-sm font-semibold text-foreground">{t('settings.voice.sttTitle')}</p>
            <Badge variant="neutral">{sttBackend}</Badge>
          </div>
          <div className="grid grid-cols-1 gap-3 lg:grid-cols-2 xl:grid-cols-4">
            <label className="min-w-0 space-y-1.5">
              <SettingLabel>{t('settings.form.backend')}</SettingLabel>
              <SelectSetting
                value={sttBackend}
                onValueChange={handleSttBackendChange}
                options={sttOptions}
                getOptionLabel={voiceBackendOptionLabel}
              />
            </label>
            {isExternalSttBackend(sttBackend) ? (
              <label className="min-w-0 space-y-1.5 lg:col-span-1 xl:col-span-1">
                <SettingLabel>{t('settings.form.model')}</SettingLabel>
                <Input
                  value={sttModel}
                  onChange={(event) => setSttModel(event.target.value)}
                  placeholder="whisper-1"
                  className="min-w-0 font-mono text-xs"
                />
              </label>
            ) : (
              <label className="min-w-0 space-y-1.5 lg:col-span-1 xl:col-span-1">
                <SettingLabel>{t('settings.form.model')}</SettingLabel>
                <SelectSetting value={sttModel} onValueChange={setSttModel} options={sttModelOptions} className="font-mono text-xs" />
              </label>
            )}
            <label className="min-w-0 space-y-1.5">
              <SettingLabel>{t('settings.form.language')}</SettingLabel>
              <SelectSetting value={sttLanguage} onValueChange={setSttLanguage} options={withCurrentOption(sttLanguageOptions, sttLanguage)} />
            </label>
            <label className="min-w-0 space-y-1.5">
              <SettingLabel>{t('settings.form.device')}</SettingLabel>
              <SelectSetting value={sttDevice} onValueChange={setSttDevice} options={withCurrentOption(sttDeviceOptions, sttDevice)} />
            </label>
          </div>
        </div>

        {isExternalSttBackend(sttBackend) ? (
          <div className="grid grid-cols-1 gap-3 xl:grid-cols-3">
            <label className="min-w-0 space-y-1.5 xl:col-span-2">
              <SettingLabel>{t('settings.form.apiUrl')}</SettingLabel>
              <Input
                value={sttApiBaseUrl}
                onChange={(event) => setSttApiBaseUrl(event.target.value)}
                placeholder="https://api.example.com/v1"
                className="min-w-0 font-mono text-xs"
              />
            </label>
            <label className="min-w-0 space-y-1.5">
              <SettingLabel>Timeout (seconds)</SettingLabel>
              <Input
                value={sttApiTimeout}
                onChange={(event) => setSttApiTimeout(event.target.value)}
                className="min-w-0 font-mono text-xs"
              />
            </label>
            <label className="min-w-0 space-y-1.5 xl:col-span-3">
              <SettingLabel>{t('settings.form.apiKey')}</SettingLabel>
              <Input
                value={sttApiKey}
                onChange={(event) => setSttApiKey(event.target.value)}
                placeholder={getBooleanSetting(voice, 'stt_openai_api_key_configured') ? 'Leave blank to keep existing key' : 'sk-...'}
                className="min-w-0 font-mono text-xs"
              />
            </label>
          </div>
        ) : null}

        <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
          <div className="min-w-0 space-y-1.5">
            <SettingLabel>{t('settings.voice.sttCacheDir')}</SettingLabel>
            <p className="text-xs text-muted-foreground">
              {t('settings.voice.sttCacheDirHelp')}
            </p>
            <Input
              value={sttCacheDir}
              onChange={(event) => setSttCacheDir(event.target.value)}
              placeholder={t('settings.voice.placeholder.sttCacheDir')}
              className="min-w-0 font-mono text-xs"
            />
          </div>
          <div className="min-w-0 space-y-1.5">
            <SettingLabel>{t('settings.voice.sttModelPath')}</SettingLabel>
            <p className="text-xs text-muted-foreground">
              {t('settings.voice.sttModelPathHelp')}
            </p>
            <Input
              value={sttModelPath}
              onChange={(event) => setSttModelPath(event.target.value)}
              placeholder={t('settings.voice.placeholder.sttModelPath')}
              className="min-w-0 font-mono text-xs"
            />
            <div className="flex flex-wrap gap-2">
              <Button
                type="button"
                variant="secondary"
                size="sm"
                loading={importingModel}
                onClick={() => modelFileInputRef.current?.click()}
              >
                <Upload className="h-3.5 w-3.5" />
                {t('settings.action.importLocalModel')}
              </Button>
            </div>
          </div>
        </div>

        <div className="rounded-md border border-border bg-canvas p-3">
          <div className="mb-3 flex items-center justify-between gap-3">
            <p className="text-sm font-semibold text-foreground">{t('settings.voice.ttsTitle')}</p>
            <Badge variant="neutral">{ttsBackend}</Badge>
          </div>
          <div className="grid grid-cols-1 gap-3 lg:grid-cols-2 xl:grid-cols-5">
            <label className="min-w-0 space-y-1.5">
              <SettingLabel>{t('settings.form.backend')}</SettingLabel>
              <SelectSetting
                value={ttsBackend}
                onValueChange={handleTtsBackendChange}
                options={ttsOptions}
                getOptionLabel={voiceBackendOptionLabel}
              />
            </label>
            {isExternalTtsBackend(ttsBackend) ? (
              <label className="min-w-0 space-y-1.5 xl:col-span-2">
                <SettingLabel>{t('settings.form.model')}</SettingLabel>
                <Input
                  value={ttsModel === 'none' ? '' : ttsModel}
                  onChange={(event) => setTtsModel(event.target.value || 'none')}
                  placeholder="gpt-4o-mini-tts"
                  className="min-w-0 font-mono text-xs"
                />
              </label>
            ) : (
              <label className="min-w-0 space-y-1.5 xl:col-span-2">
                <SettingLabel>{t('settings.form.model')}</SettingLabel>
                <SelectSetting value={ttsModel} onValueChange={setTtsModel} options={ttsModelOptions} className="font-mono text-xs" />
              </label>
            )}
            {isExternalTtsBackend(ttsBackend) ? (
              <label className="min-w-0 space-y-1.5 xl:col-span-2">
                <SettingLabel>{t('settings.form.voice')}</SettingLabel>
                <Input
                  value={ttsVoice}
                  onChange={(event) => setTtsVoice(event.target.value)}
                  placeholder="alloy"
                  className="min-w-0 font-mono text-xs"
                />
              </label>
            ) : (
              <label className="min-w-0 space-y-1.5 xl:col-span-2">
                <SettingLabel>{t('settings.form.voice')}</SettingLabel>
                <SelectSetting value={ttsVoice} onValueChange={setTtsVoice} options={ttsVoiceOptions} className="font-mono text-xs" />
              </label>
            )}
            <label className="min-w-0 space-y-1.5">
              <SettingLabel>{t('settings.form.language')}</SettingLabel>
              <SelectSetting value={ttsLanguage} onValueChange={setTtsLanguage} options={withCurrentOption(ttsLanguageOptions, ttsLanguage)} />
            </label>
            <label className="min-w-0 space-y-1.5">
              <SettingLabel>{t('settings.form.speed')}</SettingLabel>
              <SelectSetting value={ttsSpeed} onValueChange={setTtsSpeed} options={withCurrentOption(ttsSpeedOptions, ttsSpeed)} />
            </label>
          </div>
        </div>

        {isExternalTtsBackend(ttsBackend) ? (
          <div className="grid grid-cols-1 gap-3 xl:grid-cols-4">
            <label className="min-w-0 space-y-1.5 xl:col-span-2">
              <SettingLabel>{t('settings.form.apiUrl')}</SettingLabel>
              <Input
                value={ttsApiBaseUrl}
                onChange={(event) => setTtsApiBaseUrl(event.target.value)}
                placeholder="https://api.example.com/v1"
                className="min-w-0 font-mono text-xs"
              />
            </label>
            <label className="min-w-0 space-y-1.5">
              <SettingLabel>Timeout (seconds)</SettingLabel>
              <Input
                value={ttsApiTimeout}
                onChange={(event) => setTtsApiTimeout(event.target.value)}
                className="min-w-0 font-mono text-xs"
              />
            </label>
            <label className="min-w-0 space-y-1.5">
              <SettingLabel>Response format</SettingLabel>
              <SelectSetting
                value={ttsResponseFormat}
                onValueChange={setTtsResponseFormat}
                options={['pcm', 'wav']}
              />
            </label>
            <label className="min-w-0 space-y-1.5 xl:col-span-4">
              <SettingLabel>{t('settings.form.apiKey')}</SettingLabel>
              <Input
                value={ttsApiKey}
                onChange={(event) => setTtsApiKey(event.target.value)}
                placeholder={getBooleanSetting(voice, 'tts_openai_api_key_configured') ? 'Leave blank to keep existing key' : 'sk-...'}
                className="min-w-0 font-mono text-xs"
              />
            </label>
          </div>
        ) : null}

        {hasVoiceRecommendations ? (
          <div className="rounded-md border border-border bg-canvas p-3">
            <div className="mb-3 flex items-center justify-between gap-3">
              <p className="text-sm font-semibold text-foreground">Voice recommendations</p>
              <Badge variant="neutral">
                {localTtsRecommendations.length + externalApiTtsPresets.length} presets
              </Badge>
            </div>
            <div className="space-y-3">
              {localTtsRecommendations.length > 0 ? (
                <div className="space-y-2">
                  <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Local backends</p>
                  {localTtsRecommendations.map((recommendation) => (
                    <div key={recommendation.id} className="rounded-md border border-border bg-surface-layer px-3 py-2">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="text-sm font-medium text-foreground">{recommendation.label}</span>
                        <Badge variant="neutral">{recommendation.backend}</Badge>
                      </div>
                      {recommendation.summary ? (
                        <p className="mt-1 text-xs text-muted-foreground">{recommendation.summary}</p>
                      ) : null}
                      <p className="mt-1 font-mono text-[11px] text-muted-foreground">
                        voice={recommendation.default_voice ?? 'none'}
                        {recommendation.default_model ? ` | model=${recommendation.default_model}` : ''}
                      </p>
                    </div>
                  ))}
                </div>
              ) : null}

              {externalApiTtsPresets.length > 0 ? (
                <div className="space-y-2">
                  <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">External API presets</p>
                  {externalApiTtsPresets.map((preset) => {
                    const compatibility = preset.compatibility ?? 'openai-compatible'
                    const canApply = preset.apply_supported !== false && preset.backend === 'external-api'

                    return (
                      <div key={preset.id} className="rounded-md border border-border bg-surface-layer px-3 py-2">
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <div className="flex min-w-0 items-center gap-2">
                            <span className="truncate text-sm font-medium text-foreground">{preset.label}</span>
                            <Badge variant={canApply ? 'success' : 'warning'}>{compatibility}</Badge>
                          </div>
                          <Button
                            type="button"
                            variant="secondary"
                            size="sm"
                            disabled={!canApply}
                            onClick={() => handleApplyExternalPreset(preset)}
                          >
                            {canApply ? 'Apply' : 'Adapter required'}
                          </Button>
                        </div>
                        {preset.summary ? (
                          <p className="mt-1 text-xs text-muted-foreground">{preset.summary}</p>
                        ) : null}
                        {preset.model ? (
                          <p className="mt-1 font-mono text-[11px] text-muted-foreground">model={preset.model}</p>
                        ) : null}
                      </div>
                    )
                  })}
                </div>
              ) : null}
            </div>
          </div>
        ) : null}

        <div className="rounded-md border border-border bg-canvas p-3">
          <div className="mb-3 flex items-center justify-between gap-3">
            <p className="text-sm font-semibold text-foreground">{t('settings.voice.replyModelTitle')}</p>
            <Badge variant="neutral">{replyModelMode}</Badge>
          </div>
          <div className="grid grid-cols-1 gap-3 xl:grid-cols-3">
            <label className="min-w-0 space-y-1.5">
              <SettingLabel>{t('settings.voice.replyModelMode')}</SettingLabel>
              <SelectSetting
                value={replyModelMode}
                onValueChange={(value) => setReplyModelMode(normalizeReplyModelModeFromApi(value))}
                options={replyModelModeOptionsWithCurrent}
              />
            </label>
            <label className="min-w-0 space-y-1.5 xl:col-span-2">
              <SettingLabel>{t('settings.voice.replyModelId')}</SettingLabel>
              <Input
                value={replyModelId}
                onChange={(event) => setReplyModelId(event.target.value)}
                placeholder="e.g. ollama:qwen3:8b"
                className="min-w-0 font-mono text-xs"
                disabled={replyModelMode !== 'fixed'}
              />
            </label>
          </div>
          <p className="mt-2 text-xs text-muted-foreground">
            {t('settings.voice.replyModelHelp')}
          </p>
        </div>

        <div className="rounded-md border border-border bg-canvas p-3">
          <div className="mb-3 flex items-center justify-between gap-3">
            <p className="text-sm font-semibold text-foreground">{t('settings.voice.sessionModeTitle')}</p>
            <Badge variant="neutral">{sessionMode}</Badge>
          </div>
          <label className="min-w-0 space-y-1.5">
            <SettingLabel>{t('settings.voice.sessionMode')}</SettingLabel>
            <SelectSetting
              value={sessionMode}
              onValueChange={(value) => setSessionMode(normalizeSessionModeFromApi(value))}
              options={sessionModeOptionsWithCurrent}
            />
          </label>
          <p className="mt-2 text-xs text-muted-foreground">
            {t('settings.voice.sessionModeHelp')}
          </p>
        </div>

        <div className="rounded-md border border-border bg-canvas p-3">
          <div className="mb-3 flex items-center justify-between gap-3">
            <p className="text-sm font-semibold text-foreground">{t('settings.voice.voicePacksTitle')}</p>
            {voiceCatalogLoading ? (
              <Badge variant="neutral">{t('common.loading')}</Badge>
            ) : (
              <Badge variant={voiceCatalogUnsupported ? 'warning' : 'neutral'}>
                {voiceCatalogUnsupported
                  ? t('settings.voice.voicePacksUnavailableBadge')
                  : t('settings.voice.voicePacksCount', { count: voiceCatalog.length })}
              </Badge>
            )}
          </div>

          <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
            <div className="space-y-1.5">
              <SettingLabel>{t('settings.voice.voicePacksUpload')}</SettingLabel>
              <div className="flex flex-wrap gap-2">
                <Button
                  type="button"
                  variant="secondary"
                  size="sm"
                  loading={voicePackBusy}
                  onClick={() => voicePackInputRef.current?.click()}
                  disabled={voiceCatalogUnsupported}
                >
                  <Upload className="h-3.5 w-3.5" />
                  {t('settings.voice.voicePacksUpload')}
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => void refreshVoiceCatalog()}
                  disabled={voiceCatalogLoading}
                >
                  <RefreshCw className="h-3.5 w-3.5" />
                  {t('settings.voice.voicePacksRefresh')}
                </Button>
              </div>
            </div>
            <div className="space-y-1.5">
              <SettingLabel>{t('settings.voice.voicePacksPathLabel')}</SettingLabel>
              <PathPicker
                value={voicePackPath}
                onChange={setVoicePackPath}
                mode="file_or_directory"
                placeholder={t('settings.voice.voicePacksPathPlaceholder')}
                inputClassName="font-mono text-xs"
                disabled={voicePackBusy || voiceCatalogUnsupported}
              />
              <div className="flex justify-end">
                <Button
                  type="button"
                  variant="secondary"
                  size="sm"
                  loading={voicePackBusy}
                  onClick={() => void handleRegisterVoicePath()}
                  disabled={voiceCatalogUnsupported}
                >
                  <Save className="h-3.5 w-3.5" />
                  {t('settings.voice.voicePacksRegisterPath')}
                </Button>
              </div>
            </div>
          </div>

          {voiceCatalogUnsupported ? (
            <p className="mt-3 text-xs text-muted-foreground">
              {t('settings.voice.voicePacksUnavailable')}
            </p>
          ) : null}

          {!voiceCatalogUnsupported && voiceCatalog.length > 0 ? (
            <div className="mt-3 space-y-2">
              {voiceCatalog.map((catalogVoice) => (
                <div
                  key={catalogVoice.id}
                  className="flex items-center justify-between gap-3 rounded-md border border-border bg-surface-layer px-3 py-2"
                >
                  <div className="min-w-0">
                    <p className="truncate text-sm font-medium text-foreground">
                      {catalogVoice.name}
                    </p>
                    <p className="truncate font-mono text-xs text-muted-foreground">
                      {catalogVoice.id}
                      {catalogVoice.path ? ` · ${catalogVoice.path}` : ''}
                    </p>
                  </div>
                  <div className="flex shrink-0 items-center gap-2">
                    <Badge variant={catalogVoice.isBuiltin ? 'success' : 'neutral'}>
                      {catalogVoice.isBuiltin ? 'builtin' : (catalogVoice.source ?? 'custom')}
                    </Badge>
                    {!catalogVoice.isBuiltin ? (
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        loading={deletingVoiceId === catalogVoice.id}
                        onClick={() => void handleDeleteVoice(catalogVoice.id)}
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </Button>
                    ) : null}
                  </div>
                </div>
              ))}
            </div>
          ) : null}

          <SettingMessage message={voicePackMessage} />
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3 rounded-md border border-border bg-canvas px-3 py-2">
          <div className="flex items-center gap-2">
            <Download className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm text-foreground">{t('settings.voice.downloadMissing')}</span>
          </div>
          <Switch checked={downloadMissing} onCheckedChange={setDownloadMissing} />
        </div>

        <SettingMessage message={message} />
        {runtimeStatusLoading ? (
          <p className="text-xs text-muted-foreground">{t('settings.voice.runtimeLoading')}</p>
        ) : runtimeStatus ? (
          <div className="rounded-md border border-border bg-canvas px-3 py-2">
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <Badge variant={runtimeStatus.ready ? 'success' : 'warning'}>
                {runtimeStatus.ready ? 'ready' : (runtimeStatus.phase ?? 'unknown')}
              </Badge>
              <span className="text-muted-foreground">
                {t('settings.voice.runtimeLoaded')}: {runtimeStatus.loaded === null ? 'n/a' : String(runtimeStatus.loaded)}
              </span>
              <span className="text-muted-foreground">
                {t('settings.voice.runtimeEnabled')}: {runtimeStatus.enabled === null ? 'n/a' : String(runtimeStatus.enabled)}
              </span>
              {runtimeStatus.error ? (
                <span className="text-destructive">{runtimeStatus.error}</span>
              ) : null}
            </div>
          </div>
        ) : null}

        <div className="flex justify-end">
          <Button type="submit" variant="primary" size="sm" loading={submitting}>
            <Save className="h-3.5 w-3.5" />
            {t('settings.action.saveVoice')}
          </Button>
        </div>
      </form>
    </section>
  )
}

function MemoryStorageForm({
  memory,
  onUpdated,
}: {
  memory: Record<string, unknown>
  onUpdated: (settings: api.Settings) => void
}) {
  const { t } = useI18n()
  const [dbPath, setDbPath] = React.useState(getStringSetting(memory, 'db_path'))
  const [maxShortTerm, setMaxShortTerm] = React.useState(String(getNumberSetting(memory, 'max_short_term_messages', 50)))
  const [ftsTopK, setFtsTopK] = React.useState(String(getNumberSetting(memory, 'fts_top_k', 5)))
  const [submitting, setSubmitting] = React.useState(false)
  const [message, setMessage] = React.useState<FormMessage>(null)

  React.useEffect(() => {
    setDbPath(getStringSetting(memory, 'db_path'))
    setMaxShortTerm(String(getNumberSetting(memory, 'max_short_term_messages', 50)))
    setFtsTopK(String(getNumberSetting(memory, 'fts_top_k', 5)))
  }, [memory])

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setSubmitting(true)
    setMessage(null)

    try {
      if (typeof settingsApi.updateSettings !== 'function') {
        throw new Error('Settings update API client is unavailable.')
      }
      const settings = await settingsApi.updateSettings({
        memory: {
          db_path: dbPath,
          max_short_term_messages: Number.parseInt(maxShortTerm, 10) || 50,
          fts_top_k: Number.parseInt(ftsTopK, 10) || 5,
        },
      })
      onUpdated(settings)
      setMessage({ type: 'success', text: t('settings.memory.successSaved') })
    } catch (updateError) {
      setMessage({
        type: 'error',
        text: messageWithDetail(t('settings.memory.errorSave'), updateError),
      })
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <section className="rounded-lg border border-border bg-surface-layer">
      <div className="border-b border-border px-4 py-3">
        <div className="flex items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold text-foreground">{t('settings.memory.title')}</h3>
            <p className="mt-0.5 text-xs text-muted-foreground">{t('settings.memory.description')}</p>
          </div>
          <Database className="h-4 w-4 text-muted-foreground" />
        </div>
      </div>
      <form onSubmit={handleSubmit} className="space-y-4 px-4 py-4">
        <label className="space-y-1.5">
          <SettingLabel>{t('settings.memory.dbPath')}</SettingLabel>
          <p className="text-xs text-muted-foreground">
            {t('settings.memory.dbPathHelp')}
          </p>
          <Input
            value={dbPath}
            onChange={(event) => setDbPath(event.target.value)}
            placeholder={t('settings.memory.placeholder.dbPath')}
            className="min-w-0 font-mono text-xs"
          />
        </label>
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          <label className="space-y-1.5">
            <SettingLabel>{t('settings.memory.shortTermMessages')}</SettingLabel>
            <Input value={maxShortTerm} onChange={(event) => setMaxShortTerm(event.target.value)} className="font-mono text-xs" />
          </label>
          <label className="space-y-1.5">
            <SettingLabel>FTS Top K</SettingLabel>
            <Input value={ftsTopK} onChange={(event) => setFtsTopK(event.target.value)} className="font-mono text-xs" />
          </label>
        </div>
        <SettingMessage message={message} />
        <div className="flex justify-end">
          <Button type="submit" variant="primary" size="sm" loading={submitting}>
            <Save className="h-3.5 w-3.5" />
            {t('settings.action.saveMemory')}
          </Button>
        </div>
      </form>
    </section>
  )
}

function LearningStorageForm({
  learning,
  paths,
  onUpdated,
}: {
  learning: Record<string, unknown>
  paths: Record<string, unknown>
  onUpdated: (settings: api.Settings) => void
}) {
  const { t } = useI18n()
  const [enabled, setEnabled] = React.useState(getBooleanSetting(learning, 'enabled', true))
  const [autoExtract, setAutoExtract] = React.useState(getBooleanSetting(learning, 'auto_extract_skills', true))
  const [autoSyncFilesystem, setAutoSyncFilesystem] = React.useState(getBooleanSetting(learning, 'auto_sync_filesystem_skills', true))
  const [minSteps, setMinSteps] = React.useState(String(getNumberSetting(learning, 'min_steps_for_extraction', 3)))
  const [minToolCalls, setMinToolCalls] = React.useState(String(getNumberSetting(learning, 'min_tool_calls_for_extraction', 2)))
  const [retentionDays, setRetentionDays] = React.useState(String(getNumberSetting(learning, 'trajectory_retention_days', 30)))
  const [threshold, setThreshold] = React.useState(String(getNumberSetting(learning, 'skill_improvement_threshold', 0.7)))
  const [maxSkills, setMaxSkills] = React.useState(String(getNumberSetting(learning, 'max_skills', 500)))
  const [workspaceDir, setWorkspaceDir] = React.useState(getStringSetting(paths, 'workspace_dir', '~/.mochi'))
  const [sessionsDir, setSessionsDir] = React.useState(getStringSetting(paths, 'sessions_dir', '~/.mochi/sessions'))
  const [skillsDir, setSkillsDir] = React.useState(getStringSetting(paths, 'skills_dir', '~/.mochi/skills'))
  const [pluginsDir, setPluginsDir] = React.useState(getStringSetting(paths, 'plugins_dir', '~/.mochi/plugins'))
  const [submitting, setSubmitting] = React.useState(false)
  const [message, setMessage] = React.useState<FormMessage>(null)

  React.useEffect(() => {
    setEnabled(getBooleanSetting(learning, 'enabled', true))
    setAutoExtract(getBooleanSetting(learning, 'auto_extract_skills', true))
    setAutoSyncFilesystem(getBooleanSetting(learning, 'auto_sync_filesystem_skills', true))
    setMinSteps(String(getNumberSetting(learning, 'min_steps_for_extraction', 3)))
    setMinToolCalls(String(getNumberSetting(learning, 'min_tool_calls_for_extraction', 2)))
    setRetentionDays(String(getNumberSetting(learning, 'trajectory_retention_days', 30)))
    setThreshold(String(getNumberSetting(learning, 'skill_improvement_threshold', 0.7)))
    setMaxSkills(String(getNumberSetting(learning, 'max_skills', 500)))
    setWorkspaceDir(getStringSetting(paths, 'workspace_dir', '~/.mochi'))
    setSessionsDir(getStringSetting(paths, 'sessions_dir', '~/.mochi/sessions'))
    setSkillsDir(getStringSetting(paths, 'skills_dir', '~/.mochi/skills'))
    setPluginsDir(getStringSetting(paths, 'plugins_dir', '~/.mochi/plugins'))
  }, [learning, paths])

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setSubmitting(true)
    setMessage(null)

    try {
      if (typeof settingsApi.updateSettings !== 'function') {
        throw new Error('Settings update API client is unavailable.')
      }
      const settings = await settingsApi.updateSettings({
        learning: {
          enabled,
          auto_extract_skills: autoExtract,
          auto_sync_filesystem_skills: autoSyncFilesystem,
          min_steps_for_extraction: Number.parseInt(minSteps, 10) || 3,
          min_tool_calls_for_extraction: Number.parseInt(minToolCalls, 10) || 2,
          trajectory_retention_days: Number.parseInt(retentionDays, 10) || 30,
          skill_improvement_threshold: Number.parseFloat(threshold) || 0.7,
          max_skills: Number.parseInt(maxSkills, 10) || 500,
        },
        paths: {
          workspace_dir: workspaceDir,
          sessions_dir: sessionsDir,
          skills_dir: skillsDir,
          plugins_dir: pluginsDir,
        },
      })
      onUpdated(settings)
      setMessage({ type: 'success', text: t('settings.learning.successSaved') })
    } catch (updateError) {
      setMessage({
        type: 'error',
        text: messageWithDetail(t('settings.learning.errorSave'), updateError),
      })
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <section className="rounded-lg border border-border bg-surface-layer">
      <div className="border-b border-border px-4 py-3">
        <div className="flex items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold text-foreground">{t('settings.learning.title')}</h3>
            <p className="mt-0.5 text-xs text-muted-foreground">{t('settings.learning.description')}</p>
          </div>
          <BrainCircuit className="h-4 w-4 text-muted-foreground" />
        </div>
      </div>
      <form onSubmit={handleSubmit} className="space-y-4 px-4 py-4">
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          <div className="flex items-center justify-between gap-3 rounded-md border border-border bg-canvas px-3 py-2">
            <span className="text-sm text-foreground">{t('settings.learning.enable')}</span>
            <Switch checked={enabled} onCheckedChange={setEnabled} />
          </div>
          <div className="flex items-center justify-between gap-3 rounded-md border border-border bg-canvas px-3 py-2">
            <span className="text-sm text-foreground">{t('settings.learning.autoExtract')}</span>
            <Switch checked={autoExtract} onCheckedChange={setAutoExtract} />
          </div>
          <div className="flex items-center justify-between gap-3 rounded-md border border-border bg-canvas px-3 py-2">
            <span className="text-sm text-foreground">{t('settings.learning.autoSyncFilesystem')}</span>
            <Switch checked={autoSyncFilesystem} onCheckedChange={setAutoSyncFilesystem} />
          </div>
        </div>

        <div className="grid grid-cols-1 gap-3 xl:grid-cols-5">
          <label className="space-y-1.5">
            <SettingLabel>{t('settings.learning.minSteps')}</SettingLabel>
            <Input value={minSteps} onChange={(event) => setMinSteps(event.target.value)} className="font-mono text-xs" />
          </label>
          <label className="space-y-1.5">
            <SettingLabel>{t('settings.learning.minToolCalls')}</SettingLabel>
            <Input value={minToolCalls} onChange={(event) => setMinToolCalls(event.target.value)} className="font-mono text-xs" />
          </label>
          <label className="space-y-1.5">
            <SettingLabel>{t('settings.learning.retentionDays')}</SettingLabel>
            <Input value={retentionDays} onChange={(event) => setRetentionDays(event.target.value)} className="font-mono text-xs" />
          </label>
          <label className="space-y-1.5">
            <SettingLabel>{t('settings.learning.improvementThreshold')}</SettingLabel>
            <Input value={threshold} onChange={(event) => setThreshold(event.target.value)} className="font-mono text-xs" />
          </label>
          <label className="space-y-1.5">
            <SettingLabel>{t('settings.learning.maxSkills')}</SettingLabel>
            <Input value={maxSkills} onChange={(event) => setMaxSkills(event.target.value)} className="font-mono text-xs" />
          </label>
        </div>

        <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
          <div className="min-w-0 space-y-1.5">
            <SettingLabel>{t('settings.learning.workspaceDir')}</SettingLabel>
            <Input
              value={workspaceDir}
              onChange={(event) => setWorkspaceDir(event.target.value)}
              placeholder={t('settings.learning.placeholder.workspaceDir')}
              className="min-w-0 font-mono text-xs"
            />
          </div>
          <div className="min-w-0 space-y-1.5">
            <SettingLabel>{t('settings.learning.skillsDir')}</SettingLabel>
            <Input
              value={skillsDir}
              onChange={(event) => setSkillsDir(event.target.value)}
              placeholder={t('settings.learning.placeholder.skillsDir')}
              className="min-w-0 font-mono text-xs"
            />
          </div>
          <div className="min-w-0 space-y-1.5">
            <SettingLabel>{t('settings.learning.sessionsDir')}</SettingLabel>
            <Input
              value={sessionsDir}
              onChange={(event) => setSessionsDir(event.target.value)}
              placeholder={t('settings.learning.placeholder.sessionsDir')}
              className="min-w-0 font-mono text-xs"
            />
          </div>
          <div className="min-w-0 space-y-1.5">
            <SettingLabel>{t('settings.learning.pluginsDir')}</SettingLabel>
            <Input
              value={pluginsDir}
              onChange={(event) => setPluginsDir(event.target.value)}
              placeholder={t('settings.learning.placeholder.pluginsDir')}
              className="min-w-0 font-mono text-xs"
            />
          </div>
        </div>

        <SettingMessage message={message} />

        <div className="flex justify-end">
          <Button type="submit" variant="primary" size="sm" loading={submitting}>
            <Save className="h-3.5 w-3.5" />
            {t('settings.action.saveLearning')}
          </Button>
        </div>
      </form>
    </section>
  )
}

function SecuritySettingsForm({
  security,
  onUpdated,
}: {
  security: api.SecuritySettings | undefined
  onUpdated: (settings: api.Settings) => void
}) {
  const { t } = useI18n()
  const [autonomyMode, setAutonomyMode] = React.useState<api.SecuritySettings['autonomy_mode']>(
    security?.autonomy_mode ?? 'trusted_workspace'
  )
  const [requireShellApproval, setRequireShellApproval] = React.useState(security?.require_approval_for_shell ?? true)
  const [requireFileWriteApproval, setRequireFileWriteApproval] = React.useState(security?.require_approval_for_file_write ?? false)
  const [requireExecApproval, setRequireExecApproval] = React.useState(security?.require_approval_for_exec ?? true)
  const [agentRunDefaultMaxWallClockSec, setAgentRunDefaultMaxWallClockSec] = React.useState(
    security?.agent_run_default_max_wall_clock_sec == null ? '' : String(security.agent_run_default_max_wall_clock_sec)
  )
  const [agentRunDefaultHeartbeatTimeoutSec, setAgentRunDefaultHeartbeatTimeoutSec] = React.useState(
    security?.agent_run_default_heartbeat_timeout_sec == null ? '' : String(security.agent_run_default_heartbeat_timeout_sec)
  )
  const [agentRunDefaultCheckpointIntervalSteps, setAgentRunDefaultCheckpointIntervalSteps] = React.useState(
    String(security?.agent_run_default_checkpoint_interval_steps ?? 1)
  )
  const [agentRunDefaultMaxSubagentFailuresPerRole, setAgentRunDefaultMaxSubagentFailuresPerRole] = React.useState(
    String(security?.agent_run_default_max_subagent_failures_per_role ?? 2)
  )
  const [agentRunDefaultOnBudgetExhausted, setAgentRunDefaultOnBudgetExhausted] = React.useState<
    api.SecuritySettings['agent_run_default_on_budget_exhausted']
  >(security?.agent_run_default_on_budget_exhausted ?? 'pause')
  const [agentRunDefaultOnSubagentDisconnect, setAgentRunDefaultOnSubagentDisconnect] = React.useState<
    api.SecuritySettings['agent_run_default_on_subagent_disconnect']
  >(security?.agent_run_default_on_subagent_disconnect ?? 'retry_then_degrade')
  const [execDefaultTimeoutSec, setExecDefaultTimeoutSec] = React.useState(String(security?.exec_default_timeout_sec ?? 30))
  const [execSessionOutputLimit, setExecSessionOutputLimit] = React.useState(String(security?.exec_session_output_limit ?? 8000))
  const [fileOpsScope, setFileOpsScope] = React.useState<'workspace' | 'any'>(security?.file_ops_scope ?? 'workspace')
  const [maxFileWriteSizeMb, setMaxFileWriteSizeMb] = React.useState(String(security?.max_file_write_size_mb ?? 10.0))
  const [fileUndoMaxSizeMb, setFileUndoMaxSizeMb] = React.useState(String(security?.file_undo_max_size_mb ?? 2.0))
  const [submitting, setSubmitting] = React.useState(false)
  const [message, setMessage] = React.useState<FormMessage>(null)

  React.useEffect(() => {
    setAutonomyMode(security?.autonomy_mode ?? 'trusted_workspace')
    setRequireShellApproval(security?.require_approval_for_shell ?? true)
    setRequireFileWriteApproval(security?.require_approval_for_file_write ?? false)
    setRequireExecApproval(security?.require_approval_for_exec ?? true)
    setAgentRunDefaultMaxWallClockSec(
      security?.agent_run_default_max_wall_clock_sec == null ? '' : String(security.agent_run_default_max_wall_clock_sec)
    )
    setAgentRunDefaultHeartbeatTimeoutSec(
      security?.agent_run_default_heartbeat_timeout_sec == null ? '' : String(security.agent_run_default_heartbeat_timeout_sec)
    )
    setAgentRunDefaultCheckpointIntervalSteps(String(security?.agent_run_default_checkpoint_interval_steps ?? 1))
    setAgentRunDefaultMaxSubagentFailuresPerRole(String(security?.agent_run_default_max_subagent_failures_per_role ?? 2))
    setAgentRunDefaultOnBudgetExhausted(security?.agent_run_default_on_budget_exhausted ?? 'pause')
    setAgentRunDefaultOnSubagentDisconnect(security?.agent_run_default_on_subagent_disconnect ?? 'retry_then_degrade')
    setExecDefaultTimeoutSec(String(security?.exec_default_timeout_sec ?? 30))
    setExecSessionOutputLimit(String(security?.exec_session_output_limit ?? 8000))
    setFileOpsScope(security?.file_ops_scope ?? 'workspace')
    setMaxFileWriteSizeMb(String(security?.max_file_write_size_mb ?? 10.0))
    setFileUndoMaxSizeMb(String(security?.file_undo_max_size_mb ?? 2.0))
  }, [security])

  const handleAutonomyModeChange = (value: api.SecuritySettings['autonomy_mode']) => {
    setAutonomyMode(value)
    if (value === 'strict') {
      setRequireShellApproval(true)
      setRequireFileWriteApproval(true)
      setRequireExecApproval(true)
      setFileOpsScope('workspace')
      return
    }
    if (value === 'trusted_workspace') {
      setRequireShellApproval(true)
      setRequireFileWriteApproval(false)
      setRequireExecApproval(true)
      setFileOpsScope('workspace')
      return
    }
    if (value === 'high_autonomy') {
      setRequireShellApproval(false)
      setRequireFileWriteApproval(false)
      setRequireExecApproval(false)
      setFileOpsScope('any')
      return
    }
    setRequireShellApproval(false)
    setRequireFileWriteApproval(false)
    setRequireExecApproval(false)
    setFileOpsScope('workspace')
  }

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setSubmitting(true)
    setMessage(null)

    try {
      if (typeof settingsApi.updateSettings !== 'function') {
        throw new Error('Settings update API client is unavailable.')
      }
      const settings = await settingsApi.updateSettings({
        security: {
          autonomy_mode: autonomyMode,
          require_approval_for_shell: requireShellApproval,
          require_approval_for_file_write: requireFileWriteApproval,
          require_approval_for_exec: requireExecApproval,
          agent_run_default_max_wall_clock_sec:
            agentRunDefaultMaxWallClockSec.trim().length > 0
              ? Number.parseInt(agentRunDefaultMaxWallClockSec, 10) || 1
              : null,
          agent_run_default_heartbeat_timeout_sec:
            agentRunDefaultHeartbeatTimeoutSec.trim().length > 0
              ? Number.parseInt(agentRunDefaultHeartbeatTimeoutSec, 10) || 1
              : null,
          agent_run_default_checkpoint_interval_steps:
            Number.parseInt(agentRunDefaultCheckpointIntervalSteps, 10) || 1,
          agent_run_default_max_subagent_failures_per_role:
            Number.parseInt(agentRunDefaultMaxSubagentFailuresPerRole, 10) || 0,
          agent_run_default_on_budget_exhausted: agentRunDefaultOnBudgetExhausted,
          agent_run_default_on_subagent_disconnect: agentRunDefaultOnSubagentDisconnect,
          exec_default_timeout_sec: Number.parseInt(execDefaultTimeoutSec, 10) || 30,
          exec_session_output_limit: Number.parseInt(execSessionOutputLimit, 10) || 8000,
          file_ops_scope: fileOpsScope,
          max_file_write_size_mb: Number.parseFloat(maxFileWriteSizeMb) || 10.0,
          file_undo_max_size_mb: Number.parseFloat(fileUndoMaxSizeMb) || 2.0,
        },
      })
      onUpdated(settings)
      setMessage({ type: 'success', text: t('settings.security.successSaved') })
    } catch (updateError) {
      setMessage({
        type: 'error',
        text: messageWithDetail(t('settings.security.errorSave'), updateError),
      })
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <section className="rounded-lg border border-border bg-surface-layer">
      <div className="border-b border-border px-4 py-3">
        <div className="flex items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold text-foreground">{t('settings.security.title')}</h3>
            <p className="mt-0.5 text-xs text-muted-foreground">{t('settings.security.description')}</p>
          </div>
          <Shield className="h-4 w-4 text-muted-foreground" />
        </div>
      </div>
      <form onSubmit={handleSubmit} className="space-y-4 px-4 py-4">
        <label className="space-y-1.5">
          <SettingLabel>{t('settings.security.autonomyMode')}</SettingLabel>
          <Select value={autonomyMode} onValueChange={(value) => handleAutonomyModeChange(value as api.SecuritySettings['autonomy_mode'])}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="strict">{t('settings.security.autonomyMode.strict')}</SelectItem>
              <SelectItem value="trusted_workspace">{t('settings.security.autonomyMode.trusted_workspace')}</SelectItem>
              <SelectItem value="auto_review">{t('settings.security.autonomyMode.auto_review')}</SelectItem>
              <SelectItem value="high_autonomy">{t('settings.security.autonomyMode.high_autonomy')}</SelectItem>
            </SelectContent>
          </Select>
        </label>
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          <div className="flex items-center justify-between gap-3 rounded-md border border-border bg-canvas px-3 py-2">
            <span className="text-sm text-foreground">{t('settings.security.requireShellApproval')}</span>
            <Switch checked={requireShellApproval} onCheckedChange={setRequireShellApproval} />
          </div>
          <div className="flex items-center justify-between gap-3 rounded-md border border-border bg-canvas px-3 py-2">
            <span className="text-sm text-foreground">{t('settings.security.requireFileWriteApproval')}</span>
            <Switch checked={requireFileWriteApproval} onCheckedChange={setRequireFileWriteApproval} />
          </div>
          <div className="flex items-center justify-between gap-3 rounded-md border border-border bg-canvas px-3 py-2 md:col-span-2">
            <span className="text-sm text-foreground">{t('settings.security.requireExecApproval')}</span>
            <Switch checked={requireExecApproval} onCheckedChange={setRequireExecApproval} />
          </div>
        </div>
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          <label className="space-y-1.5">
            <SettingLabel>{t('settings.security.execDefaultTimeoutSec')}</SettingLabel>
            <Input value={execDefaultTimeoutSec} onChange={(event) => setExecDefaultTimeoutSec(event.target.value)} className="font-mono text-xs" />
          </label>
          <label className="space-y-1.5">
            <SettingLabel>{t('settings.security.execSessionOutputLimit')}</SettingLabel>
            <Input value={execSessionOutputLimit} onChange={(event) => setExecSessionOutputLimit(event.target.value)} className="font-mono text-xs" />
          </label>
        </div>
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          <label className="space-y-1.5">
            <SettingLabel>{t('settings.security.agentRunDefaultMaxWallClockSec')}</SettingLabel>
            <Input
              value={agentRunDefaultMaxWallClockSec}
              onChange={(event) => setAgentRunDefaultMaxWallClockSec(event.target.value)}
              placeholder={t('settings.security.agentRunDefaultMaxWallClockSecPlaceholder')}
              className="font-mono text-xs"
            />
          </label>
          <label className="space-y-1.5">
            <SettingLabel>{t('settings.security.agentRunDefaultHeartbeatTimeoutSec')}</SettingLabel>
            <Input
              value={agentRunDefaultHeartbeatTimeoutSec}
              onChange={(event) => setAgentRunDefaultHeartbeatTimeoutSec(event.target.value)}
              placeholder={t('settings.security.agentRunDefaultHeartbeatTimeoutSecPlaceholder')}
              className="font-mono text-xs"
            />
          </label>
        </div>
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          <label className="space-y-1.5">
            <SettingLabel>{t('settings.security.agentRunDefaultCheckpointIntervalSteps')}</SettingLabel>
            <Input
              value={agentRunDefaultCheckpointIntervalSteps}
              onChange={(event) => setAgentRunDefaultCheckpointIntervalSteps(event.target.value)}
              className="font-mono text-xs"
            />
          </label>
          <label className="space-y-1.5">
            <SettingLabel>{t('settings.security.agentRunDefaultMaxSubagentFailuresPerRole')}</SettingLabel>
            <Input
              value={agentRunDefaultMaxSubagentFailuresPerRole}
              onChange={(event) => setAgentRunDefaultMaxSubagentFailuresPerRole(event.target.value)}
              className="font-mono text-xs"
            />
          </label>
        </div>
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          <label className="space-y-1.5">
            <SettingLabel>{t('settings.security.agentRunDefaultOnBudgetExhausted')}</SettingLabel>
            <Select
              value={agentRunDefaultOnBudgetExhausted}
              onValueChange={(value) => setAgentRunDefaultOnBudgetExhausted(value as api.SecuritySettings['agent_run_default_on_budget_exhausted'])}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="pause">{t('settings.security.agentRunDefaultOnBudgetExhausted.pause')}</SelectItem>
                <SelectItem value="finalize_partial">{t('settings.security.agentRunDefaultOnBudgetExhausted.finalize_partial')}</SelectItem>
              </SelectContent>
            </Select>
          </label>
          <label className="space-y-1.5">
            <SettingLabel>{t('settings.security.agentRunDefaultOnSubagentDisconnect')}</SettingLabel>
            <Select
              value={agentRunDefaultOnSubagentDisconnect}
              onValueChange={(value) => setAgentRunDefaultOnSubagentDisconnect(value as api.SecuritySettings['agent_run_default_on_subagent_disconnect'])}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="retry_then_degrade">{t('settings.security.agentRunDefaultOnSubagentDisconnect.retry_then_degrade')}</SelectItem>
                <SelectItem value="pause">{t('settings.security.agentRunDefaultOnSubagentDisconnect.pause')}</SelectItem>
                <SelectItem value="fail">{t('settings.security.agentRunDefaultOnSubagentDisconnect.fail')}</SelectItem>
              </SelectContent>
            </Select>
          </label>
        </div>
        <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
          <label className="space-y-1.5 md:col-span-1">
            <SettingLabel>{t('settings.security.fileScope')}</SettingLabel>
            <Select value={fileOpsScope} onValueChange={(value) => setFileOpsScope(value as 'workspace' | 'any')}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="workspace">{t('settings.security.fileScope.workspace')}</SelectItem>
                <SelectItem value="any">{t('settings.security.fileScope.any')}</SelectItem>
              </SelectContent>
            </Select>
          </label>
          <label className="space-y-1.5">
            <SettingLabel>{t('settings.security.maxWriteSizeMb')}</SettingLabel>
            <Input value={maxFileWriteSizeMb} onChange={(event) => setMaxFileWriteSizeMb(event.target.value)} className="font-mono text-xs" />
          </label>
          <label className="space-y-1.5">
            <SettingLabel>{t('settings.security.undoMaxSizeMb')}</SettingLabel>
            <Input value={fileUndoMaxSizeMb} onChange={(event) => setFileUndoMaxSizeMb(event.target.value)} className="font-mono text-xs" />
          </label>
        </div>
        <SettingMessage message={message} />
        <div className="flex justify-end">
          <Button type="submit" variant="primary" size="sm" loading={submitting}>
            <Save className="h-3.5 w-3.5" />
            {t('settings.action.saveSecurity')}
          </Button>
        </div>
      </form>
    </section>
  )
}

function ToolsSettingsForm({
  tools,
  onUpdated,
}: {
  tools: api.ToolsSettings
  onUpdated: (settings: api.Settings) => void
}) {
  type FetchExtractor = 'trafilatura' | 'jina_reader' | 'htmlparser'
  type ManagedKeyField =
    | 'web_search_tavily_api_key'
    | 'web_search_serper_api_key'
    | 'web_search_brave_api_key'
    | 'web_search_jina_api_key'
    | 'web_search_exa_api_key'
    | 'web_fetch_jina_api_key'

  const engineOptions = ['tavily', 'brave', 'jina', 'serper', 'exa', 'searxng', 'duckduckgo_html', 'duckduckgo']
  const extractorOptions: FetchExtractor[] = ['trafilatura', 'jina_reader', 'htmlparser']
  const [searchEngine, setSearchEngine] = React.useState(getStringSetting(tools, 'web_search_engine', 'tavily'))
  const [fallbackEngines, setFallbackEngines] = React.useState(
    getStringOptions(tools, 'web_search_fallback_engines', ['brave', 'duckduckgo_html']).join(', ')
  )
  const [searxngBaseUrl, setSearxngBaseUrl] = React.useState(getStringSetting(tools, 'web_search_searxng_base_url'))
  const [language, setLanguage] = React.useState(getStringSetting(tools, 'web_search_language'))
  const [region, setRegion] = React.useState(getStringSetting(tools, 'web_search_region'))
  const [tavilyApiKey, setTavilyApiKey] = React.useState('')
  const [serperApiKey, setSerperApiKey] = React.useState('')
  const [braveApiKey, setBraveApiKey] = React.useState('')
  const [jinaSearchApiKey, setJinaSearchApiKey] = React.useState('')
  const [exaApiKey, setExaApiKey] = React.useState('')
  const [fetchExtractor, setFetchExtractor] = React.useState<FetchExtractor>(
    (getStringSetting(tools, 'web_fetch_extractor', 'trafilatura') as FetchExtractor) || 'trafilatura'
  )
  const [jinaFetchApiKey, setJinaFetchApiKey] = React.useState('')
  const [submitting, setSubmitting] = React.useState(false)
  const [message, setMessage] = React.useState<FormMessage>(null)

  React.useEffect(() => {
    setSearchEngine(getStringSetting(tools, 'web_search_engine', 'tavily'))
    setFallbackEngines(
      getStringOptions(tools, 'web_search_fallback_engines', ['brave', 'duckduckgo_html']).join(', ')
    )
    setSearxngBaseUrl(getStringSetting(tools, 'web_search_searxng_base_url'))
    setLanguage(getStringSetting(tools, 'web_search_language'))
    setRegion(getStringSetting(tools, 'web_search_region'))
    setFetchExtractor(
      (getStringSetting(tools, 'web_fetch_extractor', 'trafilatura') as FetchExtractor) || 'trafilatura'
    )
    setTavilyApiKey('')
    setSerperApiKey('')
    setBraveApiKey('')
    setJinaSearchApiKey('')
    setExaApiKey('')
    setJinaFetchApiKey('')
  }, [tools])

  const handleClearKey = async (
    field: ManagedKeyField,
    label: string,
    resetInput: () => void
  ) => {
    setSubmitting(true)
    setMessage(null)
    try {
      if (typeof settingsApi.updateSettings !== 'function') {
        throw new Error('Settings update API client is unavailable.')
      }
      const settings = await settingsApi.updateSettings({
        tools: {
          [field]: null,
        } as api.ToolsSettingsUpdate,
      })
      resetInput()
      onUpdated(settings)
      setMessage({ type: 'success', text: `Cleared ${label}.` })
    } catch (clearError) {
      setMessage({
        type: 'error',
        text: messageWithDetail(`Failed to clear ${label}`, clearError),
      })
    } finally {
      setSubmitting(false)
    }
  }

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setSubmitting(true)
    setMessage(null)

    try {
      if (typeof settingsApi.updateSettings !== 'function') {
        throw new Error('Settings update API client is unavailable.')
      }

      const fallback = fallbackEngines
        .split(',')
        .map((item) => item.trim())
        .filter((item) => item.length > 0)

      const toolsUpdate: api.ToolsSettingsUpdate = {
        web_search_engine: searchEngine,
        web_search_fallback_engines: fallback,
        web_search_searxng_base_url: searxngBaseUrl.trim() || null,
        web_search_language: language.trim() || null,
        web_search_region: region.trim() || null,
        web_fetch_extractor: fetchExtractor,
      }

      if (tavilyApiKey.trim().length > 0) {
        toolsUpdate.web_search_tavily_api_key = tavilyApiKey.trim()
      }
      if (serperApiKey.trim().length > 0) {
        toolsUpdate.web_search_serper_api_key = serperApiKey.trim()
      }
      if (braveApiKey.trim().length > 0) {
        toolsUpdate.web_search_brave_api_key = braveApiKey.trim()
      }
      if (jinaSearchApiKey.trim().length > 0) {
        toolsUpdate.web_search_jina_api_key = jinaSearchApiKey.trim()
      }
      if (exaApiKey.trim().length > 0) {
        toolsUpdate.web_search_exa_api_key = exaApiKey.trim()
      }
      if (jinaFetchApiKey.trim().length > 0) {
        toolsUpdate.web_fetch_jina_api_key = jinaFetchApiKey.trim()
      }

      const settings = await settingsApi.updateSettings({ tools: toolsUpdate })
      onUpdated(settings)
      setMessage({ type: 'success', text: 'Saved web search and fetch settings.' })
    } catch (updateError) {
      setMessage({
        type: 'error',
        text: messageWithDetail('Failed to save web search / fetch settings', updateError),
      })
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <section className="rounded-lg border border-border bg-surface-layer">
      <div className="border-b border-border px-4 py-3">
        <div className="flex items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold text-foreground">Web Search & Fetch</h3>
            <p className="mt-0.5 text-xs text-muted-foreground">
              Configure the primary search engine, fallback order, and page extraction behavior used by agent tools.
            </p>
          </div>
          <Globe className="h-4 w-4 text-muted-foreground" />
        </div>
      </div>
      <form onSubmit={handleSubmit} className="space-y-4 px-4 py-4">
        <div className="grid grid-cols-1 gap-3 xl:grid-cols-3">
          <label className="space-y-1.5">
            <SettingLabel>Primary search engine</SettingLabel>
            <Select value={searchEngine} onValueChange={setSearchEngine}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {engineOptions.map((option) => (
                  <SelectItem key={option} value={option}>
                    {option}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </label>
          <label className="space-y-1.5 xl:col-span-2">
            <SettingLabel>Fallback engines</SettingLabel>
            <Input
              value={fallbackEngines}
              onChange={(event) => setFallbackEngines(event.target.value)}
              placeholder="brave, duckduckgo_html"
              className="font-mono text-xs"
            />
          </label>
        </div>

        <div className="grid grid-cols-1 gap-3 xl:grid-cols-3">
          <label className="space-y-1.5">
            <SettingLabel>Language</SettingLabel>
            <Input value={language} onChange={(event) => setLanguage(event.target.value)} placeholder="zh-TW" className="font-mono text-xs" />
          </label>
          <label className="space-y-1.5">
            <SettingLabel>Region</SettingLabel>
            <Input value={region} onChange={(event) => setRegion(event.target.value)} placeholder="tw" className="font-mono text-xs" />
          </label>
          <label className="space-y-1.5">
            <SettingLabel>SearXNG base URL</SettingLabel>
            <Input value={searxngBaseUrl} onChange={(event) => setSearxngBaseUrl(event.target.value)} placeholder="https://search.example.test" className="font-mono text-xs" />
          </label>
        </div>

        <div className="grid grid-cols-1 gap-3 xl:grid-cols-3">
          <div className="space-y-1.5">
            <div className="flex items-center justify-between gap-2">
              <SettingLabel>Tavily API key</SettingLabel>
              <Button
                type="button"
                size="sm"
                variant="ghost"
                onClick={() => void handleClearKey('web_search_tavily_api_key', 'Tavily API key', () => setTavilyApiKey(''))}
                disabled={submitting || !getBooleanSetting(tools, 'web_search_tavily_api_key_configured')}
              >
                Clear key
              </Button>
            </div>
            <Input
              type="password"
              autoComplete="off"
              value={tavilyApiKey}
              onChange={(event) => setTavilyApiKey(event.target.value)}
              placeholder={getBooleanSetting(tools, 'web_search_tavily_api_key_configured') ? 'Leave blank to keep existing key' : 'tvly-...'}
              className="font-mono text-xs"
            />
            <p className="text-[11px] text-muted-foreground">Configured: {getBooleanSetting(tools, 'web_search_tavily_api_key_configured') ? 'yes' : 'no'}</p>
          </div>
          <div className="space-y-1.5">
            <div className="flex items-center justify-between gap-2">
              <SettingLabel>Serper API key</SettingLabel>
              <Button
                type="button"
                size="sm"
                variant="ghost"
                onClick={() => void handleClearKey('web_search_serper_api_key', 'Serper API key', () => setSerperApiKey(''))}
                disabled={submitting || !getBooleanSetting(tools, 'web_search_serper_api_key_configured')}
              >
                Clear key
              </Button>
            </div>
            <Input
              type="password"
              autoComplete="off"
              value={serperApiKey}
              onChange={(event) => setSerperApiKey(event.target.value)}
              placeholder={getBooleanSetting(tools, 'web_search_serper_api_key_configured') ? 'Leave blank to keep existing key' : 'serper-key'}
              className="font-mono text-xs"
            />
            <p className="text-[11px] text-muted-foreground">Configured: {getBooleanSetting(tools, 'web_search_serper_api_key_configured') ? 'yes' : 'no'}</p>
          </div>
          <div className="space-y-1.5">
            <div className="flex items-center justify-between gap-2">
              <SettingLabel>Brave API key</SettingLabel>
              <Button
                type="button"
                size="sm"
                variant="ghost"
                onClick={() => void handleClearKey('web_search_brave_api_key', 'Brave API key', () => setBraveApiKey(''))}
                disabled={submitting || !getBooleanSetting(tools, 'web_search_brave_api_key_configured')}
              >
                Clear key
              </Button>
            </div>
            <Input
              type="password"
              autoComplete="off"
              value={braveApiKey}
              onChange={(event) => setBraveApiKey(event.target.value)}
              placeholder={getBooleanSetting(tools, 'web_search_brave_api_key_configured') ? 'Leave blank to keep existing key' : 'brave-key'}
              className="font-mono text-xs"
            />
            <p className="text-[11px] text-muted-foreground">Configured: {getBooleanSetting(tools, 'web_search_brave_api_key_configured') ? 'yes' : 'no'}</p>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-3 xl:grid-cols-3">
          <div className="space-y-1.5">
            <div className="flex items-center justify-between gap-2">
              <SettingLabel>Jina search API key</SettingLabel>
              <Button
                type="button"
                size="sm"
                variant="ghost"
                onClick={() => void handleClearKey('web_search_jina_api_key', 'Jina search API key', () => setJinaSearchApiKey(''))}
                disabled={submitting || !getBooleanSetting(tools, 'web_search_jina_api_key_configured')}
              >
                Clear key
              </Button>
            </div>
            <Input
              type="password"
              autoComplete="off"
              value={jinaSearchApiKey}
              onChange={(event) => setJinaSearchApiKey(event.target.value)}
              placeholder={getBooleanSetting(tools, 'web_search_jina_api_key_configured') ? 'Leave blank to keep existing key' : 'jina-key'}
              className="font-mono text-xs"
            />
            <p className="text-[11px] text-muted-foreground">
              Provider available: {getBooleanSetting(tools, 'web_search_jina_configured', true) ? 'yes' : 'no'} · key configured: {getBooleanSetting(tools, 'web_search_jina_api_key_configured') ? 'yes' : 'no'}
            </p>
          </div>
          <div className="space-y-1.5">
            <div className="flex items-center justify-between gap-2">
              <SettingLabel>Exa API key</SettingLabel>
              <Button
                type="button"
                size="sm"
                variant="ghost"
                onClick={() => void handleClearKey('web_search_exa_api_key', 'Exa API key', () => setExaApiKey(''))}
                disabled={submitting || !getBooleanSetting(tools, 'web_search_exa_api_key_configured')}
              >
                Clear key
              </Button>
            </div>
            <Input
              type="password"
              autoComplete="off"
              value={exaApiKey}
              onChange={(event) => setExaApiKey(event.target.value)}
              placeholder={getBooleanSetting(tools, 'web_search_exa_api_key_configured') ? 'Leave blank to keep existing key' : 'exa-key'}
              className="font-mono text-xs"
            />
            <p className="text-[11px] text-muted-foreground">Configured: {getBooleanSetting(tools, 'web_search_exa_api_key_configured') ? 'yes' : 'no'}</p>
          </div>
          <label className="space-y-1.5">
            <SettingLabel>Web fetch extractor</SettingLabel>
            <Select value={fetchExtractor} onValueChange={(value) => setFetchExtractor(value as FetchExtractor)}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {extractorOptions.map((option) => (
                  <SelectItem key={option} value={option}>
                    {option}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </label>
        </div>

        <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
          <div className="space-y-1.5">
            <div className="flex items-center justify-between gap-2">
              <SettingLabel>Jina fetch API key</SettingLabel>
              <Button
                type="button"
                size="sm"
                variant="ghost"
                onClick={() => void handleClearKey('web_fetch_jina_api_key', 'Jina fetch API key', () => setJinaFetchApiKey(''))}
                disabled={submitting || !getBooleanSetting(tools, 'web_fetch_jina_api_key_configured')}
              >
                Clear key
              </Button>
            </div>
            <Input
              type="password"
              autoComplete="off"
              value={jinaFetchApiKey}
              onChange={(event) => setJinaFetchApiKey(event.target.value)}
              placeholder={getBooleanSetting(tools, 'web_fetch_jina_api_key_configured') ? 'Leave blank to keep existing key' : 'jina-key'}
              className="font-mono text-xs"
            />
            <p className="text-[11px] text-muted-foreground">Configured: {getBooleanSetting(tools, 'web_fetch_jina_api_key_configured') ? 'yes' : 'no'}</p>
          </div>
          <div className="space-y-1.5 rounded-md border border-border/60 bg-background/50 px-3 py-3 text-xs text-muted-foreground">
            <div>Provider status</div>
            <div>Jina search available: {getBooleanSetting(tools, 'web_search_jina_configured', true) ? 'yes' : 'no'}</div>
            <div>SearXNG configured: {getBooleanSetting(tools, 'web_search_searxng_configured') ? 'yes' : 'no'}</div>
            <div>DuckDuckGo HTML available: {getBooleanSetting(tools, 'web_search_duckduckgo_html_configured', true) ? 'yes' : 'no'}</div>
          </div>
        </div>

        <SettingMessage message={message} />

        <div className="flex justify-end">
          <Button type="submit" variant="primary" size="sm" loading={submitting}>
            <Save className="h-3.5 w-3.5" />
            Save Web Tools
          </Button>
        </div>
      </form>
    </section>
  )
}

function PreferencesPanel() {
  const {
    languageMode,
    setLanguageMode,
    appearanceMode,
    setAppearanceMode,
    codeTheme,
    setCodeTheme,
    fontSize,
    setFontSize,
    timezone,
    setTimezone,
    resolvedTimeZone,
    t,
  } = useI18n()

  const languageOptions: Array<{ value: UILanguageMode; label: string }> = [
    { value: AUTO_LANGUAGE, label: t('settings.preferences.language.default') },
    { value: 'zh-TW', label: t('settings.preferences.language.zhTW') },
    { value: 'en', label: t('settings.preferences.language.en') },
  ]

  const appearanceOptions: Array<{ value: UIAppearanceMode; label: string }> = [
    { value: SYSTEM_APPEARANCE, label: t('settings.preferences.appearance.system') },
    { value: 'dark', label: t('settings.preferences.appearance.dark') },
    { value: 'light', label: t('settings.preferences.appearance.light') },
  ]
  const codeThemeOptions = CODE_THEME_OPTIONS

  const fontSizeOptions: Array<{ value: UIFontSize; label: string }> = [
    { value: 'compact', label: t('settings.preferences.font.compact') },
    { value: 'default', label: t('settings.preferences.font.default') },
    { value: 'large', label: t('settings.preferences.font.large') },
  ]

  const timezoneOptions = React.useMemo<Array<{ value: string; label: string }>>(() => {
    const options: Array<{ value: string; label: string }> = [
      {
        value: AUTO_TIMEZONE,
        label: resolvedTimeZone
          ? t('settings.preferences.timezone.autoWithZone', { timezone: resolvedTimeZone })
          : t('settings.preferences.timezone.auto'),
      },
      { value: 'UTC', label: t('settings.preferences.timezone.utc') },
    ]
    if (resolvedTimeZone && !options.some((option) => option.value === resolvedTimeZone)) {
      options.push({ value: resolvedTimeZone, label: resolvedTimeZone })
    }
    if (timezone && !options.some((option) => option.value === timezone)) {
      options.push({ value: timezone, label: timezone })
    }
    return options
  }, [resolvedTimeZone, t, timezone])

  return (
    <section className="rounded-lg border border-border bg-surface-layer">
      <div className="border-b border-border px-4 py-3">
        <h3 className="text-sm font-semibold text-foreground">{t('settings.preferences.title')}</h3>
        <p className="mt-0.5 text-xs text-muted-foreground">{t('settings.preferences.description')}</p>
      </div>

      <div className="grid grid-cols-1 gap-3 px-4 py-4 md:grid-cols-2 xl:grid-cols-5">
        <label className="space-y-1.5">
          <SettingLabel>{t('settings.preferences.language')}</SettingLabel>
          <Select value={languageMode} onValueChange={(value) => setLanguageMode(value as UILanguageMode)}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {languageOptions.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </label>

        <label className="space-y-1.5">
          <SettingLabel>{t('settings.preferences.appearance')}</SettingLabel>
          <Select value={appearanceMode} onValueChange={(value) => setAppearanceMode(value as UIAppearanceMode)}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {appearanceOptions.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </label>

        <label className="space-y-1.5">
          <SettingLabel>{t('settings.preferences.fontSize')}</SettingLabel>
          <Select value={fontSize} onValueChange={(value) => setFontSize(value as UIFontSize)}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {fontSizeOptions.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </label>

        <label className="space-y-1.5">
          <SettingLabel>{t('settings.preferences.codeTheme')}</SettingLabel>
          <Select value={codeTheme} onValueChange={(value) => setCodeTheme(value as UICodeTheme)}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {codeThemeOptions.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </label>

        <label className="space-y-1.5">
          <SettingLabel>{t('settings.preferences.timezone')}</SettingLabel>
          <Select value={timezone} onValueChange={setTimezone}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {timezoneOptions.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </label>
      </div>

      <div className="border-t border-border px-4 py-2 text-xs text-muted-foreground">
        <p>{t('settings.preferences.appearanceHelp')}</p>
        <p>{t('settings.preferences.codeThemeHelp')}</p>
        <p>{t('settings.preferences.fontHelp')}</p>
        <p className="mt-1">{t('settings.preferences.timezoneHelp')}</p>
      </div>

      <div className="border-t border-border px-4 py-4">
        <div className="mb-2">
          <h4 className="text-sm font-medium text-foreground">{t('settings.preferences.codeThemePreview')}</h4>
        </div>
        <CodeThemePreview />
      </div>
    </section>
  )
}

export default function SettingsPage() {
  const { t } = useI18n()
  const router = useRouter()
  const searchParams = useSearchParams()
  const activeTab = getSettingsTabFromSearch(searchParams)
  const [settings, setSettings] = React.useState<api.Settings | null>(null)
  const [models, setModels] = React.useState<api.ModelInfo[]>([])
  const [channelsStatus, setChannelsStatus] = React.useState<api.ChannelsStatus | null>(null)
  const [channelsError, setChannelsError] = React.useState<string | null>(null)
  const [channelsRefreshing, setChannelsRefreshing] = React.useState(false)
  const [loading, setLoading] = React.useState(true)
  const [error, setError] = React.useState<string | null>(null)

  const handleTabChange = React.useCallback((nextValue: string) => {
    if (!isSettingsTab(nextValue)) {
      return
    }
    router.push(settingsTabHref(nextValue))
  }, [router])

  const refreshChannelsStatus = React.useCallback(async (showLoading = true) => {
    if (typeof settingsApi.fetchChannelsStatus !== 'function') {
      setChannelsStatus(null)
      setChannelsError(t('errors.channelsStatusUnavailable'))
      return
    }

    if (showLoading) {
      setChannelsRefreshing(true)
    }
    setChannelsError(null)

    try {
      const result = await settingsApi.fetchChannelsStatus()
      setChannelsStatus(result)
    } catch (statusError) {
      setChannelsStatus(null)
      setChannelsError(messageWithDetail(t('settings.errorLoadFailed'), statusError))
    } finally {
      if (showLoading) {
        setChannelsRefreshing(false)
      }
    }
  }, [t])

  React.useEffect(() => {
    let cancelled = false

    async function load() {
      setError(null)
      setChannelsError(null)

      try {
        const channelsPromise =
          typeof settingsApi.fetchChannelsStatus === 'function'
            ? settingsApi.fetchChannelsStatus()
                .then((result) => ({ result, error: null as string | null }))
                .catch((loadError) => ({
                  result: null,
                  error: messageWithDetail(t('settings.errorLoadFailed'), loadError),
                }))
            : Promise.resolve({
                result: null,
                error: t('errors.channelsStatusUnavailable'),
              })

        const [settingsResult, modelsResult, channelsResult] = await Promise.all([
          typeof settingsApi.fetchSettings === 'function'
            ? settingsApi.fetchSettings()
            : Promise.resolve(null),
          typeof settingsApi.fetchModels === 'function'
            ? settingsApi.fetchModels()
            : Promise.resolve([]),
          channelsPromise,
        ])

        if (cancelled) {
          return
        }

        const fetchedModels = Array.isArray(modelsResult) ? modelsResult as api.ModelInfo[] : []
        setSettings(settingsResult)
        setModels(mergeModelInfos(fetchedModels, configuredModelsFromSettings(settingsResult)))
        setChannelsStatus(channelsResult.result)
        setChannelsError(channelsResult.error)
      } catch (loadError) {
        if (cancelled) {
          return
        }

        setError(messageWithDetail(t('settings.errorLoadFailed'), loadError))
      } finally {
        if (!cancelled) {
          setLoading(false)
        }
      }
    }

    void load()

    return () => {
      cancelled = true
    }
  }, [t])

  const modelSection = React.useMemo(() => extractSection(settings, 'model'), [settings])
  const modelConfigSection = React.useMemo(() => extractSection(settings, 'model_config'), [settings])
  const agentSection = settings?.agent
  const rootModel = React.useMemo(() => getPrimaryModel(settings, models, t), [settings, models, t])
  const rawRootModel = React.useMemo(() => {
    const root = asRecord(settings)
    return root ? formatScalar(root.model, t) : null
  }, [settings, t])
  const voiceSection = React.useMemo(() => settings?.voice ?? {}, [settings])
  const memorySection = React.useMemo(() => extractSection(settings, 'memory'), [settings])
  const learningSection = React.useMemo(() => extractSection(settings, 'learning'), [settings])
  const securitySection = React.useMemo(() => settings?.security, [settings])
  const toolsSection = React.useMemo(() => settings?.tools ?? {}, [settings])
  const channelsSection = React.useMemo(() => extractSection(settings, 'channels'), [settings])
  const webSection = React.useMemo(() => extractSection(settings, 'web'), [settings])
  const pathsSection = React.useMemo(() => extractSection(settings, 'paths'), [settings])

  const connectedModels = getConnectedModelCount(models)
  const modelSummary = [
    ...(rootModel ? [{ label: t('settings.stats.primaryModel'), value: rootModel }] : []),
    ...collectSummary(modelSection, ['backend', 'model', 'provider', 'temperature', 'max_tokens'], 6, t),
  ]
  const voiceSummary = collectSummary(voiceSection, ['enabled', 'stt_backend', 'stt_model', 'stt_model_cache_dir', 'tts_backend', 'tts_voice'], 6, t)
  const memorySummary = collectSummary(memorySection, ['db_path', 'max_short_term_messages', 'fts_top_k'], 6, t)
  const learningSummary = collectSummary(learningSection, ['enabled', 'auto_extract_skills', 'auto_sync_filesystem_skills', 'trajectory_path', 'skills_db_path', 'trajectory_retention_days', 'max_skills'], 7, t)
  const toolsSummary = collectSummary(
    toolsSection,
    ['web_search_engine', 'web_search_fallback_engines', 'web_search_language', 'web_search_region', 'web_fetch_extractor'],
    6,
    t
  )
  const discordState = resolveChannelPanelState('discord', channelsSection, channelsStatus, t)
  const telegramState = resolveChannelPanelState('telegram', channelsSection, channelsStatus, t)
  const enabledChannels = [
    discordState.enabled ? 'Discord' : null,
    telegramState.enabled ? 'Telegram' : null,
  ].filter((name): name is string => name !== null)
  const channelsRunning = Boolean(discordState.running) || Boolean(telegramState.running)
  const channelReportedCount = Math.max(
    Object.keys(channelsStatus?.channels ?? {}).length,
    asRecord(channelsSection.discord) ? 1 : 0,
    asRecord(channelsSection.telegram) ? 1 : 0,
    2
  )
  const webSummary = collectSummary(webSection, ['host', 'port', 'base_url', 'cors_origins'], 6, t)

  const hasAnyData =
    modelSummary.length > 0 ||
    voiceSummary.length > 0 ||
    memorySummary.length > 0 ||
    learningSummary.length > 0 ||
    toolsSummary.length > 0 ||
    channelsStatus !== null ||
    Object.keys(channelsSection).length > 0 ||
    webSummary.length > 0 ||
    models.length > 0

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <header className="shrink-0 border-b border-border px-6 pb-4 pt-5">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 className="text-xl font-bold text-foreground">{t('settings.title')}</h1>
            <p className="mt-0.5 text-sm text-muted-foreground">
              {t('settings.subtitle')}
            </p>
          </div>
          <OverviewBadge ok={connectedModels > 0 || hasAnyData} />
        </div>

        {error ? (
          <div className="mt-3 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
            {error}
          </div>
        ) : null}
      </header>

      <div className="flex-1 overflow-y-auto px-6 py-5">
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
          <StatTile title={t('settings.stats.primaryModel')} value={getPrimaryModel(settings, models, t)} icon={Cpu} />
          <StatTile title={t('settings.stats.modelEndpoints')} value={`${connectedModels}/${models.length || 0}`} icon={Sparkles} />
          <StatTile
            title={t('settings.stats.channels')}
            value={`${enabledChannels.length}/${channelReportedCount}`}
            icon={Network}
          />
          <StatTile
            title={t('settings.stats.webSurface')}
            value={webSummary.length > 0 ? t('settings.stats.configured') : t('settings.stats.notReported')}
            icon={Globe}
          />
        </div>

        <Tabs value={activeTab} onValueChange={handleTabChange} className="mt-5 flex gap-6">
          <div className="w-40 shrink-0">
            <SettingsNav active={activeTab} />
          </div>

          <div className="min-w-0 flex-1">
            <TabsContent value="model" className="mt-0">
              <div className="space-y-4">
                <ModelConnectionForm
                  settings={settings}
                  configuredModel={rawRootModel}
                  modelConfig={modelConfigSection}
                  models={models}
                  setModels={setModels}
                  setSettings={setSettings}
                  onConfigured={(result) => {
                    const modelInfo = result.activeModel
                    const availableModels = result.availableModels.length > 0
                      ? result.availableModels
                      : [modelInfo]
                    const savedModelInfo = result.availableModels.find(
                      (entry) => modelInfoId(entry) === modelInfoId(modelInfo)
                    )
                      ?? result.availableModels[0]
                      ?? modelInfo
                    const remoteBaseUrl = baseUrlFromModelInfo(savedModelInfo)
                    const savedModelName = savedModelInfo.name || modelInfo.name
                    const savedBackendType = savedModelInfo.backendType || modelInfo.backendType
                    const isLocalModel = result.provider === 'local' || savedModelInfo.provider === 'local'
                    setModels(availableModels)
                    setSettings((current) => current ? ({
                      ...current,
                      model: isLocalModel
                        ? (savedModelInfo.modelSpec ?? savedModelName)
                        : (savedBackendType === 'ollama'
                          ? `ollama:${savedModelName}`
                          : remoteBaseUrl ?? savedModelInfo.modelSpec ?? rawRootModel ?? savedModelName),
                      model_config: {
                        ...omitConfiguredModels(current.model_config ?? {}),
                        provider: result.provider,
                        ollama_model: savedBackendType === 'ollama' ? savedModelName : '',
                        ollama_base_url: savedBackendType === 'ollama' ? withNullableString(remoteBaseUrl) : null,
                        local_model_path: isLocalModel ? withNullableString(savedModelInfo.modelSpec ?? savedModelName) : null,
                        local_model_root: isLocalModel
                          ? withNullableString(localModelRootFromModelInfo(savedModelInfo))
                          : null,
                        openai_compat_provider: result.provider === 'ollama' || isLocalModel || result.provider === 'openai_codex' ? null : result.provider,
                        openai_compat_base_url: result.provider === 'ollama' || isLocalModel || result.provider === 'openai_codex' ? null : withNullableString(remoteBaseUrl),
                        openai_compat_model: savedBackendType === 'openai_compat' ? savedModelName : '',
                        openai_codex_base_url: result.provider === 'openai_codex' ? withNullableString(remoteBaseUrl) : null,
                        openai_codex_model: result.provider === 'openai_codex' ? savedModelName : null,
                        openai_codex_auth_profile_id: result.provider === 'openai_codex' ? withNullableString(savedModelInfo.authProfileId) : null,
                        openai_codex_auth_configured: result.provider === 'openai_codex' ? Boolean(savedModelInfo.authProfileId) : false,
                      },
                      model_setup: {
                        ...(current.model_setup ?? {}),
                        configured_models: availableModels.map(configuredModelRecordFromModelInfo),
                      },
                    }) : current)
                    notifyModelsUpdated()
                  }}
                />
              <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
                <SurfaceSection
                  title={t('settings.section.modelConfig')}
                  description={t('settings.summary.modelConfig')}
                  items={modelSummary}
                />
                <SurfaceSection
                  title={t('settings.section.discoveredModels')}
                  description={t('settings.summary.discoveredModels')}
                  items={
                    models.slice(0, 6).map((entry, index) => {
                      const record = asRecord(entry)
                      const label =
                        modelInfoLabel(entry) ||
                        (record && (formatScalar(record.name, t) ?? formatScalar(record.id, t) ?? formatScalar(record.model, t))) ||
                        `${t('settings.modelFallback')} ${index + 1}`
                      const value =
                        entry.backendType ||
                        (record && (formatScalar(record.status, t) ?? summarizeValue(record.provider, t) ?? summarizeValue(record.backend, t))) ||
                        t('common.reported')
                      return { label, value }
                    })
                  }
                />
              </div>
              </div>
            </TabsContent>

            <TabsContent value="inference" className="mt-0">
              <InferenceSettingsForm
                agent={agentSection}
                onUpdated={(updatedSettings) => setSettings(updatedSettings)}
              />
            </TabsContent>

            <TabsContent value="voice" className="mt-0">
              <div className="space-y-4">
                <VoicePipelineForm
                  voice={voiceSection}
                  onUpdated={(updatedSettings) => setSettings(updatedSettings)}
                />
              <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1fr)_280px]">
                <SurfaceSection
                  title={t('settings.section.voicePipeline')}
                  description={t('settings.summary.voicePipeline')}
                  items={voiceSummary}
                />
                <div className="rounded-lg border border-border bg-surface-layer px-4 py-3">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-semibold text-foreground">{t('settings.section.runtime')}</h3>
                    <Mic className="h-4 w-4 text-muted-foreground" />
                  </div>
                  <div className="mt-4 space-y-3">
                    {[
                      { label: t('settings.runtime.voiceFields'), value: String(Object.keys(voiceSection).length) },
                      {
                        label: t('settings.runtime.settingsSource'),
                        value: voiceSummary.length > 0 ? t('settings.runtime.backend') : t('settings.runtime.unavailable'),
                      },
                    ].map((item) => (
                      <div key={item.label} className="flex items-center justify-between gap-3 text-sm">
                        <span className="text-muted-foreground">{item.label}</span>
                        <span className="font-medium text-foreground">{item.value}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
              </div>
            </TabsContent>

            <TabsContent value="memory" className="mt-0">
              <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(360px,0.9fr)]">
                <SurfaceSection
                  title={t('settings.section.memory')}
                  description={t('settings.summary.memory')}
                  items={memorySummary}
                />
                <MemoryStorageForm
                  memory={memorySection}
                  onUpdated={(updatedSettings) => setSettings(updatedSettings)}
                />
              </div>
            </TabsContent>

            <TabsContent value="learning" className="mt-0">
              <div className="space-y-4">
                <LearningStorageForm
                  learning={learningSection}
                  paths={pathsSection}
                  onUpdated={(updatedSettings) => setSettings(updatedSettings)}
                />
                <SurfaceSection
                  title={t('settings.section.learning')}
                  description={t('settings.summary.learning')}
                  items={learningSummary}
                />
              </div>
            </TabsContent>

            <TabsContent value="security" className="mt-0">
              <SecuritySettingsForm
                security={securitySection}
                onUpdated={(updatedSettings) => setSettings(updatedSettings)}
              />
            </TabsContent>

            <TabsContent value="channels" className="mt-0">
              <div className="space-y-4">
                {channelsError ? (
                  <div className="rounded-md border border-warning/30 bg-warning/10 px-3 py-2 text-sm text-warning">
                    {channelsError}
                  </div>
                ) : null}
                <DiscordSetupForm
                  channels={channelsSection}
                  onUpdated={(updatedSettings) => setSettings(updatedSettings)}
                />
                <ChannelPanel
                  channelName="discord"
                  title="Discord"
                  icon={Bot}
                  state={discordState}
                  targetLabel={t('settings.channel.allowedChannelIds')}
                  loading={channelsRefreshing}
                  onRefresh={() => void refreshChannelsStatus()}
                  onControlSuccess={() => void refreshChannelsStatus(false)}
                />
                <ChannelPanel
                  channelName="telegram"
                  title="Telegram"
                  icon={Send}
                  state={telegramState}
                  targetLabel={t('settings.channel.allowedChatIds')}
                  loading={channelsRefreshing}
                  onRefresh={() => void refreshChannelsStatus()}
                  onControlSuccess={() => void refreshChannelsStatus(false)}
                />
                <TerminalLocalPanel
                  channelsRunning={channelsRunning}
                  enabledChannels={enabledChannels}
                />
              </div>
            </TabsContent>

            <TabsContent value="web" className="mt-0">
              <div className="space-y-4">
                <PreferencesPanel />
                <ToolsSettingsForm
                  tools={toolsSection}
                  onUpdated={(updatedSettings) => setSettings(updatedSettings)}
                />
                <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
                  <SurfaceSection
                    title="Web Tools"
                    description="Current web search and page extraction settings exposed to Mochi tools."
                    items={toolsSummary}
                  />
                  <SurfaceSection
                    title={t('settings.section.web')}
                    description={t('settings.summary.web')}
                    items={webSummary}
                  />
                </div>
              </div>
            </TabsContent>
          </div>
        </Tabs>

        {!loading && !error && !hasAnyData ? (
          <div className="mt-5 rounded-lg border border-dashed border-border bg-surface-layer px-4 py-8 text-center text-sm text-muted-foreground">
            {t('settings.emptyState')}
          </div>
        ) : null}

        {loading ? (
          <div className="mt-5 grid grid-cols-1 gap-4 xl:grid-cols-2">
            {Array.from({ length: 4 }).map((_, index) => (
              <div
                key={index}
                className="h-40 animate-pulse rounded-lg border border-border bg-surface-layer"
              />
            ))}
          </div>
        ) : null}
      </div>
    </div>
  )
}
