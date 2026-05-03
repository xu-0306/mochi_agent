'use client'

import * as React from 'react'
import {
  AlertCircle,
  Bot,
  BrainCircuit,
  CheckCircle2,
  Cpu,
  Database,
  Download,
  Globe,
  KeyRound,
  Mic,
  Network,
  PlugZap,
  RefreshCw,
  Save,
  Send,
  Sparkles,
  Terminal,
  Upload,
} from 'lucide-react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import * as api from '@/lib/api'
import {
  AUTO_LANGUAGE,
  AUTO_TIMEZONE,
  SYSTEM_APPEARANCE,
  type UIAppearanceMode,
  type UIFontSize,
  type UILanguageMode,
  useI18n,
} from '@/lib/i18n'

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
  fetchOllamaModels?: (baseUrl: string) => Promise<api.OllamaModelsResult>
  importFilesystemFiles?: (input: api.FilesystemImportInput) => Promise<api.FilesystemImportResult>
  updateSettings?: (input: api.UpdateSettingsInput) => Promise<api.Settings>
}

const settingsApi = api as ApiModule
const SENSITIVE_KEY_PATTERN = /(token|secret|password|api[_-]?key|credential|authorization)/i

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
      { label: 'Text Enabled', value: boolText(summary.textEnabled, t('common.yes'), t('common.no'), t('common.notReported')) },
      { label: 'Voice Enabled', value: boolText(summary.voiceEnabled, t('common.yes'), t('common.no'), t('common.notReported')) },
      { label: 'Allowed Guild IDs', value: summarizeIds(summary.allowedGuildIds, t) },
      { label: 'Allowed Voice Channel IDs', value: summarizeIds(summary.allowedVoiceChannelIds, t) },
      { label: 'Message Mode', value: summary.messageMode ?? t('common.notReported') },
      { label: 'Auto Join Policy', value: summary.autoJoinPolicy ?? t('common.notReported') },
      {
        label: 'Active Voice Rooms',
        value: summary.activeVoiceRoomCount !== null ? String(summary.activeVoiceRoomCount) : t('common.notReported'),
      },
      {
        label: 'Voice Runtime Phase',
        value: summary.voiceRuntimePhase ?? t('common.notReported'),
      },
      {
        label: 'Playback',
        value: summary.playbackState ?? primaryRoom?.playbackState ?? t('common.notReported'),
      },
      {
        label: 'Speaking',
        value: summary.speakingState ?? primaryRoom?.speakingState ?? t('common.notReported'),
      },
      {
        label: 'Voice Error',
        value: summary.voiceRuntimeError ?? primaryRoom?.error ?? t('common.notReported'),
      },
    ]

    if (primaryRoom) {
      voiceRuntimeDetails.push(
        { label: 'Active Guild ID', value: primaryRoom.guildId ?? t('common.notReported') },
        { label: 'Active Voice Channel ID', value: primaryRoom.channelId ?? t('common.notReported') },
        { label: 'Active Session ID', value: primaryRoom.sessionId ?? t('common.notReported') },
        { label: 'Joined At', value: primaryRoom.joinedAt ?? t('common.notReported') },
        {
          label: 'Participants',
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

function getPrimaryModel(settings: unknown, models: unknown[], t?: Translator): string {
  const root = asRecord(settings)
  const rootModel = root ? formatScalar(root.model, t) : null
  if (rootModel !== null) {
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

function baseUrlFromModelInfo(modelInfo: api.ModelInfo): string | null {
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
  title,
  icon: Icon,
  state,
  targetLabel,
  loading,
  onRefresh,
}: {
  title: string
  icon: React.ComponentType<{ className?: string }>
  state: ChannelPanelState
  targetLabel: string
  loading: boolean
  onRefresh: () => void
}) {
  const { t } = useI18n()
  const hasDetails = (state.details?.length ?? 0) > 0

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
          <Button type="button" variant="secondary" size="sm" disabled title={t('settings.disabled.connectUnavailable')}>
            {t('settings.action.connect')}
          </Button>
          <Button type="button" variant="secondary" size="sm" disabled title={t('settings.disabled.testUnavailable')}>
            {t('settings.action.test')}
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
]

const defaultSttBackends = [
  'auto',
  'faster-whisper',
  'openai-api',
  'openai-whisper',
  'qwen-asr',
  'vosk',
  'whisper-cpp',
  'whisperlivekit',
]

const defaultTtsBackends = [
  'auto',
  'edge-tts',
  'openai-tts',
  'piper',
  'coqui-tts',
  'kokoro-tts',
]

const defaultSttModelsByBackend: Record<string, string[]> = {
  auto: ['tiny', 'base', 'small', 'medium', 'large-v3', 'turbo', 'distil-large-v3'],
  'faster-whisper': ['tiny', 'base', 'small', 'medium', 'large-v3', 'turbo', 'distil-large-v3'],
  'openai-whisper': ['tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large-v3', 'turbo'],
  'openai-api': ['whisper-1'],
  'qwen-asr': ['qwen3-asr-0.6b', 'qwen3-asr-1.7b'],
  vosk: ['vosk-model-small-cn-0.22', 'vosk-model-cn-0.22', 'vosk-model-small-en-us-0.15'],
  'whisper-cpp': ['tiny', 'base', 'small', 'medium', 'large-v3'],
  whisperlivekit: ['tiny', 'base', 'small', 'medium', 'large-v3', 'turbo'],
}

const defaultTtsModelsByBackend: Record<string, string[]> = {
  auto: ['none'],
  'edge-tts': ['none'],
  'openai-tts': ['gpt-4o-mini-tts', 'tts-1', 'tts-1-hd'],
  piper: ['none'],
  'coqui-tts': [
    'tts_models/en/ljspeech/tacotron2-DDC',
    'tts_models/en/ljspeech/glow-tts',
    'tts_models/multilingual/multi-dataset/xtts_v2',
  ],
  'kokoro-tts': ['none'],
}

const defaultTtsVoice = 'en-US-AriaNeural'

const defaultTtsVoicesByBackend: Record<string, string[]> = {
  auto: [defaultTtsVoice, 'zh-CN-XiaoxiaoNeural', 'zh-TW-HsiaoChenNeural'],
  'edge-tts': [defaultTtsVoice, 'zh-CN-XiaoxiaoNeural', 'zh-TW-HsiaoChenNeural'],
  'openai-tts': ['alloy', 'verse', 'aria', 'coral', 'sage', 'nova', 'shimmer'],
  piper: ['zh_CN-huayan-medium', 'en_US-lessac-medium'],
  'coqui-tts': ['default'],
  'kokoro-tts': ['af_heart', 'af_bella', 'bf_emma', 'am_adam', 'bm_george'],
}

const sttLanguageOptions = ['auto', 'zh', 'en', 'ja', 'ko', 'fr', 'de', 'es']
const sttDeviceOptions = ['auto', 'cpu', 'cuda']
const ttsLanguageOptions = ['none', 'zh', 'en', 'ja', 'ko', 'fr', 'de', 'es']
const ttsSpeedOptions = ['0.75', '0.9', '1', '1.1', '1.25', '1.5']

type FormMessage = { type: 'success' | 'error'; text: string } | null

function isProviderChoice(value: unknown): value is ProviderChoice {
  return (
    value === 'ollama' ||
    value === 'openai_compat' ||
    value === 'gemini' ||
    value === 'anthropic'
  )
}

function providerOption(provider: ProviderChoice) {
  return providerOptions.find((item) => item.value === provider) ?? providerOptions[0]
}

function providerDescription(provider: ProviderChoice, t: Translator): string {
  const keys: Record<ProviderChoice, string> = {
    ollama: 'settings.provider.ollama.description',
    openai_compat: 'settings.provider.openaiCompat.description',
    gemini: 'settings.provider.gemini.description',
    anthropic: 'settings.provider.anthropic.description',
  }
  return t(keys[provider])
}

function providerNote(provider: ProviderChoice, t: Translator): string {
  const keys: Record<ProviderChoice, string> = {
    ollama: 'settings.provider.ollama.note',
    openai_compat: 'settings.provider.openaiCompat.note',
    gemini: 'settings.provider.gemini.note',
    anthropic: 'settings.provider.anthropic.note',
  }
  return t(keys[provider])
}

function configuredProvider(modelConfig: Record<string, unknown>, configuredModel: string | null): ProviderChoice {
  const provider = modelConfig.provider
  if (isProviderChoice(provider)) {
    return provider
  }
  if (configuredModel?.startsWith('http://') || configuredModel?.startsWith('https://')) {
    const remoteProvider = modelConfig.openai_compat_provider
    return isProviderChoice(remoteProvider) && remoteProvider !== 'ollama'
      ? remoteProvider
      : 'openai_compat'
  }
  return 'ollama'
}

function configuredBaseUrl(
  provider: ProviderChoice,
  modelConfig: Record<string, unknown>
): string {
  if (provider === 'ollama') {
    return getStringSetting(modelConfig, 'ollama_base_url', providerOption(provider).defaultBaseUrl)
  }
  return getStringSetting(modelConfig, 'openai_compat_base_url', providerOption(provider).defaultBaseUrl)
}

function configuredModelName(
  provider: ProviderChoice,
  configuredModel: string | null,
  modelConfig: Record<string, unknown>
): string {
  if (provider === 'ollama') {
    return (
      getStringSetting(modelConfig, 'ollama_model') ||
      configuredModel?.replace(/^ollama:/, '') ||
      providerOption(provider).defaultModel
    )
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
  className,
}: {
  value: string
  onValueChange: (value: string) => void
  options: string[]
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
            {option}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

function ModelConnectionForm({
  configuredModel,
  modelConfig,
  onConfigured,
}: {
  configuredModel: string | null
  modelConfig: Record<string, unknown>
  onConfigured: (result: api.ConfigureModelResult) => void
}) {
  const { t } = useI18n()
  const initialProvider = configuredProvider(modelConfig, configuredModel)
  const [provider, setProvider] = React.useState<ProviderChoice>(initialProvider)
  const currentProvider = providerOption(provider)
  const [baseUrl, setBaseUrl] = React.useState(configuredBaseUrl(initialProvider, modelConfig))
  const [model, setModel] = React.useState(configuredModelName(initialProvider, configuredModel, modelConfig))
  const [apiKey, setApiKey] = React.useState('')
  const [ollamaModels, setOllamaModels] = React.useState<string[]>([])
  const [discovering, setDiscovering] = React.useState(false)
  const [discoverMessage, setDiscoverMessage] = React.useState<FormMessage>(null)
  const [submitting, setSubmitting] = React.useState(false)
  const [message, setMessage] = React.useState<{ type: 'success' | 'error'; text: string } | null>(null)

  React.useEffect(() => {
    const nextProvider = configuredProvider(modelConfig, configuredModel)
    setProvider(nextProvider)
    setBaseUrl(configuredBaseUrl(nextProvider, modelConfig))
    setModel(configuredModelName(nextProvider, configuredModel, modelConfig))
    setOllamaModels([])
    setDiscoverMessage(null)
  }, [configuredModel, modelConfig])

  const handleProviderChange = (nextProvider: ProviderChoice) => {
    const next = providerOption(nextProvider)
    setProvider(nextProvider)
    setBaseUrl(next.defaultBaseUrl)
    setModel(next.defaultModel)
    setApiKey('')
    setOllamaModels([])
    setDiscoverMessage(null)
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

  React.useEffect(() => {
    if (provider !== 'ollama') {
      return
    }
    const timer = window.setTimeout(() => {
      void discoverOllamaModels()
    }, 600)
    return () => window.clearTimeout(timer)
  }, [discoverOllamaModels, provider])

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setSubmitting(true)
    setMessage(null)

    try {
      if (typeof settingsApi.configureModel !== 'function') {
        throw new Error('Model configure API client is unavailable.')
      }

      const result = await settingsApi.configureModel({
        provider,
        model,
        baseUrl,
        apiKey,
        persist: true,
      })
      onConfigured(result)
      setApiKey('')
      setMessage({
        type: 'success',
        text: result.persisted
          ? t('settings.modelConnection.successSwitchedPersisted', { model: result.activeModel.name || model })
          : t('settings.modelConnection.successSwitched', { model: result.activeModel.name || model }),
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
          <label className="min-w-0 space-y-1.5">
            <span className="text-xs font-medium text-muted-foreground">{t('settings.form.apiUrl')}</span>
            <div className="flex min-w-0 gap-2">
              <Input
                value={baseUrl}
                onChange={(event) => {
                  setBaseUrl(event.target.value)
                  setDiscoverMessage(null)
                }}
                placeholder={currentProvider.defaultBaseUrl}
                className="min-w-0 font-mono text-xs"
              />
              {provider === 'ollama' ? (
                <Button
                  type="button"
                  variant="secondary"
                  size="icon-sm"
                  loading={discovering}
                  aria-label={t('settings.modelConnection.refreshOllama')}
                  title={t('settings.modelConnection.refreshOllama')}
                  onClick={() => void discoverOllamaModels()}
                >
                  <RefreshCw className="h-3.5 w-3.5" />
                </Button>
              ) : null}
            </div>
          </label>

          <label className="min-w-0 space-y-1.5">
            <span className="text-xs font-medium text-muted-foreground">{t('settings.form.modelName')}</span>
            {provider === 'ollama' && ollamaModels.length > 0 ? (
              <SelectSetting
                value={model}
                onValueChange={setModel}
                options={withCurrentOption(ollamaModels, model)}
                className="font-mono text-xs"
              />
            ) : (
              <Input
                value={model}
                onChange={(event) => setModel(event.target.value)}
                placeholder={currentProvider.defaultModel}
                className="min-w-0 font-mono text-xs"
              />
            )}
          </label>
        </div>

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

        <div className="rounded-md border border-border bg-canvas px-3 py-2 text-xs text-muted-foreground">
          {providerNote(currentProvider.value, t)}
        </div>

        <SettingMessage message={discoverMessage} />

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
            {t('settings.action.applyTest')}
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
  voice: Record<string, unknown>
  onUpdated: (settings: api.Settings) => void
}) {
  const { t } = useI18n()
  const sttOptions = getStringOptions(voice, 'supported_stt_backends', defaultSttBackends)
  const ttsOptions = getStringOptions(voice, 'supported_tts_backends', defaultTtsBackends)
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
  const [ttsBackend, setTtsBackend] = React.useState(getStringSetting(voice, 'tts_backend', 'edge-tts'))
  const [ttsModel, setTtsModel] = React.useState(getStringSetting(voice, 'tts_model', 'none') || 'none')
  const [ttsVoice, setTtsVoice] = React.useState(getStringSetting(voice, 'tts_voice', defaultTtsVoice))
  const [ttsLanguage, setTtsLanguage] = React.useState(getStringSetting(voice, 'tts_language', 'none') || 'none')
  const [ttsSpeed, setTtsSpeed] = React.useState(String(getNumberSetting(voice, 'tts_speed', 1)))
  const [downloadMissing, setDownloadMissing] = React.useState(true)
  const [submitting, setSubmitting] = React.useState(false)
  const [message, setMessage] = React.useState<FormMessage>(null)
  const sttModelOptions = withCurrentOption(sttModelsByBackend[sttBackend] ?? defaultSttModelsByBackend['faster-whisper'], sttModel)
  const ttsModelOptions = withCurrentOption(ttsModelsByBackend[ttsBackend] ?? ['none'], ttsModel)
  const ttsVoiceOptions = withCurrentOption(ttsVoicesByBackend[ttsBackend] ?? defaultTtsVoicesByBackend['edge-tts'], ttsVoice)
  const modelFileInputRef = React.useRef<HTMLInputElement>(null)
  const [importingModel, setImportingModel] = React.useState(false)

  React.useEffect(() => {
    setEnabled(getBooleanSetting(voice, 'enabled'))
    setSttBackend(getStringSetting(voice, 'stt_backend', 'faster-whisper'))
    setSttModel(getStringSetting(voice, 'stt_model', 'medium'))
    setSttLanguage(getStringSetting(voice, 'stt_language', 'auto'))
    setSttDevice(getStringSetting(voice, 'stt_device', 'auto'))
    setSttCacheDir(getStringSetting(voice, 'stt_model_cache_dir'))
    setSttModelPath(getStringSetting(voice, 'stt_model_path'))
    setTtsBackend(getStringSetting(voice, 'tts_backend', 'edge-tts'))
    setTtsModel(getStringSetting(voice, 'tts_model', 'none') || 'none')
    setTtsVoice(getStringSetting(voice, 'tts_voice', defaultTtsVoice))
    setTtsLanguage(getStringSetting(voice, 'tts_language', 'none') || 'none')
    setTtsSpeed(String(getNumberSetting(voice, 'tts_speed', 1)))
  }, [voice])

  const handleSttBackendChange = (backend: string) => {
    setSttBackend(backend)
    setSttModel((sttModelsByBackend[backend] ?? defaultSttModelsByBackend['faster-whisper'])[0] ?? 'medium')
    setMessage(null)
  }

  const handleTtsBackendChange = (backend: string) => {
    setTtsBackend(backend)
    setTtsModel((ttsModelsByBackend[backend] ?? ['none'])[0] ?? 'none')
    setTtsVoice((ttsVoicesByBackend[backend] ?? defaultTtsVoicesByBackend['edge-tts'])[0] ?? defaultTtsVoice)
    setMessage(null)
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
          tts_backend: ttsBackend,
          tts_model: ttsModel === 'none' ? null : ttsModel,
          tts_voice: ttsVoice,
          tts_language: ttsLanguage === 'none' ? null : ttsLanguage,
          tts_speed: Number.parseFloat(ttsSpeed) || 1,
        },
        download_missing_models: downloadMissing,
        reload_voice: true,
      })
      onUpdated(settings)
      const download = asRecord(settings.update)?.download
      const status = asRecord(download)?.status
      setMessage({
        type: 'success',
        text: status
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
              <SelectSetting value={sttBackend} onValueChange={handleSttBackendChange} options={sttOptions} />
            </label>
            <label className="min-w-0 space-y-1.5 lg:col-span-1 xl:col-span-1">
              <SettingLabel>{t('settings.form.model')}</SettingLabel>
              <SelectSetting value={sttModel} onValueChange={setSttModel} options={sttModelOptions} className="font-mono text-xs" />
            </label>
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
              <SelectSetting value={ttsBackend} onValueChange={handleTtsBackendChange} options={ttsOptions} />
            </label>
            <label className="min-w-0 space-y-1.5 xl:col-span-2">
              <SettingLabel>{t('settings.form.model')}</SettingLabel>
              <SelectSetting value={ttsModel} onValueChange={setTtsModel} options={ttsModelOptions} className="font-mono text-xs" />
            </label>
            <label className="min-w-0 space-y-1.5 xl:col-span-2">
              <SettingLabel>{t('settings.form.voice')}</SettingLabel>
              <SelectSetting value={ttsVoice} onValueChange={setTtsVoice} options={ttsVoiceOptions} className="font-mono text-xs" />
            </label>
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

        <div className="flex flex-wrap items-center justify-between gap-3 rounded-md border border-border bg-canvas px-3 py-2">
          <div className="flex items-center gap-2">
            <Download className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm text-foreground">{t('settings.voice.downloadMissing')}</span>
          </div>
          <Switch checked={downloadMissing} onCheckedChange={setDownloadMissing} />
        </div>

        <SettingMessage message={message} />

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
  const [minSteps, setMinSteps] = React.useState(String(getNumberSetting(learning, 'min_steps_for_extraction', 3)))
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
    setMinSteps(String(getNumberSetting(learning, 'min_steps_for_extraction', 3)))
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
          min_steps_for_extraction: Number.parseInt(minSteps, 10) || 3,
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
        </div>

        <div className="grid grid-cols-1 gap-3 xl:grid-cols-4">
          <label className="space-y-1.5">
            <SettingLabel>{t('settings.learning.minSteps')}</SettingLabel>
            <Input value={minSteps} onChange={(event) => setMinSteps(event.target.value)} className="font-mono text-xs" />
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

function PreferencesPanel() {
  const {
    languageMode,
    setLanguageMode,
    appearanceMode,
    setAppearanceMode,
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

      <div className="grid grid-cols-1 gap-3 px-4 py-4 md:grid-cols-2 xl:grid-cols-4">
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
        <p>{t('settings.preferences.fontHelp')}</p>
        <p className="mt-1">{t('settings.preferences.timezoneHelp')}</p>
      </div>
    </section>
  )
}

export default function SettingsPage() {
  const { t } = useI18n()
  const [settings, setSettings] = React.useState<unknown>(null)
  const [models, setModels] = React.useState<api.ModelInfo[]>([])
  const [channelsStatus, setChannelsStatus] = React.useState<api.ChannelsStatus | null>(null)
  const [channelsError, setChannelsError] = React.useState<string | null>(null)
  const [channelsRefreshing, setChannelsRefreshing] = React.useState(false)
  const [loading, setLoading] = React.useState(true)
  const [error, setError] = React.useState<string | null>(null)

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

        setSettings(settingsResult)
        setModels(Array.isArray(modelsResult) ? modelsResult as api.ModelInfo[] : [])
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
  const rootModel = React.useMemo(() => {
    const root = asRecord(settings)
    return root ? formatScalar(root.model, t) : null
  }, [settings, t])
  const voiceSection = React.useMemo(() => extractSection(settings, 'voice'), [settings])
  const memorySection = React.useMemo(() => extractSection(settings, 'memory'), [settings])
  const learningSection = React.useMemo(() => extractSection(settings, 'learning'), [settings])
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
  const learningSummary = collectSummary(learningSection, ['enabled', 'auto_extract_skills', 'trajectory_path', 'skills_db_path', 'trajectory_retention_days', 'max_skills'], 6, t)
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

        <Tabs defaultValue="model" className="mt-5 flex gap-6">
          <div className="w-40 shrink-0">
            <TabsList className="flex h-auto w-full flex-col gap-0.5 p-1">
              <TabsTrigger value="model" className="w-full justify-start">{t('settings.tabs.model')}</TabsTrigger>
              <TabsTrigger value="voice" className="w-full justify-start">{t('settings.tabs.voice')}</TabsTrigger>
              <TabsTrigger value="memory" className="w-full justify-start">{t('settings.tabs.memory')}</TabsTrigger>
              <TabsTrigger value="learning" className="w-full justify-start">{t('settings.tabs.learning')}</TabsTrigger>
              <TabsTrigger value="channels" className="w-full justify-start">{t('settings.tabs.channels')}</TabsTrigger>
              <TabsTrigger value="web" className="w-full justify-start">{t('settings.tabs.web')}</TabsTrigger>
            </TabsList>
          </div>

          <div className="min-w-0 flex-1">
            <TabsContent value="model" className="mt-0">
              <div className="space-y-4">
                <ModelConnectionForm
                  configuredModel={rootModel}
                  modelConfig={modelConfigSection}
                  onConfigured={(result) => {
                    const modelInfo = result.activeModel
                    const remoteBaseUrl = baseUrlFromModelInfo(modelInfo)
                    setModels([modelInfo])
                    setSettings((current: unknown) => ({
                      ...(asRecord(current) ?? {}),
                      model: modelInfo.backendType === 'ollama'
                        ? `ollama:${modelInfo.name}`
                        : remoteBaseUrl ?? rootModel ?? modelInfo.name,
                      model_config: {
                        ...(asRecord(asRecord(current)?.model_config) ?? {}),
                        provider: result.provider,
                        ollama_model: modelInfo.backendType === 'ollama' ? modelInfo.name : '',
                        ollama_base_url: modelInfo.backendType === 'ollama' ? remoteBaseUrl : undefined,
                        openai_compat_provider: result.provider === 'ollama' ? undefined : result.provider,
                        openai_compat_base_url: result.provider === 'ollama' ? undefined : remoteBaseUrl,
                        openai_compat_model: modelInfo.backendType === 'openai_compat' ? modelInfo.name : '',
                      },
                    }))
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
                        (record && (formatScalar(record.name, t) ?? formatScalar(record.id, t) ?? formatScalar(record.model, t))) ||
                        `${t('settings.modelFallback')} ${index + 1}`
                      const value =
                        (record && (formatScalar(record.status, t) ?? summarizeValue(record.provider, t) ?? summarizeValue(record.backend, t))) ||
                        t('common.reported')
                      return { label, value }
                    })
                  }
                />
              </div>
              </div>
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

            <TabsContent value="channels" className="mt-0">
              <div className="space-y-4">
                {channelsError ? (
                  <div className="rounded-md border border-warning/30 bg-warning/10 px-3 py-2 text-sm text-warning">
                    {channelsError}
                  </div>
                ) : null}
                <ChannelPanel
                  title="Discord"
                  icon={Bot}
                  state={discordState}
                  targetLabel={t('settings.channel.allowedChannelIds')}
                  loading={channelsRefreshing}
                  onRefresh={() => void refreshChannelsStatus()}
                />
                <ChannelPanel
                  title="Telegram"
                  icon={Send}
                  state={telegramState}
                  targetLabel={t('settings.channel.allowedChatIds')}
                  loading={channelsRefreshing}
                  onRefresh={() => void refreshChannelsStatus()}
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
                <SurfaceSection
                  title={t('settings.section.web')}
                  description={t('settings.summary.web')}
                  items={webSummary}
                />
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
