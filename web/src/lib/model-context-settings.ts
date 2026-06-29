import type { ModelInfo, Settings, UpdateSettingsInput } from './api'

export type ContextLengthSettingsKind = 'ollama' | 'gguf' | 'vllm' | null
export type ContextLengthSettingsSource = 'config' | 'runtime' | 'model' | null

export interface ContextLengthSettingsTarget {
  kind: ContextLengthSettingsKind
  value: number | null
  persistedValue: number | null
  detectedValue: number | null
  source: ContextLengthSettingsSource
}

type SettingsLike = Pick<Settings, 'model' | 'model_config' | 'ollama' | 'gguf' | 'vllm'>
type ModelInfoLike = Pick<ModelInfo, 'contextLength' | 'backendType' | 'metadata'> | Record<string, unknown> | null | undefined

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null
}

function getString(value: unknown): string | null {
  return typeof value === 'string' && value.trim().length > 0 ? value : null
}

function getNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function looksLikeGgufModel(value: string | null): boolean {
  return typeof value === 'string' && value.trim().toLowerCase().endsWith('.gguf')
}

function getActiveModelMetadata(activeModelInfo: ModelInfoLike): Record<string, unknown> | null {
  if (!activeModelInfo) {
    return null
  }
  const record = asRecord(activeModelInfo)
  return asRecord(record?.metadata)
}

function getActiveModelContextLength(activeModelInfo: ModelInfoLike): number | null {
  if (!activeModelInfo) {
    return null
  }
  const record = asRecord(activeModelInfo)
  if (record) {
    return getNumber(record.contextLength) ?? getNumber(record.context_length)
  }
  return null
}

function resolveDetectedContextValue(
  kind: Exclude<ContextLengthSettingsKind, null>,
  activeModelInfo: ModelInfoLike,
): { value: number | null; source: Exclude<ContextLengthSettingsSource, 'config' | null> | null } {
  const metadata = getActiveModelMetadata(activeModelInfo)
  const contextLength = getActiveModelContextLength(activeModelInfo)

  if (kind === 'ollama') {
    const runtimeContextLength = getNumber(metadata?.runtime_context_length)
    if (runtimeContextLength !== null) {
      return { value: runtimeContextLength, source: 'runtime' }
    }
    if (contextLength !== null) {
      return { value: contextLength, source: 'model' }
    }
    const modelMaxContextLength = getNumber(metadata?.model_max_context_length)
    if (modelMaxContextLength !== null) {
      return { value: modelMaxContextLength, source: 'model' }
    }
    return { value: null, source: null }
  }

  if (contextLength !== null) {
    return { value: contextLength, source: 'model' }
  }

  return { value: null, source: null }
}

export function resolveContextLengthSettingsTarget(
  settings: SettingsLike | null | undefined,
  activeModelInfo?: ModelInfoLike,
): ContextLengthSettingsTarget {
  if (!settings) {
    return { kind: null, value: null, persistedValue: null, detectedValue: null, source: null }
  }

  const modelConfig = asRecord(settings.model_config)
  const configuredProvider = getString(modelConfig?.provider)
  const openaiCompatProvider = getString(modelConfig?.openai_compat_provider)
  const vllmLaunchMode = getString(modelConfig?.vllm_launch_mode)
  const localModelPath = getString(modelConfig?.local_model_path) ?? settings.model

  if (configuredProvider === 'ollama') {
    const persistedValue = settings.ollama?.num_ctx ?? null
    const detected = resolveDetectedContextValue('ollama', activeModelInfo)
    return {
      kind: 'ollama',
      value: persistedValue ?? detected.value,
      persistedValue,
      detectedValue: detected.value,
      source: persistedValue !== null ? 'config' : detected.source,
    }
  }

  if ((configuredProvider === 'vllm' || openaiCompatProvider === 'vllm') && vllmLaunchMode === 'managed') {
    const persistedValue = settings.vllm?.max_model_len ?? null
    const detected = resolveDetectedContextValue('vllm', activeModelInfo)
    return {
      kind: 'vllm',
      value: persistedValue ?? detected.value,
      persistedValue,
      detectedValue: detected.value,
      source: persistedValue !== null ? 'config' : detected.source,
    }
  }

  if ((configuredProvider === 'local' || looksLikeGgufModel(localModelPath)) && looksLikeGgufModel(localModelPath)) {
    const persistedValue = settings.gguf?.n_ctx ?? null
    const detected = resolveDetectedContextValue('gguf', activeModelInfo)
    return {
      kind: 'gguf',
      value: persistedValue ?? detected.value,
      persistedValue,
      detectedValue: detected.value,
      source: persistedValue !== null ? 'config' : detected.source,
    }
  }

  return { kind: null, value: null, persistedValue: null, detectedValue: null, source: null }
}

export function buildContextLengthSettingsUpdate(
  kind: ContextLengthSettingsKind,
  value: number | null,
): UpdateSettingsInput {
  if (kind === 'ollama') {
    return {
      ollama: {
        num_ctx: value,
      },
    }
  }

  if (kind === 'gguf') {
    if (value === null) {
      throw new Error('GGUF context length requires a numeric n_ctx value.')
    }
    return {
      gguf: {
        n_ctx: value,
      },
    }
  }

  if (kind === 'vllm') {
    return {
      vllm: {
        max_model_len: value,
      },
    }
  }

  return {}
}

export function contextLengthSettingsTitle(kind: ContextLengthSettingsKind): string {
  if (kind === 'ollama' || kind === 'gguf') {
    return 'Context window'
  }
  if (kind === 'vllm') {
    return 'Max model length'
  }
  return 'Context setting'
}

export function contextLengthSettingsFieldLabel(kind: ContextLengthSettingsKind): string {
  if (kind === 'vllm') {
    return 'Max model length'
  }
  return 'Context length'
}

export function contextLengthSettingsBadge(kind: ContextLengthSettingsKind): string {
  if (kind === 'ollama') {
    return 'ollama.num_ctx'
  }
  if (kind === 'gguf') {
    return 'gguf.n_ctx'
  }
  if (kind === 'vllm') {
    return 'vllm.max_model_len'
  }
  return 'context'
}

export function contextLengthSettingsDescription(target: ContextLengthSettingsTarget): string {
  if (target.kind === 'ollama') {
    return 'Sets the Ollama runtime `num_ctx` override. Leave blank to follow the server default.'
  }
  if (target.kind === 'gguf') {
    return 'Writes to `gguf.n_ctx` for the active GGUF model.'
  }
  if (target.kind === 'vllm') {
    return 'Writes the managed vLLM startup override for `vllm.max_model_len`. Leave blank to use auto sizing.'
  }
  return ''
}

export function contextLengthSettingsPlaceholder(target: ContextLengthSettingsTarget): string {
  if (target.kind === 'gguf') {
    return '4096'
  }
  if (target.value !== null) {
    return String(target.value)
  }
  return 'auto'
}

export function contextLengthSettingsRuntimeHint(target: ContextLengthSettingsTarget): string | null {
  if (target.persistedValue !== null) {
    return 'Saved as an explicit override.'
  }
  if (target.kind === 'ollama' && target.detectedValue !== null) {
    return `Using Ollama runtime default from the server: ${target.detectedValue.toLocaleString()}. Saving will pin this value as an override.`
  }
  if (target.kind === 'vllm' && target.detectedValue !== null) {
    return `Current runtime model length: ${target.detectedValue.toLocaleString()}. Saving will pin this value as an override.`
  }
  if (target.kind === 'gguf' && target.detectedValue !== null) {
    return `Current runtime context length: ${target.detectedValue.toLocaleString()}.`
  }
  return null
}
