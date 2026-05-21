import type { Settings, UpdateSettingsInput } from './api'

export type ContextLengthSettingsKind = 'gguf' | 'vllm' | null

export interface ContextLengthSettingsTarget {
  kind: ContextLengthSettingsKind
  value: number | null
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null
}

function getString(value: unknown): string | null {
  return typeof value === 'string' && value.trim().length > 0 ? value : null
}

function looksLikeGgufModel(value: string | null): boolean {
  return typeof value === 'string' && value.trim().toLowerCase().endsWith('.gguf')
}

export function resolveContextLengthSettingsTarget(
  settings: Pick<Settings, 'model' | 'model_config' | 'gguf' | 'vllm'> | null | undefined,
): ContextLengthSettingsTarget {
  if (!settings) {
    return { kind: null, value: null }
  }

  const modelConfig = asRecord(settings.model_config)
  const configuredProvider = getString(modelConfig?.provider)
  const openaiCompatProvider = getString(modelConfig?.openai_compat_provider)
  const localModelPath = getString(modelConfig?.local_model_path) ?? settings.model

  if (configuredProvider === 'vllm' || openaiCompatProvider === 'vllm') {
    return {
      kind: 'vllm',
      value: settings.vllm?.max_model_len ?? null,
    }
  }

  if ((configuredProvider === 'local' || looksLikeGgufModel(localModelPath)) && looksLikeGgufModel(localModelPath)) {
    return {
      kind: 'gguf',
      value: settings.gguf?.n_ctx ?? null,
    }
  }

  return { kind: null, value: null }
}

export function buildContextLengthSettingsUpdate(
  kind: ContextLengthSettingsKind,
  value: number | null,
): UpdateSettingsInput {
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
