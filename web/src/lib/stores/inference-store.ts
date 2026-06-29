'use client'

import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type {
  AgentSettings,
  InferencePreset,
  ModelInfo,
  ReasoningEffort,
  Settings,
} from '@/lib/api'

const AUTO_MAX_OUTPUT_TOKENS_FALLBACK = 4096
const AUTO_RESERVE_OUTPUT_TOKENS_FALLBACK = 1024
const AUTO_MAX_OUTPUT_TOKENS_MIN = 2048
const AUTO_MAX_OUTPUT_TOKENS_MAX = 8192
const AUTO_RESERVE_OUTPUT_TOKENS_MIN = 768
const AUTO_RESERVE_OUTPUT_TOKENS_MAX = 3072
const AUTO_OUTPUT_CONTEXT_RATIO = 0.1
const AUTO_RESERVE_OUTPUT_RATIO = 0.33
const AUTO_TOKEN_ROUNDING = 256

export interface InferenceParams {
  systemPrompt: string
  temperature: number
  maxTokens: number | null
  reserveOutputTokens: number | null
  topP: number
  minP: number
  topK: number
  frequencyPenalty: number
  presencePenalty: number
  repeatPenalty: number
  reasoningEffort: ReasoningEffort | null
  showTokenStats: boolean
}

export type SessionInferenceOverride = Partial<InferenceParams>

export interface InferenceTokenControlState {
  mode: 'auto' | 'manual'
  configuredValue: number | null
  effectiveValue: number
  autoValue: number
  contextLength: number | null
  source: 'context' | 'fallback'
}

export interface InferenceTokenControlsState {
  maxTokens: InferenceTokenControlState
  reserveOutputTokens: InferenceTokenControlState
}

interface InferenceStore {
  panelOpen: boolean
  sessionOverridesById: Record<string, SessionInferenceOverride>
  setPanelOpen: (open: boolean) => void
  setSessionOverride: <K extends keyof InferenceParams>(
    sessionId: string,
    key: K,
    value: InferenceParams[K]
  ) => void
  replaceSessionOverride: (sessionId: string, override: SessionInferenceOverride) => void
  resetSessionOverride: (sessionId: string) => void
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function getNullableString(value: unknown): string | null {
  return typeof value === 'string' && value.trim().length > 0 ? value : null
}

function getNullableNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function normalizePositiveInteger(value: unknown): number | null {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return null
  }
  const parsed = Math.trunc(value)
  return parsed > 0 ? parsed : null
}

function normalizeNonnegativeInteger(value: unknown): number | null {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return null
  }
  return Math.max(0, Math.trunc(value))
}

function clampNumber(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(value, maximum))
}

function roundUpTokenBucket(value: number): number {
  if (value <= 0) {
    return AUTO_TOKEN_ROUNDING
  }
  return Math.ceil(value / AUTO_TOKEN_ROUNDING) * AUTO_TOKEN_ROUNDING
}

function looksLikeGgufModel(value: string | null): boolean {
  return typeof value === 'string' && value.trim().toLowerCase().endsWith('.gguf')
}

function resolveInferenceContextLengthHint(
  settings?: Pick<Settings, 'model' | 'model_config' | 'ollama' | 'gguf' | 'vllm'> | null,
  activeModelInfo?: Pick<ModelInfo, 'contextLength' | 'backendType' | 'metadata'> | Record<string, unknown> | null
): number | null {
  const activeModelRecord = isRecord(activeModelInfo)
    ? (activeModelInfo as Record<string, unknown>)
    : null
  const activeMetadata = isRecord(activeModelRecord?.metadata)
    ? (activeModelRecord.metadata as Record<string, unknown>)
    : null
  const activeContextLength =
    getNullableNumber(activeModelRecord?.contextLength) ??
    getNullableNumber(activeModelRecord?.context_length)

  const modelConfig = isRecord(settings?.model_config) ? settings?.model_config : null
  const configuredProvider = getNullableString(modelConfig?.provider)
  const openaiCompatProvider = getNullableString(modelConfig?.openai_compat_provider)
  const localModelPath = getNullableString(modelConfig?.local_model_path) ?? settings?.model ?? null
  const backendType =
    getNullableString(activeModelRecord?.backendType) ??
    getNullableString(activeModelRecord?.backend_type)

  if (configuredProvider === 'ollama' || backendType === 'ollama') {
    const runtimeContextLength = getNullableNumber(activeMetadata?.runtime_context_length)
    if (runtimeContextLength !== null) {
      return runtimeContextLength
    }
    if (settings?.ollama?.num_ctx !== null && settings?.ollama?.num_ctx !== undefined) {
      return settings.ollama.num_ctx
    }
    if (activeContextLength !== null) {
      return activeContextLength
    }
    return getNullableNumber(activeMetadata?.model_max_context_length)
  }

  if (
    configuredProvider === 'local' ||
    backendType === 'gguf' ||
    looksLikeGgufModel(localModelPath)
  ) {
    if (settings?.gguf?.n_ctx !== null && settings?.gguf?.n_ctx !== undefined) {
      return settings.gguf.n_ctx
    }
    return activeContextLength
  }

  if (
    configuredProvider === 'vllm' ||
    openaiCompatProvider === 'vllm' ||
    backendType === 'vllm'
  ) {
    if (settings?.vllm?.max_model_len !== null && settings?.vllm?.max_model_len !== undefined) {
      return settings.vllm.max_model_len
    }
    return activeContextLength
  }

  return activeContextLength
}

function deriveAutoMaxOutputTokens(contextLength: number | null): number {
  if (contextLength === null || contextLength <= 0) {
    return AUTO_MAX_OUTPUT_TOKENS_FALLBACK
  }
  const scaled = roundUpTokenBucket(contextLength * AUTO_OUTPUT_CONTEXT_RATIO)
  return clampNumber(
    scaled,
    AUTO_MAX_OUTPUT_TOKENS_MIN,
    AUTO_MAX_OUTPUT_TOKENS_MAX
  )
}

function deriveAutoReserveOutputTokens(
  contextLength: number | null,
  outputCap: number
): number {
  if (contextLength === null || contextLength <= 0) {
    return Math.min(outputCap, AUTO_RESERVE_OUTPUT_TOKENS_FALLBACK)
  }
  const scaled = roundUpTokenBucket(outputCap * AUTO_RESERVE_OUTPUT_RATIO)
  return Math.min(
    outputCap,
    clampNumber(
      scaled,
      AUTO_RESERVE_OUTPUT_TOKENS_MIN,
      AUTO_RESERVE_OUTPUT_TOKENS_MAX
    )
  )
}

export function resolveInferenceTokenControls(
  params: Pick<InferenceParams, 'maxTokens' | 'reserveOutputTokens'>,
  settings?: Pick<Settings, 'model' | 'model_config' | 'ollama' | 'gguf' | 'vllm'> | null,
  activeModelInfo?: Pick<ModelInfo, 'contextLength' | 'backendType' | 'metadata'> | Record<string, unknown> | null
): InferenceTokenControlsState {
  const contextLength = resolveInferenceContextLengthHint(settings, activeModelInfo)
  const autoMaxTokens = deriveAutoMaxOutputTokens(contextLength)
  const configuredMaxTokens = normalizePositiveInteger(params.maxTokens)
  const effectiveMaxTokens = configuredMaxTokens ?? autoMaxTokens
  const autoReserveOutputTokens = deriveAutoReserveOutputTokens(contextLength, effectiveMaxTokens)
  const configuredReserveOutputTokens = normalizeNonnegativeInteger(params.reserveOutputTokens)
  const effectiveReserveOutputTokens = Math.min(
    effectiveMaxTokens,
    configuredReserveOutputTokens ?? autoReserveOutputTokens
  )
  const source = contextLength !== null ? 'context' : 'fallback'

  return {
    maxTokens: {
      mode: configuredMaxTokens === null ? 'auto' : 'manual',
      configuredValue: configuredMaxTokens,
      effectiveValue: effectiveMaxTokens,
      autoValue: autoMaxTokens,
      contextLength,
      source,
    },
    reserveOutputTokens: {
      mode: configuredReserveOutputTokens === null ? 'auto' : 'manual',
      configuredValue: configuredReserveOutputTokens,
      effectiveValue: effectiveReserveOutputTokens,
      autoValue: autoReserveOutputTokens,
      contextLength,
      source,
    },
  }
}

export function inferencePresetToParams(preset: InferencePreset): InferenceParams {
  return {
    systemPrompt: preset.system_prompt,
    temperature: preset.temperature,
    maxTokens: preset.max_tokens,
    reserveOutputTokens: preset.reserve_output_tokens,
    topP: preset.top_p,
    minP: preset.min_p,
    topK: preset.top_k,
    frequencyPenalty: preset.frequency_penalty,
    presencePenalty: preset.presence_penalty,
    repeatPenalty: preset.repeat_penalty,
    reasoningEffort: preset.reasoning_effort ?? null,
    showTokenStats: false,
  }
}

export function agentSettingsToParams(agent?: AgentSettings): InferenceParams {
  return {
    systemPrompt: agent?.system_prompt ?? '',
    temperature: agent?.temperature ?? 0.7,
    maxTokens: agent?.max_tokens ?? null,
    reserveOutputTokens: agent?.reserve_output_tokens ?? null,
    topP: agent?.top_p ?? 1.0,
    minP: agent?.min_p ?? 0.0,
    topK: agent?.top_k ?? 0,
    frequencyPenalty: agent?.frequency_penalty ?? 0.0,
    presencePenalty: agent?.presence_penalty ?? 0.0,
    repeatPenalty: agent?.repeat_penalty ?? 1.0,
    reasoningEffort: agent?.reasoning_effort ?? null,
    showTokenStats: agent?.show_token_stats ?? false,
  }
}

export function getActivePreset(agent?: AgentSettings): InferencePreset | null {
  if (!agent || agent.presets.length === 0) {
    return null
  }
  return agent.presets.find((preset) => preset.name === agent.active_preset) ?? agent.presets[0] ?? null
}

export function resolveEffectiveInferenceParams(
  sessionOverride: SessionInferenceOverride | undefined,
  agent?: AgentSettings
): InferenceParams {
  const activePreset = getActivePreset(agent)
  const presetParams = activePreset ? inferencePresetToParams(activePreset) : agentSettingsToParams(agent)
  const agentParams = agentSettingsToParams(agent)
  const presetSystemPrompt = activePreset?.system_prompt || undefined

  return {
    ...agentParams,
    ...presetParams,
    systemPrompt: presetSystemPrompt ?? agentParams.systemPrompt,
    showTokenStats: agentParams.showTokenStats,
    ...sessionOverride,
  }
}

export const useInferenceStore = create<InferenceStore>()(
  persist(
    (set) => ({
      panelOpen: false,
      sessionOverridesById: {},
      setPanelOpen: (open) => set({ panelOpen: open }),
      setSessionOverride: (sessionId, key, value) =>
        set((state) => ({
          sessionOverridesById: {
            ...state.sessionOverridesById,
            [sessionId]: {
              ...(state.sessionOverridesById[sessionId] ?? {}),
              [key]: value,
            },
          },
        })),
      replaceSessionOverride: (sessionId, override) =>
        set((state) => ({
          sessionOverridesById: {
            ...state.sessionOverridesById,
            [sessionId]: { ...override },
          },
        })),
      resetSessionOverride: (sessionId) =>
        set((state) => {
          const next = { ...state.sessionOverridesById }
          delete next[sessionId]
          return { sessionOverridesById: next }
        }),
    }),
    {
      name: 'mochi.inference-ui.v1',
      partialize: (state) => ({
        panelOpen: state.panelOpen,
        sessionOverridesById: state.sessionOverridesById,
      }),
    }
  )
)
