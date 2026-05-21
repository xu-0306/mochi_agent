'use client'

import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { AgentSettings, InferencePreset } from '@/lib/api'

export interface InferenceParams {
  systemPrompt: string
  temperature: number
  maxTokens: number
  topP: number
  minP: number
  topK: number
  frequencyPenalty: number
  presencePenalty: number
  repeatPenalty: number
  showTokenStats: boolean
}

export type SessionInferenceOverride = Partial<InferenceParams>

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

export function inferencePresetToParams(preset: InferencePreset): InferenceParams {
  return {
    systemPrompt: preset.system_prompt,
    temperature: preset.temperature,
    maxTokens: preset.max_tokens,
    topP: preset.top_p,
    minP: preset.min_p,
    topK: preset.top_k,
    frequencyPenalty: preset.frequency_penalty,
    presencePenalty: preset.presence_penalty,
    repeatPenalty: preset.repeat_penalty,
    showTokenStats: false,
  }
}

export function agentSettingsToParams(agent?: AgentSettings): InferenceParams {
  return {
    systemPrompt: agent?.system_prompt ?? '',
    temperature: agent?.temperature ?? 0.7,
    maxTokens: agent?.max_tokens ?? 4096,
    topP: agent?.top_p ?? 1.0,
    minP: agent?.min_p ?? 0.0,
    topK: agent?.top_k ?? 0,
    frequencyPenalty: agent?.frequency_penalty ?? 0.0,
    presencePenalty: agent?.presence_penalty ?? 0.0,
    repeatPenalty: agent?.repeat_penalty ?? 1.0,
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

  return {
    ...agentParams,
    ...presetParams,
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
