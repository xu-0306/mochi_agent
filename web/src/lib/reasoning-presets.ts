import type { ReasoningEffort } from '@/lib/api'

export type ThinkingLevel = 'auto' | 'fast' | 'balanced' | 'deep' | 'max'

export interface ThinkingLevelOption {
  value: ThinkingLevel
  label: string
  description: string
  effort: ReasoningEffort | null
  rawEffortLabel?: string
}

const ALL_REASONING_EFFORTS: ReasoningEffort[] = [
  'none',
  'minimal',
  'low',
  'medium',
  'high',
  'xhigh',
]

const THINKING_LEVEL_DEFINITIONS: Array<{
  value: Exclude<ThinkingLevel, 'auto'>
  label: string
  description: string
  priorities: ReasoningEffort[]
}> = [
  {
    value: 'fast',
    label: 'Fast',
    description: 'Keep latency down with a lighter reasoning pass.',
    priorities: ['minimal', 'low', 'none', 'medium', 'high', 'xhigh'],
  },
  {
    value: 'balanced',
    label: 'Balanced',
    description: 'Default tradeoff for most chats and coding work.',
    priorities: ['medium', 'low', 'minimal', 'high', 'none', 'xhigh'],
  },
  {
    value: 'deep',
    label: 'Deep',
    description: 'Spend more effort on harder planning and debugging tasks.',
    priorities: ['high', 'xhigh', 'medium', 'low', 'minimal', 'none'],
  },
  {
    value: 'max',
    label: 'Max',
    description: 'Use the highest supported effort when quality matters most.',
    priorities: ['xhigh', 'high', 'medium', 'low', 'minimal', 'none'],
  },
]

export function formatReasoningEffortLabel(value: ReasoningEffort | null | undefined): string {
  if (!value) {
    return 'Auto'
  }
  if (value === 'xhigh') {
    return 'Extra High'
  }
  if (value === 'minimal') {
    return 'Minimal'
  }
  if (value === 'none') {
    return 'None'
  }
  return value.charAt(0).toUpperCase() + value.slice(1)
}

function uniqueReasoningEfforts(
  supportedEfforts: readonly ReasoningEffort[] | null | undefined
): ReasoningEffort[] {
  const source = supportedEfforts && supportedEfforts.length > 0
    ? supportedEfforts
    : ALL_REASONING_EFFORTS
  const seen = new Set<ReasoningEffort>()
  const next: ReasoningEffort[] = []
  for (const effort of source) {
    if (!ALL_REASONING_EFFORTS.includes(effort) || seen.has(effort)) {
      continue
    }
    seen.add(effort)
    next.push(effort)
  }
  return next
}

function pickThinkingEffort(
  supportedEfforts: readonly ReasoningEffort[],
  priorities: readonly ReasoningEffort[]
): ReasoningEffort | null {
  for (const effort of priorities) {
    if (supportedEfforts.includes(effort)) {
      return effort
    }
  }
  return supportedEfforts[0] ?? null
}

export function resolveThinkingLevel(value: ReasoningEffort | null | undefined): ThinkingLevel {
  if (!value) {
    return 'auto'
  }
  if (value === 'medium') {
    return 'balanced'
  }
  if (value === 'high') {
    return 'deep'
  }
  if (value === 'xhigh') {
    return 'max'
  }
  return 'fast'
}

export function buildThinkingLevelOptions(
  supportedEfforts?: readonly ReasoningEffort[] | null
): ThinkingLevelOption[] {
  const availableEfforts = uniqueReasoningEfforts(supportedEfforts)
  const options: ThinkingLevelOption[] = [
    {
      value: 'auto',
      label: 'Auto',
      description: 'Let the model decide how much extra reasoning to use.',
      effort: null,
    },
  ]
  const usedEfforts = new Set<ReasoningEffort>()

  for (const definition of THINKING_LEVEL_DEFINITIONS) {
    const effort = pickThinkingEffort(availableEfforts, definition.priorities)
    if (!effort || usedEfforts.has(effort)) {
      continue
    }
    usedEfforts.add(effort)
    options.push({
      value: definition.value,
      label: definition.label,
      description: definition.description,
      effort,
      rawEffortLabel: formatReasoningEffortLabel(effort),
    })
  }

  return options
}

export function findThinkingLevelOption(
  value: ReasoningEffort | null | undefined,
  supportedEfforts?: readonly ReasoningEffort[] | null
): ThinkingLevelOption {
  const options = buildThinkingLevelOptions(supportedEfforts)
  if (!value) {
    return options[0]
  }

  const exact = options.find((option) => option.effort === value)
  if (exact) {
    return exact
  }

  const byLevel = options.find((option) => option.value === resolveThinkingLevel(value))
  return byLevel ?? options[0]
}

export function formatThinkingLevelSummary(value: ReasoningEffort | null | undefined): string {
  if (!value) {
    return 'Auto'
  }
  const option = findThinkingLevelOption(value)
  const rawLabel = formatReasoningEffortLabel(value)
  return option.rawEffortLabel && option.rawEffortLabel !== option.label
    ? `${option.label} (${rawLabel})`
    : option.label
}
