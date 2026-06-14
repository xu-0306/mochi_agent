'use client'

import * as React from 'react'
import { Input } from '@/components/ui/input'
import { Slider } from '@/components/ui/slider'
import { Switch } from '@/components/ui/switch'
import { Textarea } from '@/components/ui/textarea'
import type { ReasoningEffort } from '@/lib/api'
import { ThinkingLevelPanelControl } from './ThinkingLevelControls'
import type { InferenceParams } from '@/lib/stores/inference-store'

interface InferenceControlsProps {
  value: InferenceParams
  onChange: <K extends keyof InferenceParams>(key: K, value: InferenceParams[K]) => void
  supportsReasoningEffort?: boolean
  showReasoningEffort?: boolean
  reasoningEffortOptions?: ReasoningEffort[]
  disabledKeys?: Array<keyof InferenceParams>
  disabledReason?: string | null
}

function NumberControl({
  label,
  value,
  min,
  max,
  step,
  onChange,
  disabled = false,
}: {
  label: string
  value: number
  min: number
  max: number
  step: number
  onChange: (next: number) => void
  disabled?: boolean
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-3">
        <span className="text-xs font-medium text-muted-foreground">{label}</span>
        <Input
          type="number"
          min={min}
          max={max}
          step={step}
          value={Number.isFinite(value) ? value : 0}
          onChange={(event) => onChange(Number(event.target.value))}
          disabled={disabled}
          className="h-8 w-28 font-mono text-xs"
        />
      </div>
      <Slider
        min={min}
        max={max}
        step={step}
        value={[value]}
        onValueChange={(values: number[]) => onChange(values[0] ?? value)}
        disabled={disabled}
      />
    </div>
  )
}

export function InferenceControls({
  value,
  onChange,
  supportsReasoningEffort = false,
  showReasoningEffort = true,
  reasoningEffortOptions,
  disabledKeys = [],
  disabledReason = null,
}: InferenceControlsProps) {
  const disabledSet = React.useMemo(() => new Set(disabledKeys), [disabledKeys])
  const isDisabled = React.useCallback((key: keyof InferenceParams) => disabledSet.has(key), [disabledSet])

  return (
    <div className="space-y-4">
      {disabledReason && disabledKeys.length > 0 ? (
        <div className="rounded-md border border-border bg-canvas px-3 py-2 text-xs text-muted-foreground">
          <p className="font-medium text-foreground">This model ignores some chat inference controls.</p>
          <p className="mt-1">{disabledReason}</p>
        </div>
      ) : null}

      <label className="block space-y-1.5">
        <span className="text-xs font-medium text-muted-foreground">System Prompt</span>
        <Textarea
          value={value.systemPrompt}
          onChange={(event) => onChange('systemPrompt', event.target.value)}
          minRows={4}
          maxRows={10}
          className="font-mono text-xs"
        />
      </label>

      <NumberControl
        label="Temperature"
        value={value.temperature}
        min={0}
        max={2}
        step={0.05}
        onChange={(next) => onChange('temperature', next)}
        disabled={isDisabled('temperature')}
      />

      <div className="space-y-1.5">
        <span className="text-xs font-medium text-muted-foreground">Max Tokens</span>
        <Input
          type="number"
          min={1}
          max={131072}
          step={1}
          value={value.maxTokens}
          onChange={(event) => onChange('maxTokens', Number(event.target.value))}
          disabled={isDisabled('maxTokens')}
          className="h-8 font-mono text-xs"
        />
      </div>

      <NumberControl
        label="Top P"
        value={value.topP}
        min={0}
        max={1}
        step={0.05}
        onChange={(next) => onChange('topP', next)}
        disabled={isDisabled('topP')}
      />

      <NumberControl
        label="Min P"
        value={value.minP}
        min={0}
        max={1}
        step={0.05}
        onChange={(next) => onChange('minP', next)}
        disabled={isDisabled('minP')}
      />

      <div className="space-y-1.5">
        <span className="text-xs font-medium text-muted-foreground">Top K</span>
        <Input
          type="number"
          min={0}
          step={1}
          value={value.topK}
          onChange={(event) => onChange('topK', Number(event.target.value))}
          disabled={isDisabled('topK')}
          className="h-8 font-mono text-xs"
        />
      </div>

      <NumberControl
        label="Frequency Penalty"
        value={value.frequencyPenalty}
        min={-2}
        max={2}
        step={0.05}
        onChange={(next) => onChange('frequencyPenalty', next)}
        disabled={isDisabled('frequencyPenalty')}
      />

      <NumberControl
        label="Presence Penalty"
        value={value.presencePenalty}
        min={-2}
        max={2}
        step={0.05}
        onChange={(next) => onChange('presencePenalty', next)}
        disabled={isDisabled('presencePenalty')}
      />

      <NumberControl
        label="Repeat Penalty"
        value={value.repeatPenalty}
        min={0}
        max={2}
        step={0.05}
        onChange={(next) => onChange('repeatPenalty', next)}
        disabled={isDisabled('repeatPenalty')}
      />

      {showReasoningEffort && supportsReasoningEffort ? (
        <div className="space-y-1.5">
          <span className="text-xs font-medium text-muted-foreground">Thinking Level</span>
          <ThinkingLevelPanelControl
            supportedEfforts={reasoningEffortOptions}
            value={value.reasoningEffort}
            disabled={isDisabled('reasoningEffort')}
            onChange={(next) => onChange('reasoningEffort', next)}
          />
        </div>
      ) : null}

      <div className="flex items-center justify-between gap-3 rounded-md border border-border bg-canvas px-3 py-2">
        <div>
          <p className="text-sm text-foreground">Show Token Stats</p>
          <p className="text-xs text-muted-foreground">Display input/output tokens and generation speed.</p>
        </div>
        <Switch
          checked={value.showTokenStats}
          onCheckedChange={(checked) => onChange('showTokenStats', checked)}
        />
      </div>
    </div>
  )
}
