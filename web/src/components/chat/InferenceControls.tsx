'use client'

import * as React from 'react'
import { Input } from '@/components/ui/input'
import { Slider } from '@/components/ui/slider'
import { Switch } from '@/components/ui/switch'
import { Textarea } from '@/components/ui/textarea'
import type { InferenceParams } from '@/lib/stores/inference-store'

interface InferenceControlsProps {
  value: InferenceParams
  onChange: <K extends keyof InferenceParams>(key: K, value: InferenceParams[K]) => void
}

function NumberControl({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string
  value: number
  min: number
  max: number
  step: number
  onChange: (next: number) => void
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
          className="h-8 w-28 font-mono text-xs"
        />
      </div>
      <Slider
        min={min}
        max={max}
        step={step}
        value={[value]}
        onValueChange={(values: number[]) => onChange(values[0] ?? value)}
      />
    </div>
  )
}

export function InferenceControls({ value, onChange }: InferenceControlsProps) {
  return (
    <div className="space-y-4">
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
      />

      <NumberControl
        label="Min P"
        value={value.minP}
        min={0}
        max={1}
        step={0.05}
        onChange={(next) => onChange('minP', next)}
      />

      <div className="space-y-1.5">
        <span className="text-xs font-medium text-muted-foreground">Top K</span>
        <Input
          type="number"
          min={0}
          step={1}
          value={value.topK}
          onChange={(event) => onChange('topK', Number(event.target.value))}
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
      />

      <NumberControl
        label="Presence Penalty"
        value={value.presencePenalty}
        min={-2}
        max={2}
        step={0.05}
        onChange={(next) => onChange('presencePenalty', next)}
      />

      <NumberControl
        label="Repeat Penalty"
        value={value.repeatPenalty}
        min={0}
        max={2}
        step={0.05}
        onChange={(next) => onChange('repeatPenalty', next)}
      />

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
