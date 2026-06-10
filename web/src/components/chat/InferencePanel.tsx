'use client'

import * as React from 'react'
import { PanelRightClose, RotateCcw, SlidersHorizontal, Sparkles } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from '@/components/ui/sheet'
import * as api from '@/lib/api'
import type { AgentSettings, InferencePreset } from '@/lib/api'
import { InferenceControls } from '@/components/chat/InferenceControls'
import type { InferenceParams } from '@/lib/stores/inference-store'
import { cn } from '@/lib/utils'
import {
  buildContextLengthSettingsUpdate,
  resolveContextLengthSettingsTarget,
} from '@/lib/model-context-settings'

interface InferencePanelProps {
  open: boolean
  mobileOpen: boolean
  onOpenChange: (open: boolean) => void
  onMobileOpenChange: (open: boolean) => void
  presets: InferencePreset[]
  activePresetName: string
  selectedPresetName: string
  onSelectedPresetChange: (name: string) => void
  value: InferenceParams
  onChange: <K extends keyof InferenceParams>(key: K, value: InferenceParams[K]) => void
  onApplyPreset: () => void
  onReset: () => void
  onSavePreset: () => void
  isSavingPreset?: boolean
  supportsReasoningEffort?: boolean
  showReasoningEffort?: boolean
  reasoningEffortOptions?: api.ReasoningEffort[]
  disabledKeys?: Array<keyof InferenceParams>
  disabledReason?: string | null
  agent?: AgentSettings
  settings?: api.Settings | null
  onSettingsUpdated?: (settings: api.Settings) => void
}

function SectionCard({
  title,
  description,
  children,
}: {
  title: string
  description?: string
  children: React.ReactNode
}) {
  return (
    <section className="rounded-2xl border border-white/8 bg-canvas/70 p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)] backdrop-blur-sm">
      <div className="mb-3">
        <h3 className="text-sm font-semibold text-foreground">{title}</h3>
        {description ? (
          <p className="mt-1 text-[11px] leading-relaxed text-muted-foreground">{description}</p>
        ) : null}
      </div>
      {children}
    </section>
  )
}

function PanelBody({
  presets,
  selectedPresetName,
  activePresetName,
  onSelectedPresetChange,
  value,
  onChange,
  onApplyPreset,
  onReset,
  onSavePreset,
  isSavingPreset,
  supportsReasoningEffort = false,
  showReasoningEffort = true,
  reasoningEffortOptions,
  disabledKeys,
  disabledReason,
  settings,
  onSettingsUpdated,
  onClose,
}: Omit<InferencePanelProps, 'open' | 'mobileOpen' | 'onOpenChange' | 'onMobileOpenChange' | 'agent'> & {
  onClose?: () => void
}) {
  const contextLengthTarget = React.useMemo(
    () => resolveContextLengthSettingsTarget(settings),
    [settings]
  )
  const [contextLengthInput, setContextLengthInput] = React.useState(
    contextLengthTarget.value === null ? '' : String(contextLengthTarget.value)
  )
  const [contextSettingsBusy, setContextSettingsBusy] = React.useState(false)
  const [contextSettingsMessage, setContextSettingsMessage] = React.useState<string | null>(null)

  React.useEffect(() => {
    setContextLengthInput(contextLengthTarget.value === null ? '' : String(contextLengthTarget.value))
    setContextSettingsMessage(null)
  }, [contextLengthTarget.kind, contextLengthTarget.value])

  const handleSaveContextSettings = React.useCallback(async () => {
    if (!contextLengthTarget.kind) {
      return
    }

    const trimmed = contextLengthInput.trim()
    let parsedValue: number | null = null

    if (contextLengthTarget.kind === 'gguf') {
      const parsed = Number.parseInt(trimmed, 10)
      if (!Number.isInteger(parsed) || parsed <= 0) {
        setContextSettingsMessage('GGUF n_ctx must be a positive integer.')
        return
      }
      parsedValue = parsed
    } else if (trimmed.length > 0) {
      const parsed = Number.parseInt(trimmed, 10)
      if (!Number.isInteger(parsed) || parsed <= 0) {
        setContextSettingsMessage('vLLM max model length must be a positive integer or left blank for auto.')
        return
      }
      parsedValue = parsed
    }

    setContextSettingsBusy(true)
    setContextSettingsMessage(null)
    try {
      const nextSettings = await api.updateSettings({
        ...buildContextLengthSettingsUpdate(contextLengthTarget.kind, parsedValue),
      })
      onSettingsUpdated?.(nextSettings)
      window.dispatchEvent(new Event('mochi:settings-updated'))
      setContextSettingsMessage(
        contextLengthTarget.kind === 'gguf'
          ? 'Saved GGUF context window.'
          : 'Saved vLLM max model length.'
      )
    } catch (error) {
      setContextSettingsMessage(
        error instanceof Error ? error.message : 'Failed to save model context settings.'
      )
    } finally {
      setContextSettingsBusy(false)
    }
  }, [contextLengthInput, contextLengthTarget.kind, onSettingsUpdated])

  return (
    <div className="flex h-full flex-col overflow-hidden rounded-[inherit] bg-[radial-gradient(circle_at_top_right,rgba(94,106,210,0.18),transparent_34%),linear-gradient(180deg,rgba(255,255,255,0.04),transparent_24%)]">
      <div className="border-b border-white/8 bg-canvas/40 px-4 py-4 backdrop-blur-xl">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="mb-2 inline-flex items-center gap-2 rounded-full border border-primary-500/20 bg-primary-500/10 px-2.5 py-1 text-[11px] font-medium uppercase tracking-[0.18em] text-primary-300">
              <SlidersHorizontal className="h-3.5 w-3.5" />
              Inference Lab
            </div>
            <h2 className="text-base font-semibold text-foreground">Session controls</h2>
            <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
              Tune the active chat without shifting the conversation layout.
            </p>
          </div>
          {onClose ? (
            <Button
              type="button"
              size="icon-sm"
              variant="ghost"
              onClick={onClose}
              title="Hide inference controls"
              aria-label="Hide inference controls"
              className="mt-0.5 rounded-full border border-white/8 bg-canvas/55 text-muted-foreground hover:bg-elevated-layer hover:text-foreground"
            >
              <PanelRightClose className="h-4 w-4" />
            </Button>
          ) : null}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-4">
        <div className="space-y-4">
          <SectionCard
            title="Preset workspace"
            description="Choose a baseline preset, then apply, store, or reset session-specific tweaks."
          >
            <div className="space-y-3">
              <label className="space-y-1.5">
                <span className="text-xs font-medium text-muted-foreground">Preset</span>
                <select
                  value={selectedPresetName}
                  onChange={(event) => onSelectedPresetChange(event.target.value)}
                  className="h-10 w-full rounded-xl border border-white/10 bg-surface-layer/85 px-3 text-sm text-foreground shadow-inner"
                >
                  {presets.map((preset) => (
                    <option key={preset.name} value={preset.name}>
                      {preset.name}{preset.name === activePresetName ? ' (active)' : ''}
                    </option>
                  ))}
                </select>
              </label>

              <div className="flex flex-wrap gap-2">
                <Button type="button" variant="secondary" size="sm" className="rounded-full" onClick={onApplyPreset}>
                  Apply Preset
                </Button>
                <Button
                  type="button"
                  variant="secondary"
                  size="sm"
                  className="rounded-full"
                  onClick={onSavePreset}
                  loading={isSavingPreset}
                >
                  Save to Preset
                </Button>
                <Button type="button" variant="ghost" size="sm" className="rounded-full" onClick={onReset}>
                  <RotateCcw className="h-3.5 w-3.5" />
                  Reset
                </Button>
              </div>
            </div>
          </SectionCard>

          {contextLengthTarget.kind ? (
            <SectionCard
              title={contextLengthTarget.kind === 'gguf' ? 'Context window' : 'Max model length'}
              description={
                contextLengthTarget.kind === 'gguf'
                  ? 'Writes to `gguf.n_ctx` for the active GGUF model.'
                  : 'Writes the managed vLLM startup override for `vllm.max_model_len`. Leave blank to use auto sizing.'
              }
            >
              <div className="space-y-3">
                <label className="space-y-1.5">
                  <span className="text-xs font-medium text-muted-foreground">
                    {contextLengthTarget.kind === 'gguf' ? 'Context length' : 'Max model length'}
                  </span>
                  <input
                    value={contextLengthInput}
                    onChange={(event) => setContextLengthInput(event.target.value)}
                    inputMode="numeric"
                    placeholder={contextLengthTarget.kind === 'gguf' ? '4096' : 'auto'}
                    className="h-10 w-full rounded-xl border border-white/10 bg-surface-layer/85 px-3 font-mono text-sm text-foreground shadow-inner"
                  />
                </label>
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="inline-flex items-center gap-1 rounded-full border border-white/8 bg-surface-layer/70 px-2.5 py-1 text-[11px] text-muted-foreground">
                    <Sparkles className="h-3 w-3 text-primary-300" />
                    Runtime-specific override
                  </div>
                  <Button
                    type="button"
                    variant="secondary"
                    size="sm"
                    className="rounded-full"
                    loading={contextSettingsBusy}
                    onClick={() => void handleSaveContextSettings()}
                  >
                    Save Context Setting
                  </Button>
                </div>
                {contextSettingsMessage ? (
                  <p className="rounded-xl border border-white/8 bg-surface-layer/60 px-3 py-2 text-xs text-muted-foreground">
                    {contextSettingsMessage}
                  </p>
                ) : null}
              </div>
            </SectionCard>
          ) : null}

          <SectionCard
            title="Sampling & response behavior"
            description="Adjust creativity, repetition, reasoning effort, and display-level diagnostics."
          >
            <InferenceControls
              value={value}
              onChange={onChange}
              supportsReasoningEffort={supportsReasoningEffort}
              showReasoningEffort={showReasoningEffort}
              reasoningEffortOptions={reasoningEffortOptions}
              disabledKeys={disabledKeys}
              disabledReason={disabledReason}
            />
          </SectionCard>
        </div>
      </div>
    </div>
  )
}

export function InferencePanel(props: InferencePanelProps) {
  const {
    open,
    mobileOpen,
    onOpenChange,
    onMobileOpenChange,
    ...bodyProps
  } = props

  return (
    <>
      <aside
        className={cn(
          'absolute right-3 top-3 bottom-3 z-30 hidden w-[23rem] overflow-hidden rounded-[28px] border border-white/10 bg-surface-layer/92 shadow-[0_28px_80px_rgba(0,0,0,0.45)] backdrop-blur-xl transition-all duration-300 ease-out-smooth md:flex md:flex-col',
          open
            ? 'pointer-events-auto translate-x-0 opacity-100'
            : 'pointer-events-none translate-x-8 opacity-0'
        )}
        aria-hidden={!open}
      >
        {open ? <PanelBody {...bodyProps} onClose={() => onOpenChange(false)} /> : null}
      </aside>

      <Sheet open={mobileOpen} onOpenChange={onMobileOpenChange}>
        <SheetContent side="right" className="w-full max-w-md p-0">
          <SheetHeader className="border-b border-border px-4 py-4">
            <SheetTitle>Inference</SheetTitle>
            <SheetDescription>Adjust session-specific inference parameters.</SheetDescription>
          </SheetHeader>
          <PanelBody {...bodyProps} />
        </SheetContent>
      </Sheet>
    </>
  )
}
