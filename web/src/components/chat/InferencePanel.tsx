'use client'

import * as React from 'react'
import { RotateCcw } from 'lucide-react'
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
}: Omit<InferencePanelProps, 'open' | 'mobileOpen' | 'onOpenChange' | 'onMobileOpenChange' | 'agent'>) {
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
    <div className="flex h-full flex-col">
      <div className="space-y-4 border-b border-border px-4 py-4">
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-muted-foreground">Preset</label>
          <select
            value={selectedPresetName}
            onChange={(event) => onSelectedPresetChange(event.target.value)}
            className="h-9 w-full rounded-md border border-border bg-canvas px-3 text-sm text-foreground"
          >
            {presets.map((preset) => (
              <option key={preset.name} value={preset.name}>
                {preset.name}{preset.name === activePresetName ? ' (active)' : ''}
              </option>
            ))}
          </select>
        </div>

        <div className="flex flex-wrap gap-2">
          <Button type="button" variant="secondary" size="sm" onClick={onApplyPreset}>
            Apply Preset
          </Button>
          <Button
            type="button"
            variant="secondary"
            size="sm"
            onClick={onSavePreset}
            loading={isSavingPreset}
          >
            Save to Preset
          </Button>
          <Button type="button" variant="ghost" size="sm" onClick={onReset}>
            <RotateCcw className="h-3.5 w-3.5" />
            Reset
          </Button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-4">
        <div className="space-y-4">
          {contextLengthTarget.kind ? (
            <div className="space-y-3 rounded-md border border-border bg-canvas px-3 py-3">
              <div>
                <p className="text-xs font-semibold text-foreground">
                  {contextLengthTarget.kind === 'gguf' ? 'GGUF Context Window' : 'vLLM Max Model Length'}
                </p>
                <p className="mt-0.5 text-[11px] text-muted-foreground">
                  {contextLengthTarget.kind === 'gguf'
                    ? 'Writes to `gguf.n_ctx` for the active GGUF model.'
                    : 'Writes to `vllm.max_model_len` for the active vLLM model. Leave blank to use auto sizing.'}
                </p>
              </div>
              <label className="space-y-1.5">
                <span className="text-xs font-medium text-muted-foreground">
                  {contextLengthTarget.kind === 'gguf' ? 'Context length' : 'Max model length'}
                </span>
                <input
                  value={contextLengthInput}
                  onChange={(event) => setContextLengthInput(event.target.value)}
                  inputMode="numeric"
                  placeholder={contextLengthTarget.kind === 'gguf' ? '4096' : 'auto'}
                  className="h-9 w-full rounded-md border border-border bg-canvas px-3 font-mono text-sm text-foreground"
                />
              </label>
              <div className="flex flex-wrap justify-end gap-2">
                <Button
                  type="button"
                  variant="secondary"
                  size="sm"
                  loading={contextSettingsBusy}
                  onClick={() => void handleSaveContextSettings()}
                >
                  Save Context Setting
                </Button>
              </div>
              {contextSettingsMessage ? (
                <p className="text-xs text-muted-foreground">{contextSettingsMessage}</p>
              ) : null}
            </div>
          ) : null}

          <InferenceControls
            value={value}
            onChange={onChange}
            supportsReasoningEffort={supportsReasoningEffort}
            showReasoningEffort={showReasoningEffort}
            reasoningEffortOptions={reasoningEffortOptions}
            disabledKeys={disabledKeys}
            disabledReason={disabledReason}
          />
        </div>
      </div>
    </div>
  )
}

export function InferencePanel(props: InferencePanelProps) {
  const {
    open,
    mobileOpen,
    onMobileOpenChange,
    ...bodyProps
  } = props

  return (
    <>
      <aside
        className={[
          'hidden border-l border-border bg-surface-layer md:flex md:h-full md:flex-col',
          open ? 'md:w-80' : 'md:w-0 md:overflow-hidden md:border-l-0',
          'transition-all duration-200',
        ].join(' ')}
      >
        {open ? <PanelBody {...bodyProps} /> : null}
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
