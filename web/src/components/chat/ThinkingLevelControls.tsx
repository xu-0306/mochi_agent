'use client'

import * as React from 'react'
import { BrainCircuit, Check, ChevronDown, Loader2 } from 'lucide-react'
import type { ReasoningEffort } from '@/lib/api'
import {
  buildThinkingLevelOptions,
  findThinkingLevelOption,
} from '@/lib/reasoning-presets'
import { cn } from '@/lib/utils'

interface BaseThinkingLevelProps {
  supportedEfforts?: readonly ReasoningEffort[] | null
  value: ReasoningEffort | null | undefined
  disabled?: boolean
  onChange?: (value: ReasoningEffort | null) => void | Promise<void>
}

interface ThinkingLevelChipControlProps extends BaseThinkingLevelProps {
  title?: string
}

interface ThinkingLevelPanelControlProps extends BaseThinkingLevelProps {
  allowInherit?: boolean
  inheritLabel?: string
  inheritDescription?: string
  onInherit?: () => void | Promise<void>
}

export function ThinkingLevelChipControl({
  supportedEfforts,
  value,
  disabled = false,
  onChange,
  title = 'Thinking Level',
}: ThinkingLevelChipControlProps) {
  const [open, setOpen] = React.useState(false)
  const [isSaving, setIsSaving] = React.useState(false)
  const menuRef = React.useRef<HTMLDivElement>(null)
  const options = React.useMemo(
    () => buildThinkingLevelOptions(supportedEfforts),
    [supportedEfforts]
  )
  const selectedOption = React.useMemo(
    () => findThinkingLevelOption(value, supportedEfforts),
    [supportedEfforts, value]
  )

  React.useEffect(() => {
    if (!open) {
      return
    }

    const handlePointerDown = (event: MouseEvent) => {
      if (!menuRef.current?.contains(event.target as Node)) {
        setOpen(false)
      }
    }

    document.addEventListener('mousedown', handlePointerDown)
    return () => document.removeEventListener('mousedown', handlePointerDown)
  }, [open])

  const handleSelect = React.useCallback(async (nextValue: ReasoningEffort | null) => {
    if (!onChange) {
      setOpen(false)
      return
    }
    if (nextValue === (value ?? null)) {
      setOpen(false)
      return
    }

    setIsSaving(true)
    try {
      await onChange(nextValue)
      setOpen(false)
    } finally {
      setIsSaving(false)
    }
  }, [onChange, value])

  return (
    <div className="relative" ref={menuRef}>
      <button
        type="button"
        onClick={() => setOpen((current) => !current)}
        disabled={disabled || isSaving}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={title}
        title={`${title}: ${selectedOption.label}`}
        className={cn(
          'flex h-7 items-center gap-1 rounded-full border border-white/10 bg-[linear-gradient(180deg,rgba(255,255,255,0.05),rgba(255,255,255,0.02))] pl-1.5 pr-2',
          'text-[10.5px] text-foreground shadow-[inset_0_1px_0_rgba(255,255,255,0.05)] backdrop-blur transition-all duration-150 hover:border-white/16 hover:bg-elevated-layer',
          'focus:outline-none focus:ring-2 focus:ring-primary-500/35',
          'disabled:cursor-not-allowed disabled:opacity-50'
        )}
      >
        <span className="flex h-4 w-4 items-center justify-center rounded-full bg-primary-500/12 text-primary-300">
          {isSaving ? <Loader2 className="h-3 w-3 animate-spin" /> : <BrainCircuit className="h-3 w-3" />}
        </span>
        <span className="whitespace-nowrap font-medium tracking-[0.01em]">Thinking: {selectedOption.label}</span>
        <ChevronDown className={cn('h-3 w-3 shrink-0 text-muted-foreground/90 transition-transform', open && 'rotate-180')} />
      </button>

      {open ? (
        <div
          className={cn(
            'absolute bottom-full left-0 z-50 mb-2 w-[20rem] max-w-[calc(100vw-2rem)] overflow-hidden rounded-[1rem]',
            'border border-white/10 bg-[linear-gradient(180deg,rgba(28,28,31,0.985),rgba(17,17,20,0.985))] shadow-[0_22px_56px_rgba(0,0,0,0.44)]',
            'animate-slide-up'
          )}
        >
          <div className="border-b border-white/8 px-3.5 py-3">
            <p className="text-[12px] font-medium tracking-[0.01em] text-slate-100">How much should Mochi think?</p>
            <p className="mt-1 text-[10.5px] leading-4 text-slate-400">
              Adjust reasoning depth for this chat without opening the full inference panel.
            </p>
          </div>

          <div className="space-y-1 px-2 py-2">
            {options.map((option) => {
              const selected = option.value === selectedOption.value

              return (
                <button
                  key={option.value}
                  type="button"
                  onClick={() => void handleSelect(option.effort)}
                  disabled={isSaving}
                  className={cn(
                    'flex w-full items-start gap-2.5 rounded-[0.85rem] border border-transparent px-2.5 py-2 text-left transition-all duration-150',
                    selected
                      ? 'border-primary-400/18 bg-[linear-gradient(180deg,rgba(83,112,255,0.16),rgba(83,112,255,0.08))] text-slate-50 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]'
                      : 'text-slate-200 hover:border-white/8 hover:bg-white/[0.035]'
                  )}
                >
                  <span
                    className={cn(
                      'mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border',
                      selected
                        ? 'border-primary-400/30 bg-primary-500/14 text-primary-200'
                        : 'border-white/8 bg-white/[0.03] text-slate-400'
                    )}
                  >
                    <BrainCircuit className="h-3.5 w-3.5" />
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="flex items-center gap-2">
                      <span className="text-[13px] font-medium">{option.label}</span>
                      {selected ? <Check className="h-3.5 w-3.5 text-primary-300" /> : null}
                      {option.rawEffortLabel ? (
                        <span className="rounded-full border border-white/10 bg-white/[0.04] px-1.5 py-0.5 text-[9px] uppercase tracking-[0.12em] text-slate-300">
                          {option.rawEffortLabel}
                        </span>
                      ) : null}
                    </span>
                    <span className="mt-0.5 block text-[10.5px] leading-4 text-slate-400">
                      {option.description}
                    </span>
                  </span>
                </button>
              )
            })}
          </div>
        </div>
      ) : null}
    </div>
  )
}

export function ThinkingLevelPanelControl({
  supportedEfforts,
  value,
  disabled = false,
  onChange,
  allowInherit = false,
  inheritLabel = 'Inherit chat setting',
  inheritDescription = 'Follow the current chat thinking level unless this workflow needs its own override.',
  onInherit,
}: ThinkingLevelPanelControlProps) {
  const options = React.useMemo(
    () => buildThinkingLevelOptions(supportedEfforts),
    [supportedEfforts]
  )
  const selectedOption = React.useMemo(
    () => findThinkingLevelOption(value, supportedEfforts),
    [supportedEfforts, value]
  )

  return (
    <div className="space-y-2">
      {allowInherit ? (
        <button
          type="button"
          onClick={() => void onInherit?.()}
          disabled={disabled}
          className={cn(
            'flex w-full items-start gap-2.5 rounded-[1rem] border border-dashed px-3 py-2.5 text-left transition-all duration-150',
            value === null
              ? 'border-primary-500/35 bg-primary-500/10 text-foreground'
              : 'border-border bg-canvas/65 text-muted-foreground hover:border-white/12 hover:bg-elevated-layer hover:text-foreground',
            disabled && 'cursor-not-allowed opacity-60'
          )}
        >
          <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-white/8 bg-white/[0.03] text-primary-300">
            <BrainCircuit className="h-3.5 w-3.5" />
          </span>
          <span className="min-w-0 flex-1">
            <span className="flex items-center gap-2">
              <span className="text-[13px] font-medium">{inheritLabel}</span>
              {value === null ? <Check className="h-3.5 w-3.5 text-primary-300" /> : null}
            </span>
            <span className="mt-0.5 block text-[10.5px] leading-4 text-muted-foreground">
              {inheritDescription}
            </span>
          </span>
        </button>
      ) : null}

      <div className="grid gap-1.5">
        {options
          .filter((option) => option.value !== 'auto' || !allowInherit)
          .map((option) => {
            const selected = value !== null && option.value === selectedOption.value

            return (
              <button
                key={option.value}
                type="button"
                onClick={() => void onChange?.(option.effort)}
                disabled={disabled}
                className={cn(
                  'flex w-full items-start gap-2.5 rounded-[1rem] border px-3 py-2.5 text-left transition-all duration-150',
                  selected
                    ? 'border-primary-500/35 bg-[linear-gradient(180deg,rgba(83,112,255,0.12),rgba(83,112,255,0.05))] text-foreground shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]'
                    : 'border-border bg-canvas/65 text-muted-foreground hover:border-white/12 hover:bg-elevated-layer hover:text-foreground',
                  disabled && 'cursor-not-allowed opacity-60'
                )}
              >
                <span
                  className={cn(
                    'mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border',
                    selected
                      ? 'border-primary-400/30 bg-primary-500/14 text-primary-200'
                      : 'border-white/8 bg-white/[0.03] text-slate-400'
                  )}
                >
                  <BrainCircuit className="h-3.5 w-3.5" />
                </span>
                <span className="min-w-0 flex-1">
                  <span className="flex items-center gap-2">
                    <span className="text-[13px] font-medium">{option.label}</span>
                    {selected ? <Check className="h-3.5 w-3.5 text-primary-300" /> : null}
                    {option.rawEffortLabel ? (
                      <span className="rounded-full border border-white/10 bg-white/[0.04] px-1.5 py-0.5 text-[9px] uppercase tracking-[0.12em] text-muted-foreground">
                        {option.rawEffortLabel}
                      </span>
                    ) : null}
                  </span>
                  <span className="mt-0.5 block text-[10.5px] leading-4 text-muted-foreground">
                    {option.description}
                  </span>
                </span>
              </button>
            )
          })}
      </div>
    </div>
  )
}
