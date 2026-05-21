'use client'

import * as React from 'react'
import { Mic, Settings, Sparkles } from 'lucide-react'
import { Button } from '@/components/ui/button'

interface EmptyStateProps {
  onPrompt: (prompt: string) => void
  onVoice: () => void
  onSettings: () => void
}

const STARTER_PROMPTS = [
  'Summarize the current project status.',
  'Help me debug a failing test in this workspace.',
  'Review the code changes in this repository.',
]

export function EmptyState({ onPrompt, onVoice, onSettings }: EmptyStateProps) {
  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col items-center rounded-3xl border border-border bg-surface-layer px-6 py-10 text-center">
      <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-primary-500/12 text-primary-400">
        <Sparkles className="h-6 w-6" />
      </div>
      <h2 className="mt-5 text-2xl font-semibold text-foreground">Start a new conversation</h2>
      <p className="mt-2 max-w-xl text-sm text-muted-foreground">
        Ask Mochi to inspect code, adjust inference settings, or work through a task with tools and reasoning.
      </p>

      <div className="mt-6 grid w-full gap-3 sm:grid-cols-3">
        {STARTER_PROMPTS.map((prompt) => (
          <button
            key={prompt}
            type="button"
            onClick={() => onPrompt(prompt)}
            className="rounded-xl border border-border bg-canvas px-4 py-4 text-left text-sm text-foreground transition-colors hover:bg-elevated-layer"
          >
            {prompt}
          </button>
        ))}
      </div>

      <div className="mt-6 flex flex-wrap items-center justify-center gap-2">
        <Button type="button" variant="secondary" size="sm" onClick={onVoice}>
          <Mic className="h-3.5 w-3.5" />
          Voice
        </Button>
        <Button type="button" variant="ghost" size="sm" onClick={onSettings}>
          <Settings className="h-3.5 w-3.5" />
          Settings
        </Button>
      </div>
    </div>
  )
}
