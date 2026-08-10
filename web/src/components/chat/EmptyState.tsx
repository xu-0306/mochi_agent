'use client'

import * as React from 'react'
import { ArrowUpRight, FileSearch, FolderKanban, Mic, Settings, TestTube2 } from 'lucide-react'
import { Button } from '@/components/ui/button'

interface EmptyStateProps {
  onPrompt: (prompt: string) => void
  onVoice: () => void
  onSettings: () => void
}

const STARTER_PROMPTS = [
  {
    label: 'Summarize this project',
    detail: 'A clear read on structure, status, and next steps.',
    icon: FolderKanban,
  },
  {
    label: 'Trace a failing test',
    detail: 'Follow the evidence from failure to a focused fix.',
    icon: TestTube2,
  },
  {
    label: 'Review recent changes',
    detail: 'Spot regressions, risky edges, and missing coverage.',
    icon: FileSearch,
  },
  {
    label: 'Plan a new feature',
    detail: 'Turn an idea into a small, verifiable implementation.',
    icon: ArrowUpRight,
  },
]

export function EmptyState({ onPrompt, onVoice, onSettings }: EmptyStateProps) {
  return (
    <div className="mx-auto flex w-full max-w-5xl flex-col items-center px-4 py-12 text-center sm:py-20">
      <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">
        Mochi workspace
      </p>
      <h2 className="mt-3 font-serif text-2xl font-semibold text-foreground sm:text-[2.15rem]">
        What are we working on?
      </h2>
      <p className="mt-3 max-w-xl text-sm leading-6 text-muted-foreground">
        Bring a question, a file, or a half-formed idea. Mochi keeps the work focused and the context close.
      </p>

      <div className="mt-9 grid w-full max-w-4xl gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {STARTER_PROMPTS.map(({ label, detail, icon: Icon }) => (
          <button
            key={label}
            type="button"
            onClick={() => onPrompt(label)}
            className="group min-h-[132px] border-y border-border/80 bg-transparent px-4 py-4 text-left transition-colors duration-200 hover:border-primary-500/45 hover:bg-muted/35"
          >
            <span className="flex h-8 w-8 items-center justify-center border border-border bg-muted/40 text-muted-foreground transition-colors group-hover:border-primary-500/35 group-hover:text-primary-500">
              <Icon className="h-4 w-4" />
            </span>
            <span className="mt-4 block text-sm font-semibold text-foreground">{label}</span>
            <span className="mt-1 block text-xs leading-5 text-muted-foreground">{detail}</span>
          </button>
        ))}
      </div>

      <div className="mt-8 flex flex-wrap items-center justify-center gap-2">
        <Button type="button" variant="outline" size="sm" onClick={onVoice}>
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
