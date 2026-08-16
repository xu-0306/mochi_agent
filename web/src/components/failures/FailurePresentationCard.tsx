'use client'

import { AlertCircle, AlertTriangle, Info } from 'lucide-react'
import type { FailureEnvelopeInput, FailurePresentation } from '@/lib/failure-presentation'
import { presentFailure } from '@/lib/failure-presentation'
import { cn } from '@/lib/utils'

const FALLBACK_PRESENTATION: FailurePresentation = {
  tone: 'error',
  title: 'Task could not continue',
  detail: 'The task could not be completed. Review it before starting another attempt.',
  allowedActions: [],
}

const ACTION_LABELS: Record<string, string> = {
  retry: 'Retry',
  retry_tool: 'Retry tool',
  request_approval: 'Request approval',
  continue: 'Continue',
  reduce_context: 'Reduce context',
  resume: 'Resume',
}

function safePresentation(failure: FailureEnvelopeInput | null | undefined): FailurePresentation {
  if (!failure) {
    return FALLBACK_PRESENTATION
  }
  try {
    return presentFailure(failure)
  } catch {
    return FALLBACK_PRESENTATION
  }
}

function toneClasses(tone: FailurePresentation['tone']): string {
  if (tone === 'warning') {
    return 'border-amber-500/35 bg-amber-500/10 text-amber-100'
  }
  if (tone === 'info') {
    return 'border-primary-500/35 bg-primary-500/10 text-primary-100'
  }
  return 'border-error/40 bg-error/10 text-foreground'
}

function ToneIcon({ tone }: { tone: FailurePresentation['tone'] }) {
  if (tone === 'warning') {
    return <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-300" aria-hidden />
  }
  if (tone === 'info') {
    return <Info className="mt-0.5 h-4 w-4 shrink-0 text-primary-300" aria-hidden />
  }
  return <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-error" aria-hidden />
}

/**
 * Renders only the versioned, user-safe failure presentation.  Raw error text
 * and diagnostics references deliberately never enter this component.
 */
export function FailurePresentationCard({
  failure,
  className,
  testId = 'failure-presentation',
}: {
  failure: FailureEnvelopeInput | null | undefined
  className?: string
  testId?: string
}) {
  const presentation = safePresentation(failure)
  const kind = typeof failure?.kind === 'string' ? failure.kind : 'unknown'

  return (
    <section
      className={cn('flex items-start gap-3 rounded-lg border px-3 py-2.5', toneClasses(presentation.tone), className)}
      data-testid={testId}
      data-failure-kind={kind}
      data-failure-tone={presentation.tone}
      role="status"
    >
      <ToneIcon tone={presentation.tone} />
      <div className="min-w-0 flex-1">
        <p className="text-sm font-medium leading-relaxed" data-testid={`${testId}-title`}>
          {presentation.title}
        </p>
        <p className="mt-1 text-sm leading-relaxed text-foreground/85" data-testid={`${testId}-detail`}>
          {presentation.detail}
        </p>
        {presentation.allowedActions.length > 0 ? (
          <ul className="mt-2 flex flex-wrap gap-1.5" aria-label="Available recovery actions">
            {presentation.allowedActions.map((action) => (
              <li
                key={action}
                className="rounded-full border border-current/25 px-2 py-0.5 text-xs font-medium"
                data-testid={`${testId}-action-${action}`}
              >
                {ACTION_LABELS[action] ?? action}
              </li>
            ))}
          </ul>
        ) : null}
      </div>
    </section>
  )
}
