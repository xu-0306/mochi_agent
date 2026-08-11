'use client'

import type { FailureEnvelopeInput } from '@/lib/failure-presentation'
import { FailurePresentationCard } from '@/components/failures/FailurePresentationCard'

/** Goal surface adapter for the shared, diagnostics-safe failure card. */
export function GoalFailurePresentation({
  failure,
  latestError,
  className,
}: {
  failure: FailureEnvelopeInput | null | undefined
  latestError: string | null | undefined
  className?: string
}) {
  if (!failure && !latestError) {
    return null
  }
  return (
    <FailurePresentationCard
      failure={failure}
      className={className}
      testId="goal-failure-presentation"
    />
  )
}
