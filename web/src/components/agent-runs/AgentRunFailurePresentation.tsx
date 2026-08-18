'use client'

import type { FailureEnvelopeInput } from '@/lib/failure-presentation'
import { FailurePresentationCard } from '@/components/failures/FailurePresentationCard'

/** Agent Run surface adapter for the shared, diagnostics-safe failure card. */
export function AgentRunFailurePresentation({
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
      testId="agent-run-failure-presentation"
    />
  )
}
