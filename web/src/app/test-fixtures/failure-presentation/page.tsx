'use client'

import * as React from 'react'
import { AgentRunFailurePresentation } from '@/components/agent-runs/AgentRunFailurePresentation'
import { ChatMessage } from '@/components/chat/ChatMessage'
import { GoalFailurePresentation } from '@/components/chat/GoalFailurePresentation'
import {
  buildMessagesFromTimelineEvents,
  fetchAgentRun,
  fetchGoal,
} from '@/lib/api'
import type { Message } from '@/lib/chat'
import type { FailureEnvelopeInput } from '@/lib/failure-presentation'

type FixturePayload = {
  chat_event: unknown
}

type LoadedFixture = {
  chatMessage: Message
  goalFailure: FailureEnvelopeInput | null
  goalLatestError: string | null
  runFailure: FailureEnvelopeInput | null
  runLatestError: string | null
}

/**
 * Browser-only host for the real failure presentation consumers.
 *
 * The Playwright fixture supplies the API-shaped Goal and AgentRun records and
 * a canonical chat timeline event.  This route intentionally imports the
 * production API normalizers and the production components rather than
 * reproducing their mapping in test JavaScript.
 */
export default function FailurePresentationFixturePage() {
  const [loaded, setLoaded] = React.useState<LoadedFixture | null>(null)
  const [error, setError] = React.useState<string | null>(null)

  React.useEffect(() => {
    let active = true
    async function load() {
      try {
        const [goal, run, chatResponse] = await Promise.all([
          fetchGoal('goal-failure-presentation'),
          fetchAgentRun('run-failure-presentation'),
          fetch('/v1/test-fixtures/failure-presentation'),
        ])
        if (!chatResponse.ok) {
          throw new Error(`Fixture chat event failed with ${chatResponse.status}`)
        }
        const payload = (await chatResponse.json()) as FixturePayload
        const chatMessage = buildMessagesFromTimelineEvents([payload.chat_event]).at(0)
        if (!chatMessage) {
          throw new Error('Fixture chat event did not produce a message.')
        }
        if (active) {
          setLoaded({
            chatMessage,
            goalFailure: goal.failure,
            goalLatestError: goal.latest_error,
            runFailure: run.failure,
            runLatestError: run.latest_error,
          })
        }
      } catch (reason) {
        if (active) {
          setError(reason instanceof Error ? reason.message : 'Fixture load failed.')
        }
      }
    }
    void load()
    return () => {
      active = false
    }
  }, [])

  if (error) {
    return <p data-testid="failure-presentation-fixture-error">{error}</p>
  }
  if (!loaded) {
    return <p data-testid="failure-presentation-fixture-loading">Loading failure presentation fixture…</p>
  }

  return (
    <main className="space-y-5 p-6" data-testid="failure-presentation-fixture">
      <section aria-label="Chat failure surface">
        <h1 className="mb-2 text-lg font-semibold">Chat</h1>
        <ChatMessage message={loaded.chatMessage} />
      </section>
      <section aria-label="Goal failure surface">
        <h2 className="mb-2 text-lg font-semibold">Goal</h2>
        <GoalFailurePresentation failure={loaded.goalFailure} latestError={loaded.goalLatestError} />
      </section>
      <section aria-label="Agent Run failure surface">
        <h2 className="mb-2 text-lg font-semibold">Agent Run</h2>
        <AgentRunFailurePresentation failure={loaded.runFailure} latestError={loaded.runLatestError} />
      </section>
    </main>
  )
}
