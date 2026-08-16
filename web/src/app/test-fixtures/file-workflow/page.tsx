'use client'

import { TaskPanel } from '@/components/chat/TaskPanel'

/**
 * Browser-only host for the production file-approval workflow.
 *
 * The accompanying Playwright fixture owns the API responses.  Keeping this
 * page deliberately small ensures that the browser imports TaskPanel, its
 * Zustand task store, and the production API client instead of reimplementing
 * subset selection in test-only browser JavaScript.
 */
export default function FileWorkflowFixturePage() {
  return (
    <TaskPanel
      open
      onOpenChange={() => undefined}
    />
  )
}
