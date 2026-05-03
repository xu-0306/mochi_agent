'use client'

import * as React from 'react'
import { registerServiceWorker } from '@/lib/pwa/register-service-worker'
import { useGlobalShortcuts } from '@/lib/shortcuts/use-global-shortcuts'

export function AppClientBootstrap() {
  React.useEffect(() => {
    registerServiceWorker()
  }, [])

  useGlobalShortcuts()

  return null
}
