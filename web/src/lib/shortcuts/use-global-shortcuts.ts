'use client'

import * as React from 'react'
import { usePathname, useRouter } from 'next/navigation'
import { useProjectStore } from '@/lib/stores/project-store'
import { useSessionStore } from '@/lib/stores/session-store'
import { useUIStore } from '@/lib/stores/ui-store'

function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) {
    return false
  }
  if (target.isContentEditable) {
    return true
  }
  const tag = target.tagName
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT'
}

function isPrimaryModifierPressed(event: KeyboardEvent): boolean {
  return event.metaKey || event.ctrlKey
}

export function useGlobalShortcuts(): void {
  const router = useRouter()
  const pathname = usePathname()
  const createDraftSession = useSessionStore((state) => state.createDraftSession)
  const activeProjectId = useProjectStore((state) => state.activeProjectId)
  const toggleSidebar = useUIStore((state) => state.toggleSidebar)

  React.useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (!isPrimaryModifierPressed(event) || event.altKey) {
        return
      }

      const key = event.key.toLowerCase()
      const editableTarget = isEditableTarget(event.target)

      if (key === '/' || key === '?') {
        event.preventDefault()
        createDraftSession(activeProjectId)
        if (pathname !== '/') {
          router.push('/')
        }
        return
      }

      if (key === ',') {
        event.preventDefault()
        if (pathname !== '/settings') {
          router.push('/settings')
        }
        return
      }

      if (key === 'k') {
        event.preventDefault()
        if (pathname !== '/') {
          router.push('/')
        }
        requestAnimationFrame(() => {
          document.getElementById('sidebar-search-input')?.focus()
        })
        return
      }

      if (key === 'v' && event.shiftKey && !editableTarget) {
        event.preventDefault()
        if (pathname !== '/') {
          router.push('/')
        }
        window.setTimeout(() => {
          window.dispatchEvent(new CustomEvent('mochi:voice-toggle'))
        }, 50)
        return
      }

      if (key === 'b' && !editableTarget) {
        event.preventDefault()
        toggleSidebar()
        return
      }

      if (key === 'l' && !editableTarget) {
        event.preventDefault()
        if (pathname !== '/') {
          router.push('/')
        }
        requestAnimationFrame(() => {
          document.getElementById('chat-input-textarea')?.focus()
        })
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [activeProjectId, createDraftSession, pathname, router, toggleSidebar])
}
