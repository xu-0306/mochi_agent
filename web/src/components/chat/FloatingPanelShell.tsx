'use client'

import * as React from 'react'
import { Sheet, SheetContent } from '@/components/ui/sheet'
import { cn } from '@/lib/utils'

type FloatingPanelBreakpoint = 'md' | 'lg'

const DESKTOP_MEDIA_QUERIES: Record<FloatingPanelBreakpoint, string> = {
  md: '(min-width: 768px)',
  lg: '(min-width: 1024px)',
}

const DESKTOP_VISIBILITY_CLASSES: Record<FloatingPanelBreakpoint, string> = {
  md: 'md:flex',
  lg: 'lg:flex',
}

const MOBILE_HIDDEN_CLASSES: Record<FloatingPanelBreakpoint, string> = {
  md: 'md:hidden',
  lg: 'lg:hidden',
}

interface FloatingPanelShellProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  children: React.ReactNode
  desktopSide: 'left' | 'right'
  desktopWidthClass: string
  desktopBreakpoint?: FloatingPanelBreakpoint
  desktopClassName?: string
  mobileSide?: 'left' | 'right'
  mobileClassName?: string
  renderDesktop?: boolean
  renderMobile?: boolean
}

function useDesktopViewport(breakpoint: FloatingPanelBreakpoint) {
  const query = DESKTOP_MEDIA_QUERIES[breakpoint]
  const [matches, setMatches] = React.useState(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') {
      return false
    }
    return window.matchMedia(query).matches
  })

  React.useEffect(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') {
      return
    }
    const mediaQuery = window.matchMedia(query)
    const syncMatches = () => {
      setMatches(mediaQuery.matches)
    }

    syncMatches()
    mediaQuery.addEventListener('change', syncMatches)
    return () => mediaQuery.removeEventListener('change', syncMatches)
  }, [query])

  return matches
}

export function FloatingPanelShell({
  open,
  onOpenChange,
  children,
  desktopSide,
  desktopWidthClass,
  desktopBreakpoint = 'lg',
  desktopClassName,
  mobileSide = 'right',
  mobileClassName,
  renderDesktop = true,
  renderMobile = true,
}: FloatingPanelShellProps) {
  const isDesktop = useDesktopViewport(desktopBreakpoint)
  const hiddenStateClass = desktopSide === 'left'
    ? '-translate-x-8 opacity-0'
    : 'translate-x-8 opacity-0'

  return (
    <>
      {renderDesktop ? (
        <aside
          className={cn(
            'absolute top-3 bottom-3 z-30 hidden overflow-hidden rounded-[28px] border border-white/10 bg-surface-layer/92 shadow-[0_28px_80px_rgba(0,0,0,0.45)] backdrop-blur-xl transition-all duration-300 ease-out-smooth',
            desktopSide === 'left' ? 'left-3' : 'right-3',
            DESKTOP_VISIBILITY_CLASSES[desktopBreakpoint],
            desktopWidthClass,
            desktopClassName,
            open
              ? 'pointer-events-auto translate-x-0 opacity-100'
              : `pointer-events-none ${hiddenStateClass}`
          )}
          aria-hidden={!open}
        >
          {open ? children : null}
        </aside>
      ) : null}

      {renderMobile && !isDesktop ? (
        <Sheet open={open} onOpenChange={onOpenChange}>
          <SheetContent
            side={mobileSide}
            className={cn('w-full max-w-md p-0', MOBILE_HIDDEN_CLASSES[desktopBreakpoint], mobileClassName)}
          >
            {children}
          </SheetContent>
        </Sheet>
      ) : null}
    </>
  )
}
