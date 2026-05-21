'use client'

import Link from 'next/link'
import { BookOpen } from 'lucide-react'
import { cn } from '@/lib/utils'
import { useI18n } from '@/lib/i18n'

export type SettingsTab = 'model' | 'inference' | 'voice' | 'memory' | 'learning' | 'channels' | 'web'
export type SettingsNavActive = SettingsTab | 'discord-guide'

export const DEFAULT_SETTINGS_TAB: SettingsTab = 'model'

export const SETTINGS_TABS: Array<{
  value: SettingsTab
  labelKey: string
}> = [
  { value: 'model', labelKey: 'settings.tabs.model' },
  { value: 'inference', labelKey: 'settings.tabs.inference' },
  { value: 'voice', labelKey: 'settings.tabs.voice' },
  { value: 'memory', labelKey: 'settings.tabs.memory' },
  { value: 'learning', labelKey: 'settings.tabs.learning' },
  { value: 'channels', labelKey: 'settings.tabs.channels' },
  { value: 'web', labelKey: 'settings.tabs.web' },
]

export function isSettingsTab(value: string | null): value is SettingsTab {
  return SETTINGS_TABS.some((tab) => tab.value === value)
}

export function settingsTabHref(tab: SettingsTab): string {
  return tab === DEFAULT_SETTINGS_TAB ? '/settings' : `/settings?tab=${tab}`
}

function navItemClass(active: boolean): string {
  return cn(
    'inline-flex min-h-8 w-full items-center justify-start gap-2 rounded-md px-3 py-1.5',
    'text-sm font-medium text-muted-foreground transition-all duration-150 ease-out-smooth',
    'hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
    active && 'bg-elevated-layer text-foreground shadow-xs'
  )
}

export function SettingsNav({
  active,
  includeGuideLink = false,
  className,
}: {
  active: SettingsNavActive
  includeGuideLink?: boolean
  className?: string
}) {
  const { t } = useI18n()

  return (
    <nav
      aria-label={t('settings.navLabel')}
      className={cn('rounded-lg border border-border bg-surface-layer p-1', className)}
    >
      <div className="flex flex-col gap-0.5">
        {SETTINGS_TABS.map((tab) => (
          <Link
            key={tab.value}
            href={settingsTabHref(tab.value)}
            className={navItemClass(active === tab.value)}
            aria-current={active === tab.value ? 'page' : undefined}
          >
            <span className="truncate">{t(tab.labelKey)}</span>
          </Link>
        ))}
      </div>

      {includeGuideLink ? (
        <div className="mt-1 border-t border-border pt-1">
          <Link
            href="/settings/discord-guide"
            className={navItemClass(active === 'discord-guide')}
            aria-current={active === 'discord-guide' ? 'page' : undefined}
          >
            <BookOpen className="h-3.5 w-3.5 shrink-0" />
            <span className="truncate">{t('discordGuide.navLabel')}</span>
          </Link>
        </div>
      ) : null}
    </nav>
  )
}
