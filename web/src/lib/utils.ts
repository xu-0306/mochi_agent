import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

/**
 * Merges Tailwind CSS class names, resolving conflicts intelligently.
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs))
}

type LocalizedDateOptions = {
  locale?: string
  timeZone?: string
  now?: Date
  format?: Intl.DateTimeFormatOptions
}

function toDate(value: Date | string): Date | null {
  const date = typeof value === 'string' ? new Date(value) : value
  if (Number.isNaN(date.getTime())) {
    return null
  }
  return date
}

function resolveTimeZone(timeZone: string | undefined): string | undefined {
  if (!timeZone || timeZone === 'auto') {
    return undefined
  }
  return timeZone
}

/**
 * Formats a date using locale/timezone display preferences.
 */
export function formatDate(value: Date | string, options: LocalizedDateOptions = {}): string {
  const date = toDate(value)
  if (!date) {
    return ''
  }

  const locale = options.locale ?? 'en-US'
  const timeZone = resolveTimeZone(options.timeZone)
  const formatterOptions: Intl.DateTimeFormatOptions = options.format ?? {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  }

  return new Intl.DateTimeFormat(locale, {
    ...formatterOptions,
    ...(timeZone ? { timeZone } : {}),
  }).format(date)
}

/**
 * Formats a date relative to now (e.g. "2 hours ago", "Yesterday")
 */
export function formatRelativeTime(value: Date | string, options: LocalizedDateOptions = {}): string {
  const date = toDate(value)
  if (!date) {
    return ''
  }

  const now = options.now ?? new Date()
  const diffSeconds = Math.round((date.getTime() - now.getTime()) / 1000)
  const absSeconds = Math.abs(diffSeconds)
  const locale = options.locale ?? 'en-US'
  const rtf = new Intl.RelativeTimeFormat(locale, { numeric: 'auto' })

  if (absSeconds < 60) {
    return rtf.format(diffSeconds, 'second')
  }

  const diffMinutes = Math.round(diffSeconds / 60)
  if (Math.abs(diffMinutes) < 60) {
    return rtf.format(diffMinutes, 'minute')
  }

  const diffHours = Math.round(diffSeconds / 3600)
  if (Math.abs(diffHours) < 24) {
    return rtf.format(diffHours, 'hour')
  }

  const diffDays = Math.round(diffSeconds / 86_400)
  if (Math.abs(diffDays) < 7) {
    return rtf.format(diffDays, 'day')
  }

  return formatDate(date, {
    locale,
    timeZone: options.timeZone,
    format: { month: 'short', day: 'numeric' },
  })
}

/**
 * Truncates text to a given number of characters, adding ellipsis.
 */
export function truncate(text: string, maxLength: number): string {
  if (text.length <= maxLength) {
    return text
  }
  return `${text.slice(0, maxLength)}...`
}
