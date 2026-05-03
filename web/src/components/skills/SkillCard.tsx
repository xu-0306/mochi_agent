'use client'

import * as React from 'react'
import {
  BadgeCheck,
  Brain,
  Database,
  FileCode2,
  Search,
  Trash2,
  type LucideIcon,
} from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { useI18n } from '@/lib/i18n'
import { formatDate } from '@/lib/utils'

type SkillRecord = {
  id: string
  name: string
  description: string
  tags: string[]
  useCount: number
  successRate: number
  version: string
  createdAt: string
}

const iconByTag: Array<[string, LucideIcon]> = [
  ['code', FileCode2],
  ['python', FileCode2],
  ['data', Database],
  ['sql', Database],
  ['memory', Brain],
  ['search', Search],
]

function getSkillIcon(skill: SkillRecord): LucideIcon {
  const haystack = [skill.name, ...skill.tags].join(' ').toLowerCase()
  const match = iconByTag.find(([token]) => haystack.includes(token))
  return match?.[1] ?? BadgeCheck
}

function formatPercent(value: number): string {
  if (!Number.isFinite(value)) {
    return '0%'
  }

  return `${Math.max(0, Math.min(100, Math.round(value)))}%`
}

function formatCreatedAt(
  createdAt: string,
  locale: string,
  timeZone: string | undefined,
  unknownLabel: string
): string {
  if (!createdAt) {
    return unknownLabel
  }

  const parsed = new Date(createdAt)
  if (Number.isNaN(parsed.getTime())) {
    return createdAt
  }

  return formatDate(parsed, {
    locale,
    timeZone,
    format: {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
    },
  })
}

interface SkillCardProps {
  skill: SkillRecord
  deleting?: boolean
  onDelete?: (id: string) => void | Promise<void>
}

export function SkillCard({ skill, deleting = false, onDelete }: SkillCardProps) {
  const { locale, resolvedTimeZone, t } = useI18n()
  const Icon = getSkillIcon(skill)
  const successVariant =
    skill.successRate >= 85 ? 'success' :
    skill.successRate >= 60 ? 'warning' :
    'neutral'

  return (
    <Card className="flex h-full flex-col gap-4 p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="flex min-w-0 items-start gap-3">
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md border border-primary-500/20 bg-primary-500/10">
            <Icon className="h-4 w-4 text-primary-400" />
          </div>

          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <h3 className="truncate text-sm font-semibold text-foreground">{skill.name}</h3>
              <Badge variant="outline" className="h-5 px-1.5 text-[10px]">
                {skill.version}
              </Badge>
            </div>
            <p className="mt-1 text-[11px] text-muted-foreground">{skill.id}</p>
          </div>
        </div>

        {onDelete ? (
          <Button
            type="button"
            variant="ghost"
            size="icon-sm"
            aria-label={t('skills.card.delete', { name: skill.name })}
            onClick={() => void onDelete(skill.id)}
            disabled={deleting}
          >
            <Trash2 className="h-3.5 w-3.5" />
          </Button>
        ) : null}
      </div>

      <p className="line-clamp-3 text-xs leading-5 text-muted-foreground">
        {skill.description || t('skills.card.noDescription')}
      </p>

      <div className="flex flex-wrap gap-1.5">
        {skill.tags.length > 0 ? (
          skill.tags.map((tag) => (
            <Badge key={tag} variant="outline" className="h-5 px-1.5 text-[10px]">
              {tag}
            </Badge>
          ))
        ) : (
          <Badge variant="neutral" className="h-5 px-1.5 text-[10px]">
            {t('skills.card.untagged')}
          </Badge>
        )}
      </div>

      <div className="mt-auto grid grid-cols-3 gap-2 border-t border-border pt-3">
        <div className="min-w-0">
          <p className="text-[10px] uppercase tracking-wide text-muted-foreground">{t('skills.card.used')}</p>
          <p className="mt-1 text-sm font-semibold text-foreground">{skill.useCount}</p>
        </div>
        <div className="min-w-0">
          <p className="text-[10px] uppercase tracking-wide text-muted-foreground">{t('skills.card.success')}</p>
          <div className="mt-1">
            <Badge variant={successVariant} className="h-5 px-1.5 text-[10px]">
              {formatPercent(skill.successRate)}
            </Badge>
          </div>
        </div>
        <div className="min-w-0 text-right">
          <p className="text-[10px] uppercase tracking-wide text-muted-foreground">{t('skills.card.created')}</p>
          <p className="mt-1 text-xs font-medium text-foreground">
            {formatCreatedAt(skill.createdAt, locale, resolvedTimeZone, t('skills.card.unknown'))}
          </p>
        </div>
      </div>
    </Card>
  )
}
