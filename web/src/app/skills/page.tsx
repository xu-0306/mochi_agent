'use client'

import * as React from 'react'
import { RefreshCw, Search } from 'lucide-react'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { SkillCard } from '@/components/skills/SkillCard'
import * as api from '@/lib/api'
import { useI18n } from '@/lib/i18n'

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

type FilterTab = 'all' | 'recent' | 'most-used' | 'highest-success'

type SkillsApiModule = typeof api & {
  deleteSkill?: (id: string) => Promise<void>
}

const skillsApi = api as SkillsApiModule

function normalizeSkill(input: unknown): SkillRecord | null {
  if (!input || typeof input !== 'object') {
    return null
  }

  const record = input as Record<string, unknown>
  const id = typeof record.id === 'string' ? record.id : ''
  const name = typeof record.name === 'string' ? record.name : id

  if (!id || !name) {
    return null
  }

  return {
    id,
    name,
    description: typeof record.description === 'string' ? record.description : '',
    tags: Array.isArray(record.tags)
      ? record.tags.filter((tag): tag is string => typeof tag === 'string')
      : [],
    useCount: typeof record.useCount === 'number' ? record.useCount : 0,
    successRate: typeof record.successRate === 'number' ? record.successRate : 0,
    version: typeof record.version === 'string' ? record.version : 'unknown',
    createdAt: typeof record.createdAt === 'string' ? record.createdAt : '',
  }
}

function sortByCreatedAt(skills: SkillRecord[]): SkillRecord[] {
  return [...skills].sort((left, right) => {
    const leftTime = left.createdAt ? new Date(left.createdAt).getTime() : 0
    const rightTime = right.createdAt ? new Date(right.createdAt).getTime() : 0
    return rightTime - leftTime
  })
}

export default function SkillsPage() {
  const { t } = useI18n()
  const [skills, setSkills] = React.useState<SkillRecord[]>([])
  const [search, setSearch] = React.useState('')
  const [activeTab, setActiveTab] = React.useState<FilterTab>('all')
  const [loading, setLoading] = React.useState(true)
  const [refreshing, setRefreshing] = React.useState(false)
  const [deletingId, setDeletingId] = React.useState<string | null>(null)
  const [error, setError] = React.useState<string | null>(null)

  const loadSkills = React.useCallback(async () => {
    setError(null)

    try {
      const data = await api.fetchSkills()
      const normalized = Array.isArray(data)
        ? data.map(normalizeSkill).filter((skill): skill is SkillRecord => skill !== null)
        : []
      setSkills(sortByCreatedAt(normalized))
    } catch (loadError) {
      const detail = loadError instanceof Error ? loadError.message : null
      setError(detail ? `${t('skills.errorLoad')}: ${detail}` : t('skills.errorLoad'))
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }, [t])

  React.useEffect(() => {
    void loadSkills()
  }, [loadSkills])

  const handleRefresh = async () => {
    setRefreshing(true)
    await loadSkills()
  }

  const handleDelete = async (id: string) => {
    setDeletingId(id)
    setError(null)

    try {
      if (typeof skillsApi.deleteSkill === 'function') {
        await skillsApi.deleteSkill(id)
        await loadSkills()
      } else {
        setSkills((current) => current.filter((skill) => skill.id !== id))
      }
    } catch (deleteError) {
      const detail = deleteError instanceof Error ? deleteError.message : null
      setError(detail ? `${t('skills.errorDelete')}: ${detail}` : t('skills.errorDelete'))
    } finally {
      setDeletingId(null)
    }
  }

  const normalizedSearch = search.trim().toLowerCase()

  const filtered = skills.filter((skill) => {
    const matchesSearch =
      !normalizedSearch ||
      skill.name.toLowerCase().includes(normalizedSearch) ||
      skill.description.toLowerCase().includes(normalizedSearch) ||
      skill.tags.some((tag) => tag.toLowerCase().includes(normalizedSearch))

    const matchesTab =
      activeTab === 'all' ||
      (activeTab === 'recent' && !!skill.createdAt) ||
      (activeTab === 'most-used' && skill.useCount > 0) ||
      (activeTab === 'highest-success' && skill.successRate >= 80)

    return matchesSearch && matchesTab
  })

  const visibleSkills = React.useMemo(() => {
    const next = [...filtered]

    if (activeTab === 'most-used') {
      next.sort((left, right) => right.useCount - left.useCount)
      return next
    }

    if (activeTab === 'highest-success') {
      next.sort((left, right) => right.successRate - left.successRate)
      return next
    }

    return sortByCreatedAt(next)
  }, [activeTab, filtered])

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <header className="shrink-0 border-b border-border px-6 pb-4 pt-5">
        <div className="mb-4 flex items-start justify-between gap-4">
          <div>
            <h1 className="text-xl font-bold text-foreground">{t('skills.title')}</h1>
            <p className="mt-0.5 text-sm text-muted-foreground">
              {t('skills.subtitle')}
            </p>
          </div>

          <div className="flex items-center gap-2">
            <div className="rounded-md border border-border bg-surface-layer px-3 py-2 text-right">
              <p className="text-[11px] uppercase tracking-wide text-muted-foreground">{t('skills.loaded')}</p>
              <p className="text-sm font-semibold text-foreground">{skills.length}</p>
            </div>
            <Button variant="secondary" size="sm" onClick={() => void handleRefresh()} loading={refreshing}>
              <RefreshCw className="h-3.5 w-3.5" />
              {t('skills.refresh')}
            </Button>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <Tabs value={activeTab} onValueChange={(value) => setActiveTab(value as FilterTab)}>
            <TabsList>
              <TabsTrigger value="all">{t('skills.tabs.all')}</TabsTrigger>
              <TabsTrigger value="recent">{t('skills.tabs.recent')}</TabsTrigger>
              <TabsTrigger value="most-used">{t('skills.tabs.mostUsed')}</TabsTrigger>
              <TabsTrigger value="highest-success">{t('skills.tabs.highestSuccess')}</TabsTrigger>
            </TabsList>
          </Tabs>

          <div className="min-w-[240px] max-w-sm flex-1">
            <Input
              placeholder={t('skills.searchPlaceholder')}
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              leftIcon={<Search className="h-3.5 w-3.5" />}
              size="sm"
              className="pl-8"
            />
          </div>
        </div>

        {error ? (
          <div className="mt-3 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
            {error}
          </div>
        ) : null}
      </header>

      <div className="flex-1 overflow-y-auto px-6 py-5">
        {loading ? (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
            {Array.from({ length: 6 }).map((_, index) => (
              <div
                key={index}
                className="h-52 rounded-lg border border-border bg-surface-layer animate-pulse"
              />
            ))}
          </div>
        ) : visibleSkills.length === 0 ? (
          <div className="flex h-48 items-center justify-center rounded-lg border border-dashed border-border bg-surface-layer text-center">
            <p className="text-sm text-muted-foreground">
              {normalizedSearch ? t('skills.noSearchResults', { query: search }) : t('skills.empty')}
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
            {visibleSkills.map((skill) => (
              <SkillCard
                key={skill.id}
                skill={skill}
                deleting={deletingId === skill.id}
                onDelete={handleDelete}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
