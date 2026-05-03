'use client'

import * as React from 'react'
import {
  ArrowUp,
  File,
  Folder,
  FolderOpen,
  HardDrive,
  RefreshCw,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import * as api from '@/lib/api'
import { useI18n } from '@/lib/i18n'

type PickerMode = 'directory' | 'file' | 'file_or_directory'

function isWindowsDriveRoot(path: string): boolean {
  return /^[A-Za-z]:[\\/]?$/.test(path.trim())
}

function normalizeDriveRoot(path: string): string {
  const driveLetter = path.trim().slice(0, 1).toUpperCase()
  return `${driveLetter}:\\`
}

function parentPathOf(path: string): string | null {
  const raw = path.trim()
  if (!raw) {
    return null
  }
  if (raw === '/') {
    return '/'
  }
  if (isWindowsDriveRoot(raw)) {
    return normalizeDriveRoot(raw)
  }

  const normalized = raw.replace(/[\\/]+$/, '')
  if (!normalized) {
    return raw.startsWith('/') ? '/' : '.'
  }

  const separatorIndex = Math.max(normalized.lastIndexOf('/'), normalized.lastIndexOf('\\'))
  if (separatorIndex >= 0) {
    const parent = normalized.slice(0, separatorIndex)
    if (!parent) {
      return normalized.startsWith('/') ? '/' : '.'
    }
    if (/^[A-Za-z]:$/.test(parent)) {
      return normalizeDriveRoot(parent)
    }
    return parent
  }

  if (/^[A-Za-z]:$/.test(normalized)) {
    return normalizeDriveRoot(normalized)
  }
  return '.'
}

interface PathPickerProps {
  value: string
  onChange: (value: string) => void
  mode: PickerMode
  placeholder?: string
  inputClassName?: string
  dialogTitle?: string
  dialogDescription?: string
  browseButtonLabel?: string
  disabled?: boolean
}

export function PathPicker({
  value,
  onChange,
  mode,
  placeholder,
  inputClassName,
  dialogTitle,
  dialogDescription,
  browseButtonLabel,
  disabled = false,
}: PathPickerProps) {
  const { t } = useI18n()
  const [open, setOpen] = React.useState(false)
  const [roots, setRoots] = React.useState<api.FilesystemRoot[]>([])
  const [currentPath, setCurrentPath] = React.useState('')
  const [parentPath, setParentPath] = React.useState<string | null>(null)
  const [items, setItems] = React.useState<api.FilesystemListItem[]>([])
  const [loadingRoots, setLoadingRoots] = React.useState(false)
  const [loadingList, setLoadingList] = React.useState(false)
  const [rootsError, setRootsError] = React.useState<string | null>(null)
  const [listError, setListError] = React.useState<string | null>(null)

  const allowDirectorySelect = mode === 'directory' || mode === 'file_or_directory'
  const allowFileSelect = mode === 'file' || mode === 'file_or_directory'

  const loadPath = React.useCallback(async (
    path: string,
    options?: { suppressError?: boolean }
  ): Promise<boolean> => {
    setLoadingList(true)
    if (!options?.suppressError) {
      setListError(null)
    }

    try {
      const result = await api.fetchFilesystemList(path)
      const sortedItems = [...result.items].sort((a, b) => {
        if (a.isDir !== b.isDir) {
          return a.isDir ? -1 : 1
        }
        return a.name.localeCompare(b.name)
      })

      setCurrentPath(result.path)
      setParentPath(result.parent)
      setItems(sortedItems)
      return true
    } catch (error) {
      if (!options?.suppressError) {
        const detail = error instanceof Error ? error.message : null
        setListError(detail ? `${t('pathPicker.errorList')}: ${detail}` : t('pathPicker.errorList'))
      }
      return false
    } finally {
      setLoadingList(false)
    }
  }, [t])

  const loadInitialPath = React.useCallback(
    async (path: string, rootsToTry: api.FilesystemRoot[]) => {
      const targetPath = path.trim()
      const candidates: string[] = []

      const pushCandidate = (candidate: string | null) => {
        if (!candidate) {
          return
        }
        const normalized = candidate.trim()
        if (!normalized || candidates.includes(normalized)) {
          return
        }
        candidates.push(normalized)
      }

      pushCandidate(targetPath)
      pushCandidate(parentPathOf(targetPath))
      pushCandidate(rootsToTry[0]?.path ?? null)

      for (let index = 0; index < candidates.length; index += 1) {
        const loaded = await loadPath(candidates[index], {
          suppressError: index < candidates.length - 1,
        })
        if (loaded) {
          return
        }
      }
    },
    [loadPath]
  )

  React.useEffect(() => {
    if (!open) {
      return
    }

    let cancelled = false
    const targetPath = value.trim()

    async function initialize() {
      setLoadingRoots(true)
      setRootsError(null)
      setListError(null)

      try {
        const nextRoots = await api.fetchFilesystemRoots()
        if (cancelled) {
          return
        }

        setRoots(nextRoots)
        const firstPath = targetPath || nextRoots[0]?.path || ''
        if (firstPath) {
          await loadInitialPath(firstPath, nextRoots)
        }
      } catch (error) {
        if (cancelled) {
          return
        }
        const detail = error instanceof Error ? error.message : null
        setRootsError(detail ? `${t('pathPicker.errorRoots')}: ${detail}` : t('pathPicker.errorRoots'))
      } finally {
        if (!cancelled) {
          setLoadingRoots(false)
        }
      }
    }

    void initialize()

    return () => {
      cancelled = true
    }
  }, [loadInitialPath, open, t, value])

  const selectPath = (path: string) => {
    onChange(path)
    setOpen(false)
  }

  return (
    <>
      <div className="flex min-w-0 items-center gap-2">
        <Input
          value={value}
          onChange={(event) => onChange(event.target.value)}
          placeholder={placeholder}
          className={inputClassName}
          disabled={disabled}
        />
        <Button
          type="button"
          variant="secondary"
          size="md"
          className="shrink-0"
          onClick={() => setOpen(true)}
          disabled={disabled}
        >
          <FolderOpen className="h-4 w-4" />
          {browseButtonLabel ?? t('pathPicker.browseBackend')}
        </Button>
      </div>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="w-[96vw] max-w-3xl p-0">
          <DialogHeader className="mb-0 border-b border-border px-4 py-3">
            <DialogTitle>{dialogTitle ?? t('pathPicker.title')}</DialogTitle>
            <DialogDescription>{dialogDescription ?? t('pathPicker.description')}</DialogDescription>
          </DialogHeader>

          <div className="space-y-3 px-4 py-3">
            <div className="flex flex-wrap gap-2">
              {roots.map((root) => (
                <Button
                  key={`${root.label}:${root.path}`}
                  type="button"
                  variant="secondary"
                  size="sm"
                  className="max-w-full"
                  onClick={() => void loadPath(root.path)}
                >
                  <HardDrive className="h-3.5 w-3.5 shrink-0" />
                  <span className="truncate">{root.label}</span>
                </Button>
              ))}
              {loadingRoots ? (
                <span className="text-xs text-muted-foreground">{t('pathPicker.loadingRoots')}</span>
              ) : null}
            </div>

            {rootsError ? (
              <div className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                {rootsError}
              </div>
            ) : null}

            <div className="flex flex-wrap items-center gap-2">
              <Button
                type="button"
                variant="secondary"
                size="sm"
                onClick={() => parentPath && void loadPath(parentPath)}
                disabled={!parentPath || loadingList}
              >
                <ArrowUp className="h-3.5 w-3.5" />
                {t('pathPicker.parent')}
              </Button>
              <Button
                type="button"
                variant="secondary"
                size="sm"
                onClick={() => currentPath && void loadPath(currentPath)}
                disabled={!currentPath || loadingList}
              >
                <RefreshCw className="h-3.5 w-3.5" />
                {t('pathPicker.refresh')}
              </Button>
              <span className="min-w-0 flex-1 truncate rounded-md border border-border bg-canvas px-2 py-1.5 font-mono text-xs text-foreground">
                {currentPath || t('pathPicker.noPath')}
              </span>
            </div>

            {listError ? (
              <div className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                {listError}
              </div>
            ) : null}

            <div className="max-h-[360px] overflow-y-auto rounded-md border border-border bg-canvas">
              {loadingList ? (
                <div className="px-3 py-4 text-sm text-muted-foreground">{t('pathPicker.loading')}</div>
              ) : items.length === 0 ? (
                <div className="px-3 py-4 text-sm text-muted-foreground">{t('pathPicker.empty')}</div>
              ) : (
                <ul className="divide-y divide-border">
                  {items.map((item) => {
                    const fileDisabled = item.isFile && !allowFileSelect
                    const itemNotSelectable =
                      (item.isDir && !allowDirectorySelect) || (item.isFile && !allowFileSelect)

                    return (
                      <li key={item.path} className="flex items-center gap-2 px-2 py-1.5">
                        <Button
                          type="button"
                          variant="ghost"
                          size="md"
                          className="min-w-0 flex-1 justify-start px-2"
                          onClick={() => {
                            if (fileDisabled) {
                              return
                            }
                            if (item.isDir) {
                              void loadPath(item.path)
                            } else if (item.isFile && allowFileSelect) {
                              selectPath(item.path)
                            }
                          }}
                          disabled={fileDisabled}
                          title={fileDisabled ? t('pathPicker.fileDisabledTitle') : undefined}
                        >
                          {item.isDir ? (
                            <Folder className="h-4 w-4 shrink-0 text-muted-foreground" />
                          ) : (
                            <File className="h-4 w-4 shrink-0 text-muted-foreground" />
                          )}
                          <span className="truncate font-mono text-xs">{item.name}</span>
                          {fileDisabled ? (
                            <span className="shrink-0 text-[11px] text-muted-foreground">
                              {t('pathPicker.directoryOnly')}
                            </span>
                          ) : null}
                        </Button>
                        <Button
                          type="button"
                          variant="secondary"
                          size="sm"
                          onClick={() => selectPath(item.path)}
                          className="shrink-0"
                          disabled={itemNotSelectable}
                          title={itemNotSelectable ? t('pathPicker.itemNotSelectableTitle') : undefined}
                        >
                          {item.isDir ? t('pathPicker.selectDirectory') : t('pathPicker.selectFile')}
                        </Button>
                      </li>
                    )
                  })}
                </ul>
              )}
            </div>
          </div>

          <DialogFooter className="mt-0 border-t border-border px-4 py-3">
            <Button type="button" variant="secondary" onClick={() => setOpen(false)}>
              {t('pathPicker.close')}
            </Button>
            {allowDirectorySelect ? (
              <Button
                type="button"
                variant="primary"
                onClick={() => currentPath && selectPath(currentPath)}
                disabled={!currentPath}
              >
                {t('pathPicker.currentFolder')}
              </Button>
            ) : null}
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}
