'use client'

import * as React from 'react'
import {
  Download,
  FileArchive,
  FileCode2,
  FileImage,
  FileSpreadsheet,
  FileText,
  Loader2,
  X,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import * as api from '@/lib/api'
import type { ChatAttachment } from '@/lib/chat'
import { cn } from '@/lib/utils'

interface ChatAttachmentsProps {
  attachments: ChatAttachment[]
  variant?: 'composer' | 'message'
  onRemove?: (attachment: ChatAttachment) => void
  className?: string
}

type PreviewMode = 'image' | 'pdf' | 'text' | 'download'

function getExtension(name: string): string {
  const parts = name.split('.')
  return parts.length > 1 ? parts[parts.length - 1].toUpperCase() : 'FILE'
}

function isImageAttachment(attachment: ChatAttachment): boolean {
  return (
    attachment.contentType?.startsWith('image/') === true ||
    /\.(png|jpe?g|gif|webp|bmp|svg)$/i.test(attachment.name)
  )
}

function isPdfAttachment(attachment: ChatAttachment): boolean {
  return attachment.contentType === 'application/pdf' || /\.pdf$/i.test(attachment.name)
}

function isTextPreviewAttachment(attachment: ChatAttachment): boolean {
  return (
    /\.(txt|md|json|ya?ml|toml|ini|cfg|py|ts|tsx|js|jsx|html|css|scss|sql|xml|log|csv|tsv|docx|pdf|ipynb)$/i.test(
      attachment.name
    )
  )
}

function getPreviewMode(attachment: ChatAttachment): PreviewMode {
  if (isImageAttachment(attachment)) {
    return 'image'
  }
  if (isPdfAttachment(attachment)) {
    return 'pdf'
  }
  if (isTextPreviewAttachment(attachment)) {
    return 'text'
  }
  return 'download'
}

function formatFileSize(size?: number | null): string {
  if (!size || size <= 0) {
    return 'Attached file'
  }
  if (size < 1024) {
    return `${size} B`
  }
  if (size < 1024 * 1024) {
    return `${(size / 1024).toFixed(1)} KB`
  }
  return `${(size / (1024 * 1024)).toFixed(1)} MB`
}

function AttachmentIcon({ attachment }: { attachment: ChatAttachment }) {
  if (isImageAttachment(attachment)) {
    return <FileImage className="h-5 w-5" />
  }
  if (/\.(csv|tsv|xlsx?)$/i.test(attachment.name)) {
    return <FileSpreadsheet className="h-5 w-5" />
  }
  if (/\.(zip|rar|7z|tar|gz)$/i.test(attachment.name)) {
    return <FileArchive className="h-5 w-5" />
  }
  if (/\.(py|ts|tsx|js|jsx|json|yaml|yml|toml|ini|cfg|html|css|scss|sql|xml|ipynb)$/i.test(attachment.name)) {
    return <FileCode2 className="h-5 w-5" />
  }
  return <FileText className="h-5 w-5" />
}

export function ChatAttachments({
  attachments,
  variant = 'message',
  onRemove,
  className,
}: ChatAttachmentsProps) {
  const [activeAttachment, setActiveAttachment] = React.useState<ChatAttachment | null>(null)
  const [previewText, setPreviewText] = React.useState<string>('')
  const [previewLoading, setPreviewLoading] = React.useState(false)
  const [previewError, setPreviewError] = React.useState<string | null>(null)
  const [previewTruncated, setPreviewTruncated] = React.useState(false)

  React.useEffect(() => {
    if (!activeAttachment || getPreviewMode(activeAttachment) !== 'text') {
      setPreviewText('')
      setPreviewLoading(false)
      setPreviewError(null)
      setPreviewTruncated(false)
      return
    }

    let cancelled = false
    setPreviewLoading(true)
    setPreviewError(null)
    setPreviewText('')
    setPreviewTruncated(false)

    void (async () => {
      try {
        const payload = await api.fetchFilesystemPreviewText(activeAttachment.path)
        if (cancelled) {
          return
        }
        setPreviewText(payload.text)
        setPreviewTruncated(payload.truncated)
      } catch (error) {
        if (!cancelled) {
          setPreviewError(error instanceof Error ? error.message : 'Preview unavailable.')
        }
      } finally {
        if (!cancelled) {
          setPreviewLoading(false)
        }
      }
    })()

    return () => {
      cancelled = true
    }
  }, [activeAttachment])

  if (attachments.length === 0) {
    return null
  }

  const previewUrl = activeAttachment ? api.buildFilesystemFileUrl(activeAttachment.path) : null
  const previewMode = activeAttachment ? getPreviewMode(activeAttachment) : null

  return (
    <>
      <div className={cn('flex flex-wrap gap-2', className)}>
        {attachments.map((attachment) => {
          const fileUrl = api.buildFilesystemFileUrl(attachment.path)
          const isImage = isImageAttachment(attachment)
          const surfaceClass =
            variant === 'composer'
              ? 'border-border/80 bg-canvas/80 hover:border-primary-500/60 hover:bg-elevated-layer'
              : 'border-white/14 bg-white/8 hover:border-white/24 hover:bg-white/12'

          return (
            <div
              key={attachment.id ?? attachment.path}
              className={cn(
                'group relative overflow-hidden rounded-2xl border shadow-[0_14px_35px_rgba(0,0,0,0.18)] transition-all duration-200',
                'w-[180px]',
                surfaceClass
              )}
            >
              <button
                type="button"
                onClick={() => setActiveAttachment(attachment)}
                className="flex w-full flex-col text-left"
              >
                <div className="relative h-28 overflow-hidden">
                  {isImage ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={fileUrl}
                      alt={attachment.name}
                      className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-[1.03]"
                    />
                  ) : (
                    <div
                      className={cn(
                        'flex h-full w-full items-center justify-between px-4 py-3 text-white',
                        variant === 'composer'
                          ? 'bg-[linear-gradient(135deg,rgba(71,85,105,0.95),rgba(15,23,42,0.96))]'
                          : 'bg-[linear-gradient(135deg,rgba(129,140,248,0.34),rgba(15,23,42,0.92))]'
                      )}
                    >
                      <div className="rounded-2xl border border-white/12 bg-black/15 p-3 text-white/90">
                        <AttachmentIcon attachment={attachment} />
                      </div>
                      <span className="rounded-full border border-white/14 bg-black/15 px-2 py-1 text-[10px] font-semibold tracking-[0.18em] text-white/88">
                        {getExtension(attachment.name)}
                      </span>
                    </div>
                  )}
                  <div className="pointer-events-none absolute inset-x-0 bottom-0 h-14 bg-gradient-to-t from-black/55 to-transparent" />
                </div>
                <div className="space-y-1 px-3 py-3">
                  <p
                    className={cn(
                      'line-clamp-2 text-sm font-medium leading-5',
                      variant === 'composer' ? 'text-foreground' : 'text-white'
                    )}
                  >
                    {attachment.name}
                  </p>
                  <p
                    className={cn(
                      'text-[11px]',
                      variant === 'composer' ? 'text-muted-foreground' : 'text-white/68'
                    )}
                  >
                    {formatFileSize(attachment.size)}
                  </p>
                </div>
              </button>
              {onRemove ? (
                <button
                  type="button"
                  onClick={() => onRemove(attachment)}
                  className={cn(
                    'absolute top-2 right-2 flex h-7 w-7 items-center justify-center rounded-full border shadow-sm backdrop-blur',
                    variant === 'composer'
                      ? 'border-border bg-canvas/90 text-muted-foreground hover:text-foreground'
                      : 'border-white/14 bg-black/25 text-white/80 hover:text-white'
                  )}
                  aria-label={`Remove ${attachment.name}`}
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              ) : null}
            </div>
          )
        })}
      </div>

      <Dialog open={activeAttachment !== null} onOpenChange={(open) => (!open ? setActiveAttachment(null) : undefined)}>
        <DialogContent className="max-w-4xl border-border/80 p-0">
          {activeAttachment ? (
            <>
              <DialogHeader className="border-b border-border/70 px-5 pt-5 pb-4">
                <DialogTitle className="pr-10">{activeAttachment.name}</DialogTitle>
                <DialogDescription className="break-all">{activeAttachment.path}</DialogDescription>
              </DialogHeader>

              <div className="max-h-[75vh] overflow-auto bg-canvas/70 px-5 py-4">
                {previewMode === 'image' && previewUrl ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={previewUrl} alt={activeAttachment.name} className="mx-auto max-h-[70vh] rounded-2xl object-contain shadow-lg" />
                ) : null}

                {previewMode === 'pdf' && previewUrl ? (
                  <iframe
                    src={previewUrl}
                    title={activeAttachment.name}
                    className="h-[70vh] w-full rounded-2xl border border-border bg-white"
                  />
                ) : null}

                {previewMode === 'text' ? (
                  <div className="rounded-2xl border border-border bg-[#0b1020] p-4 text-sm text-slate-100 shadow-inner">
                    {previewLoading ? (
                      <div className="flex items-center gap-2 text-slate-300">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        Loading preview...
                      </div>
                    ) : previewError ? (
                      <p className="text-rose-300">{previewError}</p>
                    ) : (
                      <>
                        <pre className="overflow-x-auto whitespace-pre-wrap break-words font-mono text-[13px] leading-6">
                          {previewText || 'No preview text available.'}
                        </pre>
                        {previewTruncated ? (
                          <p className="mt-3 text-xs text-slate-400">
                            Preview truncated for readability.
                          </p>
                        ) : null}
                      </>
                    )}
                  </div>
                ) : null}

                {previewMode === 'download' ? (
                  <div className="rounded-2xl border border-dashed border-border/80 bg-surface-layer px-5 py-8 text-center">
                    <AttachmentIcon attachment={activeAttachment} />
                    <p className="mt-3 text-sm text-foreground">
                      This file type does not support inline preview yet.
                    </p>
                    <p className="mt-1 text-xs text-muted-foreground">
                      You can still open the original file directly.
                    </p>
                  </div>
                ) : null}
              </div>

              <div className="flex items-center justify-between border-t border-border/70 px-5 py-4">
                <span className="text-xs text-muted-foreground">{formatFileSize(activeAttachment.size)}</span>
                {previewUrl ? (
                  <Button asChild variant="secondary" size="sm">
                    <a href={previewUrl} target="_blank" rel="noreferrer">
                      <Download className="h-3.5 w-3.5" />
                      Open original
                    </a>
                  </Button>
                ) : null}
              </div>
            </>
          ) : null}
        </DialogContent>
      </Dialog>
    </>
  )
}
