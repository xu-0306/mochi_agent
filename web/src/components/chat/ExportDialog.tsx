'use client'

import * as React from 'react'
import { Copy, Download } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Switch } from '@/components/ui/switch'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { buildChatExport, type ChatExportFormat, type ChatExportTraceEvent } from '@/lib/chat-p2'
import type { Message } from '@/lib/chat'
import { cn } from '@/lib/utils'

interface ExportDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  messages: Message[]
  traceEvents?: ChatExportTraceEvent[]
}

function downloadText(filename: string, content: string) {
  const blob = new Blob([content], { type: 'text/plain;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  URL.revokeObjectURL(url)
}

export function ExportDialog({ open, onOpenChange, messages, traceEvents = [] }: ExportDialogProps) {
  const [format, setFormat] = React.useState<ChatExportFormat>('markdown')
  const [includeReasoning, setIncludeReasoning] = React.useState(false)
  const content = React.useMemo(
    () => buildChatExport(messages, format, { includeReasoning, traceEvents }),
    [format, includeReasoning, messages, traceEvents]
  )

  const handleCopy = React.useCallback(async () => {
    await navigator.clipboard.writeText(content)
  }, [content])

  const handleDownload = React.useCallback(() => {
    downloadText(format === 'markdown' ? 'mochi-chat.md' : 'mochi-chat.json', content)
  }, [content, format])

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl">
        <DialogHeader>
          <DialogTitle>Export chat</DialogTitle>
          <DialogDescription>Download or copy the current conversation.</DialogDescription>
        </DialogHeader>

        <div className="grid gap-3 sm:grid-cols-[auto_minmax(18rem,1fr)] sm:items-stretch">
          <div
            className="inline-flex h-12 w-fit items-center gap-1 rounded-lg border border-border bg-surface-layer p-1"
            aria-label="Export format"
          >
            <Button
              type="button"
              variant={format === 'markdown' ? 'primary' : 'ghost'}
              size="sm"
              className="h-9 px-3"
              onClick={() => setFormat('markdown')}
            >
              Markdown
            </Button>
            <Button
              type="button"
              variant={format === 'json' ? 'primary' : 'ghost'}
              size="sm"
              className="h-9 px-3"
              onClick={() => setFormat('json')}
            >
              JSON
            </Button>
          </div>

          <div
            className={cn(
              'group flex min-h-12 items-center justify-between gap-4 rounded-lg border px-4 py-3 transition-colors',
              includeReasoning
                ? 'border-primary-500/40 bg-primary-500/10 shadow-[0_0_0_1px_rgba(99,102,241,0.16)]'
                : 'border-border bg-surface-layer hover:border-primary-500/30 hover:bg-secondary/20'
            )}
          >
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-2">
                <p className="text-sm font-medium text-foreground">Include reasoning/tool trace</p>
                <span className={cn(
                  'rounded-full border px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide',
                  includeReasoning
                    ? 'border-primary-500/30 bg-primary-500/15 text-primary-200'
                    : 'border-border bg-canvas text-muted-foreground'
                )}>
                  {includeReasoning ? 'Debug export on' : 'Optional'}
                </span>
              </div>
              <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                Adds saved reasoning summaries, tool calls, results, errors, and metadata to the preview and download.
              </p>
            </div>
            <Switch
              checked={includeReasoning}
              onCheckedChange={setIncludeReasoning}
              aria-label="Include reasoning and tool trace in export"
            />
          </div>
        </div>

        <pre className="max-h-[60vh] overflow-auto rounded-lg border border-border bg-canvas p-4 text-xs text-foreground whitespace-pre-wrap break-all font-mono">
          {content || 'No messages to export.'}
        </pre>

        <DialogFooter>
          <Button type="button" variant="ghost" onClick={() => void handleCopy()}>
            <Copy className="h-3.5 w-3.5" />
            Copy
          </Button>
          <Button type="button" variant="primary" onClick={handleDownload}>
            <Download className="h-3.5 w-3.5" />
            Download
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
