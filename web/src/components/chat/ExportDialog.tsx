'use client'

import * as React from 'react'
import { Copy, Download } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { buildChatExport, type ChatExportFormat } from '@/lib/chat-p2'
import type { Message } from '@/lib/chat'

interface ExportDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  messages: Message[]
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

export function ExportDialog({ open, onOpenChange, messages }: ExportDialogProps) {
  const [format, setFormat] = React.useState<ChatExportFormat>('markdown')
  const content = React.useMemo(() => buildChatExport(messages, format), [format, messages])

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

        <div className="flex items-center gap-2">
          <Button
            type="button"
            variant={format === 'markdown' ? 'primary' : 'secondary'}
            size="sm"
            onClick={() => setFormat('markdown')}
          >
            Markdown
          </Button>
          <Button
            type="button"
            variant={format === 'json' ? 'primary' : 'secondary'}
            size="sm"
            onClick={() => setFormat('json')}
          >
            JSON
          </Button>
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
