'use client'

import * as React from 'react'
import { Check, Copy } from 'lucide-react'
import { Button } from '@/components/ui/button'

interface CopyButtonProps {
  text: string
  label?: string
  className?: string
}

export function CopyButton({ text, label = 'Copy', className }: CopyButtonProps) {
  const [copied, setCopied] = React.useState(false)

  const handleCopy = React.useCallback(async () => {
    await navigator.clipboard.writeText(text)
    setCopied(true)
    window.setTimeout(() => setCopied(false), 1500)
  }, [text])

  return (
    <Button
      type="button"
      variant="ghost"
      size="icon-sm"
      className={className}
      onClick={() => void handleCopy()}
      title={copied ? 'Copied' : label}
      aria-label={copied ? 'Copied' : label}
    >
      {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
    </Button>
  )
}
