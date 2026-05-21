'use client'

import { ArrowDown } from 'lucide-react'
import { Button } from '@/components/ui/button'

interface ScrollToBottomProps {
  visible: boolean
  onClick: () => void
}

export function ScrollToBottom({ visible, onClick }: ScrollToBottomProps) {
  if (!visible) {
    return null
  }

  return (
    <div className="pointer-events-none absolute bottom-24 right-6 z-20">
      <Button
        type="button"
        variant="secondary"
        size="icon"
        onClick={onClick}
        className="pointer-events-auto rounded-full shadow-lg"
      >
        <ArrowDown className="h-4 w-4" />
      </Button>
    </div>
  )
}
