'use client'

import * as React from 'react'
import { cn } from '@/lib/utils'

export interface TextareaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  autoResize?: boolean
  minRows?: number
  maxRows?: number
}

const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, autoResize = false, minRows = 1, maxRows = 6, onChange, ...props }, ref) => {
    const textareaRef = React.useRef<HTMLTextAreaElement | null>(null)

    const combinedRef = React.useCallback(
      (node: HTMLTextAreaElement | null) => {
        textareaRef.current = node
        if (typeof ref === 'function') {
          ref(node)
        } else if (ref) {
          ref.current = node
        }
      },
      [ref]
    )

    const adjustHeight = React.useCallback(() => {
      const textarea = textareaRef.current
      if (!textarea || !autoResize) return

      textarea.style.height = 'auto'
      const lineHeight = parseInt(getComputedStyle(textarea).lineHeight)
      const minHeight = lineHeight * minRows
      const maxHeight = lineHeight * maxRows

      const newHeight = Math.min(Math.max(textarea.scrollHeight, minHeight), maxHeight)
      textarea.style.height = `${newHeight}px`
      textarea.style.overflowY = textarea.scrollHeight > maxHeight ? 'auto' : 'hidden'
    }, [autoResize, minRows, maxRows])

    const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      adjustHeight()
      onChange?.(e)
    }

    React.useEffect(() => {
      adjustHeight()
    }, [adjustHeight, props.value])

    return (
      <textarea
        ref={combinedRef}
        className={cn(
          'flex w-full rounded-md bg-surface-layer text-foreground',
          'border border-border px-3 py-2 text-sm',
          'placeholder:text-muted-foreground',
          'transition-all duration-150',
          'focus-visible:outline-none focus-visible:border-primary-500',
          'focus-visible:ring-[3px] focus-visible:ring-primary-500/25',
          'disabled:opacity-40 disabled:cursor-not-allowed',
          'resize-none',
          className
        )}
        onChange={handleChange}
        {...props}
      />
    )
  }
)
Textarea.displayName = 'Textarea'

export { Textarea }
