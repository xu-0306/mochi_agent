'use client'

import * as React from 'react'
import type { Components } from 'react-markdown'
import { cn } from '@/lib/utils'
import { CopyButton } from '@/components/chat/CopyButton'

export function extractCodeText(children: React.ReactNode): string {
  return React.Children.toArray(children)
    .map((child) => (typeof child === 'string' ? child : ''))
    .join('')
}

export function isBlockCode({
  className,
  codeText,
  node,
}: {
  className?: string
  codeText: string
  node?: { position?: { start?: { line?: number }; end?: { line?: number } } }
}): boolean {
  if (className) {
    return true
  }

  if (codeText.includes('\n')) {
    return true
  }

  const startLine = node?.position?.start?.line
  const endLine = node?.position?.end?.line
  return Number.isInteger(startLine) && Number.isInteger(endLine) && startLine !== endLine
}

export function createMarkdownCodeComponents({
  showCopyButton = true,
}: {
  showCopyButton?: boolean
} = {}): Components {
  return {
    code(props) {
      const { children, className, node, ...rest } = props
      const codeText = extractCodeText(children)

      if (isBlockCode({ className, codeText, node })) {
        return (
          <code className={cn('mochi-code-content', className)} {...rest}>
            {children}
          </code>
        )
      }

      return (
        <code
          className={cn('mochi-inline-code text-[0.9em]', className)}
          {...rest}
        >
          {children}
        </code>
      )
    },
    pre(props) {
      const { children, ...rest } = props
      const child = React.Children.only(children)
      const childProps =
        React.isValidElement<{ children?: React.ReactNode; className?: string }>(child)
          ? child.props
          : null
      const codeText = extractCodeText(childProps?.children).replace(/\n$/, '')

      return (
        <div className="mochi-code-frame relative my-4 overflow-hidden rounded-lg border">
          {showCopyButton ? (
            <div className="mochi-code-toolbar flex items-center justify-end border-b px-2 py-1">
              <CopyButton text={codeText} label="Copy code" />
            </div>
          ) : null}
          <pre className="mochi-code-block overflow-x-auto p-3" {...rest}>
            {children}
          </pre>
        </div>
      )
    },
  }
}
