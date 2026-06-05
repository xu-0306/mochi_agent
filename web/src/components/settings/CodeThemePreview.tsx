'use client'

import * as React from 'react'
import ReactMarkdown from 'react-markdown'
import rehypeHighlight from 'rehype-highlight'
import remarkGfm from 'remark-gfm'
import { createMarkdownCodeComponents } from '@/components/code/markdown-code'

const SAMPLE = `\`\`\`python
class Codec:
    def encode(self, strs):
        """Encodes a list of strings to a single string."""
        result = []
        for s in strs:
            result.append(str(len(s)) + "#" + s)
        return "".join(result)
\`\`\`
`

export function CodeThemePreview() {
  const components = React.useMemo(
    () => createMarkdownCodeComponents({ showCopyButton: false }),
    []
  )

  return (
    <div className="rounded-lg border border-border bg-canvas px-3 py-3">
      <div className="prose max-w-none text-sm leading-7 text-foreground">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          rehypePlugins={[rehypeHighlight]}
          components={components}
        >
          {SAMPLE}
        </ReactMarkdown>
      </div>
    </div>
  )
}
