'use client'

import * as React from 'react'
import { Highlight, type PrismTheme } from 'prism-react-renderer'
import {
  buildDiffDocument,
  type DiffDisplayLine,
  type DiffLineKind,
} from '@/lib/diff-lines'
import { cn } from '@/lib/utils'

interface DiffViewerProps {
  diff: string | null
  filePath?: string | null
  oldValue?: string | null
  newValue?: string | null
  className?: string
  maxHeightClassName?: string
  emptyLabel?: string
}

const syntaxTheme: PrismTheme = {
  plain: {
    color: 'var(--code-fg)',
    backgroundColor: 'transparent',
  },
  styles: [
    { types: ['comment', 'prolog', 'doctype', 'cdata'], style: { color: 'var(--code-comment)', fontStyle: 'italic' } },
    { types: ['string', 'attr-value', 'template-string'], style: { color: 'var(--code-string)' } },
    { types: ['keyword', 'selector', 'important', 'atrule'], style: { color: 'var(--code-keyword)' } },
    { types: ['function'], style: { color: 'var(--code-function)' } },
    { types: ['number', 'boolean'], style: { color: 'var(--code-number)' } },
    { types: ['builtin', 'class-name', 'constant', 'symbol'], style: { color: 'var(--code-type)' } },
    { types: ['variable', 'parameter', 'property', 'attr-name'], style: { color: 'var(--code-variable)' } },
    { types: ['operator', 'punctuation'], style: { color: 'var(--code-operator)' } },
    { types: ['tag', 'entity'], style: { color: 'var(--code-tag)' } },
  ],
}

function inferLanguage(filePath: string | null): string {
  const extension = filePath?.split('.').pop()?.toLowerCase()
  const languageMap: Record<string, string> = {
    ts: 'typescript',
    tsx: 'tsx',
    js: 'javascript',
    jsx: 'jsx',
    py: 'python',
    rs: 'rust',
    go: 'go',
    rb: 'ruby',
    json: 'json',
    yaml: 'yaml',
    yml: 'yaml',
    toml: 'toml',
    md: 'markdown',
    css: 'css',
    scss: 'scss',
    html: 'markup',
    xml: 'markup',
    sql: 'sql',
    sh: 'bash',
    bash: 'bash',
    zsh: 'bash',
  }

  return languageMap[extension ?? ''] ?? 'text'
}

function rowTone(kind: DiffLineKind): string {
  if (kind === 'added') {
    return 'bg-emerald-500/[0.12] hover:bg-emerald-500/[0.16]'
  }
  if (kind === 'removed') {
    return 'bg-rose-500/[0.12] hover:bg-rose-500/[0.16]'
  }
  return 'bg-transparent hover:bg-white/[0.025]'
}

function markerTone(kind: DiffLineKind): string {
  if (kind === 'added') {
    return 'text-emerald-300'
  }
  if (kind === 'removed') {
    return 'text-rose-300'
  }
  return 'text-muted-foreground/45'
}

function highlightLine(content: string, language: string) {
  return (
    <Highlight theme={syntaxTheme} code={content || ' '} language={language}>
      {({ tokens, getTokenProps }) => (
        <>
          {tokens.map((tokenLine, lineIndex) => (
            <React.Fragment key={lineIndex}>
              {tokenLine.map((token, tokenIndex) => (
                <span key={tokenIndex} {...getTokenProps({ token })} />
              ))}
            </React.Fragment>
          ))}
        </>
      )}
    </Highlight>
  )
}

function DiffRow({
  line,
  language,
  index,
}: {
  line: DiffDisplayLine
  language: string
  index: number
}) {
  return (
    <div
      role="row"
      data-diff-kind={line.kind}
      className={cn(
        'grid min-w-max grid-cols-[3.25rem_1.75rem_minmax(18rem,1fr)] border-b border-white/[0.045] transition-colors last:border-b-0',
        rowTone(line.kind)
      )}
    >

      <span
        role="cell"
        aria-label={line.newNumber == null ? `Old line ${line.oldNumber}` : `New line ${line.newNumber}`}
        className="select-none border-r border-white/[0.05] bg-black/[0.1] px-2 py-1.5 text-right text-[11px] tabular-nums text-muted-foreground/75"
      >
        {line.newNumber ?? line.oldNumber ?? ''}
      </span>
      <span
        role="cell"
        aria-hidden="true"
        className={cn(
          'select-none border-r border-white/[0.045] px-2 py-1.5 text-center text-xs font-semibold',
          markerTone(line.kind)
        )}
      >
        {line.marker}
      </span>
      <code
        role="cell"
        className="block min-w-[18rem] whitespace-pre px-3 py-1.5 text-[12px] leading-5 text-[var(--code-fg)]"
        data-line-index={index}
      >
        {highlightLine(line.content, language)}
      </code>
    </div>
  )
}

export function DiffViewer({
  diff,
  filePath,
  oldValue,
  newValue,
  className,
  maxHeightClassName = 'max-h-[24rem]',
  emptyLabel = 'No diff available.',
}: DiffViewerProps) {
  const diffDocument = React.useMemo(
    () => buildDiffDocument({ diff, filePath, oldValue, newValue }),
    [diff, filePath, oldValue, newValue]
  )
  const language = React.useMemo(
    () => inferLanguage(diffDocument.filePath ?? filePath ?? null),
    [diffDocument.filePath, filePath]
  )

  return (
    <div
      className={cn(
        'min-w-0 max-w-full overflow-hidden rounded-xl border border-white/10 bg-[var(--code-surface)] shadow-[inset_0_1px_0_rgba(255,255,255,0.035)]',
        className
      )}
    >
      {diffDocument.lines.length === 0 ? (
        <div className="px-4 py-4 text-sm text-muted-foreground">{emptyLabel}</div>
      ) : (
        <div
          role="table"
          aria-label={`Inline diff for ${diffDocument.filePath ?? 'file'}`}
          className={cn(
            'w-full overflow-x-auto overflow-y-auto bg-[var(--code-surface)] font-mono [scrollbar-color:rgba(148,163,184,0.35)_transparent]',
            maxHeightClassName
          )}
        >
          <div className="w-full min-w-max">
            {diffDocument.lines.map((line, index) => (
              <DiffRow
                key={`${line.kind}:${line.oldNumber ?? 'x'}:${line.newNumber ?? 'x'}:${index}`}
                line={line}
                language={language}
                index={index}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
