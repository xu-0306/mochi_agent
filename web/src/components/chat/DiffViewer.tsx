'use client'

import * as React from 'react'
import ReactDiffViewer, { DiffMethod, type ReactDiffViewerStylesOverride } from 'react-diff-viewer-continued'
import { Highlight, type PrismTheme } from 'prism-react-renderer'
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

interface ParsedDiffSource {
  filePath: string | null
  oldValue: string
  newValue: string
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
    { types: ['inserted'], style: { color: '#b7f7cf' } },
    { types: ['deleted'], style: { color: '#fecdd3' } },
  ],
}

const diffVariables = {
  diffViewerBackground: 'var(--code-surface)',
  diffViewerColor: 'var(--code-fg)',
  diffViewerTitleBackground: 'var(--code-toolbar)',
  diffViewerTitleColor: 'var(--code-fg)',
  diffViewerTitleBorderColor: 'var(--code-border)',
  addedBackground: 'rgba(22, 163, 74, 0.14)',
  addedColor: 'var(--code-fg)',
  removedBackground: 'rgba(220, 38, 38, 0.14)',
  removedColor: 'var(--code-fg)',
  wordAddedBackground: 'rgba(22, 163, 74, 0.24)',
  wordRemovedBackground: 'rgba(220, 38, 38, 0.22)',
  addedGutterBackground: 'rgba(22, 163, 74, 0.16)',
  removedGutterBackground: 'rgba(220, 38, 38, 0.16)',
  gutterBackground: 'var(--code-toolbar)',
  gutterBackgroundDark: 'var(--code-toolbar)',
  highlightBackground: 'rgba(94, 106, 210, 0.14)',
  highlightGutterBackground: 'rgba(94, 106, 210, 0.2)',
  codeFoldGutterBackground: 'rgba(255, 255, 255, 0.05)',
  codeFoldBackground: 'rgba(255, 255, 255, 0.04)',
  emptyLineBackground: 'rgba(255, 255, 255, 0.02)',
  gutterColor: 'hsl(var(--muted-foreground))',
  addedGutterColor: '#bbf7d0',
  removedGutterColor: '#fecdd3',
  codeFoldContentColor: 'hsl(var(--muted-foreground))',
} satisfies NonNullable<ReactDiffViewerStylesOverride['variables']>['dark']

const diffStyles: ReactDiffViewerStylesOverride = {
  variables: {
    dark: diffVariables,
    light: diffVariables,
  },
  diffContainer: {
    borderRadius: '0',
    borderWidth: '0',
    background: 'var(--code-surface)',
    color: 'var(--code-fg)',
    fontFamily: 'var(--font-mono, "JetBrains Mono", monospace)',
    fontSize: '12px',
    lineHeight: '1.55',
  },
  content: {
    width: '100%',
  },
  line: {
    minHeight: '1.75rem',
  },
  gutter: {
    minWidth: '2.9rem',
    padding: '0 0.75rem',
    fontSize: '11px',
  },
  lineNumber: {
    opacity: 0.85,
  },
  marker: {
    padding: '0 0.5rem',
  },
  contentText: {
    fontFamily: 'inherit',
  },
  lineContent: {
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-word',
  },
  wordDiff: {
    padding: '1px 2px',
    borderRadius: '4px',
  },
  codeFold: {
    background: 'rgba(255, 255, 255, 0.03)',
  },
}

function normalizeFilePath(value: string | null | undefined): string | null {
  if (!value) {
    return null
  }

  const trimmed = value.trim()
  if (!trimmed || trimmed === '/dev/null') {
    return null
  }

  if (trimmed.startsWith('a/') || trimmed.startsWith('b/')) {
    return trimmed.slice(2)
  }

  return trimmed
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

function highlightSyntax(content: string, language: string) {
  return (
    <Highlight theme={syntaxTheme} code={content} language={language}>
      {({ tokens, getTokenProps }) => (
        <span className="whitespace-pre-wrap break-words">
          {tokens.map((line, lineIndex) => (
            <span key={lineIndex}>
              {line.map((token, tokenIndex) => (
                <span key={tokenIndex} {...getTokenProps({ token })} />
              ))}
            </span>
          ))}
        </span>
      )}
    </Highlight>
  )
}

function parseHunkHeader(line: string): { oldStart: number; newStart: number } | null {
  const match = /^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@/.exec(line)
  if (!match) {
    return null
  }

  return {
    oldStart: Number.parseInt(match[1], 10),
    newStart: Number.parseInt(match[2], 10),
  }
}

function buildDiffSourceFromUnifiedDiff(
  diff: string,
  fallbackFilePath: string | null
): ParsedDiffSource | null {
  const oldLines: string[] = []
  const newLines: string[] = []
  let inferredFilePath = fallbackFilePath
  let sawHunk = false

  for (const line of diff.split(/\r?\n/)) {
    if (line.startsWith('--- ')) {
      const beforePath = normalizeFilePath(line.slice(4))
      if (!inferredFilePath && beforePath) {
        inferredFilePath = beforePath
      }
      continue
    }

    if (line.startsWith('+++ ')) {
      const afterPath = normalizeFilePath(line.slice(4))
      if (afterPath) {
        inferredFilePath = afterPath
      }
      continue
    }

    if (line.startsWith('@@ ')) {
      sawHunk = true
      const header = parseHunkHeader(line)
      if (!header) {
        continue
      }
      const oldGap = Math.max(0, header.oldStart - 1 - oldLines.length)
      const newGap = Math.max(0, header.newStart - 1 - newLines.length)
      const gap = Math.max(oldGap, newGap)
      for (let index = 0; index < gap; index += 1) {
        oldLines.push('')
        newLines.push('')
      }
      continue
    }

    if (!sawHunk || line.startsWith('\\ ')) {
      continue
    }

    if (line.startsWith('+') && !line.startsWith('+++')) {
      newLines.push(line.slice(1))
      continue
    }

    if (line.startsWith('-') && !line.startsWith('---')) {
      oldLines.push(line.slice(1))
      continue
    }

    if (line.startsWith(' ')) {
      const content = line.slice(1)
      oldLines.push(content)
      newLines.push(content)
    }
  }

  if (!sawHunk) {
    return null
  }

  return {
    filePath: inferredFilePath,
    oldValue: oldLines.join('\n'),
    newValue: newLines.join('\n'),
  }
}

function buildDiffSource({
  diff,
  filePath,
  oldValue,
  newValue,
}: Pick<DiffViewerProps, 'diff' | 'filePath' | 'oldValue' | 'newValue'>): ParsedDiffSource | null {
  const normalizedPath = normalizeFilePath(filePath)

  if (oldValue != null || newValue != null) {
    return {
      filePath: normalizedPath,
      oldValue: oldValue ?? '',
      newValue: newValue ?? '',
    }
  }

  if (!diff) {
    return null
  }

  return buildDiffSourceFromUnifiedDiff(diff, normalizedPath)
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
  const source = React.useMemo(
    () => buildDiffSource({ diff, filePath, oldValue, newValue }),
    [diff, filePath, oldValue, newValue]
  )
  const language = React.useMemo(() => inferLanguage(source?.filePath ?? filePath ?? null), [filePath, source?.filePath])

  return (
    <div
      className={cn(
        'overflow-hidden rounded-2xl border mochi-code-frame bg-[var(--code-surface)] shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]',
        className
      )}
    >
      <div
        className={cn(
          'overflow-auto bg-[var(--code-surface)]',
          maxHeightClassName
        )}
      >
        {!source ? (
          <div className="px-4 py-4 text-sm text-muted-foreground">{emptyLabel}</div>
        ) : (
          <ReactDiffViewer
            oldValue={source.oldValue}
            newValue={source.newValue}
            splitView={false}
            compareMethod={DiffMethod.WORDS}
            renderContent={(value) => highlightSyntax(value, language)}
            hideLineNumbers={false}
            showDiffOnly
            extraLinesSurroundingDiff={2}
            hideSummary
            useDarkTheme
            styles={diffStyles}
          />
        )}
      </div>
    </div>
  )
}
