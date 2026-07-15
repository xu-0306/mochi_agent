export type DiffLineKind = 'context' | 'added' | 'removed'

export interface DiffDisplayLine {
  kind: DiffLineKind
  oldNumber: number | null
  newNumber: number | null
  marker: ' ' | '+' | '-'
  content: string
}

export interface DiffDocument {
  filePath: string | null
  lines: DiffDisplayLine[]
  stats: {
    additions: number
    deletions: number
  }
}

interface DiffDocumentInput {
  diff?: string | null
  filePath?: string | null
  oldValue?: string | null
  newValue?: string | null
}

function normalizeFilePath(value: string | null | undefined): string | null {
  if (!value) {
    return null
  }

  const trimmed = value.trim().split('\t', 1)[0]
  if (!trimmed || trimmed === '/dev/null') {
    return null
  }

  if (trimmed.startsWith('a/') || trimmed.startsWith('b/')) {
    return trimmed.slice(2)
  }

  return trimmed
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

function statsFor(lines: DiffDisplayLine[]) {
  return lines.reduce(
    (stats, line) => ({
      additions: stats.additions + (line.kind === 'added' ? 1 : 0),
      deletions: stats.deletions + (line.kind === 'removed' ? 1 : 0),
    }),
    { additions: 0, deletions: 0 }
  )
}

function parseUnifiedDiff(
  diff: string,
  fallbackFilePath: string | null
): DiffDocument | null {
  const lines: DiffDisplayLine[] = []
  let filePath = fallbackFilePath
  let oldNumber = 0
  let newNumber = 0
  let sawHunk = false

  for (const line of diff.split(/\r?\n/)) {
    if (line.startsWith('--- ')) {
      filePath = filePath ?? normalizeFilePath(line.slice(4))
      continue
    }

    if (line.startsWith('+++ ')) {
      filePath = normalizeFilePath(line.slice(4)) ?? filePath
      continue
    }

    if (line.startsWith('@@ ')) {
      const header = parseHunkHeader(line)
      if (header) {
        oldNumber = header.oldStart
        newNumber = header.newStart
        sawHunk = true
      }
      continue
    }

    if (!sawHunk || line.startsWith('\\ ')) {
      continue
    }

    if (line.startsWith('+') && !line.startsWith('+++')) {
      lines.push({
        kind: 'added',
        oldNumber: null,
        newNumber,
        marker: '+',
        content: line.slice(1),
      })
      newNumber += 1
      continue
    }

    if (line.startsWith('-') && !line.startsWith('---')) {
      lines.push({
        kind: 'removed',
        oldNumber,
        newNumber: null,
        marker: '-',
        content: line.slice(1),
      })
      oldNumber += 1
      continue
    }

    if (line.startsWith(' ')) {
      lines.push({
        kind: 'context',
        oldNumber,
        newNumber,
        marker: ' ',
        content: line.slice(1),
      })
      oldNumber += 1
      newNumber += 1
    }
  }

  if (!sawHunk) {
    return null
  }

  return {
    filePath,
    lines,
    stats: statsFor(lines),
  }
}

function splitContent(value: string): string[] {
  return value.length === 0 ? [] : value.split(/\r?\n/)
}

function buildContentOperations(
  oldLines: string[],
  newLines: string[]
): Array<{ kind: DiffLineKind; content: string }> {
  const rowCount = oldLines.length
  const columnCount = newLines.length

  if (rowCount * columnCount > 40_000) {
    return [
      ...oldLines.map((content) => ({ kind: 'removed' as const, content })),
      ...newLines.map((content) => ({ kind: 'added' as const, content })),
    ]
  }

  const lcs: number[][] = Array.from(
    { length: rowCount + 1 },
    () => Array(columnCount + 1).fill(0)
  )

  for (let row = rowCount - 1; row >= 0; row -= 1) {
    for (let column = columnCount - 1; column >= 0; column -= 1) {
      lcs[row][column] = oldLines[row] === newLines[column]
        ? lcs[row + 1][column + 1] + 1
        : Math.max(lcs[row + 1][column], lcs[row][column + 1])
    }
  }

  const operations: Array<{ kind: DiffLineKind; content: string }> = []
  let row = 0
  let column = 0

  while (row < rowCount && column < columnCount) {
    if (oldLines[row] === newLines[column]) {
      operations.push({ kind: 'context', content: oldLines[row] })
      row += 1
      column += 1
    } else if (lcs[row + 1][column] >= lcs[row][column + 1]) {
      operations.push({ kind: 'removed', content: oldLines[row] })
      row += 1
    } else {
      operations.push({ kind: 'added', content: newLines[column] })
      column += 1
    }
  }

  while (row < rowCount) {
    operations.push({ kind: 'removed', content: oldLines[row] })
    row += 1
  }

  while (column < columnCount) {
    operations.push({ kind: 'added', content: newLines[column] })
    column += 1
  }

  return operations
}

function buildFromContents(input: DiffDocumentInput): DiffDocument {
  const oldLines = splitContent(input.oldValue ?? '')
  const newLines = splitContent(input.newValue ?? '')
  const operations = buildContentOperations(oldLines, newLines)
  const lines: DiffDisplayLine[] = []
  let oldNumber = 1
  let newNumber = 1

  for (const operation of operations) {
    if (operation.kind === 'added') {
      lines.push({
        kind: 'added',
        oldNumber: null,
        newNumber,
        marker: '+',
        content: operation.content,
      })
      newNumber += 1
    } else if (operation.kind === 'removed') {
      lines.push({
        kind: 'removed',
        oldNumber,
        newNumber: null,
        marker: '-',
        content: operation.content,
      })
      oldNumber += 1
    } else {
      lines.push({
        kind: 'context',
        oldNumber,
        newNumber,
        marker: ' ',
        content: operation.content,
      })
      oldNumber += 1
      newNumber += 1
    }
  }

  return {
    filePath: normalizeFilePath(input.filePath),
    lines,
    stats: statsFor(lines),
  }
}

export function buildDiffDocument(input: DiffDocumentInput): DiffDocument {
  const filePath = normalizeFilePath(input.filePath)

  if (input.diff) {
    const parsed = parseUnifiedDiff(input.diff, filePath)
    if (parsed) {
      return parsed
    }
  }

  return buildFromContents(input)
}
