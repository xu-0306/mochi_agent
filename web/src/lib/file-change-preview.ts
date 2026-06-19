'use client'

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function getString(value: unknown): string | null {
  return typeof value === 'string' ? value : null
}

function getBoolean(value: unknown): boolean | null {
  return typeof value === 'boolean' ? value : null
}

function getNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function getRecordArray(value: unknown): Record<string, unknown>[] {
  if (!Array.isArray(value)) {
    return []
  }
  return value.filter(isRecord)
}

function getStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return []
  }
  return value.filter((item): item is string => typeof item === 'string' && item.trim().length > 0)
}

export interface DiffStats {
  additions: number
  deletions: number
}

export interface FileChangeSummary {
  filePath: string
  relativePath: string | null
  displayPath: string
  previousFilePath: string | null
  status: string
  originalContent: string | null
  newContent: string | null
  diff: string | null
  diffAvailable: boolean
  additions: number
  deletions: number
  undoAvailable: boolean
  undoAction: 'restore' | 'delete' | null
}

export interface FileChangeGroupSummary {
  id: string
  sourceTool: string
  title: string
  patchText: string | null
  files: FileChangeSummary[]
}

export interface PatchPreviewResult {
  valid: boolean
  summary: string | null
  errors: string[]
  warnings: string[]
  patchText: string | null
  files: FileChangeSummary[]
}

export function getFileName(filePath: string): string {
  const segments = filePath.split(/[\\/]/)
  return segments[segments.length - 1] || filePath
}

export function summarizeDiffStats(diff: string | null | undefined): DiffStats {
  if (!diff) {
    return { additions: 0, deletions: 0 }
  }

  let additions = 0
  let deletions = 0
  for (const line of diff.split(/\r?\n/)) {
    if (line.startsWith('+++') || line.startsWith('---')) {
      continue
    }
    if (line.startsWith('+')) {
      additions += 1
      continue
    }
    if (line.startsWith('-')) {
      deletions += 1
    }
  }

  return { additions, deletions }
}

function deriveStatusFromContents(
  originalContent: string | null,
  newContent: string | null,
  fallback: string
): string {
  if (!originalContent && newContent) {
    return 'added'
  }
  if (originalContent && !newContent) {
    return 'deleted'
  }
  return fallback
}

function buildDiffOperations(originalLines: string[], nextLines: string[]): Array<{
  type: 'context' | 'add' | 'remove'
  value: string
}> {
  const rows = originalLines.length
  const cols = nextLines.length

  if (rows * cols > 40_000) {
    return []
  }

  const lcs: number[][] = Array.from({ length: rows + 1 }, () => Array(cols + 1).fill(0))

  for (let row = rows - 1; row >= 0; row -= 1) {
    for (let col = cols - 1; col >= 0; col -= 1) {
      if (originalLines[row] === nextLines[col]) {
        lcs[row][col] = lcs[row + 1][col + 1] + 1
      } else {
        lcs[row][col] = Math.max(lcs[row + 1][col], lcs[row][col + 1])
      }
    }
  }

  const operations: Array<{ type: 'context' | 'add' | 'remove'; value: string }> = []
  let row = 0
  let col = 0

  while (row < rows && col < cols) {
    if (originalLines[row] === nextLines[col]) {
      operations.push({ type: 'context', value: originalLines[row] })
      row += 1
      col += 1
      continue
    }

    if (lcs[row + 1][col] >= lcs[row][col + 1]) {
      operations.push({ type: 'remove', value: originalLines[row] })
      row += 1
      continue
    }

    operations.push({ type: 'add', value: nextLines[col] })
    col += 1
  }

  while (row < rows) {
    operations.push({ type: 'remove', value: originalLines[row] })
    row += 1
  }

  while (col < cols) {
    operations.push({ type: 'add', value: nextLines[col] })
    col += 1
  }

  return operations
}

function buildUnifiedDiffFromContents(
  filePath: string,
  originalContent: string | null,
  newContent: string | null
): string | null {
  if (originalContent == null || newContent == null) {
    return null
  }

  const originalLines = originalContent.split('\n')
  const nextLines = newContent.split('\n')
  const operations = buildDiffOperations(originalLines, nextLines)

  if (operations.length === 0) {
    return null
  }

  const beforeLabel = originalContent.length > 0 ? `a/${filePath}` : '/dev/null'
  const afterLabel = newContent.length > 0 ? `b/${filePath}` : '/dev/null'
  const diffLines = [
    `--- ${beforeLabel}`,
    `+++ ${afterLabel}`,
    `@@ -1,${originalLines.length} +1,${nextLines.length} @@`,
    ...operations.map((operation) => {
      if (operation.type === 'add') {
        return `+${operation.value}`
      }
      if (operation.type === 'remove') {
        return `-${operation.value}`
      }
      return ` ${operation.value}`
    }),
  ]

  return diffLines.join('\n')
}

function normalizeFileChangeRecord(
  payload: Record<string, unknown>
): FileChangeSummary | null {
  const filePath =
    getString(payload.path) ??
    getString(payload.file_path) ??
    getString(payload.relative_path) ??
    getString(payload.target_path)

  if (!filePath) {
    return null
  }

  const originalContent =
    getString(payload.original_content) ??
    getString(payload.before_text) ??
    getString(payload.old_content)
  const newContent =
    getString(payload.new_content) ??
    getString(payload.after_text) ??
    getString(payload.content)

  const diff =
    getString(payload.diff) ??
    buildUnifiedDiffFromContents(filePath, originalContent, newContent)
  const stats = summarizeDiffStats(diff)
  const previousFilePath =
    getString(payload.previous_path) ??
    getString(payload.from_path) ??
    getString(payload.original_path)
  const relativePath = getString(payload.relative_path)
  const status = deriveStatusFromContents(
    originalContent,
    newContent,
    getString(payload.status) ?? (previousFilePath && previousFilePath !== filePath ? 'renamed' : 'modified')
  )
  const undoAction = getString(payload.undo_action)

  return {
    filePath,
    relativePath,
    displayPath:
      previousFilePath && previousFilePath !== filePath
        ? `${previousFilePath} -> ${filePath}`
        : relativePath ?? filePath,
    previousFilePath,
    status,
    originalContent,
    newContent,
    diff,
    diffAvailable: getBoolean(payload.diff_available) ?? Boolean(diff),
    additions: getNumber(payload.added_lines) ?? stats.additions,
    deletions: getNumber(payload.deleted_lines) ?? stats.deletions,
    undoAvailable: getBoolean(payload.undo_available) ?? false,
    undoAction: undoAction === 'restore' || undoAction === 'delete' ? undoAction : null,
  }
}

function buildFileChangeGroup(
  id: string,
  sourceTool: string,
  title: string,
  files: FileChangeSummary[],
  patchText: string | null = null
): FileChangeGroupSummary | null {
  if (files.length === 0) {
    return null
  }

  return {
    id,
    sourceTool,
    title,
    patchText,
    files,
  }
}

function parseApplyPatchBlocks(patchText: string): Array<{
  kind: 'update' | 'add' | 'delete'
  path: string
  moveTo: string | null
  lines: string[]
}> {
  const blocks: Array<{
    kind: 'update' | 'add' | 'delete'
    path: string
    moveTo: string | null
    lines: string[]
  }> = []

  let current: {
    kind: 'update' | 'add' | 'delete'
    path: string
    moveTo: string | null
    lines: string[]
  } | null = null

  const flushCurrent = () => {
    if (current) {
      blocks.push(current)
      current = null
    }
  }

  for (const line of patchText.split(/\r?\n/)) {
    if (line === '*** Begin Patch' || line === '*** End Patch') {
      continue
    }

    if (line.startsWith('*** Update File: ')) {
      flushCurrent()
      current = {
        kind: 'update',
        path: line.slice('*** Update File: '.length).trim(),
        moveTo: null,
        lines: [],
      }
      continue
    }

    if (line.startsWith('*** Add File: ')) {
      flushCurrent()
      current = {
        kind: 'add',
        path: line.slice('*** Add File: '.length).trim(),
        moveTo: null,
        lines: [],
      }
      continue
    }

    if (line.startsWith('*** Delete File: ')) {
      flushCurrent()
      current = {
        kind: 'delete',
        path: line.slice('*** Delete File: '.length).trim(),
        moveTo: null,
        lines: [],
      }
      continue
    }

    if (line.startsWith('*** Move to: ') && current?.kind === 'update') {
      current.moveTo = line.slice('*** Move to: '.length).trim()
      continue
    }

    if (current) {
      current.lines.push(line)
    }
  }

  flushCurrent()
  return blocks
}

function buildApplyPatchDiff(
  block: ReturnType<typeof parseApplyPatchBlocks>[number]
): FileChangeSummary | null {
  const targetPath = block.moveTo ?? block.path
  const bodyLines = block.lines.filter((line) => line !== '*** End of File')
  const patchBody = bodyLines.length > 0 ? bodyLines : []

  let diffLines: string[] = []
  let status = block.kind === 'add' ? 'added' : block.kind === 'delete' ? 'deleted' : 'modified'

  if (block.kind === 'add') {
    diffLines = [
      '--- /dev/null',
      `+++ b/${targetPath}`,
      `@@ -0,0 +1,${Math.max(bodyLines.length, 1)} @@`,
      ...patchBody,
    ]
  } else if (block.kind === 'delete') {
    diffLines = [
      `--- a/${block.path}`,
      '+++ /dev/null',
      ...(patchBody.length > 0 ? [`@@ -1,${patchBody.length} +0,0 @@`, ...patchBody] : []),
    ]
  } else {
    if (block.moveTo && block.moveTo !== block.path) {
      status = 'renamed'
    }
    diffLines = [
      `--- a/${block.path}`,
      `+++ b/${targetPath}`,
      ...patchBody,
    ]
  }

  const diff = diffLines.join('\n')
  const stats = summarizeDiffStats(diff)

  return {
    filePath: targetPath,
    relativePath: null,
    displayPath:
      block.moveTo && block.moveTo !== block.path ? `${block.path} -> ${block.moveTo}` : targetPath,
    previousFilePath: block.moveTo && block.moveTo !== block.path ? block.path : null,
    status,
    originalContent: null,
    newContent: null,
    diff,
    diffAvailable: true,
    additions: stats.additions,
    deletions: stats.deletions,
    undoAvailable: false,
    undoAction: null,
  }
}

function extractPatchText(value: Record<string, unknown>): string | null {
  return (
    getString(value.patch) ??
    getString(value.patch_text) ??
    getString(value.text) ??
    getString(value.edited_patch_text)
  )
}

export function normalizeFileChanges(value: unknown): FileChangeSummary[] {
  return getRecordArray(value)
    .map((item) => normalizeFileChangeRecord(item))
    .filter((item): item is FileChangeSummary => item !== null)
}

export function extractFileChangeGroupFromToolData(input: {
  id: string
  toolName?: string
  toolArgs?: Record<string, unknown>
  toolMeta?: Record<string, unknown>
  toolResult?: unknown
}): FileChangeGroupSummary | null {
  const toolName = input.toolName ?? 'file_change'
  const title =
    toolName === 'apply_patch'
      ? 'Patch review'
      : toolName === 'file_edit'
        ? 'Edited file'
        : toolName === 'file_write'
          ? 'Wrote file'
          : 'File changes'

  const metadataFiles = normalizeFileChanges(input.toolMeta?.file_changes)
  if (metadataFiles.length > 0) {
    return buildFileChangeGroup(
      input.id,
      toolName,
      title,
      metadataFiles,
      extractPatchText(input.toolArgs ?? {})
    )
  }

  const resultFiles = isRecord(input.toolResult)
    ? normalizeFileChanges(
        input.toolResult.file_changes ??
        input.toolResult.files ??
        input.toolResult.changes
      )
    : []
  if (resultFiles.length > 0) {
    return buildFileChangeGroup(
      input.id,
      toolName,
      title,
      resultFiles,
      extractPatchText(input.toolArgs ?? {})
    )
  }

  if (toolName === 'apply_patch') {
    const patchText = extractPatchText(input.toolArgs ?? {}) ?? extractPatchText(input.toolMeta ?? {})
    if (!patchText) {
      return null
    }
    const files = parseApplyPatchBlocks(patchText)
      .map((block) => buildApplyPatchDiff(block))
      .filter((item): item is FileChangeSummary => item !== null)
    return buildFileChangeGroup(input.id, toolName, title, files, patchText)
  }

  if (toolName !== 'file_write' && toolName !== 'file_edit') {
    return null
  }

  const fileChange = normalizeFileChangeRecord({
    path:
      input.toolMeta?.file_path ??
      input.toolMeta?.path ??
      input.toolArgs?.path,
    relative_path: input.toolMeta?.relative_path,
    original_content:
      input.toolMeta?.original_content ??
      input.toolArgs?.original_content,
    new_content:
      input.toolMeta?.new_content ??
      input.toolArgs?.content ??
      input.toolArgs?.new_content,
    undo_available: input.toolMeta?.undo_available,
    undo_action: input.toolMeta?.undo_action,
    diff: input.toolMeta?.diff,
    diff_available: input.toolMeta?.diff_available,
    status: input.toolMeta?.status,
  })

  if (!fileChange) {
    return null
  }

  return buildFileChangeGroup(input.id, toolName, title, [fileChange])
}

export function extractPatchPreviewResult(value: unknown): PatchPreviewResult {
  const record = isRecord(value) ? value : {}
  const previewRecord = isRecord(record.preview) ? record.preview : null
  const fileChanges = normalizeFileChanges(
    record.file_changes ??
    record.files ??
    record.preview_files ??
    record.changes ??
    previewRecord?.file_changes ??
    previewRecord?.files
  )
  const valid =
    getBoolean(record.valid) ??
    getBoolean(record.is_valid) ??
    getBoolean(record.ok) ??
    getBoolean(previewRecord?.valid) ??
    false

  return {
    valid,
    summary:
      getString(record.summary) ??
      getString(record.message) ??
      getString(record.detail) ??
      getString(previewRecord?.summary),
    errors: [
      ...getStringArray(record.errors),
      ...getStringArray(record.validation_errors),
      ...(getString(record.error) ? [getString(record.error) as string] : []),
    ],
    warnings: [
      ...getStringArray(record.warnings),
      ...getStringArray(record.notices),
    ],
    patchText: extractPatchText(record) ?? extractPatchText(previewRecord ?? {}),
    files: fileChanges,
  }
}
