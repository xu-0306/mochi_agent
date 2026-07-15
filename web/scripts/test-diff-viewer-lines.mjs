import assert from 'node:assert/strict'
import fs from 'node:fs/promises'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

const moduleUrl = pathToFileURL(
  path.join(process.cwd(), 'src/lib/diff-lines.ts')
).href
const { buildDiffDocument } = await import(moduleUrl)
const fileChangesModuleUrl = pathToFileURL(
  path.join(process.cwd(), 'src/lib/file-change-preview.ts')
).href
const { extractFileChangeGroupFromToolData } = await import(fileChangesModuleUrl)

const added = buildDiffDocument({
  diff: [
    '--- /dev/null',
    '+++ b/notes.txt',
    '@@ -0,0 +1 @@',
    '+hi',
  ].join('\n'),
  filePath: 'notes.txt',
})

assert.equal(added.filePath, 'notes.txt')
assert.deepEqual(added.stats, { additions: 1, deletions: 0 })
assert.deepEqual(added.lines, [
  {
    kind: 'added',
    oldNumber: null,
    newNumber: 1,
    marker: '+',
    content: 'hi',
  },
])

const modified = buildDiffDocument({
  diff: [
    '--- a/app.py',
    '+++ b/app.py',
    '@@ -10,3 +10,4 @@',
    ' context',
    '-before',
    '+after',
    '+extra',
    ' tail',
  ].join('\n'),
  filePath: 'app.py',
})

assert.deepEqual(
  modified.lines.map(({ kind, oldNumber, newNumber, marker, content }) => ({
    kind,
    oldNumber,
    newNumber,
    marker,
    content,
  })),
  [
    { kind: 'context', oldNumber: 10, newNumber: 10, marker: ' ', content: 'context' },
    { kind: 'removed', oldNumber: 11, newNumber: null, marker: '-', content: 'before' },
    { kind: 'added', oldNumber: null, newNumber: 11, marker: '+', content: 'after' },
    { kind: 'added', oldNumber: null, newNumber: 12, marker: '+', content: 'extra' },
    { kind: 'context', oldNumber: 12, newNumber: 13, marker: ' ', content: 'tail' },
  ]
)
assert.deepEqual(modified.stats, { additions: 2, deletions: 1 })

const repoMapProjection = extractFileChangeGroupFromToolData({
  id: 'repo-map-result',
  toolName: 'repo_map',
  toolResult: {
    root: 'workspace',
    files: [
      { path: 'projects.json', language: 'json', kind: 'data' },
      { path: 'notes.txt', language: 'text', kind: 'file' },
    ],
  },
})
assert.equal(
  repoMapProjection,
  null,
  'Read-only repo_map results must not be projected as file mutations'
)

const source = await fs.readFile(
  path.join(process.cwd(), 'src/components/chat/DiffViewer.tsx'),
  'utf8'
)
assert.doesNotMatch(
  source,
  /react-diff-viewer-continued/,
  'DiffViewer should not depend on the fixed-width third-party table renderer'
)
assert.match(source, /data-diff-kind=/, 'Diff rows should expose stable semantic kinds')
assert.match(source, /overflow-x-auto/, 'Diff code should preserve fidelity with horizontal scrolling')
assert.match(
  source,
  /grid-cols-\[3\.25rem_1\.75rem_minmax\(18rem,1fr\)\]/,
  'Diff rows should use one compact line-number gutter instead of separate old/new columns'
)
assert.doesNotMatch(
  source,
  /No old line number/,
  'Diff rows should not render a dedicated old-line-number cell'
)
assert.match(
  source,
  /line\.newNumber \?\? line\.oldNumber/,
  'The compact gutter should prefer new line numbers and fall back to old numbers for deletions'
)

console.log('ok')
