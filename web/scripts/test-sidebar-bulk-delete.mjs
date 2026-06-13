import assert from 'node:assert/strict'
import fs from 'node:fs/promises'
import path from 'node:path'

async function readSource(relativePath) {
  return fs.readFile(path.join(process.cwd(), relativePath), 'utf8')
}

const [sidebarSource, sessionItemSource, i18nSource] = await Promise.all([
  readSource('src/components/sidebar/Sidebar.tsx'),
  readSource('src/components/sidebar/SessionItem.tsx'),
  readSource('src/lib/i18n.tsx'),
])

assert.match(
  sidebarSource,
  /const \[selectionMode, setSelectionMode\] = React\.useState\(false\)/,
  'Sidebar should track whether bulk selection mode is active'
)

assert.match(
  sidebarSource,
  /const \[selectedSessionIds, setSelectedSessionIds\] = React\.useState<string\[\]>\(\[\]\)/,
  'Sidebar should track the selected conversations for bulk deletion'
)

assert.match(
  sidebarSource,
  /handleToggleSelectAllVisible/,
  'Sidebar should expose a select-all-visible control for bulk deletion'
)

assert.match(
  sidebarSource,
  /setPendingBulkDeleteIds\(selectedSessionIds\)/,
  'Sidebar should confirm bulk deletion against the selected conversation ids'
)

assert.match(
  sessionItemSource,
  /selectionMode\?: boolean/,
  'Session items should support a dedicated selection mode'
)

assert.match(
  sessionItemSource,
  /onClick=\{selectionMode \? onToggleSelected : onClick\}/,
  'Session items should toggle selection instead of navigating while bulk mode is active'
)

for (const key of [
  "'sidebar.bulkDelete'",
  "'sidebar.bulkCancel'",
  "'sidebar.bulkSelectAll'",
  "'sidebar.bulkClearAll'",
  "'sidebar.bulkDeleteSelected'",
  "'sidebar.bulkSelectedCount'",
  "'sidebar.bulkDeleteDialogTitle'",
  "'sidebar.bulkDeleteDialogDescription'",
]) {
  assert.match(i18nSource, new RegExp(key.replaceAll('.', '\\.')), `Missing i18n key ${key}`)
}

console.log('ok')
