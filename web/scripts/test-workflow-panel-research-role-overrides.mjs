import assert from 'node:assert/strict'
import fs from 'node:fs/promises'
import path from 'node:path'

const source = await fs.readFile(
  path.join(process.cwd(), 'src/components/chat/WorkflowPanel.tsx'),
  'utf8'
)

assert.match(
  source,
  /title="Research team"[\s\S]*Lead model[\s\S]*Research worker model[\s\S]*Lead responsibilities[\s\S]*Worker responsibilities/,
  'Research workflow UI should expose clear lead and worker team defaults.'
)

assert.match(
  source,
  /title=\{workflowTemplate === 'research_debate' \? 'Role overrides' : 'Agent roles'\}/,
  'Research workflow should rename raw agent roles into role overrides.'
)

assert.match(
  source,
  /SelectItem value=\{CUSTOM_ROLE_VALUE\}>Custom role<\/SelectItem>/,
  'Role editor should expose a dropdown with a custom role fallback.'
)

assert.match(
  source,
  /getRoleDefaultModelLabel\(selectedRoleOption\.defaultModel\)/,
  'Known roles should surface their recommended default model guidance.'
)

console.log('ok')
