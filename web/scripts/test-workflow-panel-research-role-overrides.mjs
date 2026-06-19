import assert from 'node:assert/strict'
import fs from 'node:fs/promises'
import path from 'node:path'

const source = await fs.readFile(
  path.join(process.cwd(), 'src/components/chat/WorkflowPanel.tsx'),
  'utf8'
)

assert.match(
  source,
  /title="Research team"[\s\S]*Smart model[\s\S]*Research worker model[\s\S]*Research-only roles[\s\S]*Execution lane/,
  'Research workflow UI should expose clear Smart-model defaults plus research-only versus execution-lane guidance.'
)

assert.match(
  source,
  /title=\{workflowTemplate === 'research_debate' \? 'Role overrides' : 'Agent roles'\}/,
  'Research workflow should rename raw agent roles into role overrides.'
)

assert.match(
  source,
  /Role preset[\s\S]*Choose role preset[\s\S]*Custom role/,
  'Role editor should expose preset-driven dropdown choices with a custom role fallback.'
)

assert.match(
  source,
  /getRoleDefaultModelLabel\(selectedRoleOption\.defaultModel\)[\s\S]*getRoleCapabilityDescription\(selectedRoleOption\.value\)/,
  'Known roles should surface both default-model guidance and capability descriptions.'
)

assert.match(
  source,
  /Shared default for planner, judge, verifier, and synthesizer\./,
  'Smart model copy should explicitly call out the shared planner/judge/verifier/synthesizer default.'
)

console.log('ok')
