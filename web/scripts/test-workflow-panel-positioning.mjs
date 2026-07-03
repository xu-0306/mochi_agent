import assert from 'node:assert/strict'
import fs from 'node:fs/promises'
import path from 'node:path'

const source = await fs.readFile(
  path.join(process.cwd(), 'src/components/chat/WorkflowPanel.tsx'),
  'utf8'
)
const normalizedSource = source.replace(/\r\n/g, '\n')

assert.match(
  normalizedSource,
  /const workflowUiSuppressed =\s*sessionGoalRuntimeContext\.interactionMode === 'goal' &&\s*sessionGoalRuntimeContext\.executionTopology === 'single_agent'/s,
  'WorkflowPanel should keep the single-agent goal suppression guard.'
)

const branchStart = normalizedSource.indexOf('{workflowUiSuppressed ? (')
const elseMarker = normalizedSource.indexOf('\n        ) : (\n', branchStart)
const branchEnd = normalizedSource.indexOf('\n        )}\n      </div>', elseMarker)

assert.notEqual(branchStart, -1, 'WorkflowPanel should keep the suppression branch start.')
assert.notEqual(elseMarker, -1, 'WorkflowPanel should keep the suppressed/full branch boundary.')
assert.notEqual(branchEnd, -1, 'WorkflowPanel should keep the suppression branch end.')

const suppressedBranch = normalizedSource.slice(branchStart, elseMarker)
const fullControlsBranch = normalizedSource.slice(elseMarker, branchEnd)

assert.doesNotMatch(
  suppressedBranch,
  /\/workflow|slash command|workflow <request>/i,
  'Suppressed single-agent goal copy should not teach workflow command ceremony.'
)

assert.match(
  suppressedBranch,
  /title="Goal path"[\s\S]*chat-first[\s\S]*operator inspection[\s\S]*Active runtime[\s\S]*Chat-first single-agent goal[\s\S]*On-file workflow settings/s,
  'Suppressed branch should frame the panel as chat-first inspection and distinguish the active goal runtime from on-file workflow settings.'
)

assert.doesNotMatch(
  suppressedBranch,
  /Selected strategy/,
  'Suppressed branch should not present stored workflow settings as the active strategy.'
)

assert.match(
  suppressedBranch,
  /On-file workflow setting:/,
  'Suppressed branch should mark workflow protocol details as on-file workflow settings.'
)

assert.match(
  suppressedBranch,
  /Open run detail/,
  'Suppressed branch should keep deep inspection affordances when a run is already bound.'
)

assert.doesNotMatch(
  suppressedBranch,
  /Enable explicit override|Enable workflow override|New run|Switch between the ordinary goal path and the bound workflow lane\./,
  'Suppressed branch should not surface the full workflow override controls.'
)

assert.match(
  fullControlsBranch,
  /title="Explicit override"[\s\S]*description="Operator control for routing this chat into the workflow runtime when the selected strategy actually needs it\."[\s\S]*Enable explicit override/s,
  'Visible workflow controls should remain framed as operator override, not the default goal path.'
)

assert.match(
  normalizedSource,
  /Inspect the bound workflow runtime and apply explicit overrides when this strategy actually needs them\./,
  'Panel header should describe this surface as advanced inspection plus explicit override.'
)

console.log('ok')
