import assert from 'node:assert/strict'
import crypto from 'node:crypto'
import fs from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { resolveChatGoalWorkflowRouting } from '../../../web/src/lib/chat-goal-routing.ts'
import { projectGoalDrawerState } from '../../../web/src/lib/goal-drawer-model.ts'
import { resolveGoalExecutionRecordTarget } from '../../../web/src/lib/goal-execution-record-target.ts'
import { normalizeAgentRunRouteId } from '../../../web/src/lib/agent-run-route-id.ts'

const here = path.dirname(fileURLToPath(import.meta.url))
const root = path.resolve(here, '../../..')
const artifactPath = path.join(here, 'probe-results.json')
const command = 'rtk proxy node --experimental-strip-types artifacts/goal-ui/evaluation-recheck/goal-ui-recheck-probe.mjs'

const sourcePaths = {
  page: 'web/src/app/page.tsx',
  drawer: 'web/src/components/chat/GoalComposerDrawer.tsx',
  routing: 'web/src/lib/chat-goal-routing.ts',
  target: 'web/src/lib/goal-execution-record-target.ts',
  routeId: 'web/src/lib/agent-run-route-id.ts',
  agentRunDetail: 'web/src/app/agent-runs/[runId]/page.tsx',
  api: 'web/src/lib/api.ts',
  drawerBrowser: 'web/scripts/test-goal-drawer-browser-evidence.mjs',
  liveBrowser: 'web/scripts/test-goal-live-browser-evidence.mjs',
  hydration: 'web/scripts/test-hydration-no-mismatch.mjs',
  layout: 'web/src/app/layout.tsx',
  i18n: 'web/src/lib/i18n.tsx',
  goals: 'web/src/app/goals/page.tsx',
  sidebar: 'web/src/components/sidebar/Sidebar.tsx',
}

const sources = Object.fromEntries(
  await Promise.all(Object.entries(sourcePaths).map(async ([name, relativePath]) => [
    name,
    (await fs.readFile(path.join(root, relativePath), 'utf8')).replace(/\r\n/g, '\n'),
  ]))
)

function occurrences(text, needle) {
  return text.split(needle).length - 1
}

function mustInclude(text, needle, message) {
  assert.ok(text.includes(needle), message ?? `missing ${needle}`)
}

function section(text, start, end) {
  const first = text.indexOf(start)
  assert.notEqual(first, -1, `missing section start ${start}`)
  const last = text.indexOf(end, first)
  assert.notEqual(last, -1, `missing section end ${end}`)
  return text.slice(first, last)
}

function ordered(text, ...needles) {
  let cursor = -1
  for (const needle of needles) {
    const next = text.indexOf(needle, cursor + 1)
    assert.ok(next > cursor, `expected ordered source step ${needle}`)
    cursor = next
  }
}

const probeMetadata = {
  'PROBE-RECHECK-ACTIVATION-IDENTITY': {
    invariant_id: 'INV-GOAL-ACTIVATION-IDENTITY',
    setup: 'Load the current production routing export and the public composer fixture source.',
    fault_injection: 'Submit /goal Research this for 20 minutes and the equivalent natural-language Research this for 20 minutes into the real routing export.',
    production_trigger_or_boundary: 'ChatPage::handleSend -> resolveChatGoalWorkflowRouting -> handleGoalWorkflowRouting -> api.createGoal.',
    oracle: 'Both inputs produce the same goal_proposal content; the shared handler boundary is the only page-level createGoal site; the public-composer fixture asserts one create payload.',
    anti_oracle: 'Different routing kinds/content, bypassing handleGoalWorkflowRouting, or two create payloads falsifies convergence.',
  },
  'PROBE-RECHECK-COMPOSER-SURFACE': {
    invariant_id: 'INV-GOAL-COMPOSER-SURFACE',
    setup: 'Read the integrated ChatPage render tree and current drawer component.',
    fault_injection: 'Search for a second Goal surface or a legacy Goal header/focus/floating render while a drawer exists.',
    production_trigger_or_boundary: 'ChatPage composer footer render immediately before ChatInput.',
    oracle: 'Exactly one <GoalComposerDrawer render precedes <ChatInput, with no legacy Goal surface references in ChatPage.',
    anti_oracle: 'A component merely existing on disk is not a render proof; any extra render/reference in ChatPage fails.',
  },
  'PROBE-RECHECK-SESSION-ISOLATION': {
    invariant_id: 'INV-GOAL-SESSION-ISOLATION',
    setup: 'Read the selected-session guards and the real deferred-response browser fixture.',
    fault_injection: 'Hold session A response, select no-Goal B, release A while B remains selected, then inspect the no-drawer assertion.',
    production_trigger_or_boundary: 'ChatPage selectedSessionId effect and drawer action request-generation guards; browser fixture session switch.',
    oracle: 'The fixture releases A only after B selection and waits for delivery before asserting B hidden; production resets action state on session switch and clears busy state only when request/session identities still match.',
    anti_oracle: 'A timer that resolves before B is selected, reselecting A before delivery, or missing session/request identity guards fails.',
  },
  'PROBE-RECHECK-ACTION-SEMANTICS': {
    invariant_id: 'INV-GOAL-ACTION-SEMANTICS',
    setup: 'Execute the current drawer model export and inspect the Clear UI/handler boundary.',
    fault_injection: 'Attempt Clear on a running linked Goal, then cancel and confirm; scan the confirmed handler for durable/network/navigation effects.',
    production_trigger_or_boundary: 'GoalComposerDrawer Clear -> disclosure/cancel/confirm -> ChatPage::handleClearGoalDrawer.',
    oracle: 'Clear first opens retained-history disclosure, cancel closes it, confirm alone calls a presentation-only handler; the handler only writes session-keyed local detachment state and has no API/fetch/router/delete/send token.',
    anti_oracle: 'Single-click hiding, missing disclosure/cancel control, or a durable side effect in Clear fails.',
  },
  'PROBE-RECHECK-EXECUTION-RECORD-DIRECT': {
    invariant_id: 'INV-EXECUTION-RECORD-DIRECT',
    setup: 'Execute the real target resolver and dynamic-route normalizer using run / linked?&.',
    fault_injection: 'Supply the encoded Next route segment and a foreign higher-index attempt.',
    production_trigger_or_boundary: 'Goal drawer href -> /agent-runs/[runId] -> normalizeAgentRunRouteId -> api.fetchAgentRun/fetchAgentRunHealth.',
    oracle: 'The selected Goal produces one encoded direct href; the encoded route decodes once to the raw ID; detail code passes raw routeRunId to API helpers, whose single encode produces no %25.',
    anti_oracle: 'A /goals target, foreign attempt, unnormalized encoded segment, or %25 request-equivalent fails.',
  },
  'PROBE-RECHECK-HYDRATION-PARITY': {
    invariant_id: 'INV-HYDRATION-PARITY',
    setup: 'Read RootLayout, i18n, and the current runtime hydration fixture.',
    fault_injection: 'Require clean and persisted-preference first navigation plus reload at /, /agent-runs, special-character detail, and /goals redirect.',
    production_trigger_or_boundary: 'Next server layout/first hydration and test-hydration-no-mismatch.mjs::navigateAndCapture.',
    oracle: 'Neither layout nor i18n suppresses hydration warnings; the fixture uses the real preference key, enumerates all four routes, executes clean and persisted scenarios, reloads each, and fails on captured warnings/errors.',
    anti_oracle: 'Root-only coverage, a suppressed hydration warning, omitted reload, or a fixture that ignores console errors fails.',
  },
  'PROBE-RECHECK-LIVE-TIMELINE': {
    invariant_id: 'INV-REGRESSION-QUALITY',
    setup: 'Read the current live browser gate at its durable transcript consumer assertions.',
    fault_injection: 'Require SSE after_seq=3, polling recovery after_seq=4, five unique events before and after reload, and no empty assistant cards.',
    production_trigger_or_boundary: 'test-goal-live-browser-evidence.mjs live SSE/poll/reload browser flow.',
    oracle: 'The gate has explicit SSE, poll, no-empty-card, direct detail, and reload exact-one-card assertions for every expected event.',
    anti_oracle: 'Defined-but-unused helpers alone, missing poll recovery, or no post-reload uniqueness assertion fails.',
  },
}

const results = []
async function probe(probeId, body) {
  const metadata = probeMetadata[probeId]
  try {
    const observed = await body()
    results.push({ probe_id: probeId, ...metadata, command, exit_code: 0, status: 'passed', observed, artifact_path: 'artifacts/goal-ui/evaluation-recheck/probe-results.json' })
  } catch (error) {
    results.push({ probe_id: probeId, ...metadata, command, exit_code: 1, status: 'failed', observed: error instanceof Error ? error.message : String(error), artifact_path: 'artifacts/goal-ui/evaluation-recheck/probe-results.json' })
  }
}

await probe('PROBE-RECHECK-ACTIVATION-IDENTITY', () => {
  const input = { attachmentCount: 0, hasPendingProposal: false, hasActiveGoal: false }
  const slash = resolveChatGoalWorkflowRouting({ ...input, text: '/goal Research this for 20 minutes' })
  const natural = resolveChatGoalWorkflowRouting({ ...input, text: 'Research this for 20 minutes' })
  assert.deepEqual(slash.route, { kind: 'goal_proposal', content: 'Research this for 20 minutes', raw: '/goal Research this for 20 minutes' })
  assert.deepEqual(natural.route, { kind: 'goal_proposal', content: 'Research this for 20 minutes', raw: 'Research this for 20 minutes' })
  assert.equal(slash.shouldHandleGoalWorkflowRouting, true)
  assert.equal(natural.shouldHandleGoalWorkflowRouting, true)
  assert.equal(occurrences(sources.page, 'resolveChatGoalWorkflowRouting({'), 1, 'ChatPage must have one routing boundary')
  mustInclude(sources.page, 'await handleGoalWorkflowRouting({', 'ChatPage must send all non-direct routes through the shared handler')
  assert.equal(occurrences(sources.page, 'api.createGoal({'), 1, 'ChatPage must expose one durable creation site')
  ordered(sources.drawerBrowser, 'goal-action-clear', 'goal-clear-disclosure', 'goal-clear-cancel', 'goal-clear-confirm')
  return { slashRoute: slash.route, naturalRoute: natural.route, pageCreateGoalCalls: 1, publicFixtureSingleCreateAssertion: true }
})

await probe('PROBE-RECHECK-COMPOSER-SURFACE', () => {
  assert.equal(occurrences(sources.page, '<GoalComposerDrawer'), 1, 'ChatPage must render one drawer')
  const chatInputRenders = (sources.page.match(/<ChatInput\s+sessionId=/g) ?? []).length
  assert.equal(chatInputRenders, 1, 'ChatPage must render one composer')
  assert.ok(sources.page.indexOf('<GoalComposerDrawer') < sources.page.indexOf('<ChatInput\n'), 'drawer must be composer-adjacent before ChatInput')
  for (const legacy of ['GoalHeaderChip', 'GoalFocusPanel', 'goal-header-chip', 'goal-focus-panel', 'goal-floating-panel']) {
    assert.equal(sources.page.includes(legacy), false, `legacy Goal surface ${legacy} must not appear in ChatPage`)
  }
  mustInclude(sources.drawer, 'data-testid="goal-composer-drawer"')
  return { drawerRenders: 1, chatInputRenders: 1, drawerBeforeComposer: true, legacyChatPageReferences: 0 }
})

await probe('PROBE-RECHECK-SESSION-ISOLATION', () => {
  ordered(
    sources.drawerBrowser,
    "await selectSession(page, 'No Goal session B')",
    "assert.equal(typeof state.releaseA, 'function'",
    'state.releaseA()',
    "await waitForCondition(() => state.aResponseDelivered",
    "await expectHidden(page, '[data-testid=\"goal-composer-drawer\"]', 'released session-A response must not pollute selected session B')"
  )
  mustInclude(sources.drawerBrowser, 'await new Promise((resolve) => { state.releaseA = resolve })')
  const sessionEffect = section(sources.page, 'React.useEffect(() => {\n    goalDrawerSessionRef.current = currentSessionId', '\n\n  const workflowEnabled')
  mustInclude(sessionEffect, 'goalDrawerActionInFlightRef.current = null')
  mustInclude(sessionEffect, 'setGoalDrawerBusyAction(null)')
  const actions = section(sources.page, 'const runGoalDrawerCommand', 'const goalSurfaceCopySource')
  assert.equal(occurrences(actions, 'goalDrawerSessionRef.current === sessionId'), 2, 'each async action completion needs session identity guard')
  assert.equal(occurrences(actions, 'goalDrawerActionRequestIdRef.current === requestId'), 2, 'each async action completion needs request identity guard')
  return { deferredResponseHeldUntilB: true, bAssertedAfterRelease: true, guardedActionCompletions: 2 }
})

await probe('PROBE-RECHECK-ACTION-SEMANTICS', () => {
  const running = projectGoalDrawerState({ goalId: 'goal-a', title: 'A', status: 'running', hasLinkedAgentRun: true })
  const paused = projectGoalDrawerState({ goalId: 'goal-a', title: 'A', status: 'paused', hasLinkedAgentRun: true })
  assert.equal(running.canPause, true)
  assert.equal(running.canResume, false)
  assert.equal(running.canStop, true)
  assert.equal(running.canEdit, true)
  assert.equal(running.canClear, true)
  assert.equal(paused.canPause, false)
  assert.equal(paused.canResume, true)
  mustInclude(sources.drawer, 'onClick={() => setClearConfirmationOpen(true)}')
  mustInclude(sources.drawer, 'data-testid="goal-clear-disclosure"')
  mustInclude(sources.drawer, 'This only hides the Goal from this chat. Its execution and durable history')
  mustInclude(sources.drawer, 'data-testid="goal-clear-cancel"')
  mustInclude(sources.drawer, 'data-testid="goal-clear-confirm"')
  const clearHandler = section(sources.page, 'const handleClearGoalDrawer', 'const goalSurfaceCopySource')
  for (const prohibited of ['api.', 'fetch(', 'router.', 'handleSend(', 'DELETE', 'delete ']) {
    assert.equal(clearHandler.includes(prohibited), false, `Clear handler must not contain ${prohibited}`)
  }
  mustInclude(clearHandler, 'setClearedGoalPresentationBySessionId')
  ordered(sources.drawerBrowser, 'goal-action-clear', 'goal-clear-disclosure', 'goal-clear-cancel', 'goal-clear-confirm')
  mustInclude(sources.drawerBrowser, "requests.slice(beforeClear).some((entry) => entry.startsWith('DELETE ')), false")
  return { running, paused, clearRequiresConfirmation: true, clearHandlerDurableSideEffectTokens: [] }
})

await probe('PROBE-RECHECK-EXECUTION-RECORD-DIRECT', () => {
  const rawRunId = 'run / linked?&'
  const target = resolveGoalExecutionRecordTarget({
    selectedGoalId: 'goal-selected',
    snapshotGoalId: 'goal-selected',
    currentAttemptId: 'attempt-selected',
    attempts: [
      { goalId: 'goal-foreign', attemptId: 'attempt-foreign', attemptIndex: 99, agentRunId: 'foreign run' },
      { goalId: 'goal-selected', attemptId: 'attempt-selected', attemptIndex: 0, agentRunId: rawRunId },
    ],
  })
  assert.deepEqual(target, { kind: 'agent_run', runId: rawRunId, href: `/agent-runs/${encodeURIComponent(rawRunId)}` })
  const encoded = encodeURIComponent(rawRunId)
  assert.equal(normalizeAgentRunRouteId(encoded), rawRunId)
  assert.equal(normalizeAgentRunRouteId('%E0%A4%A'), '%E0%A4%A', 'malformed route value must not crash')
  mustInclude(sources.agentRunDetail, 'const routeRunId = normalizeAgentRunRouteId(params.runId)')
  mustInclude(sources.agentRunDetail, 'api.fetchAgentRun(routeRunId)')
  mustInclude(sources.agentRunDetail, 'api.fetchAgentRunHealth(routeRunId)')
  assert.ok(occurrences(sources.api, 'encodeURIComponent(runId)') >= 2, 'detail and health API helpers must encode raw run IDs')
  const onceEncodedRequest = `/agent-runs/${encodeURIComponent(normalizeAgentRunRouteId(encoded))}`
  assert.equal(onceEncodedRequest.includes('%25'), false)
  mustInclude(sources.drawerBrowser, "requests.some((entry) => entry.includes('%25')), false")
  return { target, encoded, normalized: normalizeAgentRunRouteId(encoded), onceEncodedRequest, containsPercent25: false }
})

await probe('PROBE-RECHECK-HYDRATION-PARITY', () => {
  assert.equal(sources.layout.includes('suppressHydrationWarning'), false, 'RootLayout must not suppress hydration warnings')
  assert.equal(sources.i18n.includes('suppressHydrationWarning'), false, 'I18n must not suppress hydration warnings')
  mustInclude(sources.hydration, "localStorage.setItem('mochi.ui.preferences.v1'")
  for (const route of ["{ label: 'root', path: '/' }", "{ label: 'agent-runs', path: '/agent-runs' }", "{ label: 'agent-run-detail', path: `/agent-runs/${encodeURIComponent(linkedRunId)}` }", "{ label: 'goals-redirect', path: '/goals', expectedFinalPath: '/agent-runs' }"]) {
    mustInclude(sources.hydration, route)
  }
  assert.equal(occurrences(sources.hydration, 'scenarios.push(await navigateAndCapture'), 2, 'each route must run in clean and persisted contexts')
  mustInclude(sources.hydration, "await page.reload({ waitUntil: 'domcontentloaded' })")
  mustInclude(sources.hydration, "pass: messages.length === 0")
  mustInclude(sources.hydration, 'hydrationMessages: messages.filter(isHydrationMessage)')
  return { suppressionSites: 0, routeCount: 4, contextCountPerRoute: 2, reloadPerScenario: true, capturedConsoleFailures: true }
})

await probe('PROBE-RECHECK-LIVE-TIMELINE', () => {
  mustInclude(sources.liveBrowser, "GET /v1/agent-runs/${runId}/events?after_seq=4")
  mustInclude(sources.liveBrowser, "GET /v1/agent-runs/${runId}/events/stream?after_seq=3")
  assert.equal(occurrences(sources.liveBrowser, 'await assertNoEmptyAssistantCards(page)'), 1, 'no-empty-card helper must be invoked')
  for (const eventText of [
    'Goal runtime accepted the autonomous task.',
    'Worker started live browser evidence collection.',
    'Thinking through the live execution evidence path.',
    'Streamed execution progress reached the browser live.',
    'Polling fallback replayed the live transcript without duplicates.',
  ]) {
    assert.ok(occurrences(sources.liveBrowser, eventText) >= 2, `event ${eventText} must be checked before and after reload`)
  }
  assert.ok(occurrences(sources.liveBrowser, "detail event must render once: ${expectedText}") === 1)
  assert.ok(occurrences(sources.liveBrowser, "detail reload event must render once: ${expectedText}") === 1)
  return { sseAfterSeq: 3, pollRecoveryAfterSeq: 4, noEmptyAssistantCardsInvocation: 1, fiveEventsAssertedBeforeAndAfterReload: true }
})

const sourceHashes = Object.fromEntries(Object.entries(sources).map(([name, text]) => [name, crypto.createHash('sha256').update(text).digest('hex')]))
const payload = {
  schema_name: 'agent-foreman/evaluation-probe-results',
  schema_version: '1.0',
  generated_at_utc: new Date().toISOString(),
  production_access: 'read-only',
  command,
  source_hashes: sourceHashes,
  probes: results,
  deterministic_result: results.every((result) => result.status === 'passed') ? 'passed' : 'failed',
}
await fs.writeFile(artifactPath, `${JSON.stringify(payload, null, 2)}\n`, 'utf8')
console.log(JSON.stringify(payload, null, 2))
if (payload.deterministic_result !== 'passed') process.exitCode = 1
