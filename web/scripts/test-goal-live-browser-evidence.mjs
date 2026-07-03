import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { createServer } from 'node:http'
import { createRequire } from 'node:module'
import fs from 'node:fs'
import path from 'node:path'

const WEB_DIR = process.cwd()
const PORT = Number(process.env.MOCHI_GOAL_LIVE_FIXTURE_PORT ?? 3220)
const BASE_URL = `http://127.0.0.1:${PORT}`
const APP_URL = `${BASE_URL}/`
const require = createRequire(import.meta.url)
const SSE_CHUNK_DELAY_MS = 80
let stoppingDevServer = false

const sessionId = 'session-goal-live-evidence'
const goalId = 'goal-live-evidence'
const runId = 'run-goal-live-evidence'
const subagentId = 'goal-live-worker'
const objective = 'WS8 live browser evidence goal'

function requireLocalPlaywright() {
  const candidates = [
    path.join(WEB_DIR, 'node_modules'),
    'C:/Users/Xu/AppData/Local/npm-cache/_npx/e41f203b7505f1fb/node_modules',
  ]
  for (const candidate of candidates) {
    try {
      return require(require.resolve('playwright', { paths: [candidate] }))
    } catch {
      // Try the next known local install.
    }
  }
  throw new Error('Playwright is not available locally.')
}

function findChromiumExecutable() {
  const candidates = [
    process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE,
    'C:/Users/Xu/AppData/Local/ms-playwright/chromium-1179/chrome-win/chrome.exe',
    'C:/Program Files/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
    'C:/Program Files/Microsoft/Edge/Application/msedge.exe',
  ].filter(Boolean)
  return candidates.find((candidate) => fs.existsSync(candidate))
}

function startDevServer(directApiBaseUrl) {
  const command = process.platform === 'win32' ? 'npm.cmd' : 'npm'
  const child = spawn(command, ['run', 'dev', '--', '--hostname', '127.0.0.1', '--port', String(PORT)], {
    cwd: WEB_DIR,
    env: {
      ...process.env,
      NEXT_TELEMETRY_DISABLED: '1',
      NEXT_PUBLIC_MOCHI_API_BASE_URL: directApiBaseUrl,
    },
    shell: process.platform === 'win32',
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  child.on('exit', (code, signal) => {
    if (stoppingDevServer) {
      return
    }
    if (code !== null && code !== 0) {
      console.error(`Next dev server exited with code ${code}`)
    } else if (signal) {
      console.error(`Next dev server exited with signal ${signal}`)
    }
  })
  return child
}

async function stopDevServer(child) {
  if (!child || child.killed) {
    return
  }
  stoppingDevServer = true
  child.stdout?.destroy()
  child.stderr?.destroy()
  if (process.platform === 'win32') {
    await new Promise((resolve) => {
      const killer = spawn('taskkill.exe', ['/PID', String(child.pid), '/T', '/F'], {
        stdio: 'ignore',
      })
      killer.on('exit', resolve)
      killer.on('error', resolve)
    })
    return
  }
  child.kill()
  await new Promise((resolve) => setTimeout(resolve, 800))
  child.kill('SIGKILL')
}

async function waitForServer(url, timeoutMs = 45_000) {
  const startedAt = Date.now()
  while (Date.now() - startedAt < timeoutMs) {
    try {
      const response = await fetch(url)
      if (response.ok) {
        return
      }
    } catch {
      // Server is not ready yet.
    }
    await sleep(500)
  }
  throw new Error(`Timed out waiting for ${url}`)
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

async function waitForCondition(predicate, label, timeoutMs = 30_000) {
  const startedAt = Date.now()
  while (Date.now() - startedAt < timeoutMs) {
    if (predicate()) {
      return
    }
    await sleep(100)
  }
  throw new Error(`Timed out waiting for ${label}`)
}

function json(body) {
  return {
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify(body),
  }
}

function escapeRegex(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function buildRowKey(event) {
  return (
    event.event_id ??
    event.eventId ??
    event.dedupe_key ??
    event.dedupeKey ??
    `${event.seq ?? 'seq'}:${event.type}:${event.subagent_id ?? event.subagentId ?? 'runtime'}:${event.created_at ?? event.createdAt ?? 'unknown'}`
  )
}

function serializeSseFrame(event) {
  return `id: ${event.event_id ?? event.seq ?? 'event'}\nevent: goal_surface\ndata: ${JSON.stringify(event)}\n\n`
}

async function writeChunkedSseFrame(response, event) {
  const frame = serializeSseFrame(event)
  const chunkSize = Math.max(1, Math.ceil(frame.length / 3))
  for (let index = 0; index < frame.length; index += chunkSize) {
    if (response.destroyed || response.writableEnded) {
      return
    }
    response.write(frame.slice(index, index + chunkSize))
    await sleep(SSE_CHUNK_DELAY_MS)
  }
}

const goalSummary = {
  goal_id: goalId,
  objective,
  status: 'running',
  execution_mode: 'autonomous_single_agent',
  interaction_mode: 'goal',
  execution_topology: 'single_agent',
  strategy_id: 'autonomous_single_agent',
  selection_source: 'explicit',
  selection_reason: 'WS8 live browser fixture',
  protocol_selection: 'autonomous_single_agent',
  selection_rationale: 'WS8 live browser fixture',
  runtime_mode: 'autonomous_single_agent',
  models: [{ role: 'executor', model: 'fixture-model' }],
  current_attempt_id: 'attempt-live-1',
  attempts: [
    {
      attempt_id: 'attempt-live-1',
      status: 'running',
      agent_run_id: runId,
      created_at: '2026-07-03T03:01:00.000Z',
      updated_at: '2026-07-03T03:02:15.000Z',
    },
  ],
  created_at: '2026-07-03T03:00:00.000Z',
  updated_at: '2026-07-03T03:02:15.000Z',
}

const sessionGoalState = {
  active_goal_id: goalId,
  active_goal_status: 'running',
  execution_mode: 'autonomous_single_agent',
  interaction_mode: 'goal',
  execution_topology: 'single_agent',
  strategy_id: 'autonomous_single_agent',
  selection_source: 'explicit',
  selection_reason: 'WS8 live browser fixture',
  bound_run_id: runId,
  protocol_selection: 'autonomous_single_agent',
  selection_rationale: 'WS8 live browser fixture',
  default_route: 'goal',
  last_goal_summary: {
    ...goalSummary,
    bound_run_id: runId,
  },
  pending_proposal: null,
}

const sessionEvents = [
  {
    type: 'message',
    role: 'user',
    content: objective,
    timestamp: '2026-07-03T03:00:00.000Z',
  },
]

const transcriptSeedEvents = [
  {
    type: 'run_started',
    seq: 1,
    event_id: 'evt-run-started',
    dedupe_key: `${runId}:run_started`,
    visibility: 'visible',
    durability: 'transient',
    projection_lane: 'goal_surface',
    parent_type: 'agent_run',
    parent_id: runId,
    status: 'running',
    title: 'Goal runtime started',
    summary: 'Goal runtime accepted the autonomous task.',
    timestamp: '2026-07-03T03:01:00.000Z',
  },
  {
    type: 'subagent_started',
    seq: 2,
    event_id: 'evt-worker-started',
    dedupe_key: `${runId}:${subagentId}:started`,
    visibility: 'visible',
    durability: 'transient',
    projection_lane: 'goal_surface',
    parent_type: 'agent_run',
    parent_id: runId,
    subagent_id: subagentId,
    role_id: 'executor',
    title: 'Goal live fixture worker',
    model_id: 'fixture-model',
    status: 'running',
    summary: 'Worker started live browser evidence collection.',
    timestamp: '2026-07-03T03:01:15.000Z',
  },
  {
    type: 'subagent_thinking',
    seq: 3,
    event_id: 'evt-worker-thinking',
    dedupe_key: `${runId}:${subagentId}:thinking`,
    visibility: 'visible',
    durability: 'transient',
    projection_lane: 'goal_surface',
    parent_type: 'agent_run',
    parent_id: runId,
    subagent_id: subagentId,
    role_id: 'executor',
    title: 'Goal live fixture worker',
    model_id: 'fixture-model',
    status: 'running',
    summary: 'Thinking through the live execution evidence path.',
    timestamp: '2026-07-03T03:01:30.000Z',
  },
]

const streamedGoalSurfaceEvent = {
  type: 'subagent_progress',
  seq: 4,
  event_id: 'evt-worker-progress-live',
  dedupe_key: `${runId}:${subagentId}:progress:live`,
  visibility: 'visible',
  durability: 'transient',
  projection_lane: 'goal_surface',
  parent_type: 'agent_run',
  parent_id: runId,
  subagent_id: subagentId,
  role_id: 'executor',
  title: 'Goal live fixture worker',
  model_id: 'fixture-model',
  status: 'running',
  summary: 'Streamed execution progress reached the browser live.',
  timestamp: '2026-07-03T03:01:45.000Z',
}

const reconnectPollEvent = {
  type: 'subagent_thinking',
  seq: 5,
  event_id: 'evt-worker-thinking-reconnect',
  dedupe_key: `${runId}:${subagentId}:thinking:reconnect`,
  visibility: 'visible',
  durability: 'transient',
  projection_lane: 'goal_surface',
  parent_type: 'agent_run',
  parent_id: runId,
  subagent_id: subagentId,
  role_id: 'executor',
  title: 'Goal live fixture worker',
  model_id: 'fixture-model',
  status: 'running',
  summary: 'Polling fallback replayed the live transcript without duplicates.',
  timestamp: '2026-07-03T03:02:00.000Z',
}

const expectedTimelineRowKeys = [
  transcriptSeedEvents[0],
  transcriptSeedEvents[1],
  transcriptSeedEvents[2],
  streamedGoalSurfaceEvent,
  reconnectPollEvent,
].map((event) => buildRowKey(event))

function buildTranscriptHistory(state) {
  const events = [...transcriptSeedEvents]
  if (state.streamDelivered) {
    events.push(streamedGoalSurfaceEvent)
  }
  if (state.reconnectDelivered) {
    events.push(reconnectPollEvent)
  }
  return events
}

function buildTranscriptPayload(state, afterSeq) {
  if (afterSeq === null) {
    if (state.reconnectDelivered) {
      return [
        ...transcriptSeedEvents,
        streamedGoalSurfaceEvent,
        reconnectPollEvent,
        { ...streamedGoalSurfaceEvent },
        { ...reconnectPollEvent },
      ]
    }
    return [
      ...transcriptSeedEvents,
      { ...transcriptSeedEvents[1] },
      { ...transcriptSeedEvents[2] },
    ]
  }

  if (afterSeq >= 5) {
    return [{ ...reconnectPollEvent }]
  }

  if (afterSeq >= 4) {
    state.reconnectDelivered = true
    return [
      { ...streamedGoalSurfaceEvent },
      reconnectPollEvent,
      { ...reconnectPollEvent },
    ]
  }

  if (afterSeq >= 3) {
    return [{ ...streamedGoalSurfaceEvent }]
  }

  return buildTranscriptHistory(state)
}

function buildAgentRunDetail(state) {
  const events = buildTranscriptHistory(state)
  return {
    run_id: runId,
    status: 'running',
    protocol_id: 'autonomous_single_agent',
    objective,
    created_at: '2026-07-03T03:01:00.000Z',
    updated_at: events.at(-1)?.timestamp ?? '2026-07-03T03:01:30.000Z',
    events,
  }
}

function buildSubagentSummary(state) {
  const events = buildTranscriptHistory(state)
  return {
    subagent_id: subagentId,
    agent_run_id: runId,
    parent_type: 'agent_run',
    parent_id: runId,
    session_id: sessionId,
    role_id: 'executor',
    title: 'Goal live fixture worker',
    model_id: 'fixture-model',
    status: 'running',
    prompt_preview: 'Collect live browser evidence.',
    summary: 'Worker started live browser evidence collection.',
    output_preview: null,
    event_count: events.length,
    created_at: '2026-07-03T03:01:15.000Z',
    updated_at: events.at(-1)?.timestamp ?? '2026-07-03T03:01:30.000Z',
  }
}

async function startMockApiServer(requestLog) {
  const state = {
    reconnectDelivered: false,
    streamConnectionCount: 0,
    streamDelivered: false,
  }

  const server = createServer((request, response) => {
    void (async () => {
      const url = new URL(request.url ?? '/', 'http://127.0.0.1')
      requestLog.push(`${request.method ?? 'GET'} ${url.pathname}${url.search}`)
      response.setHeader('Access-Control-Allow-Origin', '*')
      response.setHeader('Access-Control-Allow-Methods', 'GET,OPTIONS')
      response.setHeader('Access-Control-Allow-Headers', '*')

      if (request.method === 'OPTIONS') {
        response.writeHead(204)
        response.end()
        return
      }

      if (url.pathname !== `/v1/agent-runs/${runId}/events/stream`) {
        response.writeHead(404, { 'Content-Type': 'application/json' })
        response.end(JSON.stringify({ detail: 'Not found' }))
        return
      }

      state.streamConnectionCount += 1
      const connectionNumber = state.streamConnectionCount
      response.writeHead(200, {
        'Cache-Control': 'no-cache, no-transform',
        Connection: 'keep-alive',
        'Content-Type': 'text/event-stream; charset=utf-8',
        'X-Accel-Buffering': 'no',
      })

      await sleep(150)
      if (connectionNumber === 1) {
        state.streamDelivered = true
        await writeChunkedSseFrame(response, streamedGoalSurfaceEvent)
      } else if (connectionNumber === 2) {
        await writeChunkedSseFrame(response, reconnectPollEvent)
      }

      await sleep(100)
      if (!response.destroyed && !response.writableEnded) {
        response.write('data: [DONE]\n\n')
        response.end()
      }
    })().catch((error) => {
      if (!response.destroyed) {
        response.destroy(error)
      }
    })
  })

  await new Promise((resolve, reject) => {
    server.once('error', reject)
    server.listen(0, '127.0.0.1', resolve)
  })
  const address = server.address()
  if (!address || typeof address === 'string') {
    throw new Error('Failed to bind mock SSE server.')
  }
  return {
    directApiBaseUrl: `http://127.0.0.1:${address.port}`,
    server,
    state,
  }
}

async function stopMockApiServer(server) {
  if (!server || !server.listening) {
    return
  }
  await new Promise((resolve) => server.close(resolve))
}

async function installRoutes(page, requestLog, fixtureState) {
  const apiPattern = new RegExp(`^${escapeRegex(BASE_URL)}/v1/`)
  await page.route('**/favicon.ico', (route) => route.fulfill({ status: 204, body: '' }))
  await page.route(apiPattern, (route) => {
    const request = route.request()
    const url = new URL(request.url())
    const pathname = url.pathname
    requestLog.push(`${request.method()} ${pathname}${url.search}`)

    if (pathname === '/v1/sessions') {
      return route.fulfill(json({
        items: [
          {
            session_id: sessionId,
            title: 'Goal Live Evidence Session',
            updated_at: '2026-07-03T03:02:15.000Z',
            event_count: sessionEvents.length,
            goal: sessionGoalState,
          },
        ],
      }))
    }
    if (pathname === `/v1/sessions/${sessionId}`) {
      return route.fulfill(json({
        type: 'session',
        session_id: sessionId,
        title: 'Goal Live Evidence Session',
        goal: sessionGoalState,
        workflow: {
          enabled: false,
          bound_run_id: null,
          config: {},
        },
        events: sessionEvents,
      }))
    }
    if (pathname === `/v1/sessions/${sessionId}/events`) {
      return route.fulfill(json({
        type: 'session',
        session_id: sessionId,
        title: 'Goal Live Evidence Session',
        goal: sessionGoalState,
        workflow: {
          enabled: false,
          bound_run_id: null,
          config: {},
        },
        events: sessionEvents,
      }))
    }
    if (pathname === `/v1/sessions/${sessionId}/subagents`) {
      return route.fulfill(json({ subagents: [] }))
    }
    if (pathname === `/v1/agent-runs/${runId}`) {
      return route.fulfill(json(buildAgentRunDetail(fixtureState)))
    }
    if (pathname === `/v1/agent-runs/${runId}/events`) {
      const afterSeqText = url.searchParams.get('after_seq')
      const afterSeq =
        afterSeqText === null || afterSeqText === ''
          ? null
          : Number.isFinite(Number(afterSeqText))
            ? Number(afterSeqText)
            : null
      return route.fulfill(json({
        events: buildTranscriptPayload(fixtureState, afterSeq),
      }))
    }
    if (pathname === `/v1/agent-runs/${runId}/subagents`) {
      return route.fulfill(json({ subagents: [buildSubagentSummary(fixtureState)] }))
    }
    if (pathname === '/v1/goals') {
      return route.fulfill(json([goalSummary]))
    }
    if (pathname === `/v1/goals/${goalId}`) {
      return route.fulfill(json(goalSummary))
    }
    if (pathname === `/v1/goals/${goalId}/health`) {
      return route.fulfill(json({
        goal_id: goalId,
        status: 'running',
        blocker: null,
        approval_state: {
          pending_count: 0,
          approval_ids: [],
        },
      }))
    }
    if (pathname.endsWith('/projects') || pathname.endsWith('/tasks') || pathname.endsWith('/approvals') || pathname.endsWith('/agent-runs')) {
      return route.fulfill(json([]))
    }
    if (pathname.endsWith('/settings')) {
      return route.fulfill(json({ ok: true }))
    }
    if (pathname.endsWith('/models')) {
      return route.fulfill(json({ configured: [], available: [] }))
    }
    if (pathname === '/v1/goals/strategies' || pathname.endsWith('/goal-strategies')) {
      return route.fulfill(json({
        strategies: [
          {
            strategy_id: 'autonomous_single_agent',
            label: 'Autonomous single agent',
            description: 'Fixture strategy',
          },
        ],
      }))
    }
    return route.fulfill(json({}))
  })
}

async function waitForVisibleText(page, text, label, requestLog = []) {
  await page.waitForFunction(
    (expected) =>
      Array.from(document.querySelectorAll('body *')).some((element) => {
        const rect = element.getBoundingClientRect()
        const style = window.getComputedStyle(element)
        return (
          element.textContent?.includes(expected) &&
          rect.width > 0 &&
          rect.height > 0 &&
          style.visibility !== 'hidden' &&
          style.display !== 'none'
        )
      }),
    text,
    { timeout: 30_000 }
  ).catch(async (error) => {
    const body = await page.locator('body').innerText({ timeout: 2_000 }).catch(() => '')
    throw new Error(
      `${label}: visible text not found: ${text}. requests=${JSON.stringify(requestLog)} body=${JSON.stringify(body.slice(0, 1500))}. ${error.message}`
    )
  })
}

async function readTimelineRowKeys(page) {
  return page.locator('[data-testid="execution-timeline-row"]').evaluateAll((rows) =>
    rows.map((row) => row.getAttribute('data-row-key') ?? '')
  )
}

async function assertTimelineRows(page, expectedKeys, label) {
  const actualKeys = await readTimelineRowKeys(page)
  assert.deepEqual(
    actualKeys,
    expectedKeys,
    `${label}: expected rendered timeline row keys ${JSON.stringify(expectedKeys)}, saw ${JSON.stringify(actualKeys)}`
  )
  assert.equal(
    new Set(actualKeys).size,
    actualKeys.length,
    `${label}: duplicate timeline rows should not render`
  )
}

async function assertNoEmptyAssistantCards(page) {
  const emptyAssistantCards = await page.evaluate(() => {
    return Array.from(document.querySelectorAll('.prose.prose-invert'))
      .filter((element) => {
        const rect = element.getBoundingClientRect()
        const style = window.getComputedStyle(element)
        return (
          rect.width > 0 &&
          rect.height > 0 &&
          style.visibility !== 'hidden' &&
          style.display !== 'none' &&
          (element.textContent ?? '').trim().length === 0
        )
      })
      .length
  })
  assert.equal(emptyAssistantCards, 0, 'no visible empty generic assistant markdown cards should render')
}

async function assertNoConsoleNoise(page, badMessages) {
  assert.deepEqual(badMessages, [], 'console warnings/errors')
  const overflow = await page.evaluate(() => ({
    documentWidth: document.documentElement.scrollWidth,
    viewportWidth: document.documentElement.clientWidth,
  }))
  assert.ok(
    overflow.documentWidth <= overflow.viewportWidth + 2,
    `document should not overflow horizontally: ${JSON.stringify(overflow)}`
  )
}

async function main() {
  const requestLog = []
  const mockApi = await startMockApiServer(requestLog)
  const server = startDevServer(mockApi.directApiBaseUrl)
  try {
    await waitForServer(APP_URL)
    const { chromium } = requireLocalPlaywright()
    const browser = await chromium.launch({
      headless: true,
      executablePath: findChromiumExecutable(),
    })
    try {
      const page = await browser.newPage({ viewport: { width: 1280, height: 900 } })
      const badMessages = []
      page.on('console', (message) => {
        if (message.type() === 'warning' || message.type() === 'error') {
          badMessages.push(`${message.type()}: ${message.text()}`)
        }
      })
      page.on('pageerror', (error) => badMessages.push(`pageerror: ${error.message}`))
      await installRoutes(page, requestLog, mockApi.state)

      await page.goto(APP_URL, { waitUntil: 'domcontentloaded' })
      await waitForVisibleText(page, objective, 'initial goal title', requestLog)
      await waitForVisibleText(page, 'Execution highlights', 'initial execution lane', requestLog)
      await waitForVisibleText(page, 'Goal runtime accepted the autonomous task.', 'initial run progress', requestLog)
      await waitForVisibleText(page, 'Thinking through the live execution evidence path.', 'initial thinking progress', requestLog)
      await waitForVisibleText(page, 'Goal live fixture worker', 'initial subagent progress', requestLog)
      await waitForVisibleText(page, 'Streamed execution progress reached the browser live.', 'live streamed progress row', requestLog)
      await waitForVisibleText(page, 'Polling fallback replayed the live transcript without duplicates.', 'poll fallback progress row', requestLog)
      await assertNoEmptyAssistantCards(page)
      await assertTimelineRows(page, expectedTimelineRowKeys, 'after initial stream and poll recovery')

      await page.reload({ waitUntil: 'domcontentloaded' })
      await waitForVisibleText(page, objective, 'reload goal title', requestLog)
      await waitForVisibleText(page, 'Execution highlights', 'reload execution lane', requestLog)
      await waitForVisibleText(page, 'Goal runtime accepted the autonomous task.', 'reload run progress', requestLog)
      await waitForVisibleText(page, 'Streamed execution progress reached the browser live.', 'reload streamed progress row', requestLog)
      await waitForVisibleText(page, 'Polling fallback replayed the live transcript without duplicates.', 'reload poll replay row', requestLog)
      await waitForCondition(
        () => requestLog.includes(`GET /v1/agent-runs/${runId}/events/stream?after_seq=5`),
        'reload SSE replay request'
      )
      await sleep(400)
      await assertNoEmptyAssistantCards(page)
      await assertTimelineRows(page, expectedTimelineRowKeys, 'after reload reconstruction and stream replay')

      assert.ok(
        requestLog.includes(`GET /v1/agent-runs/${runId}/events/stream?after_seq=3`),
        'browser should open the live SSE stream after initial transcript hydration'
      )
      assert.ok(
        requestLog.includes(`GET /v1/agent-runs/${runId}/events?after_seq=4`),
        'browser should poll transcript recovery after the first stream closes'
      )
      assert.ok(
        requestLog.filter((item) => item === `GET /v1/agent-runs/${runId}/events`).length >= 2,
        'reload should reconstruct from transcript history again'
      )
      assert.ok(
        requestLog.filter((item) => item === `GET /v1/sessions/${sessionId}`).length >= 2,
        'reload should refetch the session detail'
      )
      await assertNoConsoleNoise(page, badMessages)
      await page.close()

      console.log(JSON.stringify({
        ok: true,
        checks: {
          visibleExecutionHighlights: true,
          visibleStreamedProgress: true,
          visiblePollReplayProgress: true,
          noEmptyAssistantCards: true,
          reloadAndReconnectDedupeStable: true,
        },
        counts: {
          expectedTimelineRows: expectedTimelineRowKeys.length,
          streamConnections: mockApi.state.streamConnectionCount,
          requestLogEntries: requestLog.length,
        },
        requests: requestLog,
      }, null, 2))
    } finally {
      await browser.close()
    }
  } finally {
    await stopDevServer(server)
    await stopMockApiServer(mockApi.server)
  }
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error)
    process.exit(1)
  })
