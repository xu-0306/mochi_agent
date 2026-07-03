import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { createRequire } from 'node:module'
import fs from 'node:fs'
import path from 'node:path'

const WEB_DIR = process.cwd()
const PORT = Number(process.env.MOCHI_GOAL_LIVE_FIXTURE_PORT ?? 3220)
const BASE_URL = `http://127.0.0.1:${PORT}`
const APP_URL = `${BASE_URL}/`
const require = createRequire(import.meta.url)
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

function startDevServer() {
  const command = process.platform === 'win32' ? 'npm.cmd' : 'npm'
  const child = spawn(command, ['run', 'dev', '--', '--hostname', '127.0.0.1', '--port', String(PORT)], {
    cwd: WEB_DIR,
    env: {
      ...process.env,
      NEXT_TELEMETRY_DISABLED: '1',
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
    await new Promise((resolve) => setTimeout(resolve, 500))
  }
  throw new Error(`Timed out waiting for ${url}`)
}

function json(body) {
  return {
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify(body),
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
      updated_at: '2026-07-03T03:02:00.000Z',
    },
  ],
  created_at: '2026-07-03T03:00:00.000Z',
  updated_at: '2026-07-03T03:02:00.000Z',
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

const transcriptEvents = [
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

const duplicatedTranscriptPayload = {
  events: [
    ...transcriptEvents,
    { ...transcriptEvents[1], seq: 22 },
    { ...transcriptEvents[2], seq: 23 },
  ],
}

const subagentSummary = {
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
  event_count: 3,
  created_at: '2026-07-03T03:01:15.000Z',
  updated_at: '2026-07-03T03:01:30.000Z',
}

const agentRunDetail = {
  run_id: runId,
  status: 'running',
  protocol_id: 'autonomous_single_agent',
  objective,
  created_at: '2026-07-03T03:01:00.000Z',
  updated_at: '2026-07-03T03:01:30.000Z',
  events: transcriptEvents,
}

async function installRoutes(page, requestLog) {
  await page.route('**/favicon.ico', (route) => route.fulfill({ status: 204, body: '' }))
  await page.route('**/v1/**', (route) => {
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
            updated_at: '2026-07-03T03:02:00.000Z',
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
      return route.fulfill(json(agentRunDetail))
    }
    if (pathname === `/v1/agent-runs/${runId}/events`) {
      return route.fulfill(json(duplicatedTranscriptPayload))
    }
    if (pathname === `/v1/agent-runs/${runId}/events/stream`) {
      return route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body: '',
      })
    }
    if (pathname === `/v1/agent-runs/${runId}/subagents`) {
      return route.fulfill(json({ subagents: [subagentSummary] }))
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

async function visibleTextCount(page, text) {
  return page.evaluate((expected) => {
    return Array.from(document.querySelectorAll('body *')).filter((element) => {
      const rect = element.getBoundingClientRect()
      const style = window.getComputedStyle(element)
      const visible = (
        element.textContent?.includes(expected) &&
        rect.width > 0 &&
        rect.height > 0 &&
        style.visibility !== 'hidden' &&
        style.display !== 'none'
      )
      if (!visible) {
        return false
      }
      return !Array.from(element.children).some((child) => child.textContent?.includes(expected))
    }).length
  }, text)
}

async function visibleTextContainerCount(page, text) {
  return page.evaluate((expected) => {
    return Array.from(document.querySelectorAll('body *')).filter((element) => {
      const rect = element.getBoundingClientRect()
      const style = window.getComputedStyle(element)
      return (
        element.textContent?.includes(expected) &&
        rect.width > 0 &&
        rect.height > 0 &&
        style.visibility !== 'hidden' &&
        style.display !== 'none'
      )
    }).length
  }, text)
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
  const server = startDevServer()
  try {
    await waitForServer(APP_URL)
    const { chromium } = requireLocalPlaywright()
    const browser = await chromium.launch({
      headless: true,
      executablePath: findChromiumExecutable(),
    })
    try {
      const page = await browser.newPage({ viewport: { width: 1280, height: 900 } })
      const requestLog = []
      const badMessages = []
      page.on('console', (message) => {
        if (message.type() === 'warning' || message.type() === 'error') {
          badMessages.push(`${message.type()}: ${message.text()}`)
        }
      })
      page.on('pageerror', (error) => badMessages.push(`pageerror: ${error.message}`))
      await installRoutes(page, requestLog)

      await page.goto(APP_URL, { waitUntil: 'domcontentloaded' })
      await waitForVisibleText(page, objective, 'initial goal title', requestLog)
      await waitForVisibleText(page, 'Execution highlights', 'initial execution lane', requestLog)
      await waitForVisibleText(page, 'Goal runtime accepted the autonomous task.', 'initial run progress', requestLog)
      await waitForVisibleText(page, 'Thinking through the live execution evidence path.', 'initial thinking progress', requestLog)
      await waitForVisibleText(page, 'Goal live fixture worker', 'initial subagent progress', requestLog)
      await assertNoEmptyAssistantCards(page)

      const initialThinkingRows = await visibleTextCount(page, 'Thinking through the live execution evidence path.')
      const initialWorkerCards = await visibleTextCount(page, 'Goal live fixture worker')
      const initialThinkingContainers = await visibleTextContainerCount(page, 'Thinking through the live execution evidence path.')

      await page.reload({ waitUntil: 'domcontentloaded' })
      await waitForVisibleText(page, objective, 'reload goal title', requestLog)
      await waitForVisibleText(page, 'Execution highlights', 'reload execution lane', requestLog)
      await waitForVisibleText(page, 'Goal runtime accepted the autonomous task.', 'reload run progress', requestLog)
      await waitForVisibleText(page, 'Thinking through the live execution evidence path.', 'reload thinking progress', requestLog)
      await assertNoEmptyAssistantCards(page)

      const reloadedThinkingRows = await visibleTextCount(page, 'Thinking through the live execution evidence path.')
      const reloadedWorkerCards = await visibleTextCount(page, 'Goal live fixture worker')
      const reloadedThinkingContainers = await visibleTextContainerCount(page, 'Thinking through the live execution evidence path.')

      assert.equal(
        reloadedThinkingRows,
        initialThinkingRows,
        'reload should reconstruct thinking rows without duplicates'
      )
      assert.equal(
        reloadedWorkerCards,
        initialWorkerCards,
        'reload should reconstruct subagent cards without duplicates'
      )
      assert.ok(
        initialThinkingContainers > initialThinkingRows,
        'fixture should verify row text inside rendered containers, not only a detached string'
      )
      assert.ok(
        requestLog.some((item) => item.startsWith(`GET /v1/agent-runs/${runId}/events`)),
        'browser should request run transcript events'
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
          visibleThinkingProgress: true,
          noEmptyAssistantCards: true,
          reloadDedupeStable: true,
        },
        counts: {
          initialThinkingRows,
          reloadedThinkingRows,
          initialWorkerCards,
          reloadedWorkerCards,
          initialThinkingContainers,
          reloadedThinkingContainers,
        },
      }, null, 2))
    } finally {
      await browser.close()
    }
  } finally {
    await stopDevServer(server)
  }
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error)
    process.exit(1)
  })
