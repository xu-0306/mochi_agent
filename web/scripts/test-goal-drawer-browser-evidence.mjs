import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { createRequire } from 'node:module'
import fs from 'node:fs'
import path from 'node:path'
import net from 'node:net'

const WEB_DIR = process.cwd()
const ARTIFACT_PATH = path.resolve(WEB_DIR, '..', 'artifacts', 'goal-ui', 'goal-drawer-browser-evidence.json')
const NEXT_DIST_DIR = process.env.MOCHI_NEXT_DIST_DIR ?? '.next-fixture-goal-drawer'
const require = createRequire(import.meta.url)
const sessionA = 'session-goal-drawer-a'
const sessionB = 'session-goal-drawer-b'
const goalId = 'goal-drawer-a'
const runId = 'run / linked?&'
let port = 0
let devServer = null
let failureDiagnostic = null

function sleep(ms) { return new Promise((resolve) => setTimeout(resolve, ms)) }
async function waitForCondition(predicate, label, timeoutMs = 10_000) {
  const startedAt = Date.now()
  while (Date.now() - startedAt < timeoutMs) { if (predicate()) return; await sleep(25) }
  throw new Error(`Timed out waiting for ${label}`)
}
function apiResponse(body) { return { status: 200, contentType: 'application/json', body: JSON.stringify(body) } }
function routePath(request) { const url = new URL(request.url()); return `${request.method()} ${url.pathname}${url.search}` }
function requirePlaywright() {
  for (const candidate of [path.join(WEB_DIR, 'node_modules'), 'C:/Users/Xu/AppData/Local/npm-cache/_npx/e41f203b7505f1fb/node_modules']) {
    try { return require(require.resolve('playwright', { paths: [candidate] })) } catch { /* next */ }
  }
  throw new Error('Playwright is not available locally.')
}
function chromiumExecutable() {
  return [process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE, 'C:/Users/Xu/AppData/Local/ms-playwright/chromium-1179/chrome-win/chrome.exe', 'C:/Program Files/Google/Chrome/Application/chrome.exe'].find((item) => item && fs.existsSync(item))
}
async function reservePort() {
  const probe = net.createServer()
  await new Promise((resolve, reject) => { probe.once('error', reject); probe.listen(0, '127.0.0.1', resolve) })
  port = probe.address().port
  await new Promise((resolve) => probe.close(resolve))
}
function startServer() {
  const nextCli = require.resolve('next/dist/bin/next')
  return spawn(process.execPath, [nextCli, 'dev', '--hostname', '127.0.0.1', '--port', String(port)], {
    cwd: WEB_DIR, env: { ...process.env, NEXT_TELEMETRY_DISABLED: '1', MOCHI_NEXT_DIST_DIR: NEXT_DIST_DIR },
    shell: false, windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'],
  })
}
async function stopServer(child) {
  if (!child || child.exitCode !== null) return
  if (process.platform === 'win32') {
    await new Promise((resolve) => { const killer = spawn('taskkill.exe', ['/PID', String(child.pid), '/T', '/F'], { stdio: 'ignore', windowsHide: true }); killer.once('exit', resolve); killer.once('error', resolve) })
  } else child.kill('SIGTERM')
}
async function waitForServer(url) {
  const started = Date.now()
  while (Date.now() - started < 45_000) { try { if ((await fetch(url)).ok) return } catch { /* booting */ }; await sleep(300) }
  throw new Error(`Timed out waiting for ${url}`)
}
function goalSummary(status) {
  return {
    goal_id: goalId, objective: 'Goal drawer browser evidence', status,
    execution_mode: 'autonomous_single_agent', interaction_mode: 'goal', execution_topology: 'single_agent',
    strategy_id: 'autonomous_single_agent', selection_source: 'explicit', selection_reason: 'fixture',
    protocol_selection: 'autonomous_single_agent', selection_rationale: 'fixture', runtime_mode: 'autonomous_single_agent',
    models: [{ role: 'executor', model: 'fixture-model' }], bound_run_id: runId, current_attempt_id: 'attempt-a',
    attempts: [{ attempt_id: 'attempt-a', status, agent_run_id: runId, created_at: '2026-08-20T00:00:00.000Z', updated_at: '2026-08-20T00:00:00.000Z' }],
    created_at: '2026-08-20T00:00:00.000Z', updated_at: '2026-08-20T00:00:00.000Z',
  }
}
function sessionGoal(status) {
  const summary = goalSummary(status)
  return { active_goal_id: goalId, active_goal_status: status, execution_mode: summary.execution_mode, interaction_mode: 'goal', execution_topology: 'single_agent', strategy_id: summary.strategy_id, selection_source: 'explicit', selection_reason: 'fixture', bound_run_id: runId, protocol_selection: summary.protocol_selection, selection_rationale: summary.selection_rationale, default_route: 'goal', last_goal_summary: { ...summary, bound_run_id: runId }, pending_proposal: null }
}
function sessionDetail(id, status, active) {
  const hasGoal = id === sessionA && active
  return { type: 'session', session_id: id, title: hasGoal ? 'Goal drawer session A' : 'No Goal session B', goal: hasGoal ? sessionGoal(status) : { active_goal_id: null, active_goal_status: null, execution_mode: null, interaction_mode: null, execution_topology: null, bound_run_id: null, last_goal_summary: null, pending_proposal: null }, workflow: { enabled: false, bound_run_id: null, config: {} }, events: [] }
}
async function installRoutes(page, requests, state) {
  await page.route('**/favicon.ico', (route) => route.fulfill({ status: 204, body: '' }))
  await page.route('**/v1/**', async (route) => {
    const request = route.request(); const url = new URL(request.url()); const pathname = url.pathname; const method = request.method()
    requests.push(routePath(request))
    if (pathname === '/v1/sessions') return route.fulfill(apiResponse({ items: [
      { session_id: sessionA, title: 'Goal drawer session A', updated_at: '2026-08-20T00:00:00.000Z', event_count: 0, goal: state.active ? sessionGoal(state.status) : sessionDetail(sessionA, state.status, false).goal },
      { session_id: sessionB, title: 'No Goal session B', updated_at: '2026-08-20T00:00:01.000Z', event_count: 0, goal: sessionDetail(sessionB, state.status, false).goal },
    ] }))
    if (pathname === `/v1/sessions/${sessionA}` && method === 'PATCH') {
      const body = request.postDataJSON()
      if (body && typeof body === 'object' && body.goal) state.persistedGoal = body.goal
      return route.fulfill(apiResponse({ type: 'session', session_id: sessionA, title: 'Goal drawer session A', goal: state.persistedGoal ?? sessionDetail(sessionA, state.status, state.active).goal, workflow: { enabled: false, bound_run_id: null, config: {} }, events: [] }))
    }
    if (pathname === `/v1/sessions/${sessionA}/events` && method === 'POST') return route.fulfill(apiResponse({ type: 'session', session_id: sessionA, title: 'Goal drawer session A', goal: state.persistedGoal ?? sessionDetail(sessionA, state.status, state.active).goal, workflow: { enabled: false, bound_run_id: null, config: {} }, events: [] }))
    if (pathname === `/v1/sessions/${sessionA}` || pathname === `/v1/sessions/${sessionA}/events`) {
      if (state.delayAOnce) {
        state.delayAOnce = false
        await new Promise((resolve) => { state.releaseA = resolve })
        state.aResponseDelivered = true
      }
      return route.fulfill(apiResponse(sessionDetail(sessionA, state.status, state.active)))
    }
    if (pathname === `/v1/sessions/${sessionB}` || pathname === `/v1/sessions/${sessionB}/events`) return route.fulfill(apiResponse(sessionDetail(sessionB, state.status, false)))
    if (pathname === `/v1/sessions/${sessionA}/subagents` || pathname === `/v1/sessions/${sessionB}/subagents`) return route.fulfill(apiResponse({ subagents: [] }))
    if (pathname === '/v1/goals/pending-proposal-intent' && method === 'POST') return route.fulfill(apiResponse({ intent: 'confirm_start', confidence: 1, rationale: 'fixture confirmation' }))
    if (pathname === '/v1/goals' && method === 'POST') { state.createPayloads.push(request.postDataJSON()); return route.fulfill(apiResponse(goalSummary('proposed'))) }
    if (pathname === `/v1/goals/${goalId}/start` && method === 'POST') { state.active = true; return route.fulfill(apiResponse(goalSummary(state.status))) }
    if (pathname === `/v1/goals/${goalId}` && method === 'GET') return route.fulfill(apiResponse(goalSummary(state.status)))
    if (pathname === `/v1/goals/${goalId}/health`) return route.fulfill(apiResponse({ goal_id: goalId, status: state.status, blocker: null, approval_state: { pending_count: 0, approval_ids: [] } }))
    if (pathname === `/v1/goals/${goalId}/pause` && method === 'POST') { state.status = 'paused'; return route.fulfill(apiResponse(goalSummary(state.status))) }
    if (pathname === `/v1/goals/${goalId}/resume` && method === 'POST') { state.status = 'running'; return route.fulfill(apiResponse(goalSummary(state.status))) }
    if (pathname === `/v1/goals/${goalId}/cancel` && method === 'POST') { state.status = 'stopped'; return route.fulfill(apiResponse(goalSummary(state.status))) }
    if (pathname === `/v1/goals/${goalId}/turn-decision` && method === 'POST') return route.fulfill(apiResponse({ action: 'guidance', message: 'fixture steer accepted' }))
    if (pathname === `/v1/agent-runs/${encodeURIComponent(runId)}` || pathname === `/v1/agent-runs/${runId}`) return route.fulfill(apiResponse({ run_id: runId, status: state.status, objective: 'Goal drawer browser evidence', events: [] }))
    if (pathname === '/v1/goals') return route.fulfill(apiResponse([goalSummary(state.status)]))
    if (pathname.endsWith('/models')) return route.fulfill(apiResponse({ type: 'models_status', configured_model: 'fixture-model', supported_model_spec_formats: [], active_model: { id: 'fixture-model', name: 'Fixture model', provider: 'openai', supports_tool_calling: true }, available_models: [{ id: 'fixture-model', name: 'Fixture model', provider: 'openai', supports_tool_calling: true }] }))
    if (pathname.endsWith('/settings')) return route.fulfill(apiResponse({ ok: true }))
    if (pathname === '/v1/goals/strategies' || pathname.endsWith('/goal-strategies')) return route.fulfill(apiResponse({ strategies: [{ strategy_id: 'autonomous_single_agent', label: 'Autonomous single agent', description: 'Fixture strategy' }] }))
    if (pathname.endsWith('/projects') || pathname.endsWith('/tasks') || pathname.endsWith('/approvals') || pathname.endsWith('/agent-runs')) return route.fulfill(apiResponse([]))
    return route.fulfill(apiResponse([]))
  })
}
async function expectVisible(page, selector, label) { await page.locator(selector).waitFor({ state: 'visible', timeout: 30_000 }).catch((error) => { throw new Error(`${label}: ${error.message}`) }) }
async function expectHidden(page, selector, label) { await page.locator(selector).waitFor({ state: 'hidden', timeout: 10_000 }).catch((error) => { throw new Error(`${label}: ${error.message}`) }) }
async function selectSession(page, title) {
  const candidate = page.getByText(title, { exact: true }).filter({ visible: true }).first()
  await candidate.click({ timeout: 20_000 })
}
async function writeArtifact(payload) { await fs.promises.mkdir(path.dirname(ARTIFACT_PATH), { recursive: true }); await fs.promises.writeFile(ARTIFACT_PATH, `${JSON.stringify(payload, null, 2)}\n`, 'utf8') }

async function main() {
  await reservePort(); const appUrl = `http://127.0.0.1:${port}/`; devServer = startServer()
  const requests = []; const consoleMessages = []; const state = { status: 'running', delayAOnce: true, active: false, createPayloads: [], persistedGoal: null, releaseA: null, aResponseDelivered: false }; failureDiagnostic = { requests, state, consoleMessages }
  try {
    await waitForServer(appUrl)
    const { chromium } = requirePlaywright(); const browser = await chromium.launch({ headless: true, executablePath: chromiumExecutable() })
    try {
      const page = await browser.newPage({ viewport: { width: 1280, height: 900 } })
      page.on('console', (message) => { if (message.type() === 'warning' || message.type() === 'error') consoleMessages.push(`${message.type()}: ${message.text()}`) })
      page.on('pageerror', (error) => consoleMessages.push(`pageerror: ${error.message}`))
      await installRoutes(page, requests, state)
      await page.goto(appUrl, { waitUntil: 'domcontentloaded' })
      await expectHidden(page, '[data-testid="goal-composer-drawer"]', 'unactivated session should not show drawer')
      await selectSession(page, 'No Goal session B')
      assert.equal(typeof state.releaseA, 'function', 'session-A response must be held before B is selected')
      state.releaseA()
      await waitForCondition(() => state.aResponseDelivered, 'deferred session-A response release')
      await expectHidden(page, '[data-testid="goal-composer-drawer"]', 'released session-A response must not pollute selected session B')
      await selectSession(page, 'Goal drawer session A')
      await expectHidden(page, '[data-testid="goal-composer-drawer"]', 'session A starts without a Goal')
      const composer = page.locator('#chat-input-textarea')
      await composer.fill('/goal Research this for 20 minutes')
      await composer.press('Escape')
      await composer.press('Enter')
      await page.waitForTimeout(250)
      await composer.fill('Research this for 20 minutes')
      await composer.press('Enter')
      await page.waitForTimeout(250)
      await composer.fill('start it')
      await composer.press('Enter')
      await expectVisible(page, '[data-testid="goal-composer-drawer"]', 'Goal activated through public composer')
      assert.equal(state.createPayloads.length, 1, 'slash and equivalent natural-language activation must converge to one durable Goal create')
      assert.equal(state.createPayloads[0]?.objective, 'Research this for 20 minutes', 'durable Goal objective must use the shared normalized content')
      assert.ok(requests.includes(`POST /v1/goals/${goalId}/start`), 'public confirmation must start the single created Goal')
      assert.equal(await page.locator('[data-testid="goal-composer-drawer"]').count(), 1, 'exactly one goal drawer should render')
      assert.equal(await page.locator('[data-testid="goal-header-chip"], [data-testid="goal-focus-panel"], [data-testid="goal-floating-panel"]').count(), 0, 'legacy Goal surfaces must not render')
      assert.equal(await page.getByText('Execution highlights', { exact: true }).count(), 0, 'legacy execution transcript surface must not render')
      assert.equal(await page.getByText('Goals', { exact: true }).count(), 0, 'sidebar or navigation must not expose Goals')
      await expectVisible(page, '[data-testid="goal-action-pause"]', 'running pause action')
      await expectVisible(page, '[data-testid="goal-action-stop"]', 'running stop action')
      await expectVisible(page, '[data-testid="goal-action-steer"]', 'running steer action')
      await expectVisible(page, '[data-testid="goal-action-clear"]', 'running clear action')
      await expectVisible(page, '[data-testid="goal-action-execution-record"]', 'linked execution record action')
      await expectHidden(page, '[data-testid="goal-action-resume"]', 'running resume action must be absent')
      await page.locator('[data-testid="goal-action-pause"]').click()
      await expectVisible(page, '[data-testid="goal-action-resume"]', 'paused resume action')
      assert.ok(requests.includes(`POST /v1/goals/${goalId}/pause`), 'pause must POST only to the selected Goal')
      await page.locator('[data-testid="goal-action-resume"]').click()
      await expectVisible(page, '[data-testid="goal-action-pause"]', 'resumed pause action')
      assert.ok(requests.includes(`POST /v1/goals/${goalId}/resume`), 'resume must POST only to the selected Goal')
      await page.locator('[data-testid="goal-steer-input"]').fill('Keep the same Goal identity')
      await page.locator('[data-testid="goal-action-steer"]').click()
      await page.waitForTimeout(300)
      assert.equal(requests.some((entry) => entry.startsWith('DELETE /v1/goals/') || entry.startsWith('DELETE /v1/agent-runs/')), false, 'steer must not destructively delete Goal history')
      const beforeClear = requests.length
      await page.locator('[data-testid="goal-action-clear"]').click()
      await expectVisible(page, '[data-testid="goal-clear-disclosure"]', 'clear must disclose presentation-only semantics')
      await expectVisible(page, '[data-testid="goal-composer-drawer"]', 'clear disclosure must keep drawer visible')
      await page.locator('[data-testid="goal-clear-cancel"]').click()
      await expectHidden(page, '[data-testid="goal-clear-disclosure"]', 'clear cancel closes disclosure')
      await expectVisible(page, '[data-testid="goal-composer-drawer"]', 'clear cancel preserves drawer')
      await page.locator('[data-testid="goal-action-clear"]').click()
      await page.locator('[data-testid="goal-clear-confirm"]').click()
      await expectHidden(page, '[data-testid="goal-composer-drawer"]', 'confirmed clear detaches composer presentation')
      assert.equal(requests.slice(beforeClear).some((entry) => entry.startsWith('DELETE ')), false, 'clear must not delete durable Goal or AgentRun history')
      await page.reload({ waitUntil: 'domcontentloaded' })
      await expectVisible(page, '[data-testid="goal-composer-drawer"]', 'reload restores server-backed Goal presentation')
      const navigation = page.waitForURL(new RegExp(`/agent-runs/${encodeURIComponent(runId).replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`), { timeout: 20_000 })
      await page.locator('[data-testid="goal-action-execution-record"]').click(); await navigation
      assert.equal(new URL(page.url()).pathname, `/agent-runs/${encodeURIComponent(runId)}`, 'execution record must directly navigate to encoded linked AgentRun')
      assert.ok(requests.includes(`GET /v1/agent-runs/${encodeURIComponent(runId)}`), 'AgentRun detail must load the selected once-encoded raw run ID')
      assert.equal(requests.some((entry) => entry.includes('%25')), false, 'AgentRun detail must never double-encode the selected run ID')
      await page.goto(appUrl, { waitUntil: 'domcontentloaded' }); await expectVisible(page, '[data-testid="goal-composer-drawer"]', 'drawer after return')
      await page.locator('[data-testid="goal-action-stop"]').click()
      assert.ok(requests.includes(`POST /v1/goals/${goalId}/cancel`), 'stop must POST cancel to the selected Goal')
      await page.goto(`${appUrl}goals`, { waitUntil: 'domcontentloaded' })
      await page.waitForURL(/\/agent-runs(?:\/)?$/, { timeout: 20_000 })
      assert.equal(new URL(page.url()).pathname, '/agent-runs', '/goals must be a compatibility redirect to Agent Runs')
      assert.deepEqual(consoleMessages, [], 'browser must have no warning/error/pageerror noise')
      const payload = { ok: true, checks: { oneComposerDrawer: true, legacyGoalUiAbsent: true, lifecycleRequestsNonDestructive: true, clearPreservesHistory: true, encodedDirectExecutionRecord: true, delayedSessionIsolation: true, goalsRedirect: true, consoleClean: true }, requests, consoleMessages }
      await writeArtifact(payload); console.log(JSON.stringify(payload, null, 2)); await page.close()
    } finally { await browser.close() }
  } finally { await stopServer(devServer) }
}

main().catch(async (error) => { await writeArtifact({ ok: false, error: error.message, diagnostic: failureDiagnostic }).catch(() => {}); console.error(error); process.exit(1) })
