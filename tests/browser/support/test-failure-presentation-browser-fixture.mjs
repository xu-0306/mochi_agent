import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { spawn } from 'node:child_process'
import { createRequire } from 'node:module'
import { createServer } from 'node:net'
import fs from 'node:fs'
import path from 'node:path'

const WEB_DIR = process.cwd()
const require = createRequire(import.meta.url)
const NEXT_DIST_DIR = process.env.MOCHI_NEXT_DIST_DIR ?? '.next-fixture-failure-presentation'
const artifactDir = process.env.MOCHI_FAILURE_PRESENTATION_ARTIFACT_DIR
const RAW_LATEST_ERROR = 'raw latest_error: /srv/secrets/model-token'
const DIAGNOSTICS_REF = 'restricted diagnostics_ref: /srv/secrets/model-token'
const EXPECTED_TITLE = 'Tool did not complete'
const EXPECTED_TONE = 'error'
const EXPECTED_ACTION = 'retry_tool'
let port = 0
let baseUrl = ''
let stoppingDevServer = false

function canonicalEnvelope() {
  return {
    schema_version: '1.7',
    kind: 'tool_error',
    origin: 'runtime',
    recoverability: 'manual_retry',
    retry_policy: 'manual',
    terminal: true,
    inject_into_model_context: false,
    telemetry_key: 'failure.tool_error',
    ui_hint: 'retry',
    diagnostics_ref: DIAGNOSTICS_REF,
  }
}

function goalPayload() {
  return {
    goal_id: 'goal-failure-presentation',
    objective: 'Render a safe failure card',
    title: 'Failure fixture goal',
    status: 'failed',
    latest_error: RAW_LATEST_ERROR,
    failure: canonicalEnvelope(),
    attempts: [],
    created_at: '2026-08-09T00:00:00.000Z',
    updated_at: '2026-08-09T00:00:00.000Z',
  }
}

function agentRunPayload() {
  return {
    run_id: 'run-failure-presentation',
    protocol_id: 'controlled_subagent_execution',
    title: 'Failure fixture run',
    status: 'failed',
    latest_error: RAW_LATEST_ERROR,
    failure: canonicalEnvelope(),
    created_at: '2026-08-09T00:00:00.000Z',
    updated_at: '2026-08-09T00:00:00.000Z',
    events: [],
  }
}

function chatPayload() {
  return {
    chat_event: {
      type: 'turn_event',
      turn_id: 'turn-failure-presentation',
      timestamp: '2026-08-09T00:00:00.000Z',
      phase: 'error',
      payload: {
        error: RAW_LATEST_ERROR,
        failure: canonicalEnvelope(),
      },
    },
  }
}

async function configurePort() {
  const probe = createServer()
  await new Promise((resolve, reject) => {
    probe.once('error', reject)
    probe.listen(0, '127.0.0.1', resolve)
  })
  const address = probe.address()
  port = typeof address === 'object' && address ? address.port : 0
  await new Promise((resolve) => probe.close(resolve))
  baseUrl = `http://127.0.0.1:${port}`
}

function startDevServer() {
  let nextCli
  try {
    nextCli = require.resolve('next/dist/bin/next', { paths: [path.join(WEB_DIR, 'node_modules')] })
  } catch {
    throw new Error('The failure presentation fixture requires web/node_modules with Next and React installed.')
  }
  return spawn(process.execPath, [nextCli, 'dev', '--hostname', '127.0.0.1', '--port', String(port)], {
    cwd: WEB_DIR,
    env: { ...process.env, NEXT_TELEMETRY_DISABLED: '1', MOCHI_NEXT_DIST_DIR: NEXT_DIST_DIR },
    stdio: ['ignore', 'pipe', 'pipe'],
    windowsHide: true,
  })
}

async function stopDevServer(child) {
  if (!child || child.exitCode !== null) return
  stoppingDevServer = true
  child.stdout?.destroy()
  child.stderr?.destroy()
  if (process.platform === 'win32') {
    await new Promise((resolve) => {
      const killer = spawn('taskkill.exe', ['/PID', String(child.pid), '/T', '/F'], { stdio: 'ignore', windowsHide: true })
      const timeout = setTimeout(resolve, 5_000)
      timeout.unref?.()
      killer.once('exit', () => { clearTimeout(timeout); resolve() })
      killer.once('error', () => { clearTimeout(timeout); resolve() })
    })
    child.unref()
    return
  }
  child.kill('SIGTERM')
}

async function waitForServer(url) {
  const startedAt = Date.now()
  while (Date.now() - startedAt < 45_000) {
    try {
      const response = await fetch(url)
      if (response.ok) return
    } catch {
      // Next is compiling the production fixture route.
    }
    await new Promise((resolve) => setTimeout(resolve, 250))
  }
  throw new Error(`Timed out waiting for ${url}`)
}

function requirePlaywright() {
  for (const candidate of [path.join(WEB_DIR, 'node_modules'), 'C:/Users/xu/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules']) {
    try {
      return require(require.resolve('playwright', { paths: [candidate] }))
    } catch {
      // Try the next pinned runtime location.
    }
  }
  throw new Error('Playwright is unavailable for the failure presentation fixture.')
}

function findChromiumExecutable(chromium) {
  return [
    process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE,
    chromium.executablePath(),
    '/usr/bin/google-chrome',
    '/usr/bin/google-chrome-stable',
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser',
    'C:/Program Files/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
    'C:/Program Files/Microsoft/Edge/Application/msedge.exe',
  ].filter(Boolean).find((candidate) => fs.existsSync(candidate))
}

async function installApiRoutes(page, state) {
  await page.route('**/v1/**', async (route) => {
    const request = route.request()
    const pathname = new URL(request.url()).pathname
    const json = (body, status = 200) => route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) })
    state.requests.push(`${request.method()} ${pathname}`)
    if (request.method() !== 'GET') return json({ detail: 'Unexpected fixture method' }, 405)
    if (pathname === '/v1/projects' || pathname === '/v1/sessions') return json([])
    if (pathname === '/v1/goals/goal-failure-presentation') return json(goalPayload())
    if (pathname === '/v1/agent-runs/run-failure-presentation') return json(agentRunPayload())
    if (pathname === '/v1/test-fixtures/failure-presentation') return json(chatPayload())
    return json({ detail: `Unhandled fixture request: ${request.method()} ${pathname}` }, 404)
  })
}

function sha256(file) {
  return createHash('sha256').update(fs.readFileSync(file)).digest('hex')
}

async function runBrowserAssertions() {
  assert.ok(artifactDir, 'MOCHI_FAILURE_PRESENTATION_ARTIFACT_DIR must be set by the focused browser test.')
  await fs.promises.mkdir(artifactDir, { recursive: true })
  const { chromium } = requirePlaywright()
  const executablePath = findChromiumExecutable(chromium)
  assert.ok(executablePath, 'No Chromium executable is available for the production failure fixture.')
  const browser = await chromium.launch({ headless: true, executablePath })
  const manifest = { status: 'passed', baseline_update_policy: 'never', raw_latest_error_rendered: false, diagnostics_ref_rendered: false, viewports: [] }
  try {
    for (const viewport of [
      { name: 'desktop', width: 1440, height: 900 },
      { name: 'mobile', width: 390, height: 844 },
    ]) {
      const page = await browser.newPage({ viewport: { width: viewport.width, height: viewport.height } })
      page.setDefaultTimeout(10_000)
      const state = { requests: [] }
      const consoleErrors = []
      page.on('console', (message) => {
        if (message.type() === 'error' && !message.text().startsWith('Failed to load resource:')) consoleErrors.push(message.text())
      })
      page.on('pageerror', (error) => consoleErrors.push(error.message))
      await installApiRoutes(page, state)
      const response = await page.goto(`${baseUrl}/test-fixtures/failure-presentation`, { waitUntil: 'networkidle' })
      assert.ok(response?.ok(), `${viewport.name}: production fixture did not load`)
      await page.getByTestId('failure-presentation-fixture').waitFor()
      assert.deepEqual(state.requests.sort(), [
        'GET /v1/agent-runs/run-failure-presentation',
        'GET /v1/goals/goal-failure-presentation',
        'GET /v1/projects',
        'GET /v1/sessions',
        'GET /v1/test-fixtures/failure-presentation',
      ])
      for (const testId of ['chat-failure-presentation', 'goal-failure-presentation', 'agent-run-failure-presentation']) {
        const card = page.getByTestId(testId)
        assert.equal(await card.count(), 1, `${viewport.name}: ${testId} must render once`)
        assert.equal(await card.getAttribute('data-failure-kind'), 'tool_error')
        assert.equal(await card.getAttribute('data-failure-tone'), EXPECTED_TONE)
        assert.equal(await page.getByTestId(`${testId}-title`).innerText(), EXPECTED_TITLE)
        assert.equal(await page.getByTestId(`${testId}-action-${EXPECTED_ACTION}`).count(), 1)
      }
      const horizontalBounds = []
      for (const testId of ['chat-failure-presentation', 'goal-failure-presentation', 'agent-run-failure-presentation']) {
        for (const target of [testId, `${testId}-action-${EXPECTED_ACTION}`]) {
          const boundingBox = await page.getByTestId(target).boundingBox()
          assert.ok(boundingBox, `${viewport.name}: ${target} must have a visible bounding box`)
          const right = boundingBox.x + boundingBox.width
          const overflowPx = Math.max(0, -boundingBox.x, right - viewport.width)
          horizontalBounds.push({ test_id: target, x: boundingBox.x, width: boundingBox.width, right, overflow_px: overflowPx })
          assert.ok(overflowPx <= 0.5, `${viewport.name}: ${target} overflows horizontally by ${overflowPx}px`)
        }
      }
      const documentBounds = await page.evaluate(() => ({
        client_width: document.documentElement.clientWidth,
        scroll_width: document.documentElement.scrollWidth,
      }))
      const documentOverflowPx = Math.max(0, documentBounds.scroll_width - documentBounds.client_width)
      assert.ok(documentOverflowPx <= 0.5, `${viewport.name}: document overflows horizontally by ${documentOverflowPx}px`)
      const renderedText = await page.locator('body').innerText()
      const rawLatestErrorRendered = renderedText.includes(RAW_LATEST_ERROR)
      const diagnosticsRefRendered = renderedText.includes(DIAGNOSTICS_REF)
      assert.equal(rawLatestErrorRendered, false, `${viewport.name}: raw latest_error must never render`)
      assert.equal(diagnosticsRefRendered, false, `${viewport.name}: diagnostics_ref must never render`)
      assert.deepEqual(consoleErrors, [], `${viewport.name}: browser errors`)
      const screenshot = `${viewport.name}.png`
      const screenshotPath = path.join(artifactDir, screenshot)
      await page.screenshot({ path: screenshotPath, fullPage: true })
      const screenshotBytes = fs.readFileSync(screenshotPath)
      assert.equal(screenshotBytes.includes(Buffer.from(RAW_LATEST_ERROR)), false, `${viewport.name}: screenshot leaked raw latest_error`)
      assert.equal(screenshotBytes.includes(Buffer.from(DIAGNOSTICS_REF)), false, `${viewport.name}: screenshot leaked diagnostics_ref`)
      manifest.viewports.push({
        viewport,
        status: 'passed',
        screenshot,
        screenshot_sha256: sha256(screenshotPath),
        raw_latest_error_rendered: rawLatestErrorRendered,
        diagnostics_ref_rendered: diagnosticsRefRendered,
        horizontal_bounds: horizontalBounds,
        document_bounds: documentBounds,
        horizontal_overflow_px: Math.max(documentOverflowPx, ...horizontalBounds.map((item) => item.overflow_px)),
      })
      await page.close()
    }
  } finally {
    await browser.close()
  }
  const manifestText = JSON.stringify(manifest, null, 2)
  assert.equal(manifestText.includes(RAW_LATEST_ERROR), false, 'manifest must not include raw latest_error')
  assert.equal(manifestText.includes(DIAGNOSTICS_REF), false, 'manifest must not include diagnostics_ref')
  await fs.promises.writeFile(path.join(artifactDir, 'manifest.json'), `${manifestText}\n`, 'utf8')
}

async function main() {
  await configurePort()
  const child = startDevServer()
  child.once('exit', (code) => {
    if (!stoppingDevServer && code !== 0) console.error(`Next development server exited with code ${code}`)
  })
  try {
    await waitForServer(`${baseUrl}/test-fixtures/failure-presentation`)
    await runBrowserAssertions()
    console.log('ok')
  } finally {
    console.error('[failure-presentation-fixture] stopping Next')
    await stopDevServer(child)
    console.error('[failure-presentation-fixture] cleanup complete')
  }
}

main().catch((error) => { console.error(error); process.exit(1) })
