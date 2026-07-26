import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { createRequire } from 'node:module'
import { createServer } from 'node:net'
import fs from 'node:fs'
import path from 'node:path'

const WEB_DIR = process.cwd()
const configuredPort = Number(process.env.MOCHI_RELOAD_FIXTURE_PORT ?? 0)
let PORT = Number.isInteger(configuredPort) && configuredPort > 0 ? configuredPort : 0
let BASE_URL = ''
let APP_URL = ''
const NEXT_DIST_DIR = process.env.MOCHI_NEXT_DIST_DIR ?? '.next-fixture-reload'
const require = createRequire(import.meta.url)
let stoppingDevServer = false

function reportPhase(phase) {
  console.error(`[reload-browser-fixture] phase=${phase}`)
}

async function configurePort() {
  if (PORT === 0) {
    const probe = createServer()
    await new Promise((resolve, reject) => {
      probe.once('error', reject)
      probe.listen(0, '127.0.0.1', resolve)
    })
    const address = probe.address()
    PORT = typeof address === 'object' && address ? address.port : 0
    await new Promise((resolve) => probe.close(resolve))
  }
  BASE_URL = `http://127.0.0.1:${PORT}`
  APP_URL = `${BASE_URL}/`
}

const sessionId = 'session-reload-subagent'
const subagentId = 'reload-subagent-approval'
const approvalId = 'reload-approval-1'

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
  const nextCli = require.resolve('next/dist/bin/next')
  const child = spawn(process.execPath, [nextCli, 'dev', '--hostname', '127.0.0.1', '--port', String(PORT)], {
    cwd: WEB_DIR,
    env: {
      ...process.env,
      NEXT_TELEMETRY_DISABLED: '1',
      MOCHI_NEXT_DIST_DIR: NEXT_DIST_DIR,
    },
    shell: false,
    windowsHide: true,
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
  if (!child || child.exitCode !== null) {
    return
  }
  stoppingDevServer = true
  child.stdout?.destroy()
  child.stderr?.destroy()
  if (process.platform === 'win32') {
    await new Promise((resolve) => {
      let settled = false
      const finish = () => {
        if (settled) {
          return
        }
        settled = true
        clearTimeout(timeout)
        resolve()
      }
      const killer = spawn('taskkill.exe', ['/PID', String(child.pid), '/T', '/F'], {
        stdio: 'ignore',
      })
      const timeout = setTimeout(() => {
        killer.kill()
        if (!child.killed) {
          child.kill()
        }
        finish()
      }, 5_000)
      timeout.unref?.()
      killer.on('exit', finish)
      killer.on('error', finish)
    })
    await new Promise((resolve) => {
      if (child.exitCode !== null) {
        resolve()
        return
      }
      const timeout = setTimeout(resolve, 3_000)
      timeout.unref?.()
      child.once('exit', () => {
        clearTimeout(timeout)
        resolve()
      })
    })
    return
  }
  child.kill()
  await new Promise((resolve) => setTimeout(resolve, 800))
  child.kill('SIGKILL')
}

async function cleanupNextDistDir() {
  if (process.env.MOCHI_NEXT_DIST_DIR) {
    return
  }
  await fs.promises.rm(path.join(WEB_DIR, NEXT_DIST_DIR), { recursive: true, force: true })
}

async function waitForServer(url, timeoutMs = 45_000) {
  const startedAt = Date.now()
  while (Date.now() - startedAt < timeoutMs) {
    const controller = new AbortController()
    const probeTimeout = setTimeout(() => controller.abort(), 2_000)
    try {
      const response = await fetch(url, { signal: controller.signal })
      if (response.ok) {
        return
      }
    } catch {
      // Server is not ready yet.
    } finally {
      clearTimeout(probeTimeout)
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

const sessionEvents = [
  {
    type: 'message',
    role: 'user',
    content: 'Reload the session and restore the delegated subagent card.',
    timestamp: '2026-06-30T04:00:00.000Z',
  },
  {
    type: 'subagent_started',
    seq: 1,
    session_id: sessionId,
    parent_type: 'chat_turn',
    parent_id: 'turn-reload-1',
    subagent_id: subagentId,
    role_id: 'reload_researcher',
    title: 'Reload persistence researcher',
    model_id: 'qwen3-coder-fixture',
    status: 'running',
    summary: 'Reload persistence subagent started.',
    timestamp: '2026-06-30T04:01:00.000Z',
  },
  {
    type: 'subagent_tool_result',
    seq: 2,
    session_id: sessionId,
    parent_type: 'chat_turn',
    parent_id: 'turn-reload-1',
    subagent_id: subagentId,
    role_id: 'reload_researcher',
    title: 'Reload persistence researcher',
    model_id: 'qwen3-coder-fixture',
    status: 'approval_required',
    summary: 'Approval required during reload persistence probe.',
    timestamp: '2026-06-30T04:02:00.000Z',
    metadata: {
      approval_id: approvalId,
      approval_ids: [approvalId],
      blocker_type: 'approval',
      tool_names: ['exec_command'],
      pending_approvals: [
        {
          approval_id: approvalId,
          tool_name: 'exec_command',
          reason: 'Reload persistence approval fixture.',
          approval_kind: 'exec',
          approval_scope: 'workspace',
          security_decision: 'require_approval',
          allowed_decisions: ['approve_once', 'reject'],
        },
      ],
    },
  },
]

const subagentSummary = {
  subagent_id: subagentId,
  parent_type: 'chat_turn',
  parent_id: 'turn-reload-1',
  session_id: sessionId,
  role_id: 'reload_researcher',
  title: 'Reload persistence researcher',
  model_id: 'qwen3-coder-fixture',
  status: 'awaiting_approval',
  prompt_preview: 'Reload prompt preview sentinel.',
  summary: 'Reload summary from API should survive event-derived summaries.',
  output_preview: 'Reload output preview sentinel.',
  event_count: 4,
  created_at: '2026-06-30T04:01:00.000Z',
  updated_at: '2026-06-30T04:03:00.000Z',
}

const subagentDetail = {
  ...subagentSummary,
  system_prompt: 'System reload sentinel prompt. Keep approval metadata visible after reload.',
  user_prompt:
    'User reload sentinel prompt with a long-token reload-persistence-super-long-token-that-must-wrap-without-overflow.',
  events: [
    ...sessionEvents.slice(1),
    {
      type: 'runtime_blocked',
      seq: 3,
      session_id: sessionId,
      parent_type: 'chat_turn',
      parent_id: 'turn-reload-1',
      subagent_id: subagentId,
      role_id: 'reload_researcher',
      title: 'Reload persistence researcher',
      model_id: 'qwen3-coder-fixture',
      status: 'blocked',
      summary: 'Runtime blocked on reload approval.',
      timestamp: '2026-06-30T04:03:00.000Z',
      metadata: {
        approval_ids: [approvalId],
        blocker_type: 'approval',
        tool_names: ['exec_command'],
        recommended_action: 'resolve_approval',
      },
    },
  ],
}

async function installRoutes(page, requestLog) {
  let guidanceSent = false
  const detailWithGuidance = () => ({
    ...subagentDetail,
    events: guidanceSent
      ? [
          ...subagentDetail.events,
          {
            type: 'subagent_guidance',
            seq: 4,
            session_id: sessionId,
            parent_type: 'chat_turn',
            parent_id: 'turn-reload-1',
            subagent_id: subagentId,
            status: 'queued',
            summary: 'Guidance queued after reload.',
            timestamp: '2026-06-30T04:04:00.000Z',
          },
        ]
      : subagentDetail.events,
  })

  await page.route('**/favicon.ico', (route) => route.fulfill({ status: 204, body: '' }))
  await page.route('**/v1/**', (route) => {
    const request = route.request()
    const url = new URL(request.url())
    requestLog.push(`${request.method()} ${url.pathname}`)
    const pathname = url.pathname

    if (pathname === '/v1/sessions') {
      return route.fulfill(json({
        items: [
          {
            session_id: sessionId,
            title: 'Reload Subagent Session',
            updated_at: '2026-06-30T04:04:00.000Z',
            event_count: sessionEvents.length,
          },
        ],
      }))
    }
    if (pathname === `/v1/sessions/${sessionId}`) {
      return route.fulfill(json({
        session_id: sessionId,
        title: 'Reload Subagent Session',
        events: sessionEvents,
      }))
    }
    if (pathname === `/v1/sessions/${sessionId}/subagents`) {
      return route.fulfill(json({ subagents: [subagentSummary] }))
    }
    if (pathname === `/v1/sessions/${sessionId}/subagents/${subagentId}`) {
      return route.fulfill(json(detailWithGuidance()))
    }
    if (pathname === `/v1/sessions/${sessionId}/subagents/${subagentId}/messages`) {
      guidanceSent = true
      return route.fulfill(json(detailWithGuidance()))
    }
    if (pathname.endsWith('/projects') || pathname.endsWith('/goals') || pathname.endsWith('/tasks') || pathname.endsWith('/approvals') || pathname.endsWith('/agent-runs')) {
      return route.fulfill(json([]))
    }
    if (pathname.endsWith('/settings')) {
      return route.fulfill(json({ ok: true }))
    }
    if (pathname.endsWith('/models')) {
      return route.fulfill(json({ configured: [], available: [] }))
    }
    return route.fulfill(json({}))
  })
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

async function waitForVisibleText(page, text, label) {
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
  ).catch((error) => {
    throw new Error(`${label}: visible text not found: ${text}. ${error.message}`)
  })
}

async function main() {
  reportPhase('configure-port')
  await configurePort()
  reportPhase('start-next-dev-server')
  const server = startDevServer()
  try {
    await waitForServer(APP_URL)
    reportPhase('next-dev-server-ready')
    const { chromium } = requireLocalPlaywright()
    const browser = await chromium.launch({
      headless: true,
      executablePath: findChromiumExecutable(),
    })
    reportPhase('browser-launched')
    try {
      const page = await browser.newPage({ viewport: { width: 390, height: 844 } })
      const requestLog = []
      const badMessages = []
      page.on('console', (message) => {
        if (message.type() === 'warning' || message.type() === 'error') {
          badMessages.push(`${message.type()}: ${message.text()}`)
        }
      })
      page.on('pageerror', (error) => badMessages.push(`pageerror: ${error.message}`))
      await installRoutes(page, requestLog)

      await page.goto(APP_URL, { waitUntil: 'networkidle' })
      await waitForVisibleText(page, 'Reload persistence researcher', 'initial load')
      await waitForVisibleText(page, '4 events', 'initial load')

      await page.reload({ waitUntil: 'networkidle' })
      await waitForVisibleText(page, 'Reload persistence researcher', 'after reload')
      await waitForVisibleText(page, 'Reload summary from API should survive event-derived summaries.', 'after reload')
      await waitForVisibleText(page, '4 events', 'after reload')

      await page.getByRole('button', { name: /Open Reload persistence researcher/i }).first().click()
      const dialog = page.locator('[role="dialog"]')
      await dialog.getByText('Waiting for approval', { exact: true }).waitFor()
      await dialog.getByRole('button', { name: /Prompt/i }).first().click()
      await page.getByText('System reload sentinel prompt').waitFor()
      await page.getByText('reload-persistence-super-long-token', { exact: false }).waitFor()
      await page.getByText('Reload output preview sentinel.').waitFor()
      await waitForVisibleText(page, 'Runtime blocked on reload approval.', 'drawer timeline')
      await page.getByPlaceholder('Ask for follow-up changes').fill('Reload guidance sentinel')
      await page.getByRole('button', { name: /Send follow-up guidance/i }).click()
      try {
        await page.getByText('Guidance queued after reload.').waitFor({ timeout: 8_000 })
      } catch (error) {
        const body = await page.locator('body').innerText({ timeout: 2_000 }).catch(() => '')
        throw new Error(`Guidance result did not render. requests=${JSON.stringify(requestLog)} body=${JSON.stringify(body.slice(0, 1000))}`)
      }

      const requiredRequests = [
        `GET /v1/sessions`,
        `GET /v1/sessions/${sessionId}`,
        `GET /v1/sessions/${sessionId}/subagents`,
        `GET /v1/sessions/${sessionId}/subagents/${subagentId}`,
        `POST /v1/sessions/${sessionId}/subagents/${subagentId}/messages`,
      ]
      for (const required of requiredRequests) {
        assert.ok(requestLog.includes(required), `expected request ${required}`)
      }
      assert.ok(
        requestLog.filter((item) => item === `GET /v1/sessions/${sessionId}/subagents`).length >= 2,
        'reload should refetch session subagents'
      )
      await assertNoConsoleNoise(page, badMessages)
      await page.close()
      reportPhase('assertions-complete')
    } finally {
      reportPhase('close-browser')
      await browser.close()
    }
    console.log('ok')
  } finally {
    reportPhase('cleanup-runtime')
    await stopDevServer(server)
    await cleanupNextDistDir()
    reportPhase('cleanup-complete')
  }
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error)
    process.exit(1)
  })
