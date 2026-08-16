import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { createRequire } from 'node:module'
import { createServer } from 'node:net'
import fs from 'node:fs'
import path from 'node:path'

const WEB_DIR = process.cwd()
const require = createRequire(import.meta.url)
const NEXT_DIST_DIR = process.env.MOCHI_NEXT_DIST_DIR ?? '.next-fixture-file-workflow'
let port = 0
let baseUrl = ''
let stoppingDevServer = false

function file(path, entryId, dependencyGroup = null) {
  return {
    path,
    relative_path: path,
    status: 'modified',
    original_content: 'before',
    new_content: 'after',
    change_set_id: 'change-parent',
    entry_id: entryId,
    request_digest: 'a'.repeat(64),
    dependency_group: dependencyGroup,
    change_contract_mode: 'enforce',
  }
}

function approval({ id, status = 'pending', supersedes = null, supersededBy = null, files }) {
  return {
    approval_id: id,
    task_id: null,
    status,
    tool_name: 'apply_patch',
    arguments: {},
    created_at: '2026-08-09T00:00:00.000Z',
    requires_approval: true,
    approval_kind: 'workspace_write',
    approval_scope: 'workspace',
    replay_safe: false,
    allowed_decisions: ['approve_once', 'reject'],
    file_change_groups: [{
      id: `${id}:files`,
      source_tool: 'apply_patch',
      title: 'Pending file review',
      files,
    }],
    change_set_id: `change-${id}`,
    request_digest: id === 'approval-parent' ? 'a'.repeat(64) : 'b'.repeat(64),
    change_contract_mode: 'enforce',
    change_expires_at: '2026-08-09T01:00:00.000Z',
    change_policy_version: 'file-policy-v1:fixture',
    approval_state: status === 'pending' ? 'replacement_pending' : 'superseded',
    supersedes_approval_id: supersedes,
    superseded_by_approval_id: supersededBy,
  }
}

const parentApproval = approval({
  id: 'approval-parent',
  files: [
    file('src/a.ts', 'entry-a', 'rename'),
    file('src/b.ts', 'entry-b', 'rename'),
    file('docs/c.md', 'entry-c'),
  ],
})
const replacementApproval = approval({
  id: 'approval-replacement',
  supersedes: 'approval-parent',
  files: [file('docs/c.md', 'entry-c'), file('docs/d.md', 'entry-d')],
})

function subsetPreview({ replacementApprovalId, requestDigest }) {
  return {
    valid: true,
    summary: 'Server created a replacement approval.',
    errors: [],
    warnings: [],
    file_changes: [],
    change_set_id: 'change-replacement',
    request_digest: requestDigest,
    expires_at: '2026-08-09T01:00:00.000Z',
    policy_version: 'file-policy-v1:fixture',
    change_contract_mode: 'enforce',
    replacement_approval_id: replacementApprovalId,
    approval_state: 'replacement_pending',
    would_reject_edited_patch: false,
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
    nextCli = require.resolve('next/dist/bin/next', {
      paths: [path.join(WEB_DIR, 'node_modules')],
    })
  } catch {
    throw new Error('The production TaskPanel fixture requires web/node_modules with Next and React installed.')
  }
  return spawn(process.execPath, [nextCli, 'dev', '--hostname', '127.0.0.1', '--port', String(port)], {
    cwd: WEB_DIR,
    env: {
      ...process.env,
      NEXT_TELEMETRY_DISABLED: '1',
      MOCHI_NEXT_DIST_DIR: NEXT_DIST_DIR,
    },
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
      const killer = spawn('taskkill.exe', ['/PID', String(child.pid), '/T', '/F'], {
        stdio: 'ignore',
        windowsHide: true,
      })
      const timeout = setTimeout(resolve, 5_000)
      timeout.unref?.()
      killer.once('exit', () => {
        clearTimeout(timeout)
        resolve()
      })
      killer.once('error', () => {
        clearTimeout(timeout)
        resolve()
      })
    })
    child.unref()
    return
  }
  child.kill('SIGTERM')
}

async function cleanupNextDistDir() {
  if (!process.env.MOCHI_NEXT_DIST_DIR) {
    await fs.promises.rm(path.join(WEB_DIR, NEXT_DIST_DIR), { recursive: true, force: true })
  }
}

async function waitForServer(url) {
  const startedAt = Date.now()
  while (Date.now() - startedAt < 45_000) {
    try {
      const response = await fetch(url)
      if (response.ok) return
    } catch {
      // The Next development server is still compiling the production fixture.
    }
    await new Promise((resolve) => setTimeout(resolve, 250))
  }
  throw new Error(`Timed out waiting for ${url}`)
}

function requirePlaywright() {
  const candidates = [
    path.join(WEB_DIR, 'node_modules'),
    'C:/Users/xu/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules',
  ]
  for (const candidate of candidates) {
    try {
      return require(require.resolve('playwright', { paths: [candidate] }))
    } catch {
      // Try the next explicit runtime location.
    }
  }
  throw new Error('Playwright is unavailable for the production TaskPanel fixture.')
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
    const url = new URL(request.url())
    const json = (body, status = 200) => route.fulfill({
      status,
      contentType: 'application/json',
      body: JSON.stringify(body),
    })

    if (request.method() === 'GET' && url.pathname === '/v1/tasks') return json([])
    if (request.method() === 'GET' && url.pathname === '/v1/sessions') return json([])
    if (request.method() === 'GET' && url.pathname === '/v1/projects') return json([])
    if (request.method() === 'GET' && url.pathname === '/v1/approvals') {
      return json(state.replacementCreated
        ? [
            approval({
              id: 'approval-parent',
              status: 'superseded',
              supersededBy: 'approval-replacement',
              files: parentApproval.file_change_groups[0].files,
            }),
            replacementApproval,
          ]
        : [parentApproval])
    }
    if (request.method() === 'POST' && url.pathname === '/v1/workspace/patch/subset-preview') {
      const body = JSON.parse(request.postData() ?? '{}')
      state.subsetRequests.push(body)
      if (body.approval_id === 'approval-parent') {
        assert.deepEqual(body, {
          approval_id: 'approval-parent',
          selected_entry_ids: ['entry-c'],
        })
        state.replacementCreated = true
        return json(subsetPreview({
          replacementApprovalId: 'approval-replacement',
          requestDigest: 'b'.repeat(64),
        }))
      }
      if (body.approval_id === 'approval-replacement') {
        assert.deepEqual(body, {
          approval_id: 'approval-replacement',
          selected_entry_ids: ['entry-c'],
        })
        return json({ detail: 'replacement conflicted' }, 409)
      }
    }
    return json({ detail: `Unhandled fixture request: ${request.method()} ${url.pathname}` }, 404)
  })
}

async function runBrowserAssertions() {
  const { chromium } = requirePlaywright()
  const executablePath = findChromiumExecutable(chromium)
  assert.ok(executablePath, 'No Chromium executable is available for the production TaskPanel fixture.')
  const browser = await chromium.launch({ headless: true, executablePath })
  try {
    for (const viewport of [{ width: 1440, height: 900 }, { width: 390, height: 844 }]) {
      const page = await browser.newPage({ viewport })
      page.setDefaultTimeout(10_000)
      console.error(`[file-workflow-fixture] viewport=${viewport.width} start`)
      const state = { replacementCreated: false, subsetRequests: [] }
      const consoleErrors = []
      const responseErrors = []
      page.on('console', (message) => {
        if (
          message.type() === 'error' &&
          !message.text().startsWith('Failed to load resource:')
        ) {
          consoleErrors.push(message.text())
        }
      })
      page.on('pageerror', (error) => consoleErrors.push(error.message))
      page.on('response', (resourceResponse) => {
        const resourceUrl = new URL(resourceResponse.url())
        const expectedSubsetConflict =
          resourceResponse.status() === 409 &&
          resourceUrl.pathname === '/v1/workspace/patch/subset-preview'
        if (resourceResponse.status() >= 400 && !expectedSubsetConflict) {
          responseErrors.push(`${resourceResponse.status()} ${resourceUrl.pathname}`)
        }
      })
      await installApiRoutes(page, state)
      const response = await page.goto(`${baseUrl}/test-fixtures/file-workflow`, { waitUntil: 'networkidle' })
      assert.ok(response?.ok(), `${viewport.width}px: production fixture did not load`)
      await page.getByTestId('subset-selection').filter({ visible: true }).waitFor()
      await page.getByTestId('subset-entry-entry-a').filter({ visible: true }).locator('input').uncheck()
      assert.equal(
        await page.getByTestId('subset-entry-entry-b').filter({ visible: true }).locator('input').isChecked(),
        false,
      )
      await page.getByTestId('subset-excluded').filter({ visible: true }).waitFor()
      await page
        .getByText('Digest:', { exact: false })
        .filter({ hasText: 'b'.repeat(64), visible: true })
        .waitFor()
      await page.getByTestId('subset-entry-entry-d').filter({ visible: true }).locator('input').uncheck()
      await page
        .getByText('it will not retry automatically.', { exact: false })
        .filter({ visible: true })
        .waitFor()
      await new Promise((resolve) => setTimeout(resolve, 300))
      assert.equal(state.subsetRequests.length, 2, 'a subset conflict must not auto-retry')
      assert.deepEqual(consoleErrors, [], `${viewport.width}px: browser errors`)
      assert.deepEqual(responseErrors, [], `${viewport.width}px: unexpected HTTP responses`)
      await page.close()
      console.error(`[file-workflow-fixture] viewport=${viewport.width} passed`)
    }
  } finally {
    await browser.close()
    console.error('[file-workflow-fixture] browser closed')
  }
}

async function main() {
  await configurePort()
  const child = startDevServer()
  child.once('exit', (code) => {
    if (!stoppingDevServer && code !== 0) {
      console.error(`Next development server exited with code ${code}`)
    }
  })
  try {
    await waitForServer(`${baseUrl}/test-fixtures/file-workflow`)
    await runBrowserAssertions()
    console.log('ok')
  } finally {
    console.error('[file-workflow-fixture] stopping Next')
    await stopDevServer(child)
    await cleanupNextDistDir()
    console.error('[file-workflow-fixture] cleanup complete')
  }
}

main().catch((error) => {
  console.error(error)
  process.exit(1)
})
