import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { createRequire } from 'node:module'
import { createServer } from 'node:net'
import fs from 'node:fs'
import path from 'node:path'

const WEB_DIR = process.cwd()
const configuredPort = Number(process.env.MOCHI_SUBAGENT_FIXTURE_PORT ?? 0)
let PORT = Number.isInteger(configuredPort) && configuredPort > 0 ? configuredPort : 0
let BASE_URL = ''
let FIXTURE_URL = ''
const NEXT_DIST_DIR = process.env.MOCHI_NEXT_DIST_DIR ?? '.next-fixture-subagent'
const require = createRequire(import.meta.url)
let stoppingDevServer = false

function reportPhase(phase) {
  console.error(`[subagent-browser-fixture] phase=${phase}`)
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
  FIXTURE_URL = `${BASE_URL}/test-fixtures/subagent-runtime`
}

function requireLocalPlaywright() {
  const candidates = [
    path.join(WEB_DIR, 'node_modules'),
    'C:/Users/Xu/AppData/Local/npm-cache/_npx/e41f203b7505f1fb/node_modules',
  ]
  for (const candidate of candidates) {
    try {
      const resolved = require.resolve('playwright', { paths: [candidate] })
      return require(resolved)
    } catch {
      // Try the next known local install.
    }
  }
  throw new Error('Playwright is not available locally. Install or expose a local playwright package before running this fixture.')
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

async function waitForServer(url, timeoutMs = 45_000) {
  const startedAt = Date.now()
  let lastError = null
  while (Date.now() - startedAt < timeoutMs) {
    const controller = new AbortController()
    const probeTimeout = setTimeout(() => controller.abort(), 2_000)
    try {
      const response = await fetch(url, { method: 'GET', signal: controller.signal })
      if (response.ok) {
        return
      }
      lastError = new Error(`HTTP ${response.status}`)
    } catch (error) {
      lastError = error
    } finally {
      clearTimeout(probeTimeout)
    }
    await new Promise((resolve) => setTimeout(resolve, 500))
  }
  throw new Error(`Timed out waiting for ${url}: ${lastError?.message ?? 'unknown error'}`)
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
  let output = ''
  child.stdout.on('data', (chunk) => {
    output += chunk.toString()
  })
  child.stderr.on('data', (chunk) => {
    output += chunk.toString()
  })
  child.on('exit', (code, signal) => {
    if (stoppingDevServer) {
      return
    }
    if (code !== null && code !== 0) {
      console.error(output)
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
  if (child.stdout) {
    child.stdout.destroy()
  }
  if (child.stderr) {
    child.stderr.destroy()
  }
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

async function assertNoHorizontalOverflow(page, label) {
  const overflow = await page.evaluate(() => ({
    documentWidth: document.documentElement.scrollWidth,
    viewportWidth: document.documentElement.clientWidth,
    bodyWidth: document.body.scrollWidth,
    offenders: Array.from(document.querySelectorAll('body *'))
      .map((element) => {
        const rect = element.getBoundingClientRect()
        return {
          tag: element.tagName,
          className: typeof element.className === 'string' ? element.className.slice(0, 160) : '',
          text: element.textContent?.trim().slice(0, 80) ?? '',
          left: Math.round(rect.left),
          right: Math.round(rect.right),
          width: Math.round(rect.width),
        }
      })
      .filter((item) => item.width > 0 && (item.left < -2 || item.right > window.innerWidth + 2))
      .slice(0, 8),
  }))
  assert.ok(
    overflow.documentWidth <= overflow.viewportWidth + 2,
    `${label}: document has horizontal overflow ${JSON.stringify(overflow)}`
  )
  assert.ok(
    overflow.bodyWidth <= overflow.viewportWidth + 2,
    `${label}: body has horizontal overflow ${JSON.stringify(overflow)}`
  )
}

async function assertNoNestedInteractiveControls(page, label) {
  const nested = await page.evaluate(() => {
    const selectors = [
      'button button',
      'button a',
      'a button',
      'button textarea',
      'button input',
      'button select',
    ]
    return selectors.flatMap((selector) =>
      Array.from(document.querySelectorAll(selector)).map((element) => ({
        selector,
        text: element.textContent?.trim().slice(0, 80) ?? '',
      }))
    )
  })
  assert.deepEqual(nested, [], `${label}: nested interactive controls found`)
}

async function assertElementWithinViewport(page, selector, label) {
  const result = await page.locator(selector).first().evaluate((element) => {
    const rect = element.getBoundingClientRect()
    return {
      left: rect.left,
      right: rect.right,
      top: rect.top,
      bottom: rect.bottom,
      width: rect.width,
      height: rect.height,
      viewportWidth: window.innerWidth,
      viewportHeight: window.innerHeight,
    }
  })
  assert.ok(result.width > 0 && result.height > 0, `${label}: element is empty`)
  assert.ok(result.left >= -2, `${label}: element overflows left ${JSON.stringify(result)}`)
  assert.ok(result.right <= result.viewportWidth + 2, `${label}: element overflows right ${JSON.stringify(result)}`)
  assert.ok(result.top >= -2, `${label}: element overflows top ${JSON.stringify(result)}`)
  assert.ok(result.bottom <= result.viewportHeight + 2, `${label}: element overflows bottom ${JSON.stringify(result)}`)
}

async function assertNoTabs(page, label) {
  const tabCount = await page.locator('[role="tablist"], [role="tab"], [role="tabpanel"]').count()
  assert.equal(tabCount, 0, `${label}: subagent side thread should not render tab controls`)
}

async function assertPromptCardExpands(page, label) {
  const dialog = page.locator('[role="dialog"]')
  await dialog.getByRole('button', { name: /Prompt/i }).first().click()
  await dialog.getByText('approval-fixture-super-long-identifier', { exact: false }).first().waitFor()
  await assertNoHorizontalOverflow(page, `${label} prompt card`)
}

async function assertOutputCardWraps(page, label) {
  const dialog = page.locator('[role="dialog"]')
  await dialog.getByText('output-preview-super-long-token', { exact: false }).waitFor()
  await assertNoHorizontalOverflow(page, `${label} output card`)
}

async function assertGuidanceWorks(page, label) {
  const textbox = page.getByPlaceholder('Ask for follow-up changes')
  await textbox.fill(`fixture guidance ${label}`)
  await page.getByRole('button', { name: /Send follow-up guidance/i }).click()
  await page.getByText(`fixture guidance ${label}`).waitFor()
}

async function assertDeliveryStatesRender(page, label) {
  const dialog = page.locator('[role="dialog"]')
  await dialog.getByText('Prioritize the transcript delivery state before output polish.', { exact: true }).waitFor()
  await dialog.getByText('applied', { exact: true }).waitFor()
  await dialog.getByText('queued', { exact: true }).waitFor()
  await dialog.getByText('deferred', { exact: true }).waitFor()
  await assertNoHorizontalOverflow(page, `${label} delivery states`)
}

async function assertCancellationProtocolStatesRender(page, label) {
  const response = await page.goto(FIXTURE_URL, { waitUntil: 'domcontentloaded' })
  assert.ok(response?.ok(), `${label}: fixture reload for cancellation states failed`)
  await page.getByText('Subagent runtime fixture', { exact: true }).waitFor({ timeout: 8_000 })
  await page.locator('[role="dialog"]').waitFor()
  await page.keyboard.press('Escape')
  await page.locator('[role="dialog"]').waitFor({ state: 'hidden' })
  await page.locator('button:visible').filter({ hasText: 'Cancellation observer' }).first().click()
  const dialog = page.locator('[role="dialog"]')
  await dialog.waitFor()
  await dialog.getByText('Cancellation observer', { exact: true }).waitFor()
  for (const text of [
    'Tool cancel requested',
    'cancel requested',
    'Tool cancel deferred',
    'cancel deferred',
    'Tool cancelled',
    'Thread interrupted',
    'interrupted',
    'Cancel the current tool and pause for operator review.',
    'cancelled',
  ]) {
    const item = dialog.getByText(text, { exact: true }).first()
    await item.scrollIntoViewIfNeeded()
    await item.waitFor()
  }
  await assertNoHorizontalOverflow(page, `${label} cancellation states`)
}

async function assertExpandableCards(page, label) {
  await page.keyboard.press('Escape')
  await page.locator('[role="dialog"]').waitFor({ state: 'hidden' })
  await page.getByText('Execution highlights', { exact: true }).first().waitFor()
  await page.getByRole('button', { name: /Collapse Approval parity researcher/i }).first().click()
  await page.getByRole('button', { name: /Expand Approval parity researcher/i }).first().waitFor()
  await page.getByRole('button', { name: /Expand Approval parity researcher/i }).first().click()
  await page.getByText('Execution highlights', { exact: true }).first().waitFor()
  await assertNoNestedInteractiveControls(page, `${label} expandable cards`)
  await page.getByRole('button', { name: /Open Approval parity researcher/i }).first().click()
  await page.locator('[role="dialog"]').getByText('Waiting for approval', { exact: true }).waitFor()
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
    return page.locator('body').innerText({ timeout: 2_000 })
      .catch(() => '')
      .then((body) => {
        throw new Error(`${label}: visible text not found: ${text}. body=${JSON.stringify(body.slice(0, 1200))}. ${error.message}`)
      })
  })
}

async function assertApprovalLinkOpensReview(page, label) {
  const dialog = page.locator('[role="dialog"]')
  await dialog.getByText('Waiting for approval', { exact: true }).waitFor()
  const reviewButton = dialog.getByRole('button', { name: /Review approvals?/i }).first()
  await reviewButton.click()
  try {
    await waitForVisibleText(page, 'Focused approvals', label)
    await waitForVisibleText(page, 'No matching pending approvals.', label)
  } catch {
    await assertNoHorizontalOverflow(page, `${label} approval review fallback`)
    return
  }
  await assertNoHorizontalOverflow(page, `${label} approval review`)
  await assertNoNestedInteractiveControls(page, `${label} approval review`)
}

async function runViewport(page, viewport) {
  const label = `${viewport.width}x${viewport.height}`
  const consoleMessages = []
  const badResponses = []
  page.on('console', (message) => {
    if (message.type() === 'warning' || message.type() === 'error') {
      consoleMessages.push(`${message.type()}: ${message.text()}`)
    }
  })
  page.on('pageerror', (error) => {
    consoleMessages.push(`pageerror: ${error.message}`)
  })
  page.on('response', (response) => {
    const status = response.status()
    const url = response.url()
    if (status >= 400 && !url.endsWith('/favicon.ico')) {
      badResponses.push(`${status}: ${url}`)
    }
  })

  await page.route('**/favicon.ico', (route) =>
    route.fulfill({
      status: 204,
      body: '',
    })
  )
  await page.route('**/v1/**', (route) => {
    const url = new URL(route.request().url())
    const pathname = url.pathname
    let body = {}
    if (
      pathname.endsWith('/sessions') ||
      pathname.endsWith('/projects') ||
      pathname.endsWith('/goals') ||
      pathname.endsWith('/tasks') ||
      pathname.endsWith('/approvals') ||
      pathname.endsWith('/agent-runs')
    ) {
      body = []
    } else if (pathname.endsWith('/settings')) {
      body = { ok: true }
    } else if (pathname.endsWith('/models')) {
      body = { configured: [], available: [] }
    }
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(body),
    })
  })

  await page.setViewportSize(viewport)
  const response = await page.goto(FIXTURE_URL, { waitUntil: 'domcontentloaded' })
  assert.ok(response, `${label}: fixture navigation did not return a response`)
  assert.ok(response.ok(), `${label}: fixture navigation failed with HTTP ${response.status()}`)
  try {
    await page.getByText('Subagent runtime fixture', { exact: true }).waitFor({ timeout: 8_000 })
  } catch (error) {
    const title = await page.title().catch(() => '')
    const body = await page.locator('body').innerText({ timeout: 2_000 }).catch(() => '')
    throw new Error(`${label}: fixture heading did not render. title=${JSON.stringify(title)} body=${JSON.stringify(body.slice(0, 1000))}`)
  }
  await page.getByText('Waiting for approval', { exact: true }).waitFor()
  const dialog = page.locator('[role="dialog"]')
  await dialog.getByText('ID exec-approval-fixture-primary', { exact: false }).waitFor()
  await dialog.getByPlaceholder('Ask for follow-up changes').waitFor()

  await assertElementWithinViewport(page, '[role="dialog"]', `${label} drawer`)
  await assertNoTabs(page, label)
  await assertNoHorizontalOverflow(page, label)
  await assertNoNestedInteractiveControls(page, label)
  await assertDeliveryStatesRender(page, label)

  const approvalBanner = dialog.getByText('Waiting for approval', { exact: true }).locator('xpath=ancestor::section[1]')
  const bannerBox = await approvalBanner.boundingBox()
  assert.ok(bannerBox && bannerBox.width <= viewport.width + 2, `${label}: approval banner should fit viewport`)

  await assertExpandableCards(page, label)
  await assertPromptCardExpands(page, label)
  await assertOutputCardWraps(page, label)
  await assertGuidanceWorks(page, label)
  await assertApprovalLinkOpensReview(page, label)
  await assertCancellationProtocolStatesRender(page, label)
  await assertNoNestedInteractiveControls(page, `${label} after interactions`)

  assert.deepEqual(badResponses, [], `${label}: HTTP errors`)
  assert.deepEqual(consoleMessages, [], `${label}: console warnings/errors`)
}

async function main() {
  reportPhase('configure-port')
  await configurePort()
  let server = null
  reportPhase('start-next-dev-server')
  server = startDevServer()
  try {
    await waitForServer(FIXTURE_URL)
    reportPhase('next-dev-server-ready')

    const { chromium } = await requireLocalPlaywright()
    const executablePath = findChromiumExecutable()
    const browser = await chromium.launch({
      headless: true,
      executablePath,
    })
    reportPhase('browser-launched')
    try {
      for (const viewport of [
        { width: 390, height: 844 },
        { width: 768, height: 900 },
        { width: 1440, height: 1000 },
      ]) {
        const page = await browser.newPage()
        try {
          await runViewport(page, viewport)
        } finally {
          await page.close()
        }
      }
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
