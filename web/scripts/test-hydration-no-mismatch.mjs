import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { createRequire } from 'node:module'
import fs from 'node:fs'
import path from 'node:path'

const WEB_DIR = process.cwd()
const ARTIFACT_PATH = path.resolve(WEB_DIR, '..', 'artifacts', 'goal-ui', 'hydration-no-mismatch.json')
const NEXT_DIST_DIR = process.env.MOCHI_NEXT_DIST_DIR ?? '.next-fixture-hydration'
const require = createRequire(import.meta.url)
let port = 0
let server = null
const linkedRunId = 'run / linked?&'

function sleep(ms) { return new Promise((resolve) => setTimeout(resolve, ms)) }

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
  const net = await import('node:net')
  const probe = net.createServer()
  await new Promise((resolve, reject) => { probe.once('error', reject); probe.listen(0, '127.0.0.1', resolve) })
  port = probe.address().port
  await new Promise((resolve) => probe.close(resolve))
}

function startServer() {
  const nextCli = require.resolve('next/dist/bin/next')
  return spawn(process.execPath, [nextCli, 'dev', '--hostname', '127.0.0.1', '--port', String(port)], {
    cwd: WEB_DIR,
    env: { ...process.env, NEXT_TELEMETRY_DISABLED: '1', MOCHI_NEXT_DIST_DIR: NEXT_DIST_DIR },
    shell: false,
    windowsHide: true,
    stdio: ['ignore', 'pipe', 'pipe'],
  })
}

async function stopServer(child) {
  if (!child || child.exitCode !== null) return
  if (process.platform === 'win32') {
    await new Promise((resolve) => {
      const killer = spawn('taskkill.exe', ['/PID', String(child.pid), '/T', '/F'], { stdio: 'ignore', windowsHide: true })
      killer.once('exit', resolve); killer.once('error', resolve)
    })
  } else child.kill('SIGTERM')
}

async function waitForServer(url) {
  const started = Date.now()
  while (Date.now() - started < 45_000) {
    try { if ((await fetch(url)).ok) return } catch { /* booting */ }
    await sleep(300)
  }
  throw new Error(`Timed out waiting for ${url}`)
}

function response(body) { return { status: 200, contentType: 'application/json', body: JSON.stringify(body) } }

async function mockApi(page) {
  await page.route('**/favicon.ico', (route) => route.fulfill({ status: 204, body: '' }))
  await page.route('**/v1/**', (route) => {
    const { pathname } = new URL(route.request().url())
    if (pathname === '/v1/sessions') return route.fulfill(response({ items: [] }))
    if (pathname.endsWith('/models')) return route.fulfill(response({ configured: [], available: [] }))
    if (pathname.endsWith('/settings')) return route.fulfill(response({ ok: true }))
    if (pathname === '/v1/goals/strategies' || pathname.endsWith('/goal-strategies')) return route.fulfill(response({ strategies: [] }))
    if (pathname === `\/v1\/agent-runs\/${encodeURIComponent(linkedRunId)}`) return route.fulfill(response({ run_id: linkedRunId, status: 'completed', objective: 'Hydration linked run fixture', events: [] }))
    if (pathname.startsWith(`\/v1\/agent-runs\/${encodeURIComponent(linkedRunId)}\/`)) return route.fulfill(response([]))
    return route.fulfill(response([]))
  })
}

function isHydrationMessage(text) {
  return /hydrated but some attributes|hydration failed|hydration mismatch|server rendered html.*didn.t match/i.test(text)
}

async function navigateAndCapture(context, url, label, expectedHtmlAttributes, expectedFinalPath = null) {
  const page = await context.newPage()
  const messages = []
  page.on('console', (message) => {
    if (message.type() === 'warning' || message.type() === 'error') messages.push(`${message.type()}: ${message.text()}`)
  })
  page.on('pageerror', (error) => messages.push(`pageerror: ${error.message}`))
  await mockApi(page)
  const assertAttributes = async () => {
    if (!expectedHtmlAttributes) { await page.waitForTimeout(500); return null }
    await page.waitForFunction((expected) => {
      const html = document.documentElement
      return html.lang === expected.lang
        && html.dataset.fontSize === expected.fontSize
        && html.dataset.theme === expected.theme
        && html.dataset.codeTheme === expected.codeTheme
        && html.style.colorScheme === expected.colorScheme
    }, expectedHtmlAttributes, { timeout: 10_000 })
    return await page.evaluate(() => {
      const html = document.documentElement
      return { lang: html.lang, fontSize: html.dataset.fontSize ?? null, theme: html.dataset.theme ?? null, codeTheme: html.dataset.codeTheme ?? null, colorScheme: html.style.colorScheme, bodyClass: document.body.className }
    })
  }
  await page.goto(url, { waitUntil: 'domcontentloaded' })
  if (expectedFinalPath) await page.waitForURL((nextUrl) => nextUrl.pathname === expectedFinalPath, { timeout: 10_000 })
  const firstNavigationAttributes = await assertAttributes()
  await page.reload({ waitUntil: 'domcontentloaded' })
  if (expectedFinalPath) await page.waitForURL((nextUrl) => nextUrl.pathname === expectedFinalPath, { timeout: 10_000 })
  const reloadAttributes = await assertAttributes()
  const actualPath = new URL(page.url()).pathname
  await page.close()
  return { label, actualPath, messages, hydrationMessages: messages.filter(isHydrationMessage), expectedHtmlAttributes, firstNavigationAttributes, reloadAttributes, pass: messages.length === 0 && (!expectedFinalPath || actualPath === expectedFinalPath) }
}

async function writeArtifact(payload) {
  await fs.promises.mkdir(path.dirname(ARTIFACT_PATH), { recursive: true })
  await fs.promises.writeFile(ARTIFACT_PATH, `${JSON.stringify(payload, null, 2)}\n`, 'utf8')
}

async function main() {
  await reservePort()
  const url = `http://127.0.0.1:${port}/`
  server = startServer()
  try {
    await waitForServer(url)
    const { chromium } = requirePlaywright()
    const browser = await chromium.launch({ headless: true, executablePath: chromiumExecutable() })
    try {
      const clean = await browser.newContext()
      const persisted = await browser.newContext()
      await persisted.addInitScript(() => {
        localStorage.setItem('mochi.ui.preferences.v1', JSON.stringify({
          languageMode: 'zh-TW',
          fontSize: 'large',
          appearanceMode: 'light',
          codeTheme: 'github-light',
          timezone: 'Asia/Taipei',
        }))
      })
      const persistedHtmlAttributes = { lang: 'zh-TW', fontSize: 'large', theme: 'light', codeTheme: 'github-light', colorScheme: 'light' }
      const routes = [
        { label: 'root', path: '/' },
        { label: 'agent-runs', path: '/agent-runs' },
        { label: 'agent-run-detail', path: `/agent-runs/${encodeURIComponent(linkedRunId)}` },
        { label: 'goals-redirect', path: '/goals', expectedFinalPath: '/agent-runs' },
      ]
      const scenarios = []
      for (const route of routes) {
        const routeUrl = new URL(route.path, url).toString()
        scenarios.push(await navigateAndCapture(clean, routeUrl, `clean:${route.label}:first-navigation-and-reload`, null, route.expectedFinalPath))
        scenarios.push(await navigateAndCapture(persisted, routeUrl, `persisted:${route.label}:first-navigation-and-reload`, persistedHtmlAttributes, route.expectedFinalPath))
      }
      await clean.close(); await persisted.close()
      const payload = { ok: scenarios.every((scenario) => scenario.pass), scenarios }
      await writeArtifact(payload)
      assert.equal(payload.ok, true, `console/page errors or warnings: ${JSON.stringify(scenarios)}`)
      console.log(JSON.stringify(payload, null, 2))
    } finally { await browser.close() }
  } finally { await stopServer(server) }
}

main().catch(async (error) => { await writeArtifact({ ok: false, error: error.message }).catch(() => {}); console.error(error); process.exit(1) })
