import { spawn } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'

const WEB_DIR = process.cwd()
const [fixtureArgument, distArgument] = process.argv.slice(2)
const DEFAULT_TIMEOUT_MS = 120_000
const CLEANUP_TIMEOUT_MS = 15_000
const TERMINATION_TIMEOUT_MS = 10_000

if (!fixtureArgument || !distArgument) {
  console.error('Usage: node run-browser-fixture.mjs <fixture-script> <owned-dist-dir>')
  process.exit(2)
}

const configuredTimeout = Number(process.env.MOCHI_BROWSER_FIXTURE_TIMEOUT_MS ?? DEFAULT_TIMEOUT_MS)
const timeoutMs =
  Number.isFinite(configuredTimeout) && configuredTimeout >= 30_000
    ? Math.min(configuredTimeout, 600_000)
    : DEFAULT_TIMEOUT_MS
const fixturePath = path.resolve(WEB_DIR, fixtureArgument)
const distPath = path.resolve(WEB_DIR, distArgument)
const relativeDistPath = path.relative(WEB_DIR, distPath)

if (
  !relativeDistPath ||
  relativeDistPath.startsWith(`..${path.sep}`) ||
  path.isAbsolute(relativeDistPath) ||
  !path.basename(distPath).startsWith('.next-fixture-')
) {
  console.error(`Refusing to manage unsafe browser fixture dist path: ${distPath}`)
  process.exit(2)
}

function withTimeout(promise, durationMs, label) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`${label} exceeded ${durationMs}ms`)), durationMs)
    timer.unref?.()
    promise.then(
      (value) => {
        clearTimeout(timer)
        resolve(value)
      },
      (error) => {
        clearTimeout(timer)
        reject(error)
      }
    )
  })
}

async function terminateProcessTree(child) {
  if (!child || child.exitCode !== null) {
    return
  }
  if (process.platform === 'win32') {
    await withTimeout(
      new Promise((resolve) => {
        const killer = spawn('taskkill.exe', ['/PID', String(child.pid), '/T', '/F'], {
          stdio: 'ignore',
          windowsHide: true,
        })
        killer.once('exit', resolve)
        killer.once('error', resolve)
      }),
      TERMINATION_TIMEOUT_MS,
      'browser fixture process-tree termination'
    ).catch((error) => {
      console.error(error.message)
      child.kill()
    })
    return
  }

  try {
    process.kill(-child.pid, 'SIGTERM')
  } catch {
    child.kill('SIGTERM')
  }
  await new Promise((resolve) => setTimeout(resolve, 800))
  if (child.exitCode === null) {
    try {
      process.kill(-child.pid, 'SIGKILL')
    } catch {
      child.kill('SIGKILL')
    }
  }
}

async function cleanupOwnedDist() {
  await withTimeout(
    fs.promises.rm(distPath, { recursive: true, force: true }),
    CLEANUP_TIMEOUT_MS,
    `browser fixture cleanup for ${relativeDistPath}`
  )
}

console.error(
  `[browser-fixture] start script=${path.relative(WEB_DIR, fixturePath)} ` +
    `dist=${relativeDistPath} timeout_ms=${timeoutMs}`
)

const child = spawn(process.execPath, [fixturePath], {
  cwd: WEB_DIR,
  env: {
    ...process.env,
    MOCHI_NEXT_DIST_DIR: relativeDistPath,
  },
  stdio: 'inherit',
  shell: false,
  windowsHide: true,
  detached: process.platform !== 'win32',
})

let timeoutHandle
const outcome = await Promise.race([
  new Promise((resolve) => {
    child.once('exit', (code, signal) => resolve({ kind: 'exit', code, signal }))
    child.once('error', (error) => resolve({ kind: 'error', error }))
  }),
  new Promise((resolve) => {
    timeoutHandle = setTimeout(() => resolve({ kind: 'timeout' }), timeoutMs)
  }),
])
clearTimeout(timeoutHandle)

let exitCode = 1
if (outcome.kind === 'timeout') {
  console.error(
    `[browser-fixture] timeout script=${path.relative(WEB_DIR, fixturePath)} ` +
      `after_ms=${timeoutMs}`
  )
  await terminateProcessTree(child)
  exitCode = 124
} else if (outcome.kind === 'error') {
  console.error(`[browser-fixture] spawn failed: ${outcome.error.message}`)
} else if (outcome.code === 0) {
  exitCode = 0
} else {
  console.error(
    `[browser-fixture] child failed code=${String(outcome.code)} signal=${String(outcome.signal)}`
  )
}

try {
  await cleanupOwnedDist()
  console.error(`[browser-fixture] cleanup complete dist=${relativeDistPath}`)
} catch (error) {
  console.error(`[browser-fixture] cleanup failed: ${error.message}`)
  exitCode = 1
}

process.exit(exitCode)
