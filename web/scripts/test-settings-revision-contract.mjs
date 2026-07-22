import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

const api = readFileSync(new URL('../src/lib/api.ts', import.meta.url), 'utf8')
const settingsPage = readFileSync(new URL('../src/app/settings/page.tsx', import.meta.url), 'utf8')

assert.match(api, /export class SettingsRevisionConflict/)
assert.match(api, /revision: string/)
assert.match(api, /latestSettingsRevision = payload\.revision/)
assert.match(api, /expectedRevision: string \| null = latestSettingsRevision/)
assert.match(api, /'If-Match': `"\$\{expectedRevision\}"`/)
assert.match(api, /current_revision/)
assert.match(settingsPage, /error instanceof api\.SettingsRevisionConflict/)
assert.match(settingsPage, /Your unsaved edits were kept/)

console.log('settings revision contract checks passed')
