import assert from 'node:assert/strict'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

const moduleUrl = pathToFileURL(
  path.join(process.cwd(), 'src/lib/current-model-selection.ts')
).href

const { resolvePreferredCurrentModelId } = await import(moduleUrl)

assert.equal(
  resolvePreferredCurrentModelId(
    {
      configuredModel: 'openai_compat:gpt-5.4-mini',
      activeModelId: 'ollama:qwen3.5:4b',
    },
    [
      { id: 'ollama:qwen3.5:4b' },
      { id: 'openai_compat:gpt-5.4-mini' },
    ]
  ),
  'ollama:qwen3.5:4b'
)

assert.equal(
  resolvePreferredCurrentModelId(
    {
      configuredModel: 'openai_compat:gpt-5.4-mini',
      activeModelId: 'qwen3.5:4b',
    },
    [
      { id: 'ollama:qwen3.5:4b' },
      { id: 'openai_compat:gpt-5.4-mini' },
    ]
  ),
  'ollama:qwen3.5:4b'
)

assert.equal(
  resolvePreferredCurrentModelId(
    {
      configuredModel: 'openai_compat:gpt-5.4-mini',
      activeModelId: null,
    },
    [
      { id: 'ollama:qwen3.5:4b' },
      { id: 'openai_compat:gpt-5.4-mini' },
    ]
  ),
  'openai_compat:gpt-5.4-mini'
)

console.log('ok')
