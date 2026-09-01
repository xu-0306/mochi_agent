import assert from 'node:assert/strict'
import test from 'node:test'
import { modelTargetId } from './model-target-id.ts'

test('duplicate display labels resolve to distinct endpoint identities', () => {
  const first = modelTargetId({
    id: 'gpt-5.4 (openai_compat)',
    label: 'gpt-5.4 (openai_compat)',
    provider: 'openai_compat',
    model: 'gpt-5.4',
    model_spec: 'https://one.example/v1',
    base_url: 'https://one.example/v1',
  })
  const second = modelTargetId({
    id: 'gpt-5.4 (openai_compat)',
    label: 'gpt-5.4 (openai_compat)',
    provider: 'openai_compat',
    model: 'gpt-5.4',
    model_spec: 'https://two.example/v1',
    base_url: 'https://two.example/v1',
  })

  assert.notEqual(first, second)
  assert.notEqual(first, 'gpt-5.4 (openai_compat)')
})

test('explicit target_id wins over a mutable label', () => {
  assert.equal(
    modelTargetId({ target_id: 'model:openai_compat:abc', label: 'same label' }),
    'model:openai_compat:abc'
  )
})

test('legacy ollama model ids include the serving endpoint', () => {
  const first = modelTargetId({
    id: 'ollama:qwen2.5',
    label: 'qwen2.5',
    provider: 'ollama',
    model: 'qwen2.5',
    model_spec: 'ollama:qwen2.5',
    base_url: 'http://one.example:11434',
  })
  const second = modelTargetId({
    id: 'ollama:qwen2.5',
    label: 'qwen2.5',
    provider: 'ollama',
    model: 'qwen2.5',
    model_spec: 'ollama:qwen2.5',
    base_url: 'http://two.example:11434',
  })
  assert.notEqual(first, second)
})

test('OAuth profiles and model specs are part of the fallback identity', () => {
  const first = modelTargetId({
    id: 'gpt-5.4 (openai_codex)',
    label: 'gpt-5.4 (openai_codex)',
    provider: 'openai_codex',
    model: 'gpt-5.4',
    model_spec: 'https://chatgpt.com/backend-api',
    base_url: 'https://chatgpt.com/backend-api',
    auth_profile_id: 'profile-a',
  })
  const second = modelTargetId({
    id: 'gpt-5.4 (openai_codex)',
    label: 'gpt-5.4 (openai_codex)',
    provider: 'openai_codex',
    model: 'gpt-5.4',
    model_spec: 'https://chatgpt.com/backend-api',
    base_url: 'https://chatgpt.com/backend-api',
    auth_profile_id: 'profile-b',
  })
  assert.notEqual(first, second)
})

test('endpoint credentials are not copied into a fallback identity', () => {
  const target = modelTargetId({
    id: 'gpt-5.4 (openai_compat)',
    label: 'gpt-5.4 (openai_compat)',
    provider: 'openai_compat',
    model: 'gpt-5.4',
    model_spec: 'https://user:password@example.test/v1?api_key=query-secret&region=tw#fragment',
    base_url: 'https://user:password@example.test/v1?api_key=query-secret&region=tw#fragment',
  })
  assert.equal(target.includes('password'), false)
  assert.equal(target.includes('query-secret'), false)
  assert.equal(target.includes('fragment'), false)
  assert.equal(target.includes('region%3Dtw'), true)
})
