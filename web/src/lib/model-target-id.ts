/** Stable model identity helpers shared by model selectors.
 *
 * A display label is not an identity: providers can expose the same model
 * name, and labels may be translated or changed without changing the target.
 */

export type ModelIdentityLike = {
  id?: unknown
  target_id?: unknown
  targetId?: unknown
  provider?: unknown
  backend_type?: unknown
  backendType?: unknown
  model?: unknown
  name?: unknown
  label?: unknown
  model_spec?: unknown
  modelSpec?: unknown
  base_url?: unknown
  baseUrl?: unknown
  auth_profile_id?: unknown
  authProfileId?: unknown
}

function stringValue(value: unknown): string | null {
  return typeof value === 'string' && value.trim() ? value.trim() : null
}

function normalizeEndpoint(value: string | null): string {
  if (!value) return ''
  try {
    const parsed = new URL(value)
    parsed.protocol = parsed.protocol.toLowerCase()
    parsed.hostname = parsed.hostname.toLowerCase()
    parsed.username = ''
    parsed.password = ''
    parsed.pathname = parsed.pathname.replace(/\/+$/, '')
    const sensitiveQueryNames = new Set([
      'api_key',
      'apikey',
      'authorization',
      'key',
      'password',
      'secret',
      'token',
      'access_token',
      'refresh_token',
    ])
    for (const name of Array.from(parsed.searchParams.keys())) {
      if (sensitiveQueryNames.has(name.toLowerCase().replaceAll('-', '_'))) {
        parsed.searchParams.delete(name)
      }
    }
    parsed.hash = ''
    return parsed.toString().replace(/\/+$/, '')
  } catch {
    return value.replace(/\/+$/, '')
  }
}

function isDisplayLabel(id: string, model: ModelIdentityLike): boolean {
  const label = stringValue(model.label)
  const name = stringValue(model.name) ?? stringValue(model.model)
  const provider = stringValue(model.provider) ?? stringValue(model.backend_type) ?? stringValue(model.backendType)
  if (label && id === label) return true
  if (name && id === name) return true
  if (name && provider && id === name + ' (' + provider + ')') return true
  // Older Ollama entries used ollama:<model> and therefore collided when the
  // same model was served by more than one endpoint.
  if (name && provider && id === provider + ':' + name) return true
  // Rebuild readable aliases when the current model contains identity
  // dimensions that an older alias did not encode.
  if (id.startsWith('target:')) return true
  return false
}

/** Return the canonical target id used for React keys, Select values, and API actions. */
export function modelTargetId(model: ModelIdentityLike | null | undefined): string {
  if (!model) return ''

  const explicit = stringValue(model.target_id) ?? stringValue(model.targetId)
  if (explicit) return explicit

  const id = stringValue(model.id)
  const provider = stringValue(model.provider) ?? stringValue(model.backend_type) ?? stringValue(model.backendType)
  const name = stringValue(model.model) ?? stringValue(model.name)
  const modelSpec = normalizeEndpoint(stringValue(model.model_spec) ?? stringValue(model.modelSpec))
  const baseUrl = normalizeEndpoint(stringValue(model.base_url) ?? stringValue(model.baseUrl))
  const endpoint = baseUrl || modelSpec
  const authProfileId = stringValue(model.auth_profile_id) ?? stringValue(model.authProfileId)
  const backendType = stringValue(model.backend_type) ?? stringValue(model.backendType)

  // Keep existing opaque/canonical ids for compatibility. Legacy label-shaped
  // ids are rebuilt from provider + endpoint + model so duplicate labels cannot
  // collapse into one React child.
  if (id && !isDisplayLabel(id, model)) return id
  if (provider && endpoint && name) {
    const parts = [
      'target',
      encodeURIComponent(provider),
      encodeURIComponent(endpoint),
      encodeURIComponent(name),
    ]
    if (modelSpec && modelSpec !== endpoint) {
      parts.push('spec', encodeURIComponent(modelSpec))
    }
    if (backendType) {
      parts.push('backend', encodeURIComponent(backendType))
    }
    if (authProfileId) {
      parts.push('auth', encodeURIComponent(authProfileId))
    }
    return parts.join(':')
  }
  if (provider && name) {
    const parts = ['target', encodeURIComponent(provider), encodeURIComponent(name)]
    if (backendType) {
      parts.push('backend', encodeURIComponent(backendType))
    }
    if (authProfileId) {
      parts.push('auth', encodeURIComponent(authProfileId))
    }
    return parts.join(':')
  }
  return id ?? modelSpec ?? name ?? ''
}
