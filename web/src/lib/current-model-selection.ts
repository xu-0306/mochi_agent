interface CurrentModelPayload {
  configuredModel?: string | null
  activeModelId?: string | null
}

interface CurrentModelOption {
  id: string
}

function normalizeModelId(value: string | null | undefined): string | null {
  if (typeof value !== 'string') {
    return null
  }
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : null
}

export function resolvePreferredCurrentModelId(
  payload: CurrentModelPayload,
  options: CurrentModelOption[]
): string | null {
  const activeModelId = normalizeModelId(payload.activeModelId)
  if (activeModelId) {
    const activeOption = options.find(
      (option) => option.id === activeModelId || option.id.endsWith(`:${activeModelId}`)
    )
    return activeOption?.id ?? activeModelId
  }

  const configuredModel = normalizeModelId(payload.configuredModel)
  if (configuredModel) {
    const configuredOption = options.find((option) => option.id === configuredModel)
    if (configuredOption) {
      return configuredOption.id
    }
  }

  return configuredModel ?? options[0]?.id ?? null
}
