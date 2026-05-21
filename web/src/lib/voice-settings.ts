export type VoiceReplyModelModeUi = 'global' | 'fixed'
export type VoiceSessionModeUi = 'shared' | 'isolated'

export function normalizeReplyModelModeFromApi(value: string | null | undefined): VoiceReplyModelModeUi {
  if (value === 'configured_model' || value === 'fixed') {
    return 'fixed'
  }
  return 'global'
}

export function normalizeReplyModelModeForApi(
  value: VoiceReplyModelModeUi
): 'inherit_active' | 'configured_model' {
  return value === 'fixed' ? 'configured_model' : 'inherit_active'
}

export function normalizeSessionModeFromApi(value: string | null | undefined): VoiceSessionModeUi {
  if (value === 'isolated_voice' || value === 'isolated' || value === 'voice_room') {
    return 'isolated'
  }
  return 'shared'
}

export function normalizeSessionModeForApi(
  value: VoiceSessionModeUi
): 'append_current' | 'isolated_voice' {
  return value === 'isolated' ? 'isolated_voice' : 'append_current'
}
