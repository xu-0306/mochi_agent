'use client'

export type UICodeTheme =
  | 'vscode-dark-plus'
  | 'github-dark'
  | 'monokai'
  | 'vscode-light-plus'
  | 'github-light'

export const DEFAULT_CODE_THEME: UICodeTheme = 'vscode-dark-plus'

export const CODE_THEME_OPTIONS: Array<{ value: UICodeTheme; label: string }> = [
  { value: 'vscode-dark-plus', label: 'VS Code Dark+' },
  { value: 'github-dark', label: 'GitHub Dark' },
  { value: 'monokai', label: 'Monokai' },
  { value: 'vscode-light-plus', label: 'VS Code Light+' },
  { value: 'github-light', label: 'GitHub Light' },
]

export function normalizeCodeTheme(value: unknown): UICodeTheme {
  return CODE_THEME_OPTIONS.some((option) => option.value === value)
    ? (value as UICodeTheme)
    : DEFAULT_CODE_THEME
}
