import type { SessionSecurityOverride } from '@/lib/api'

export function resolveMaterializedSecurityOverride(
  detailOverride: SessionSecurityOverride | null,
  createdOverride: SessionSecurityOverride | null
): SessionSecurityOverride | null {
  return detailOverride ?? createdOverride
}
