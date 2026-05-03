'use client'

export function registerServiceWorker(): void {
  if (typeof window === 'undefined' || !('serviceWorker' in navigator)) {
    return
  }

  const register = async () => {
    try {
      await navigator.serviceWorker.register('/sw.js', {
        scope: '/',
        updateViaCache: 'none',
      })
    } catch {
      // noop: keep app usable even when SW registration fails
    }
  }

  if (document.readyState === 'complete') {
    void register()
    return
  }

  const onLoad = () => {
    void register()
  }
  window.addEventListener('load', onLoad, { once: true })
}
