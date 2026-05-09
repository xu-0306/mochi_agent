'use client'

export function registerServiceWorker(): void {
  if (typeof window === 'undefined' || !('serviceWorker' in navigator)) {
    return
  }

  if (process.env.NODE_ENV !== 'production') {
    const unregisterDevelopmentWorkers = async () => {
      try {
        const registrations = await navigator.serviceWorker.getRegistrations()
        await Promise.all(registrations.map((registration) => registration.unregister()))
        if ('caches' in window) {
          const keys = await window.caches.keys()
          await Promise.all(
            keys
              .filter((key) => key.startsWith('mochi-web-'))
              .map((key) => window.caches.delete(key))
          )
        }
      } catch {
        // noop: local development should remain usable even if cleanup fails
      }
    }

    void unregisterDevelopmentWorkers()
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
