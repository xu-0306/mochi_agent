const CACHE_NAME = 'mochi-web-v2'
const APP_SHELL_URLS = ['/', '/settings', '/skills', '/manifest.webmanifest']
const LOCAL_DEV_HOSTS = new Set(['localhost', '127.0.0.1', '[::1]'])
const IS_LOCAL_DEV = LOCAL_DEV_HOSTS.has(self.location.hostname)

self.addEventListener('install', (event) => {
  if (IS_LOCAL_DEV) {
    self.skipWaiting()
    return
  }

  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(APP_SHELL_URLS))
  )
  self.skipWaiting()
})

self.addEventListener('activate', (event) => {
  if (IS_LOCAL_DEV) {
    event.waitUntil(
      caches.keys().then((keys) =>
        Promise.all(
          keys
            .filter((key) => key.startsWith('mochi-web-'))
            .map((key) => caches.delete(key))
        )
      ).then(() => self.registration.unregister())
    )
    self.clients.claim()
    return
  }

  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys
          .filter((key) => key !== CACHE_NAME)
          .map((key) => caches.delete(key))
      )
    )
  )
  self.clients.claim()
})

self.addEventListener('fetch', (event) => {
  if (IS_LOCAL_DEV) {
    return
  }

  const request = event.request

  if (request.method !== 'GET') {
    return
  }

  const url = new URL(request.url)
  const isSameOrigin = url.origin === self.location.origin
  if (!isSameOrigin) {
    return
  }

  const isApiRequest = url.pathname.startsWith('/v1/')
  const isStaticAsset = request.destination === 'style' ||
    request.destination === 'script' ||
    request.destination === 'font' ||
    request.destination === 'image' ||
    url.pathname.startsWith('/_next/')

  if (isApiRequest) {
    return
  }

  if (request.mode === 'navigate') {
    event.respondWith(
      fetch(request).catch(() => caches.match('/'))
    )
    return
  }

  if (isStaticAsset) {
    event.respondWith(
      caches.match(request).then((cached) => {
        if (cached) {
          return cached
        }
        return fetch(request).then((response) => {
          const shouldCache = response.ok
          if (shouldCache) {
            const copy = response.clone()
            void caches.open(CACHE_NAME).then((cache) => cache.put(request, copy))
          }
          return response
        })
      })
    )
  }
})
