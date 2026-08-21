import type { Metadata, Viewport } from 'next'
import Script from 'next/script'
import '@/styles/globals.css'
import { Sidebar } from '@/components/sidebar/Sidebar'
import { AppClientBootstrap } from '@/components/app/AppClientBootstrap'
import { I18nProvider } from '@/lib/i18n'

export const metadata: Metadata = {
  title: 'Mochi - Personal AI Agent',
  description: 'Talk to your continuously learning AI agent across text, voice, and channels.',
  manifest: '/manifest.webmanifest',
  icons: {
    icon: '/favicon.ico',
  },
}

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  themeColor: '#0B0B0F',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  const serviceWorkerCleanupScript = process.env.NODE_ENV !== 'production'
    ? `
      try {
        if ('serviceWorker' in navigator) {
          navigator.serviceWorker.getRegistrations()
            .then(function (registrations) {
              return Promise.all(registrations.map(function (registration) {
                return registration.unregister();
              }));
            })
            .catch(function () {});
        }
        if ('caches' in window) {
          window.caches.keys()
            .then(function (keys) {
              return Promise.all(keys
                .filter(function (key) { return key.indexOf('mochi-web-') === 0; })
                .map(function (key) { return window.caches.delete(key); }));
            })
            .catch(function () {});
        }
      } catch {}
    `
    : null

  return (
    <html
      lang="en"
      data-theme="light"
      data-code-theme="vscode-dark-plus"
    >
      <body
        className="h-screen overflow-hidden bg-canvas text-foreground font-sans antialiased"
      >
        {serviceWorkerCleanupScript ? (
          <Script id="mochi-dev-service-worker-cleanup" strategy="beforeInteractive">
            {serviceWorkerCleanupScript}
          </Script>
        ) : null}
        <I18nProvider>
          <AppClientBootstrap />
          <div className="flex h-full">
            <Sidebar />
            <main className="flex-1 min-w-0 overflow-hidden flex flex-col">
              {children}
            </main>
          </div>
        </I18nProvider>
      </body>
    </html>
  )
}
