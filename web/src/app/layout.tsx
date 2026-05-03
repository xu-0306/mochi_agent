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
  const themeBootstrapScript = `
    try {
      var storageKey = 'mochi.ui.preferences.v1';
      var raw = window.localStorage.getItem(storageKey);
      var parsed = raw ? JSON.parse(raw) : null;
      var mode = parsed && typeof parsed === 'object'
        ? (parsed.appearanceMode ?? parsed.appearance_mode ?? parsed.appearance ?? parsed.theme ?? parsed.colorScheme)
        : null;
      var theme = mode === 'dark' || mode === 'light'
        ? mode
        : (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
      document.documentElement.dataset.theme = theme;
      document.documentElement.style.colorScheme = theme;
      var metaTheme = document.querySelector('meta[name="theme-color"]');
      if (metaTheme) {
        metaTheme.setAttribute('content', theme === 'dark' ? '#0B0B0F' : '#FBFBFC');
      }
    } catch {}
  `

  return (
    <html
      lang="en"
      data-theme="light"
      suppressHydrationWarning
    >
      <body
        className="h-screen overflow-hidden bg-canvas text-foreground font-sans antialiased"
        suppressHydrationWarning
      >
        <Script id="mochi-theme-bootstrap" strategy="beforeInteractive">
          {themeBootstrapScript}
        </Script>
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
