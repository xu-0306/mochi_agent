'use client'

import Link from 'next/link'
import * as React from 'react'
import { ArrowLeft, BookOpen, ExternalLink } from 'lucide-react'
import { SettingsNav } from '@/components/settings/SettingsNav'
import { Button } from '@/components/ui/button'
import { useI18n } from '@/lib/i18n'

function GuideCodeBlock({
  title,
  lines,
}: {
  title: string
  lines: string[]
}) {
  return (
    <section className="rounded-lg border border-border bg-surface-layer">
      <div className="border-b border-border px-4 py-3">
        <h2 className="text-sm font-semibold text-foreground">{title}</h2>
      </div>
      <pre className="overflow-x-auto px-4 py-4 text-xs text-foreground">
        <code>{lines.join('\n')}</code>
      </pre>
    </section>
  )
}

export default function DiscordGuidePage() {
  const { t } = useI18n()

  const steps = Array.from({ length: 8 }, (_, index) =>
    t(`discordGuide.step${index + 1}`)
  )

  const powershellLines = [
    '$env:DISCORD_BOT_TOKEN="your-token"',
    'uv sync --extra channels --active',
    'uv run mochi channels run',
  ]

  const configLines = [
    'channels:',
    '  discord:',
    '    enabled: true',
    '    text_enabled: true',
    '    voice_enabled: true',
    '    bot_token: null',
    '    message_mode: "mentions_only"',
    '    auto_join_policy: "manual_only"',
    '    voice_auto_reply: true',
    '    voice_stt_enabled: true',
    '    voice_tts_enabled: true',
    'voice:',
    '  enabled: true',
    '  stt_backend: "faster-whisper"',
    '  tts_backend: "kokoro-tts"',
    '  sample_rate: 16000',
    '  channels: 1',
  ]

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <header className="shrink-0 border-b border-border px-6 pb-4 pt-5">
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <BookOpen className="h-5 w-5 text-muted-foreground" />
              <h1 className="text-xl font-bold text-foreground">{t('discordGuide.title')}</h1>
            </div>
            <p className="mt-1 text-sm text-muted-foreground">
              {t('discordGuide.subtitle')}
            </p>
          </div>
          <Button asChild type="button" variant="secondary" size="sm">
            <Link href="/settings?tab=channels">
              <ArrowLeft className="h-3.5 w-3.5" />
              {t('discordGuide.backToSettings')}
            </Link>
          </Button>
        </div>
      </header>

      <div className="flex-1 overflow-y-auto px-6 py-5">
        <div className="flex gap-6">
          <div className="w-40 shrink-0">
            <SettingsNav active="discord-guide" includeGuideLink />
          </div>

          <article className="min-w-0 flex-1 space-y-4">
            <section className="rounded-lg border border-border bg-surface-layer">
              <div className="border-b border-border px-4 py-3">
                <h2 className="text-sm font-semibold text-foreground">{t('discordGuide.stepsTitle')}</h2>
              </div>
              <ol className="list-decimal space-y-2 px-8 py-4 text-sm text-foreground">
                {steps.map((step) => (
                  <li key={step}>{step}</li>
                ))}
              </ol>
            </section>

            <GuideCodeBlock title={t('discordGuide.powershellTitle')} lines={powershellLines} />
            <GuideCodeBlock title={t('discordGuide.configTitle')} lines={configLines} />

            <section className="rounded-lg border border-border bg-surface-layer px-4 py-4">
              <p className="text-sm text-muted-foreground">
                {t('discordGuide.footerNote')}
              </p>
              <div className="mt-3">
                <a
                  href="https://discord.com/developers/applications"
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-2 text-sm font-medium text-primary-500 hover:text-primary-400"
                >
                  <ExternalLink className="h-3.5 w-3.5" />
                  Discord Developer Portal
                </a>
              </div>
            </section>
          </article>
        </div>
      </div>
    </div>
  )
}
