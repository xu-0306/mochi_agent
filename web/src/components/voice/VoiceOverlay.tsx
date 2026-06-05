'use client'

import * as React from 'react'
import { AlertTriangle, Brain, Loader2, Mic, MicOff, Volume2, XCircle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { useI18n } from '@/lib/i18n'
import { cn } from '@/lib/utils'
import { buildVoiceMeterAmplitudes } from '@/lib/voice-capture'
import { type VoiceRuntimePhase, type VoiceVadState } from '@/lib/voice-ws'
import { Waveform } from './Waveform'

interface VoiceOverlayProps {
  open: boolean
  phase: VoiceRuntimePhase
  isRecording: boolean
  inputLevel: number
  hasInputSignal: boolean
  microphoneLabel: string | null
  vadState: VoiceVadState | null
  partialTranscription: string
  finalTranscription: string
  assistantText: string
  captureWarning: string | null
  errorMessage: string | null
  onToggleRecording: () => void
  onInterrupt: () => void
  onClose: () => void
}

function phaseLabel(phase: VoiceRuntimePhase, t: (key: string) => string): string {
  switch (phase) {
    case 'connecting':
      return t('chat.voice.phase.connecting')
    case 'ready':
      return t('chat.voice.phase.ready')
    case 'listening':
      return t('chat.voice.phase.listening')
    case 'transcribing':
      return t('chat.voice.phase.transcribing')
    case 'thinking':
      return t('chat.voice.phase.thinking')
    case 'synthesizing':
      return t('chat.voice.phase.synthesizing')
    case 'error':
      return t('chat.voice.phase.error')
    default:
      return t('chat.voice.phase.idle')
  }
}

export function VoiceOverlay({
  open,
  phase,
  isRecording,
  inputLevel,
  hasInputSignal,
  microphoneLabel,
  vadState,
  partialTranscription,
  finalTranscription,
  assistantText,
  captureWarning,
  errorMessage,
  onToggleRecording,
  onInterrupt,
  onClose,
}: VoiceOverlayProps) {
  const { t } = useI18n()
  const activeTranscript = partialTranscription || finalTranscription
  const showSpinner = phase === 'connecting' || phase === 'transcribing'
  const showThinking = phase === 'thinking'
  const showSynthesis = phase === 'synthesizing'
  const waveformAmplitudes = React.useMemo(
    () => buildVoiceMeterAmplitudes(inputLevel),
    [inputLevel]
  )

  return (
    <Dialog open={open} onOpenChange={(nextOpen) => (!nextOpen ? onClose() : undefined)}>
      <DialogContent className="max-w-xl p-0">
        <DialogHeader className="border-b border-border px-5 pt-5 pb-4">
          <DialogTitle className="flex items-center gap-2">
            <Mic className="h-4 w-4 text-primary-400" />
            {t('chat.voice.title')}
          </DialogTitle>
          <DialogDescription className="text-xs">
            {phaseLabel(phase, t)}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 px-5 py-4">
          <div className="rounded-lg border border-border bg-surface-layer px-4 py-4">
            <Waveform
              amplitudes={isRecording ? waveformAmplitudes : undefined}
              isActive={isRecording || showSpinner || showSynthesis}
              className="mx-auto"
            />
            <div className="mt-3 flex items-center justify-center gap-2 text-xs text-muted-foreground">
              {showSpinner ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : null}
              {showThinking ? <Brain className="h-3.5 w-3.5 text-primary-400" /> : null}
              {showSynthesis ? <Volume2 className="h-3.5 w-3.5 text-primary-400" /> : null}
              <span>{phaseLabel(phase, t)}</span>
            </div>
            <div className="mt-2 flex flex-wrap items-center justify-center gap-2 text-[11px] text-muted-foreground">
              <span className="rounded-full border border-border px-2 py-1">
                {microphoneLabel
                  ? `${t('chat.voice.microphone')}: ${microphoneLabel}`
                  : t('chat.voice.deviceUnknown')}
              </span>
              {isRecording ? (
                <span className="rounded-full border border-border px-2 py-1">
                  {hasInputSignal
                    ? t('chat.voice.signalDetected')
                    : t('chat.voice.signalWaiting')}
                </span>
              ) : null}
              {vadState === 'speech_started' ? (
                <span className="rounded-full border border-border px-2 py-1">
                  {t('chat.voice.vadDetected')}
                </span>
              ) : null}
            </div>
          </div>

          {captureWarning ? (
            <div className="flex items-start gap-2 rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-sm text-foreground">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-300" />
              <span className="break-words">{captureWarning}</span>
            </div>
          ) : null}

          <div className="space-y-2">
            <p className="text-[11px] uppercase text-muted-foreground">{t('chat.voice.transcription')}</p>
            <div
              className={cn(
                'min-h-16 rounded-lg border border-border px-3 py-2 text-sm text-foreground',
                activeTranscript ? 'bg-surface-layer' : 'bg-canvas text-muted-foreground',
              )}
            >
              {activeTranscript || t('chat.voice.startPrompt')}
            </div>
          </div>

          <div className="space-y-2">
            <p className="text-[11px] uppercase text-muted-foreground">{t('chat.voice.assistant')}</p>
            <div
              className={cn(
                'min-h-16 rounded-lg border border-border px-3 py-2 text-sm text-foreground',
                assistantText ? 'bg-surface-layer' : 'bg-canvas text-muted-foreground',
              )}
            >
              {assistantText || t('chat.voice.waiting')}
            </div>
          </div>

          {errorMessage ? (
            <div className="flex items-start gap-2 rounded-lg border border-error/40 bg-error/10 px-3 py-2 text-sm text-foreground">
              <XCircle className="mt-0.5 h-4 w-4 shrink-0 text-error" />
              <span className="break-words">{errorMessage}</span>
            </div>
          ) : null}
        </div>

        <DialogFooter className="border-t border-border px-5 py-4 sm:justify-between">
          <Button variant="ghost" size="sm" onClick={onInterrupt}>
            {t('chat.voice.interrupt')}
          </Button>
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" onClick={onClose}>
              {t('chat.voice.close')}
            </Button>
            <Button variant={isRecording ? 'destructive' : 'primary'} size="sm" onClick={onToggleRecording}>
              {isRecording ? (
                <>
                  <MicOff className="h-3.5 w-3.5" />
                  {t('chat.voice.stop')}
                </>
              ) : (
                <>
                  <Mic className="h-3.5 w-3.5" />
                  {t('chat.voice.record')}
                </>
              )}
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
