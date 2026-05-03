import * as React from 'react'
import { cn } from '@/lib/utils'

interface WaveformProps {
  /** Array of amplitude values 0-1 for each bar. Defaults to static demo heights. */
  amplitudes?: number[]
  isActive?: boolean
  barCount?: number
  className?: string
}

const STATIC_AMPLITUDES = [
  0.2, 0.4, 0.6, 0.8, 0.5, 0.9, 0.7, 0.3, 0.6, 0.8,
  0.4, 0.7, 1.0, 0.6, 0.8, 0.9, 0.7, 1.0, 0.8, 0.5,
  0.9, 0.7, 0.4, 0.6, 0.8, 0.5, 0.3, 0.6, 0.4, 0.2,
  0.5, 0.3,
]

export function Waveform({ amplitudes = STATIC_AMPLITUDES, isActive = false, barCount = 32, className }: WaveformProps) {
  const bars = amplitudes.slice(0, barCount)

  const minHeight = 4
  const maxHeight = 40

  return (
    <div
      className={cn(
        'flex items-center justify-center gap-[3px]',
        isActive && 'animate-pulse-glow rounded-full',
        className
      )}
      aria-hidden="true"
    >
      <svg
        width={barCount * 5}
        height={48}
        viewBox={`0 0 ${barCount * 5} 48`}
        className="overflow-visible"
      >
        {bars.map((amp, i) => {
          const h = minHeight + amp * (maxHeight - minHeight)
          const y = (48 - h) / 2
          return (
            <rect
              key={i}
              x={i * 5}
              y={y}
              width={2}
              height={h}
              rx={1}
              className={cn(
                isActive ? 'fill-accent' : 'fill-muted-foreground/40'
              )}
            />
          )
        })}
      </svg>
    </div>
  )
}
