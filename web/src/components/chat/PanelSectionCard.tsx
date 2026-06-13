'use client'

import * as React from 'react'

interface PanelSectionCardProps {
  title: string
  description?: string
  children: React.ReactNode
}

export function PanelSectionCard({ title, description, children }: PanelSectionCardProps) {
  return (
    <section className="rounded-2xl border border-white/8 bg-canvas/70 p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)] backdrop-blur-sm">
      <div className="mb-3">
        <h3 className="text-sm font-semibold text-foreground">{title}</h3>
        {description ? (
          <p className="mt-1 text-[11px] leading-relaxed text-muted-foreground">{description}</p>
        ) : null}
      </div>
      {children}
    </section>
  )
}
