import * as React from 'react'
import { cva, type VariantProps } from 'class-variance-authority'
import { cn } from '@/lib/utils'

const badgeVariants = cva(
  'inline-flex items-center gap-1 rounded-full h-[22px] px-2 text-xs font-medium transition-colors duration-150',
  {
    variants: {
      variant: {
        neutral: 'bg-muted text-muted-foreground',
        primary: 'bg-primary-500/15 text-primary-400 border border-primary-500/30',
        success: 'bg-success/15 text-success border border-success/30',
        warning: 'bg-warning/15 text-warning border border-warning/30',
        error: 'bg-destructive/15 text-destructive border border-destructive/30',
        outline: 'border border-border text-muted-foreground bg-transparent',
        web: 'bg-primary-500/15 text-primary-400 border border-primary-500/30',
        cli: 'bg-muted text-muted-foreground border border-border',
        discord: 'bg-[#5865F2]/15 text-[#7B8AFF] border border-[#5865F2]/30',
        telegram: 'bg-[#0088CC]/15 text-[#29B5E8] border border-[#0088CC]/30',
      },
    },
    defaultVariants: {
      variant: 'neutral',
    },
  }
)

export interface BadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return <span className={cn(badgeVariants({ variant }), className)} {...props} />
}

export { Badge, badgeVariants }
