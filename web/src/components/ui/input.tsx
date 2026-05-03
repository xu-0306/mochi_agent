'use client'

import * as React from 'react'
import { cva, type VariantProps } from 'class-variance-authority'
import { cn } from '@/lib/utils'

const inputVariants = cva(
  [
    'flex w-full rounded-md bg-surface-layer text-foreground',
    'border border-border',
    'placeholder:text-muted-foreground',
    'transition-all duration-150',
    'focus-visible:outline-none focus-visible:border-primary-500',
    'focus-visible:ring-[3px] focus-visible:ring-primary-500/25',
    'disabled:opacity-40 disabled:cursor-not-allowed',
    'file:border-0 file:bg-transparent file:text-sm file:font-medium',
  ],
  {
    variants: {
      size: {
        sm: 'h-9 px-3 text-sm',
        md: 'h-10 px-3 text-sm',
        lg: 'h-12 px-4 text-base',
      },
      state: {
        default: '',
        error: 'border-destructive focus-visible:ring-destructive/25',
      },
    },
    defaultVariants: {
      size: 'md',
      state: 'default',
    },
  }
)

export interface InputProps
  extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'size'>,
    VariantProps<typeof inputVariants> {
  errorMessage?: string
  leftIcon?: React.ReactNode
  rightElement?: React.ReactNode
}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, size, state, errorMessage, leftIcon, rightElement, type, ...props }, ref) => {
    const hasError = state === 'error' || !!errorMessage

    if (leftIcon || rightElement) {
      return (
        <div className="relative flex items-center">
          {leftIcon && (
            <span className="absolute left-3 text-muted-foreground pointer-events-none">
              {leftIcon}
            </span>
          )}
          <input
            type={type}
            ref={ref}
            className={cn(
              inputVariants({ size, state: hasError ? 'error' : 'default', className }),
              leftIcon ? 'pl-9' : '',
              rightElement ? 'pr-9' : ''
            )}
            {...props}
          />
          {rightElement && (
            <span className="absolute right-3 text-muted-foreground">{rightElement}</span>
          )}
          {errorMessage && (
            <p className="mt-1 text-xs text-destructive">{errorMessage}</p>
          )}
        </div>
      )
    }

    return (
      <div className="w-full">
        <input
          type={type}
          ref={ref}
          className={cn(inputVariants({ size, state: hasError ? 'error' : 'default', className }))}
          {...props}
        />
        {errorMessage && (
          <p className="mt-1 text-xs text-destructive">{errorMessage}</p>
        )}
      </div>
    )
  }
)
Input.displayName = 'Input'

export { Input, inputVariants }
