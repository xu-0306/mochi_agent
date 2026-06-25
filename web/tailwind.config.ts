import type { Config } from 'tailwindcss'
import typography from '@tailwindcss/typography'
import tailwindcssAnimate from 'tailwindcss-animate'

export default {
  darkMode: ['class', '[data-theme="dark"]'],
  content: ['./src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        border: 'hsl(var(--border))',
        input: 'hsl(var(--input))',
        ring: 'hsl(var(--ring))',
        background: 'hsl(var(--background))',
        foreground: 'hsl(var(--foreground))',
        primary: {
          DEFAULT: '#5E6AD2',
          50: '#EEF0FF',
          100: '#DEE2FF',
          200: '#C3C9FA',
          300: '#9EA6F2',
          400: '#7B83E4',
          500: '#5E6AD2',
          600: '#4C57B8',
          700: '#3D4796',
          900: '#1F2458',
          foreground: '#FFFFFF',
        },
        secondary: { DEFAULT: '#6B8AFD', foreground: '#FFFFFF' },
        accent: { DEFAULT: '#F76D8E', glow: 'rgba(247,109,142,0.35)' },
        success: '#16A34A',
        warning: '#D97706',
        error: '#DC2626',
        info: '#0284C7',
        surface: 'hsl(var(--bg-surface))',
        elevated: 'hsl(var(--bg-elevated))',
        sidebar: 'hsl(var(--bg-sidebar))',
        muted: {
          DEFAULT: 'hsl(var(--muted))',
          foreground: 'hsl(var(--muted-foreground))',
        },
        card: {
          DEFAULT: 'hsl(var(--card))',
          foreground: 'hsl(var(--card-foreground))',
        },
        popover: {
          DEFAULT: 'hsl(var(--popover))',
          foreground: 'hsl(var(--popover-foreground))',
        },
        destructive: {
          DEFAULT: 'hsl(var(--destructive))',
          foreground: 'hsl(var(--destructive-foreground))',
        },
      },
      fontFamily: {
        sans: ['Inter', 'Noto Sans TC', 'sans-serif'],
        mono: ['JetBrains Mono', 'Noto Sans Mono CJK TC', 'monospace'],
      },
      borderRadius: {
        sm: '4px',
        md: '8px',
        lg: '12px',
        xl: '16px',
        '2xl': '24px',
      },
      boxShadow: {
        xs: '0 1px 2px rgba(0,0,0,0.18)',
        sm: '0 2px 4px rgba(0,0,0,0.22), 0 1px 2px rgba(0,0,0,0.14)',
        md: '0 6px 16px rgba(0,0,0,0.28), 0 2px 4px rgba(0,0,0,0.18)',
        lg: '0 16px 32px rgba(0,0,0,0.32), 0 4px 8px rgba(0,0,0,0.22)',
        xl: '0 28px 56px rgba(0,0,0,0.40), 0 8px 16px rgba(0,0,0,0.26)',
        glow: '0 0 0 4px rgba(247,109,142,0.18), 0 0 24px 4px rgba(247,109,142,0.55)',
      },
      transitionTimingFunction: {
        'out-smooth': 'cubic-bezier(0.22, 1, 0.36, 1)',
        spring: 'cubic-bezier(0.34, 1.56, 0.64, 1)',
      },
      keyframes: {
        pulseGlow: {
          '0%, 100%': { boxShadow: '0 0 0 4px rgba(247,109,142,0.15), 0 0 16px rgba(247,109,142,0.35)' },
          '50%': { boxShadow: '0 0 0 6px rgba(247,109,142,0.25), 0 0 32px rgba(247,109,142,0.70)' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(12px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        blink: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0' },
        },
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        dialogIn: {
          '0%': { opacity: '0', transform: 'translate(-50%, calc(-50% + 12px)) scale(0.96)' },
          '100%': { opacity: '1', transform: 'translate(-50%, -50%) scale(1)' },
        },
        dialogOut: {
          '0%': { opacity: '1', transform: 'translate(-50%, -50%) scale(1)' },
          '100%': { opacity: '0', transform: 'translate(-50%, calc(-50% + 8px)) scale(0.98)' },
        },
        'accordion-down': {
          from: { height: '0' },
          to: { height: 'var(--radix-accordion-content-height)' },
        },
        'accordion-up': {
          from: { height: 'var(--radix-accordion-content-height)' },
          to: { height: '0' },
        },
      },
      animation: {
        'pulse-glow': 'pulseGlow 1.2s ease-in-out infinite',
        'slide-up': 'slideUp 200ms cubic-bezier(0.22, 1, 0.36, 1)',
        blink: 'blink 600ms step-start infinite',
        'fade-in': 'fadeIn 200ms ease-out',
        'dialog-in': 'dialogIn 220ms cubic-bezier(0.22, 1, 0.36, 1)',
        'dialog-out': 'dialogOut 180ms ease-in forwards',
        'accordion-down': 'accordion-down 200ms ease-out',
        'accordion-up': 'accordion-up 200ms ease-out',
      },
    },
  },
  plugins: [tailwindcssAnimate, typography],
} satisfies Config
