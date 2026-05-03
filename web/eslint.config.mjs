import nextVitals from 'eslint-config-next/core-web-vitals'
import tseslint from 'typescript-eslint'

const eslintConfig = [
  ...nextVitals,
  {
    files: ['**/*.{ts,tsx}'],
    plugins: {
      '@typescript-eslint': tseslint.plugin,
    },
    rules: {
      '@typescript-eslint/no-explicit-any': 'error',
      '@typescript-eslint/no-unused-vars': ['warn', { argsIgnorePattern: '^_' }],
      'react-hooks/set-state-in-effect': 'off',
      'react-hooks/static-components': 'off',
      'react/no-inline-styles': 'off',
    },
  },
  {
    ignores: [
      '.next/**',
      'node_modules/**',
      'out/**',
      'build/**',
      'next-env.d.ts',
    ],
  },
]

export default eslintConfig
