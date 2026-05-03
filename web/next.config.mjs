function normalizeApiOrigin(origin) {
  if (!origin) {
    return null
  }
  return origin.replace(/\/+$/, '')
}

function resolveApiOrigin() {
  const configuredOrigin =
    process.env.MOCHI_API_BASE_URL ||
    process.env.NEXT_PUBLIC_MOCHI_API_BASE_URL
  if (configuredOrigin && configuredOrigin.trim().length > 0) {
    return normalizeApiOrigin(configuredOrigin.trim())
  }

  if (process.env.NODE_ENV === 'development') {
    return 'http://127.0.0.1:8000'
  }
  return null
}

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    const apiOrigin = resolveApiOrigin()
    if (!apiOrigin) {
      return []
    }

    return [
      {
        source: '/v1/:path*',
        destination: `${apiOrigin}/v1/:path*`,
      },
    ]
  },
  typedRoutes: false,
}

export default nextConfig
