import type { NextConfig } from "next";

// The browser only ever talks to this server; /api/* is proxied to the FastAPI service, so the
// API needs no public URL and no CORS in deployment. Rewrites are resolved at build time.
const apiUrl = process.env.API_INTERNAL_URL ?? "http://localhost:8000";

const nextConfig: NextConfig = {
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${apiUrl}/:path*` }];
  },
  experimental: {
    // Mitigation searches with the ensemble can take minutes; the proxy default is 30 s.
    proxyTimeout: 300_000,
  },
};

export default nextConfig;
