import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  reactStrictMode: true,
  async rewrites() {
    return [{ source: "/analysis/:path*", destination: `${process.env.ANALYSIS_URL ?? "http://localhost:8000"}/:path*` }];
  },
};

export default nextConfig;
