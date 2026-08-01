import type { NextConfig } from "next";

const basePath = process.env.NEXT_PUBLIC_PASTVIDEO_BASE?.replace(/\/$/, "") ?? "";

const nextConfig: NextConfig = {
  basePath,
};

export default nextConfig;
