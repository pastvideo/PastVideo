import type { Metadata } from "next";
import "./globals.css";

const publicBase = process.env.NEXT_PUBLIC_PASTVIDEO_BASE?.replace(/\/$/, "") ?? "";

export const metadata: Metadata = {
  title: "PastVideo — Local semantic video search",
  description: "Search local footage by describing the moment with Qwen3-VL.",
  openGraph: {
    title: "PastVideo — Find the moment, not the filename",
    description: "Private, on-device semantic video search powered by Qwen3-VL.",
    images: [{ url: `${publicBase}/pastvideo-social.png`, width: 1732, height: 909 }],
  },
  twitter: {
    card: "summary_large_image",
    title: "PastVideo — Local semantic video search",
    description: "Describe the scene. Search the footage. Keep it local.",
    images: [`${publicBase}/pastvideo-social.png`],
  },
  icons: {
    icon: `${publicBase}/favicon.svg`,
    shortcut: `${publicBase}/favicon.svg`,
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
