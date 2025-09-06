import type React from "react"
import type { Metadata } from "next"
import { GeistSans } from "geist/font/sans"
import { GeistMono } from "geist/font/mono"
import { Analytics } from "@vercel/analytics/next"
import WavyNeonBackground from "@/components/wavy-neon-background"
import "./globals.css"
import { Suspense } from "react"

export const metadata: Metadata = {
  title: "DeepSea eDNA AI - Marine Biodiversity Research",
  description: "AI-driven identification of eukaryotic taxa from deep-sea environmental DNA",
  generator: "v0.app",
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body className={`font-sans ${GeistSans.variable} ${GeistMono.variable}`}>
        <Suspense fallback={null}>
          <WavyNeonBackground />
        </Suspense>
        {children}
        <Analytics />
      </body>
    </html>
  )
}
