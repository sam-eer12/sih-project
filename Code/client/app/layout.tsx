import type React from "react"
import type { Metadata } from "next"
import { GeistSans } from "geist/font/sans"
import { GeistMono } from "geist/font/mono"
import { Analytics } from "@vercel/analytics/next"
import WavyNeonBackground from "@/components/wavy-neon-background"
import "./globals.css"
import { Suspense } from "react"

export const metadata: Metadata = {
  title: "SIH25042 Project",
  description: "AI-driven identification of eukaryotic taxa from deep-sea environmental DNA",
  generator: "systummmmmmmm",
  icons: {
    icon: "/favicon3.png",
    
  },
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <head>
        <link rel="icon" href="/favicon.ico" />
      </head>
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
