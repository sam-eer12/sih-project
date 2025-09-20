"use client"

import { useEffect, useRef } from "react"

export default function WavyNeonBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const mouseRef = useRef({ x: 0, y: 0 })
  const animationRef = useRef<number>()

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Set canvas size
    const resizeCanvas = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
    }
    resizeCanvas()
    window.addEventListener("resize", resizeCanvas)

    // Mouse tracking
    const handleMouseMove = (e: MouseEvent) => {
      mouseRef.current.x = e.clientX
      mouseRef.current.y = e.clientY
    }
    window.addEventListener("mousemove", handleMouseMove)

    let time = 0
    const waves = [
      { amplitude: 80, frequency: 0.015, speed: 0.015, offset: 0, color: "rgba(177, 48, 204, 0.41)" },
      { amplitude: 120, frequency: 0.012, speed: 0.018, offset: Math.PI / 4, color: "rgba(168, 85, 247, 0.12)" },
      { amplitude: 100, frequency: 0.018, speed: 0.022, offset: Math.PI / 2, color: "rgba(124, 58, 237, 0.1)" },
      { amplitude: 90, frequency: 0.008, speed: 0.012, offset: Math.PI / 1.5, color: "rgba(147, 51, 234, 0.08)" },
      { amplitude: 110, frequency: 0.025, speed: 0.028, offset: Math.PI, color: "rgba(126, 34, 206, 0.06)" },
    ]

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // Calculate mouse influence
      const mouseInfluence = {
        x: (mouseRef.current.x / canvas.width) * 2 - 1,
        y: (mouseRef.current.y / canvas.height) * 2 - 1,
      }

      waves.forEach((wave, index) => {
        for (let layer = 0; layer < 3; layer++) {
          ctx.beginPath()
          const layerOpacity = wave.color.replace(
            /[\d.]+\)$/,
            `${Number.parseFloat(wave.color.match(/[\d.]+\)$/)?.[0].slice(0, -1) || "0.1") * (1 - layer * 0.3)})`,
          )
          ctx.strokeStyle = layerOpacity
          ctx.lineWidth = 2 + layer * 3
          ctx.shadowColor = layerOpacity
          ctx.shadowBlur = 30 + layer * 20

          for (let x = 0; x <= canvas.width; x += 1) {
            const normalizedX = x / canvas.width

            // Base wave with multiple harmonics for more organic movement
            let y =
              canvas.height / 2 +
              Math.sin(normalizedX * Math.PI * 2 * wave.frequency + time * wave.speed + wave.offset) * wave.amplitude +
              Math.sin(normalizedX * Math.PI * 4 * wave.frequency + time * wave.speed * 1.3) * wave.amplitude * 0.4 +
              Math.sin(normalizedX * Math.PI * 6 * wave.frequency + time * wave.speed * 0.8) * wave.amplitude * 0.2

            const distanceFromMouse = Math.sqrt(
              Math.pow((x - mouseRef.current.x) / canvas.width, 2) +
                Math.pow((y - mouseRef.current.y) / canvas.height, 2),
            )

            // Smoother mouse influence with exponential falloff
            const mouseEffect = Math.exp(-distanceFromMouse * 3) * 150
            y += mouseEffect * mouseInfluence.y * (index + 1) * 0.8

            y +=
              Math.sin(normalizedX * Math.PI * 8 * wave.frequency + time * wave.speed * 2) *
              wave.amplitude *
              0.15 *
              (1 + mouseEffect * 0.02)

            // Vertical offset variation for each layer
            y += layer * 20 * Math.sin(time * 0.01 + index)

            if (x === 0) {
              ctx.moveTo(x, y)
            } else {
              ctx.lineTo(x, y)
            }
          }

          ctx.stroke()
        }

        for (let glowLayer = 0; glowLayer < 2; glowLayer++) {
          ctx.beginPath()
          const glowOpacity = wave.color.replace(
            /[\d.]+\)$/,
            `${Number.parseFloat(wave.color.match(/[\d.]+\)$/)?.[0].slice(0, -1) || "0.05") * 0.3}`,
          )
          ctx.strokeStyle = glowOpacity
          ctx.lineWidth = 15 + glowLayer * 10
          ctx.shadowBlur = 60 + glowLayer * 30

          for (let x = 0; x <= canvas.width; x += 3) {
            const normalizedX = x / canvas.width
            let y =
              canvas.height / 2 +
              Math.sin(normalizedX * Math.PI * 2 * wave.frequency + time * wave.speed + wave.offset) * wave.amplitude +
              Math.sin(normalizedX * Math.PI * 4 * wave.frequency + time * wave.speed * 1.3) * wave.amplitude * 0.4

            const distanceFromMouse = Math.sqrt(
              Math.pow((x - mouseRef.current.x) / canvas.width, 2) +
                Math.pow((y - mouseRef.current.y) / canvas.height, 2),
            )

            const mouseEffect = Math.exp(-distanceFromMouse * 3) * 150
            y += mouseEffect * mouseInfluence.y * (index + 1) * 0.8

            if (x === 0) {
              ctx.moveTo(x, y)
            } else {
              ctx.lineTo(x, y)
            }
          }

          ctx.stroke()
        }
      })

      time += 1
      animationRef.current = requestAnimationFrame(animate)
    }

    animate()

    return () => {
      window.removeEventListener("resize", resizeCanvas)
      window.removeEventListener("mousemove", handleMouseMove)
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 pointer-events-none z-0"
      style={{
        background: "transparent",
        mixBlendMode: "screen",
      }}
    />
  )
}
