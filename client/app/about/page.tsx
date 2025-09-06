import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { ArrowLeft, Waves, Fish, Microscope, Globe, Users, Target } from "lucide-react"
import Link from "next/link"

export default function AboutPage() {
  return (
    <div className="min-h-screen bg-background">
      {/* Navigation */}
      <nav className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <Link href="/" className="flex items-center space-x-2">
              <ArrowLeft className="h-5 w-5 text-muted-foreground" />
              <span className="text-muted-foreground hover:text-foreground transition-colors">Back to Home</span>
            </Link>
            <div className="flex items-center space-x-8">
              <Link href="/background" className="text-muted-foreground hover:text-foreground transition-colors">
                Background
              </Link>
              <Link href="/solution" className="text-muted-foreground hover:text-foreground transition-colors">
                Solution
              </Link>
              <Link href="/dashboard" className="text-muted-foreground hover:text-foreground transition-colors">
                Dashboard
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="py-20 lg:py-32">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <Badge variant="secondary" className="mb-6">
              About Our Research
            </Badge>
            <h1 className="text-4xl md:text-5xl font-bold text-foreground mb-6 text-balance">
              Revolutionizing Deep-Sea <span className="text-primary">Biodiversity Research</span>
            </h1>
            <p className="text-xl text-muted-foreground text-pretty">
              Understanding the vast, unexplored ecosystems of our planet's deep oceans through cutting-edge AI
              technology
            </p>
          </div>
        </div>
      </section>

      {/* Mission & Vision */}
      <section className="py-16 bg-card/30">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid md:grid-cols-2 gap-12">
            <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
              <CardHeader>
                <Target className="h-12 w-12 text-primary mb-4" />
                <CardTitle className="text-2xl">Our Mission</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-base leading-relaxed">
                  To accelerate the discovery and understanding of marine biodiversity in Earth's most unexplored
                  ecosystems through innovative AI-driven analysis of environmental DNA. We aim to provide researchers
                  with powerful tools that can identify both known and novel species from deep-sea samples.
                </CardDescription>
              </CardContent>
            </Card>

            <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
              <CardHeader>
                <Globe className="h-12 w-12 text-accent mb-4" />
                <CardTitle className="text-2xl">Our Vision</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-base leading-relaxed">
                  A future where marine biodiversity is fully understood and protected through comprehensive,
                  AI-enhanced monitoring systems. We envision enabling rapid, accurate species identification that
                  supports conservation efforts and scientific discovery in the world's oceans.
                </CardDescription>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Research Context */}
      <section className="py-20">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-4">The Deep-Sea Challenge</h2>
            <p className="text-lg text-muted-foreground max-w-3xl mx-auto">
              The deep ocean represents the largest habitat on Earth, yet it remains one of the least understood
              ecosystems
            </p>
          </div>

          <div className="grid lg:grid-cols-3 gap-8 mb-16">
            <Card className="border-border/50 bg-gradient-to-br from-primary/10 to-transparent">
              <CardHeader className="text-center">
                <Waves className="h-16 w-16 text-primary mx-auto mb-4" />
                <CardTitle className="text-xl">Vast Unexplored Territory</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-center">
                  Over 80% of the ocean remains unmapped and unexplored, with the deep sea containing countless
                  undiscovered species and ecosystems.
                </CardDescription>
              </CardContent>
            </Card>

            <Card className="border-border/50 bg-gradient-to-br from-accent/10 to-transparent">
              <CardHeader className="text-center">
                <Fish className="h-16 w-16 text-accent mx-auto mb-4" />
                <CardTitle className="text-xl">Hidden Biodiversity</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-center">
                  Scientists estimate that 91% of marine species remain undescribed, with the deep sea harboring the
                  greatest potential for new discoveries.
                </CardDescription>
              </CardContent>
            </Card>

            <Card className="border-border/50 bg-gradient-to-br from-primary/10 to-transparent">
              <CardHeader className="text-center">
                <Microscope className="h-16 w-16 text-primary mx-auto mb-4" />
                <CardTitle className="text-xl">eDNA Revolution</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-center">
                  Environmental DNA sampling allows us to detect species presence without physical specimens,
                  revolutionizing biodiversity assessment methods.
                </CardDescription>
              </CardContent>
            </Card>
          </div>

          {/* Key Statistics */}
          <div className="bg-gradient-to-r from-primary/20 to-accent/20 rounded-lg p-8">
            <h3 className="text-2xl font-bold text-center text-foreground mb-8">Research Impact</h3>
            <div className="grid md:grid-cols-4 gap-8 text-center">
              <div>
                <div className="text-3xl font-bold text-primary mb-2">2M+</div>
                <div className="text-muted-foreground">Marine Species Estimated</div>
              </div>
              <div>
                <div className="text-3xl font-bold text-accent mb-2">91%</div>
                <div className="text-muted-foreground">Species Undescribed</div>
              </div>
              <div>
                <div className="text-3xl font-bold text-primary mb-2">80%</div>
                <div className="text-muted-foreground">Ocean Unexplored</div>
              </div>
              <div>
                <div className="text-3xl font-bold text-accent mb-2">95%</div>
                <div className="text-muted-foreground">Deep Sea Unknown</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CMLRE Partnership */}
      <section className="py-20 bg-card/30">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
            <CardHeader className="text-center">
              <Users className="h-16 w-16 text-primary mx-auto mb-4" />
              <CardTitle className="text-3xl">Partnership with CMLRE</CardTitle>
              <CardDescription className="text-lg mt-4">Central Marine Living Resources and Ecology</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <p className="text-muted-foreground leading-relaxed">
                Our research is conducted in partnership with the Central Marine Living Resources and Ecology (CMLRE), a
                leading institution in marine biodiversity research. This collaboration ensures that our AI-driven tools
                meet the real-world needs of marine biologists and conservation scientists.
              </p>
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-semibold text-foreground mb-2">Research Excellence</h4>
                  <p className="text-sm text-muted-foreground">
                    CMLRE brings decades of expertise in marine ecology and biodiversity assessment to guide our AI
                    development efforts.
                  </p>
                </div>
                <div>
                  <h4 className="font-semibold text-foreground mb-2">Field Validation</h4>
                  <p className="text-sm text-muted-foreground">
                    Real-world testing and validation of our AI tools in diverse marine environments ensures practical
                    applicability.
                  </p>
                </div>
              </div>
              <div className="text-center pt-4">
                <Button variant="outline">Learn More About CMLRE</Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </section>
    </div>
  )
}
