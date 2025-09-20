import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { ArrowRight, Microscope, Database, BarChart3, Dna } from "lucide-react"
import Link from "next/link"

export default function HomePage() {
  return (
    <div className="min-h-screen bg-background">
      {/* Navigation */}
      <nav className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-2">
              <Dna className="h-8 w-8 text-primary" />
              <span className="text-xl font-bold text-foreground">DeepSea eDNA AI</span>
            </div>
            <div className="hidden md:flex items-center space-x-8">
              <Link href="/about" className="text-muted-foreground hover:text-foreground transition-colors">
                About
              </Link>
              <Link href="/background" className="text-muted-foreground hover:text-foreground transition-colors">
                Background
              </Link>
              <Link href="/solution" className="text-muted-foreground hover:text-foreground transition-colors">
                Solution
              </Link>
              <Link href="/dashboard" className="text-muted-foreground hover:text-foreground transition-colors">
                Dashboard
              </Link>
              <Button variant="outline" size="sm">
                Get Started
              </Button>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative py-20 lg:py-32">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <Badge variant="secondary" className="mb-6">
              AI-Powered Marine Biology Research
            </Badge>
            <h1 className="text-4xl md:text-6xl font-bold text-foreground mb-6 text-balance">
              AI-Driven Identification of <span className="text-primary">Eukaryotic Taxa</span> from Deep-Sea eDNA
            </h1>
            <p className="text-xl text-muted-foreground mb-8 max-w-3xl mx-auto text-pretty">
              Revolutionizing marine biodiversity research with advanced machine learning algorithms that identify
              unknown species from environmental DNA samples in the deep ocean.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button size="lg" className="text-lg px-8">
                Explore Dashboard
                <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
              <Button variant="outline" size="lg" className="text-lg px-8 bg-transparent">
                Learn More
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* Key Features */}
      <section className="py-20 bg-card/30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-4">Advancing Marine Discovery</h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Our AI pipeline transforms raw eDNA data into actionable biodiversity insights
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
              <CardHeader>
                <Microscope className="h-12 w-12 text-primary mb-4" />
                <CardTitle className="text-xl">eDNA Processing</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-base">
                  Advanced cleaning and denoising of environmental DNA reads from deep-sea samples
                </CardDescription>
              </CardContent>
            </Card>

            <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
              <CardHeader>
                <Database className="h-12 w-12 text-accent mb-4" />
                <CardTitle className="text-xl">AI Clustering</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-base">
                  Machine learning embeddings group sequences into putative taxa, even without database matches
                </CardDescription>
              </CardContent>
            </Card>

            <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
              <CardHeader>
                <Dna className="h-12 w-12 text-primary mb-4" />
                <CardTitle className="text-xl">Novel Detection</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-base">
                  Identify previously unknown species alongside known taxa from reference databases
                </CardDescription>
              </CardContent>
            </Card>

            <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
              <CardHeader>
                <BarChart3 className="h-12 w-12 text-accent mb-4" />
                <CardTitle className="text-xl">Biodiversity Metrics</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-base">
                  Comprehensive analysis of species richness, abundance, and diversity indices
                </CardDescription>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      
      <section id="about" className="py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <div>
              <Badge variant="outline" className="mb-4">
                Research Challenge
              </Badge>
              <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-6">
                The Deep Sea Biodiversity Challenge
              </h2>
              <p className="text-lg text-muted-foreground mb-6">
                The deep sea contains a vast array of organisms, most of which remain unknown or poorly studied.
                Traditional eDNA analysis tools rely heavily on incomplete reference databases, leaving many sequences
                unclassified and biodiversity underestimated.
              </p>
              <ul className="space-y-3 text-muted-foreground">
                <li className="flex items-start">
                  <div className="w-2 h-2 bg-primary rounded-full mt-2 mr-3 flex-shrink-0"></div>
                  Incomplete reference databases limit species identification
                </li>
                <li className="flex items-start">
                  <div className="w-2 h-2 bg-primary rounded-full mt-2 mr-3 flex-shrink-0"></div>
                  Long processing times delay critical research insights
                </li>
                <li className="flex items-start">
                  <div className="w-2 h-2 bg-primary rounded-full mt-2 mr-3 flex-shrink-0"></div>
                  Novel taxa remain undetected in biodiversity assessments
                </li>
              </ul>
            </div>
            <div className="relative">
              <Card className="border-border/50 bg-gradient-to-br from-primary/10 to-accent/10 backdrop-blur-sm">
                <CardHeader>
                  <CardTitle className="text-2xl text-center">Current Limitations</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-muted-foreground">Unclassified Sequences</span>
                    <Badge variant="destructive">60-80%</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-muted-foreground">Processing Time</span>
                    <Badge variant="secondary">Weeks</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-muted-foreground">Novel Species Detection</span>
                    <Badge variant="destructive">Limited</Badge>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border bg-card/30 py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <div className="flex items-center justify-center space-x-2 mb-4">
              <Dna className="h-6 w-6 text-primary" />
              <span className="text-lg font-semibold text-foreground">DeepSea eDNA AI</span>
            </div>
            <p className="text-muted-foreground">
              Advancing marine biodiversity research through artificial intelligence
            </p>
            <p className="text-sm text-muted-foreground mt-2">
              Developed for Central Marine Living Resources and Ecology (CMLRE)
            </p>
          </div>
        </div>
      </footer>
    </div>
  )
}
