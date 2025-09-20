import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { ArrowLeft, Dna, Database, AlertTriangle, Clock, TrendingUp } from "lucide-react"
import Link from "next/link"

export default function BackgroundPage() {
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
              <Link href="/about" className="text-muted-foreground hover:text-foreground transition-colors">
                About
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
              Scientific Background
            </Badge>
            <h1 className="text-4xl md:text-5xl font-bold text-foreground mb-6 text-balance">
              Understanding <span className="text-primary">Environmental DNA</span> in Deep-Sea Research
            </h1>
            <p className="text-xl text-muted-foreground text-pretty">
              Exploring the challenges and opportunities in marine biodiversity assessment through eDNA analysis
            </p>
          </div>
        </div>
      </section>

      {/* What is eDNA */}
      <section className="py-16">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid lg:grid-cols-2 gap-12 items-center mb-20">
            <div>
              <Badge variant="outline" className="mb-4">
                Environmental DNA
              </Badge>
              <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-6">What is eDNA?</h2>
              <p className="text-lg text-muted-foreground mb-6 leading-relaxed">
                Environmental DNA (eDNA) refers to genetic material obtained directly from environmental samples such as
                water, sediment, or air, without isolating any target organisms. In marine environments, organisms
                continuously shed DNA through skin cells, mucus, feces, and other biological materials.
              </p>
              <ul className="space-y-3 text-muted-foreground">
                <li className="flex items-start">
                  <div className="w-2 h-2 bg-primary rounded-full mt-2 mr-3 flex-shrink-0"></div>
                  Non-invasive sampling method requiring no physical specimens
                </li>
                <li className="flex items-start">
                  <div className="w-2 h-2 bg-primary rounded-full mt-2 mr-3 flex-shrink-0"></div>
                  Detects presence of species even in low abundance
                </li>
                <li className="flex items-start">
                  <div className="w-2 h-2 bg-primary rounded-full mt-2 mr-3 flex-shrink-0"></div>
                  Enables biodiversity assessment across large spatial scales
                </li>
              </ul>
            </div>
            <Card className="border-border/50 bg-gradient-to-br from-primary/10 to-accent/10 backdrop-blur-sm">
              <CardHeader>
                <Dna className="h-12 w-12 text-primary mb-4" />
                <CardTitle className="text-2xl">eDNA Process</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 bg-primary/20 rounded-full flex items-center justify-center text-sm font-semibold text-primary">
                    1
                  </div>
                  <span className="text-muted-foreground">Water/sediment sample collection</span>
                </div>
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 bg-primary/20 rounded-full flex items-center justify-center text-sm font-semibold text-primary">
                    2
                  </div>
                  <span className="text-muted-foreground">DNA extraction and purification</span>
                </div>
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 bg-primary/20 rounded-full flex items-center justify-center text-sm font-semibold text-primary">
                    3
                  </div>
                  <span className="text-muted-foreground">PCR amplification and sequencing</span>
                </div>
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 bg-primary/20 rounded-full flex items-center justify-center text-sm font-semibold text-primary">
                    4
                  </div>
                  <span className="text-muted-foreground">Bioinformatic analysis and identification</span>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Current Challenges */}
      <section className="py-20 bg-card/30">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-4">Current Challenges in eDNA Analysis</h2>
            <p className="text-lg text-muted-foreground max-w-3xl mx-auto">
              Despite its potential, traditional eDNA analysis faces significant limitations that hinder marine
              biodiversity research
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
              <CardHeader>
                <Database className="h-12 w-12 text-destructive mb-4" />
                <CardTitle className="text-xl">Incomplete Reference Databases</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-base leading-relaxed">
                  Current databases like NCBI RefSeq contain sequences for only a fraction of marine species, leaving
                  60-80% of eDNA reads unclassified in typical studies.
                </CardDescription>
              </CardContent>
            </Card>

            <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
              <CardHeader>
                <Clock className="h-12 w-12 text-destructive mb-4" />
                <CardTitle className="text-xl">Long Processing Times</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-base leading-relaxed">
                  Traditional bioinformatic pipelines can take weeks to process large eDNA datasets, delaying critical
                  research insights and conservation decisions.
                </CardDescription>
              </CardContent>
            </Card>

            <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
              <CardHeader>
                <AlertTriangle className="h-12 w-12 text-destructive mb-4" />
                <CardTitle className="text-xl">Novel Species Detection</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-base leading-relaxed">
                  Standard tools struggle to identify novel or poorly characterized species, leading to underestimation
                  of biodiversity in unexplored ecosystems.
                </CardDescription>
              </CardContent>
            </Card>
          </div>

          {/* Impact Statistics */}
          <div className="mt-16 bg-gradient-to-r from-destructive/10 to-destructive/5 rounded-lg p-8">
            <h3 className="text-2xl font-bold text-center text-foreground mb-8">The Scale of the Problem</h3>
            <div className="grid md:grid-cols-3 gap-8 text-center">
              <div>
                <div className="text-4xl font-bold text-destructive mb-2">60-80%</div>
                <div className="text-muted-foreground">Unclassified eDNA Sequences</div>
                <div className="text-sm text-muted-foreground mt-1">in typical marine studies</div>
              </div>
              <div>
                <div className="text-4xl font-bold text-destructive mb-2">2-4 weeks</div>
                <div className="text-muted-foreground">Processing Time</div>
                <div className="text-sm text-muted-foreground mt-1">for large datasets</div>
              </div>
              <div>
                <div className="text-4xl font-bold text-destructive mb-2">Limited</div>
                <div className="text-muted-foreground">Novel Detection</div>
                <div className="text-sm text-muted-foreground mt-1">capability with current tools</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* The Opportunity */}
      <section className="py-20">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <Badge variant="secondary" className="mb-4">
              The AI Opportunity
            </Badge>
            <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-4">Transforming eDNA Analysis with AI</h2>
            <p className="text-lg text-muted-foreground max-w-3xl mx-auto">
              Machine learning and AI offer unprecedented opportunities to overcome traditional limitations
            </p>
          </div>

          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <Card className="border-border/50 bg-gradient-to-br from-primary/10 to-accent/10 backdrop-blur-sm">
              <CardHeader>
                <TrendingUp className="h-12 w-12 text-primary mb-4" />
                <CardTitle className="text-2xl">AI Advantages</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-start space-x-3">
                  <div className="w-2 h-2 bg-primary rounded-full mt-2 flex-shrink-0"></div>
                  <div>
                    <div className="font-semibold text-foreground">Database Independence</div>
                    <div className="text-sm text-muted-foreground">Identify species without exact database matches</div>
                  </div>
                </div>
                <div className="flex items-start space-x-3">
                  <div className="w-2 h-2 bg-accent rounded-full mt-2 flex-shrink-0"></div>
                  <div>
                    <div className="font-semibold text-foreground">Pattern Recognition</div>
                    <div className="text-sm text-muted-foreground">Detect novel taxa through sequence clustering</div>
                  </div>
                </div>
                <div className="flex items-start space-x-3">
                  <div className="w-2 h-2 bg-primary rounded-full mt-2 flex-shrink-0"></div>
                  <div>
                    <div className="font-semibold text-foreground">Rapid Processing</div>
                    <div className="text-sm text-muted-foreground">Reduce analysis time from weeks to hours</div>
                  </div>
                </div>
                <div className="flex items-start space-x-3">
                  <div className="w-2 h-2 bg-accent rounded-full mt-2 flex-shrink-0"></div>
                  <div>
                    <div className="font-semibold text-foreground">Scalable Analysis</div>
                    <div className="text-sm text-muted-foreground">Handle large-scale biodiversity assessments</div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <div className="space-y-6">
              <h3 className="text-2xl font-bold text-foreground">Expected Improvements</h3>
              <div className="space-y-4">
                <div className="flex justify-between items-center p-4 bg-card/50 rounded-lg border border-border/50">
                  <span className="text-muted-foreground">Classification Rate</span>
                  <div className="flex items-center space-x-2">
                    <Badge variant="destructive" className="text-xs">
                      20-40%
                    </Badge>
                    <span className="text-muted-foreground">→</span>
                    <Badge variant="default" className="text-xs bg-primary">
                      80-95%
                    </Badge>
                  </div>
                </div>
                <div className="flex justify-between items-center p-4 bg-card/50 rounded-lg border border-border/50">
                  <span className="text-muted-foreground">Processing Time</span>
                  <div className="flex items-center space-x-2">
                    <Badge variant="destructive" className="text-xs">
                      Weeks
                    </Badge>
                    <span className="text-muted-foreground">→</span>
                    <Badge variant="default" className="text-xs bg-primary">
                      Hours
                    </Badge>
                  </div>
                </div>
                <div className="flex justify-between items-center p-4 bg-card/50 rounded-lg border border-border/50">
                  <span className="text-muted-foreground">Novel Detection</span>
                  <div className="flex items-center space-x-2">
                    <Badge variant="destructive" className="text-xs">
                      Limited
                    </Badge>
                    <span className="text-muted-foreground">→</span>
                    <Badge variant="default" className="text-xs bg-primary">
                      Enhanced
                    </Badge>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Call to Action */}
      <section className="py-20 bg-card/30">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-6">Ready to Explore Our Solution?</h2>
          <p className="text-lg text-muted-foreground mb-8">
            Discover how our AI-driven pipeline addresses these challenges and revolutionizes marine biodiversity
            research
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button size="lg" asChild>
              <Link href="/solution">Explore Our Solution</Link>
            </Button>
            <Button variant="outline" size="lg" asChild>
              <Link href="/dashboard">View Dashboard</Link>
            </Button>
          </div>
        </div>
      </section>
    </div>
  )
}
