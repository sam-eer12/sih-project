import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { ArrowLeft, Brain, Cpu, Zap, Target, ArrowRight, CheckCircle, TrendingUp } from "lucide-react"
import Link from "next/link"

export default function SolutionPage() {
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
              <Link href="/background" className="text-muted-foreground hover:text-foreground transition-colors">
                Background
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
              AI-Powered Solution
            </Badge>
            <h1 className="text-4xl md:text-5xl font-bold text-foreground mb-6 text-balance">
              Our <span className="text-primary">AI-Driven Pipeline</span> for eDNA Analysis
            </h1>
            <p className="text-xl text-muted-foreground text-pretty">
              A comprehensive machine learning solution that transforms raw eDNA data into actionable biodiversity
              insights
            </p>
          </div>
        </div>
      </section>

      {/* Solution Overview */}
      <section className="py-16">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid lg:grid-cols-2 gap-12 items-center mb-20">
            <div>
              <Badge variant="outline" className="mb-4">
                Core Innovation
              </Badge>
              <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-6">
                Database-Independent Species Identification
              </h2>
              <p className="text-lg text-muted-foreground mb-6 leading-relaxed">
                Our AI pipeline uses advanced machine learning to identify species from eDNA sequences without relying
                solely on reference databases. By generating embeddings and clustering sequences, we can detect both
                known and novel taxa in marine environments.
              </p>
              <ul className="space-y-3 text-muted-foreground">
                <li className="flex items-start">
                  <CheckCircle className="w-5 h-5 text-primary mt-0.5 mr-3 flex-shrink-0" />
                  Unsupervised clustering identifies putative taxa
                </li>
                <li className="flex items-start">
                  <CheckCircle className="w-5 h-5 text-primary mt-0.5 mr-3 flex-shrink-0" />
                  Deep learning embeddings capture sequence patterns
                </li>
                <li className="flex items-start">
                  <CheckCircle className="w-5 h-5 text-primary mt-0.5 mr-3 flex-shrink-0" />
                  Novel species detection alongside known taxa
                </li>
              </ul>
            </div>
            <Card className="border-border/50 bg-gradient-to-br from-primary/10 to-accent/10 backdrop-blur-sm">
              <CardHeader>
                <Brain className="h-12 w-12 text-primary mb-4" />
                <CardTitle className="text-2xl">Key Capabilities</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">Classification Rate</span>
                  <Badge variant="default" className="bg-primary">
                    80-95%
                  </Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">Processing Speed</span>
                  <Badge variant="default" className="bg-accent">
                    Hours
                  </Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">Novel Detection</span>
                  <Badge variant="default" className="bg-primary">
                    Enhanced
                  </Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">Database Dependency</span>
                  <Badge variant="default" className="bg-accent">
                    Reduced
                  </Badge>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Pipeline Workflow */}
      <section className="py-20 bg-card/30">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-4">AI Pipeline Workflow</h2>
            <p className="text-lg text-muted-foreground max-w-3xl mx-auto">
              Our comprehensive pipeline processes raw eDNA data through multiple AI-enhanced stages
            </p>
          </div>

          <div className="space-y-8">
            {/* Step 1 */}
            <div className="flex flex-col lg:flex-row items-center gap-8">
              <div className="lg:w-1/3">
                <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
                  <CardHeader className="text-center">
                    <div className="w-12 h-12 bg-primary/20 rounded-full flex items-center justify-center mx-auto mb-4">
                      <span className="text-xl font-bold text-primary">1</span>
                    </div>
                    <CardTitle className="text-xl">Data Preprocessing</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <CardDescription className="text-center">
                      Clean and denoise raw eDNA reads using advanced quality control algorithms
                    </CardDescription>
                  </CardContent>
                </Card>
              </div>
              <div className="lg:w-2/3">
                <div className="space-y-3">
                  <h3 className="text-xl font-semibold text-foreground">Quality Control & Filtering</h3>
                  <p className="text-muted-foreground">
                    Raw eDNA sequences undergo rigorous quality assessment, removing low-quality reads, adapter
                    sequences, and potential contaminants to ensure high-quality input data.
                  </p>
                  <ul className="text-sm text-muted-foreground space-y-1">
                    <li>• Quality score filtering and trimming</li>
                    <li>• Adapter and primer removal</li>
                    <li>• Chimera detection and removal</li>
                    <li>• Length and complexity filtering</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="flex justify-center">
              <ArrowRight className="h-8 w-8 text-muted-foreground" />
            </div>

            {/* Step 2 */}
            <div className="flex flex-col lg:flex-row-reverse items-center gap-8">
              <div className="lg:w-1/3">
                <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
                  <CardHeader className="text-center">
                    <div className="w-12 h-12 bg-accent/20 rounded-full flex items-center justify-center mx-auto mb-4">
                      <span className="text-xl font-bold text-accent">2</span>
                    </div>
                    <CardTitle className="text-xl">ML Embeddings</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <CardDescription className="text-center">
                      Generate high-dimensional embeddings that capture sequence patterns and relationships
                    </CardDescription>
                  </CardContent>
                </Card>
              </div>
              <div className="lg:w-2/3">
                <div className="space-y-3">
                  <h3 className="text-xl font-semibold text-foreground">Deep Learning Feature Extraction</h3>
                  <p className="text-muted-foreground">
                    Advanced neural networks transform DNA sequences into meaningful numerical representations that
                    capture evolutionary relationships and taxonomic patterns.
                  </p>
                  <ul className="text-sm text-muted-foreground space-y-1">
                    <li>• Transformer-based sequence encoding</li>
                    <li>• Evolutionary distance preservation</li>
                    <li>• Multi-scale pattern recognition</li>
                    <li>• Dimensionality optimization</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="flex justify-center">
              <ArrowRight className="h-8 w-8 text-muted-foreground" />
            </div>

            {/* Step 3 */}
            <div className="flex flex-col lg:flex-row items-center gap-8">
              <div className="lg:w-1/3">
                <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
                  <CardHeader className="text-center">
                    <div className="w-12 h-12 bg-primary/20 rounded-full flex items-center justify-center mx-auto mb-4">
                      <span className="text-xl font-bold text-primary">3</span>
                    </div>
                    <CardTitle className="text-xl">AI Clustering</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <CardDescription className="text-center">
                      Group sequences into putative taxa using unsupervised clustering algorithms
                    </CardDescription>
                  </CardContent>
                </Card>
              </div>
              <div className="lg:w-2/3">
                <div className="space-y-3">
                  <h3 className="text-xl font-semibold text-foreground">Intelligent Sequence Clustering</h3>
                  <p className="text-muted-foreground">
                    Machine learning algorithms identify natural groupings in the embedding space, representing
                    potential taxonomic units without requiring database matches.
                  </p>
                  <ul className="text-sm text-muted-foreground space-y-1">
                    <li>• Density-based clustering (DBSCAN)</li>
                    <li>• Hierarchical clustering analysis</li>
                    <li>• Optimal cluster number detection</li>
                    <li>• Noise and outlier handling</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="flex justify-center">
              <ArrowRight className="h-8 w-8 text-muted-foreground" />
            </div>

            {/* Step 4 */}
            <div className="flex flex-col lg:flex-row-reverse items-center gap-8">
              <div className="lg:w-1/3">
                <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
                  <CardHeader className="text-center">
                    <div className="w-12 h-12 bg-accent/20 rounded-full flex items-center justify-center mx-auto mb-4">
                      <span className="text-xl font-bold text-accent">4</span>
                    </div>
                    <CardTitle className="text-xl">Taxonomic Assignment</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <CardDescription className="text-center">
                      Label clusters using available databases while preserving novel taxa information
                    </CardDescription>
                  </CardContent>
                </Card>
              </div>
              <div className="lg:w-2/3">
                <div className="space-y-3">
                  <h3 className="text-xl font-semibold text-foreground">Hybrid Classification Approach</h3>
                  <p className="text-muted-foreground">
                    Combine database matching with cluster analysis to provide taxonomic labels for known species while
                    flagging novel or uncharacterized taxa for further investigation.
                  </p>
                  <ul className="text-sm text-muted-foreground space-y-1">
                    <li>• BLAST-based database matching</li>
                    <li>• Confidence score calculation</li>
                    <li>• Novel taxa flagging system</li>
                    <li>• Taxonomic hierarchy assignment</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Technology Stack */}
      <section className="py-20">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <Badge variant="secondary" className="mb-4">
              Technology Stack
            </Badge>
            <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-4">Cutting-Edge AI Technologies</h2>
            <p className="text-lg text-muted-foreground max-w-3xl mx-auto">
              Our pipeline leverages state-of-the-art machine learning frameworks and bioinformatics tools
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
              <CardHeader>
                <Brain className="h-12 w-12 text-primary mb-4" />
                <CardTitle className="text-xl">Deep Learning</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-base leading-relaxed">
                  Transformer architectures and convolutional neural networks for sequence analysis and pattern
                  recognition in genomic data.
                </CardDescription>
              </CardContent>
            </Card>

            <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
              <CardHeader>
                <Cpu className="h-12 w-12 text-accent mb-4" />
                <CardTitle className="text-xl">High-Performance Computing</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-base leading-relaxed">
                  GPU-accelerated processing and distributed computing for handling large-scale eDNA datasets
                  efficiently.
                </CardDescription>
              </CardContent>
            </Card>

            <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
              <CardHeader>
                <Zap className="h-12 w-12 text-primary mb-4" />
                <CardTitle className="text-xl">Bioinformatics Tools</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-base leading-relaxed">
                  Integration with established bioinformatics pipelines and databases for comprehensive sequence
                  analysis and validation.
                </CardDescription>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Expected Impact */}
      <section className="py-20 bg-card/30">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <Badge variant="outline" className="mb-4">
              Expected Impact
            </Badge>
            <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-4">Transforming Marine Research</h2>
            <p className="text-lg text-muted-foreground max-w-3xl mx-auto">
              Our AI solution will revolutionize how researchers study and protect marine biodiversity
            </p>
          </div>

          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <div className="space-y-6">
              <div className="flex items-start space-x-4">
                <div className="w-12 h-12 bg-primary/20 rounded-lg flex items-center justify-center flex-shrink-0">
                  <Target className="h-6 w-6 text-primary" />
                </div>
                <div>
                  <h3 className="text-xl font-semibold text-foreground mb-2">Accelerated Discovery</h3>
                  <p className="text-muted-foreground">
                    Enable rapid identification of new species in deep-sea ecosystems, accelerating the pace of marine
                    biodiversity discovery and taxonomic research.
                  </p>
                </div>
              </div>

              <div className="flex items-start space-x-4">
                <div className="w-12 h-12 bg-accent/20 rounded-lg flex items-center justify-center flex-shrink-0">
                  <TrendingUp className="h-6 w-6 text-accent" />
                </div>
                <div>
                  <h3 className="text-xl font-semibold text-foreground mb-2">Enhanced Monitoring</h3>
                  <p className="text-muted-foreground">
                    Provide faster, more comprehensive biodiversity assessments for long-term ecosystem monitoring and
                    conservation planning.
                  </p>
                </div>
              </div>

              <div className="flex items-start space-x-4">
                <div className="w-12 h-12 bg-primary/20 rounded-lg flex items-center justify-center flex-shrink-0">
                  <CheckCircle className="h-6 w-6 text-primary" />
                </div>
                <div>
                  <h3 className="text-xl font-semibold text-foreground mb-2">Research Accessibility</h3>
                  <p className="text-muted-foreground">
                    Make advanced eDNA analysis accessible to researchers without extensive bioinformatics expertise
                    through user-friendly interfaces.
                  </p>
                </div>
              </div>
            </div>

            <Card className="border-border/50 bg-gradient-to-br from-primary/10 to-accent/10 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="text-2xl text-center">Projected Improvements</CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="text-center">
                  <div className="text-4xl font-bold text-primary mb-2">10x</div>
                  <div className="text-muted-foreground">Faster Processing</div>
                </div>
                <div className="text-center">
                  <div className="text-4xl font-bold text-accent mb-2">3x</div>
                  <div className="text-muted-foreground">Higher Classification Rate</div>
                </div>
                <div className="text-center">
                  <div className="text-4xl font-bold text-primary mb-2">5x</div>
                  <div className="text-muted-foreground">More Novel Species Detected</div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Call to Action */}
      <section className="py-20">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-6">Ready to See Our Solution in Action?</h2>
          <p className="text-lg text-muted-foreground mb-8">
            Explore our interactive dashboard to see how AI transforms eDNA data into biodiversity insights
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button size="lg" asChild>
              <Link href="/dashboard">
                View Interactive Dashboard
                <ArrowRight className="ml-2 h-5 w-5" />
              </Link>
            </Button>
            <Button variant="outline" size="lg" asChild>
              <Link href="/about">Learn More About Our Research</Link>
            </Button>
          </div>
        </div>
      </section>
    </div>
  )
}
