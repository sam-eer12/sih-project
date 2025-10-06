import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { ArrowLeft, Waves, Fish, Microscope, Globe, Users, Target, Brain, Database, Cpu, GitBranch, Mail, Linkedin } from "lucide-react"
import Link from "next/link"
import Image from "next/image"

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
              Smart India Hackathon 2024
            </Badge>
            <h1 className="text-4xl md:text-5xl font-bold text-foreground mb-6 text-balance">
              AI-Driven Deep-Sea <span className="text-primary">Biodiversity Assessment</span>
            </h1>
            <p className="text-xl text-muted-foreground text-pretty">
              Revolutionary eDNA analysis pipeline using machine learning to discover and classify marine life in Earth's most unexplored ecosystems
            </p>
          </div>
        </div>
      </section>

      {/* Project Overview */}
      <section className="py-16 bg-card/30">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-4">Project Overview</h2>
            <p className="text-lg text-muted-foreground max-w-3xl mx-auto">
              Addressing the critical challenge of poor database representation for deep-sea organisms in traditional reference databases
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-12 mb-16">
            <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
              <CardHeader>
                <Target className="h-12 w-12 text-primary mb-4" />
                <CardTitle className="text-2xl">The Problem</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-base leading-relaxed">
                  Traditional bioinformatic pipelines rely heavily on sequence alignment to databases built from terrestrial and shallow-water species. This leads to significant misclassifications and underestimation of deep-sea biodiversity, as reference databases like SILVA, PR2, and NCBI lack comprehensive deep-sea organism representation.
                </CardDescription>
              </CardContent>
            </Card>

            <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
              <CardHeader>
                <Brain className="h-12 w-12 text-accent mb-4" />
                <CardTitle className="text-2xl">Our Solution</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-base leading-relaxed">
                  An AI-driven pipeline using deep learning and unsupervised learning algorithms to identify eukaryotic taxa directly from raw eDNA reads. Our system minimizes reliance on reference databases while enabling discovery of novel taxa and reducing computational time through optimized workflows.
                </CardDescription>
              </CardContent>
            </Card>
          </div>

          {/* Project Images Section */}
          <div className="mb-16">
            <h3 className="text-2xl font-bold text-center text-foreground mb-8">Project Showcase</h3>
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
              <Card className="border-border/50 bg-card/50 backdrop-blur-sm overflow-hidden">
                <div className="aspect-video bg-gradient-to-br from-primary/20 to-accent/20 flex items-center justify-center">
                  <div className="text-center">
                    <Microscope className="h-16 w-16 text-primary mx-auto mb-2" />
                    <p className="text-sm text-muted-foreground">eDNA Sampling Process</p>
                  </div>
                </div>
              </Card>
              <Card className="border-border/50 bg-card/50 backdrop-blur-sm overflow-hidden">
                <div className="aspect-video bg-gradient-to-br from-accent/20 to-primary/20 flex items-center justify-center">
                  <div className="text-center">
                    <Brain className="h-16 w-16 text-accent mx-auto mb-2" />
                    <p className="text-sm text-muted-foreground">AI Pipeline Architecture</p>
                  </div>
                </div>
              </Card>
              <Card className="border-border/50 bg-card/50 backdrop-blur-sm overflow-hidden">
                <div className="aspect-video bg-gradient-to-br from-primary/20 to-accent/20 flex items-center justify-center">
                  <div className="text-center">
                    <Fish className="h-16 w-16 text-primary mx-auto mb-2" />
                    <p className="text-sm text-muted-foreground">Species Classification Results</p>
                  </div>
                </div>
              </Card>
            </div>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="py-20">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-4">How Our System Works</h2>
            <p className="text-lg text-muted-foreground max-w-3xl mx-auto">
              A comprehensive AI pipeline that transforms raw eDNA data into actionable biodiversity insights
            </p>
          </div>

          <div className="grid lg:grid-cols-4 gap-8 mb-16">
            <Card className="border-border/50 bg-gradient-to-br from-primary/10 to-transparent">
              <CardHeader className="text-center">
                <Database className="h-16 w-16 text-primary mx-auto mb-4" />
                <CardTitle className="text-xl">Data Collection</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-center">
                  Environmental DNA samples are collected from deep-sea environments and processed to extract 18S rRNA and COI marker genes for analysis.
                </CardDescription>
              </CardContent>
            </Card>

            <Card className="border-border/50 bg-gradient-to-br from-accent/10 to-transparent">
              <CardHeader className="text-center">
                <Cpu className="h-16 w-16 text-accent mx-auto mb-4" />
                <CardTitle className="text-xl">AI Processing</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-center">
                  Advanced machine learning algorithms including K-means, DBSCAN, and hierarchical clustering analyze sequence data without relying on traditional databases.
                </CardDescription>
              </CardContent>
            </Card>

            <Card className="border-border/50 bg-gradient-to-br from-primary/10 to-transparent">
              <CardHeader className="text-center">
                <GitBranch className="h-16 w-16 text-primary mx-auto mb-4" />
                <CardTitle className="text-xl">Classification</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-center">
                  Unsupervised learning identifies and classifies taxa across major groups: Annelida, Arthropoda, Chordata, Cnidaria, Echinodermata, Mollusca, and Porifera.
                </CardDescription>
              </CardContent>
            </Card>

            <Card className="border-border/50 bg-gradient-to-br from-accent/10 to-transparent">
              <CardHeader className="text-center">
                <Globe className="h-16 w-16 text-accent mx-auto mb-4" />
                <CardTitle className="text-xl">Discovery</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-center">
                  The system estimates abundance, identifies novel taxa, and provides comprehensive biodiversity assessments for conservation and research.
                </CardDescription>
              </CardContent>
            </Card>
          </div>

          {/* Technical Features */}
          <div className="bg-gradient-to-r from-primary/20 to-accent/20 rounded-lg p-8">
            <h3 className="text-2xl font-bold text-center text-foreground mb-8">Technical Capabilities</h3>
            <div className="grid md:grid-cols-3 gap-8 text-center">
              <div>
                <div className="text-3xl font-bold text-primary mb-2">3</div>
                <div className="text-muted-foreground">ML Algorithms</div>
                <div className="text-sm text-muted-foreground mt-1">K-means, DBSCAN, Hierarchical</div>
              </div>
              <div>
                <div className="text-3xl font-bold text-accent mb-2">7</div>
                <div className="text-muted-foreground">Taxonomic Groups</div>
                <div className="text-sm text-muted-foreground mt-1">Major marine phyla coverage</div>
              </div>
              <div>
                <div className="text-3xl font-bold text-primary mb-2">2</div>
                <div className="text-muted-foreground">Marker Genes</div>
                <div className="text-sm text-muted-foreground mt-1">18S rRNA & COI</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Team Members */}
      {/* <section className="py-20 bg-card/30">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-4">Meet Our Team</h2>
            <p className="text-lg text-muted-foreground max-w-3xl mx-auto">
              A dedicated group of researchers and developers working to revolutionize marine biodiversity assessment
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            
            <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
              <CardHeader className="text-center">
                <div className="w-24 h-24 mx-auto mb-4 rounded-full bg-gradient-to-br from-primary/20 to-accent/20 flex items-center justify-center">
                  <img src="/sameer.jpg" alt="Sameer Gupta" className="w-24 h-24 rounded-full" / >
                </div>
                <CardTitle className="text-xl">Sameer Gupta</CardTitle>
                <CardDescription>Project Lead & AI Researcher</CardDescription>
              </CardHeader>
              <CardContent className="text-center">
                <p className="text-sm text-muted-foreground mb-4">
                  Specializes in machine learning algorithms and deep-sea biodiversity research. Leading the development of our AI-driven eDNA analysis pipeline.
                </p>

                <div className="flex justify-center space-x-2">
                <Link href="mailto:sameer.gupta.ug24@nsut.ac.in">
                  <Button variant="outline" size="sm">
                    <Mail className="h-4 w-4" />
                  </Button></Link>
                  <Link href="https://linkedin.com/in/sameer-gupta-768b28312/" target="_blank" rel="noopener noreferrer">
                  <Button  variant="outline" size="sm">
                    <Linkedin  className="h-4 w-4" />
                  </Button>
                  </Link>
                </div>
              </CardContent>
            </Card>

            
            <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
              <CardHeader className="text-center">
                <div className="w-24 h-24 mx-auto mb-4 rounded-full bg-gradient-to-br from-accent/20 to-primary/20 flex items-center justify-center">
                  <img src="/anmol.jpg" alt="Anmol Mittal" className="w-24 h-24 rounded-full" />
                </div>
                <CardTitle className="text-xl">Anmol Mittal</CardTitle>
                <CardDescription>Bioinformatics Specialist</CardDescription>
              </CardHeader>
              <CardContent className="text-center">
                <p className="text-sm text-muted-foreground mb-4">
                  Expert in genomic data analysis and environmental DNA processing. Responsible for developing the core bioinformatics workflows and data preprocessing.
                </p>
                <div className="flex justify-center space-x-2">
                <Link href="mailto:anmol.mittal.ug24@nsut.ac.in">
                  <Button variant="outline" size="sm">
                    <Mail className="h-4 w-4" />
                  </Button>
                  </Link>
                  <Link href="https://www.linkedin.com/in/anmol-mittal-095b79312/?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app" target="_blank" rel="noopener noreferrer">
                  <Button variant="outline" size="sm">
                    <Linkedin className="h-4 w-4" />
                  </Button>
                  </Link>
                </div>
              </CardContent>
            </Card>


            <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
              <CardHeader className="text-center">
                <div className="w-24 h-24 mx-auto mb-4 rounded-full bg-gradient-to-br from-primary/20 to-accent/20 flex items-center justify-center">
                  <img src="/veda.jpg" alt="veda joshi" className="w-24 h-24 rounded-full" />
                </div>
                <CardTitle className="text-xl">Veda Joshi</CardTitle>
                <CardDescription>Full-Stack Developer</CardDescription>
              </CardHeader>
              <CardContent className="text-center">
                <p className="text-sm text-muted-foreground mb-4">
                  Develops the web application interface and backend systems. Creates intuitive user experiences for researchers to interact with our AI tools.
                </p>
                <div className="flex justify-center space-x-2">
                  <Link href="mailto:veda.joshi.ug24@nsut.ac.in">
                  <Button variant="outline" size="sm">
                    <Mail className="h-4 w-4" />
                  </Button></Link>
                  <Link href="https://www.linkedin.com/in/veda-joshi1409?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app" target="_blank" rel="noopener noreferrer">
                  <Button variant="outline" size="sm">
                    <Linkedin className="h-4 w-4" />
                  </Button>
                  </Link>
                </div>
              </CardContent>
            </Card>

            
            <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
              <CardHeader className="text-center">
                <div className="w-24 h-24 mx-auto mb-4 rounded-full bg-gradient-to-br from-accent/20 to-primary/20 flex items-center justify-center">
                  <img src="/sanidhya.jpg" alt="sanidhya upadhyay" className="w-24 h-24 rounded-full" />
                </div>
                <CardTitle className="text-xl">Sanidhya Upadhyay</CardTitle>
                <CardDescription>Marine Biology Consultant</CardDescription>
              </CardHeader>
              <CardContent className="text-center">
                <p className="text-sm text-muted-foreground mb-4">
                  Provides domain expertise in marine ecology and taxonomy. Ensures our AI models align with biological principles and conservation needs.
                </p>
                <div className="flex justify-center space-x-2">
                  <Link href="mailto:sanidhya.upadhyay.ug24@nsut.ac.in">
                  <Button variant="outline" size="sm">
                    <Mail className="h-4 w-4" />
                  </Button>
                  </Link>
                  <Link href="https://www.linkedin.com/in/sanidhyaupadhyay?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app" target="_blank" rel="noopener noreferrer">
                  <Button variant="outline" size="sm">
                    <Linkedin className="h-4 w-4" />
                  </Button>
                  </Link>
                </div>
              </CardContent>
            </Card>

           
            <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
              <CardHeader className="text-center">
                <div className="w-24 h-24 mx-auto mb-4 rounded-full bg-gradient-to-br from-primary/20 to-accent/20 flex items-center justify-center">
                  <img src="/kunal.jpg" alt="kunal verma" className="w-24 h-24 rounded-full" />
                </div>
                <CardTitle className="text-xl">Kunal Verma</CardTitle>
                <CardDescription>Data Scientist</CardDescription>
              </CardHeader>
              <CardContent className="text-center">
                <p className="text-sm text-muted-foreground mb-4">
                  Focuses on statistical analysis and model validation. Develops metrics and evaluation frameworks for assessing AI performance in biodiversity assessment.
                </p>
                <div className="flex justify-center space-x-2">
                  <Link href="mailto:kunal.verma.ug24@nsut.ac.in">
                  <Button variant="outline" size="sm">
                    <Mail className="h-4 w-4" />
                  </Button>
                  </Link>
                  <Link href="https://www.linkedin.com/in/kunalverma10/" target="_blank" rel="noopener noreferrer">
                  <Button variant="outline" size="sm">
                    <Linkedin className="h-4 w-4" />
                  </Button>
                  </Link>
                </div>
              </CardContent>
            </Card>

           
            <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
              <CardHeader className="text-center">
                <div className="w-24 h-24 mx-auto mb-4 rounded-full bg-gradient-to-br from-accent/20 to-primary/20 flex items-center justify-center">
                  <img src="/aniket.jpg" alt="aniket raj" className="w-24 h-24 rounded-full" />
                </div>
                <CardTitle className="text-xl">Aniket Raj</CardTitle>
                <CardDescription>UI/UX Designer</CardDescription>
              </CardHeader>
              <CardContent className="text-center">
                <p className="text-sm text-muted-foreground mb-4">
                  Designs intuitive interfaces and user experiences. Ensures our complex AI tools are accessible and user-friendly for marine researchers worldwide.
                </p>
                <div className="flex justify-center space-x-2">
                  <Link href="mailto:aniket.raj.ug24@nsut.ac.in">
                  <Button variant="outline" size="sm">
                    <Mail className="h-4 w-4" />
                  </Button>
                  </Link>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </section> */}

    </div>
  )
}
