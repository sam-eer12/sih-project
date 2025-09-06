"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { ArrowLeft, Search, Filter, Download, Eye, Info, ExternalLink, Microscope } from "lucide-react"
import Link from "next/link"
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts"

// Sample data for demonstration
const biodiversityData = [
  {
    name: "Protists",
    value: 35,
    count: 142,
    color: "hsl(var(--chart-1))",
    description: "Single-celled eukaryotic organisms including foraminifera, radiolarians, and various microbes",
  },
  {
    name: "Cnidaria",
    value: 18,
    count: 73,
    color: "hsl(var(--chart-2))",
    description: "Marine invertebrates including jellyfish, corals, sea anemones, and hydroids",
  },
  {
    name: "Arthropoda",
    value: 15,
    count: 61,
    color: "hsl(var(--chart-3))",
    description: "Joint-legged invertebrates including deep-sea crustaceans, copepods, and amphipods",
  },
  {
    name: "Mollusca",
    value: 12,
    count: 49,
    color: "hsl(var(--chart-4))",
    description: "Soft-bodied invertebrates including deep-sea clams, snails, and cephalopods",
  },
  {
    name: "Fungi",
    value: 8,
    count: 32,
    color: "hsl(var(--chart-5))",
    description: "Marine fungi including yeasts and filamentous species adapted to deep-sea conditions",
  },
  {
    name: "Unknown/Novel",
    value: 12,
    count: 48,
    color: "hsl(var(--primary))",
    description: "Unclassified sequences representing potentially novel taxa requiring further investigation",
  },
]

const diversityMetrics = [
  { metric: "Species Richness", value: 405, description: "Total number of identified taxa" },
  { metric: "Shannon Index", value: 3.42, description: "Measure of species diversity" },
  { metric: "Simpson Index", value: 0.89, description: "Probability of diversity" },
  { metric: "Novel Taxa", value: 48, description: "Potentially new species detected" },
]

const abundanceData = [
  { depth: "0-200m", species: 45, abundance: 1250 },
  { depth: "200-500m", species: 78, abundance: 2100 },
  { depth: "500-1000m", species: 92, abundance: 1850 },
  { depth: "1000-2000m", species: 115, abundance: 1650 },
  { depth: "2000m+", species: 75, abundance: 980 },
]

const taxaData = [
  {
    id: 1,
    taxonName: "Bathymodiolus thermophilus",
    classification: "Mollusca > Bivalvia > Mytilida",
    abundance: 8.5,
    novelty: "Known",
    confidence: 0.95,
    habitat: "Hydrothermal vents",
    clusterSize: 127,
    description:
      "A deep-sea mussel species that forms symbiotic relationships with chemosynthetic bacteria, commonly found around hydrothermal vents.",
    ecologicalRole: "Primary consumer in chemosynthetic ecosystems",
  },
  {
    id: 2,
    taxonName: "Alvinella pompejana",
    classification: "Annelida > Polychaeta > Terebellida",
    abundance: 6.2,
    novelty: "Known",
    confidence: 0.92,
    habitat: "Deep-sea vents",
    clusterSize: 89,
    description:
      "The Pompeii worm, one of the most heat-tolerant complex animals known, living in extreme temperatures near hydrothermal vents.",
    ecologicalRole: "Extremophile grazer and biofilm consumer",
  },
  {
    id: 3,
    taxonName: "Novel Cnidarian Cluster A",
    classification: "Cnidaria > Unknown > Unknown",
    abundance: 4.8,
    novelty: "Novel",
    confidence: 0.78,
    habitat: "Abyssal plains",
    clusterSize: 156,
    description:
      "An uncharacterized cnidarian group showing unique genetic signatures not matching any known species in current databases.",
    ecologicalRole: "Unknown - requires further taxonomic investigation",
  },
  {
    id: 4,
    taxonName: "Pyrococcus furiosus",
    classification: "Archaea > Thermococci > Thermococcales",
    abundance: 12.3,
    novelty: "Known",
    confidence: 0.98,
    habitat: "Extreme thermophile",
    clusterSize: 203,
    description:
      "A hyperthermophilic archaeon capable of growing at temperatures up to 100°C, important for understanding life in extreme conditions.",
    ecologicalRole: "Primary producer in high-temperature environments",
  },
  {
    id: 5,
    taxonName: "Deep-sea Protist Sp. 1",
    classification: "Protista > Unknown > Unknown",
    abundance: 3.7,
    novelty: "Novel",
    confidence: 0.65,
    habitat: "Hadal zone",
    clusterSize: 94,
    description:
      "A potentially novel protist species from the deepest ocean trenches, representing an unexplored branch of eukaryotic diversity.",
    ecologicalRole: "Likely microbial predator in hadal ecosystems",
  },
  {
    id: 6,
    taxonName: "Calyptogena magnifica",
    classification: "Mollusca > Bivalvia > Venerida",
    abundance: 5.9,
    novelty: "Known",
    confidence: 0.89,
    habitat: "Cold seeps",
    clusterSize: 112,
    description:
      "A large deep-sea clam that harbors chemosynthetic bacteria in its gills, forming the foundation of cold seep ecosystems.",
    ecologicalRole: "Ecosystem engineer and primary consumer",
  },
]

export default function DashboardPage() {
  const [searchTerm, setSearchTerm] = useState("")
  const [noveltyFilter, setNoveltyFilter] = useState("all")
  const [selectedTaxon, setSelectedTaxon] = useState<(typeof taxaData)[0] | null>(null)
  const [hoveredSegment, setHoveredSegment] = useState<(typeof biodiversityData)[0] | null>(null)

  const filteredTaxa = taxaData.filter((taxon) => {
    const matchesSearch =
      taxon.taxonName.toLowerCase().includes(searchTerm.toLowerCase()) ||
      taxon.classification.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesNovelty = noveltyFilter === "all" || taxon.novelty.toLowerCase() === noveltyFilter
    return matchesSearch && matchesNovelty
  })

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload
      return (
        <Card className="border-border/50 bg-card/95 backdrop-blur-sm shadow-lg max-w-xs">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-white">{data.name}</CardTitle>
          </CardHeader>
          <CardContent className="pt-0">
            <p className="text-xs text-white/80 mb-2">{data.description}</p>
            <div className="space-y-1">
              <div className="flex justify-between text-xs">
                <span className="text-white/70">Abundance:</span>
                <span className="font-semibold text-white">{data.value}%</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-white/70">Species Count:</span>
                <span className="font-semibold text-white">{data.count}</span>
              </div>
            </div>
          </CardContent>
        </Card>
      )
    }
    return null
  }

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
              <Link href="/solution" className="text-muted-foreground hover:text-foreground transition-colors">
                Solution
              </Link>
              <Button variant="outline" size="sm">
                <Download className="h-4 w-4 mr-2" />
                Export Data
              </Button>
            </div>
          </div>
        </div>
      </nav>

      {/* Dashboard Header */}
      <section className="py-12 bg-card/30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-8">
            <Badge variant="secondary" className="mb-4">
              Interactive Dashboard
            </Badge>
            <h1 className="text-3xl md:text-4xl font-bold text-foreground mb-4">
              Deep-Sea eDNA <span className="text-primary">Biodiversity Analysis</span>
            </h1>
            <p className="text-lg text-muted-foreground max-w-3xl mx-auto">
              Explore AI-generated insights from environmental DNA samples collected from deep-sea ecosystems
            </p>
          </div>

          {/* Key Metrics */}
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {diversityMetrics.map((metric, index) => (
              <Card key={index} className="border-border/50 bg-card/50 backdrop-blur-sm">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm font-medium text-muted-foreground">{metric.metric}</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold text-foreground mb-1">{metric.value}</div>
                  <p className="text-xs text-muted-foreground">{metric.description}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Main Dashboard Content */}
      <section className="py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid lg:grid-cols-2 gap-8 mb-12">
            {/* Biodiversity Pie Chart */}
            <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="text-xl">Taxonomic Distribution</CardTitle>
                <CardDescription>
                  Relative abundance of major taxonomic groups identified in deep-sea samples
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={biodiversityData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, value }) => `${name}: ${value}%`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                      onMouseEnter={(data) => setHoveredSegment(data)}
                      onMouseLeave={() => setHoveredSegment(null)}
                    >
                      {biodiversityData.map((entry, index) => (
                        <Cell
                          key={`cell-${index}`}
                          fill={entry.color}
                          stroke={hoveredSegment?.name === entry.name ? "#ffffff" : "none"}
                          strokeWidth={hoveredSegment?.name === entry.name ? 2 : 0}
                        />
                      ))}
                    </Pie>
                    <Tooltip content={<CustomTooltip />} />
                  </PieChart>
                </ResponsiveContainer>

                <div className="grid grid-cols-2 gap-2 mt-4">
                  {biodiversityData.map((item, index) => (
                    <div
                      key={index}
                      className="flex items-center space-x-2 text-sm p-2 rounded hover:bg-muted/50 cursor-pointer transition-colors"
                      onMouseEnter={() => setHoveredSegment(item)}
                      onMouseLeave={() => setHoveredSegment(null)}
                    >
                      <div className="w-3 h-3 rounded-full" style={{ backgroundColor: item.color }}></div>
                      <span className="text-muted-foreground">{item.name}</span>
                      <span className="text-xs text-muted-foreground">({item.count})</span>
                    </div>
                  ))}
                </div>

                {hoveredSegment && (
                  <Card className="mt-4 border-border/50 bg-primary/5">
                    <CardContent className="pt-4">
                      <h4 className="font-semibold text-foreground mb-2">{hoveredSegment.name}</h4>
                      <p className="text-sm text-muted-foreground">{hoveredSegment.description}</p>
                    </CardContent>
                  </Card>
                )}
              </CardContent>
            </Card>

            {/* Depth Distribution */}
            <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="text-xl">Species Abundance by Depth</CardTitle>
                <CardDescription>Distribution of species richness across different depth zones</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={abundanceData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
                    <XAxis dataKey="depth" stroke="#ffffff" fontSize={12} />
                    <YAxis stroke="#ffffff" fontSize={12} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "hsl(var(--card))",
                        border: "1px solid hsl(var(--border))",
                        borderRadius: "6px",
                        color: "#ffffff",
                      }}
                    />
                    <Bar dataKey="species" fill="hsl(var(--primary))" name="Species Count" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>

          {/* Taxa Table */}
          <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
            <CardHeader>
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                <div>
                  <CardTitle className="text-xl">Identified Taxa</CardTitle>
                  <CardDescription>
                    Detailed list of species identified through AI analysis with confidence scores
                  </CardDescription>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      placeholder="Search taxa..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="pl-9 w-64"
                    />
                  </div>
                  <Select value={noveltyFilter} onValueChange={setNoveltyFilter}>
                    <SelectTrigger className="w-32">
                      <Filter className="h-4 w-4 mr-2" />
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All</SelectItem>
                      <SelectItem value="known">Known</SelectItem>
                      <SelectItem value="novel">Novel</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="rounded-md border border-border/50">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Taxon Name</TableHead>
                      <TableHead>Classification</TableHead>
                      <TableHead>Abundance (%)</TableHead>
                      <TableHead>Novelty</TableHead>
                      <TableHead>Confidence</TableHead>
                      <TableHead>Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {filteredTaxa.map((taxon) => (
                      <TableRow
                        key={taxon.id}
                        className="hover:bg-muted/50 cursor-pointer"
                        onClick={() => setSelectedTaxon(taxon)}
                      >
                        <TableCell className="font-medium">{taxon.taxonName}</TableCell>
                        <TableCell className="text-sm text-muted-foreground">{taxon.classification}</TableCell>
                        <TableCell>{taxon.abundance}%</TableCell>
                        <TableCell>
                          <Badge
                            variant={taxon.novelty === "Novel" ? "destructive" : "secondary"}
                            className={taxon.novelty === "Novel" ? "bg-primary text-primary-foreground" : ""}
                          >
                            {taxon.novelty}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          <div className="flex items-center space-x-2">
                            <div className="w-16 bg-muted rounded-full h-2">
                              <div
                                className="bg-primary h-2 rounded-full"
                                style={{ width: `${taxon.confidence * 100}%` }}
                              ></div>
                            </div>
                            <span className="text-sm text-muted-foreground">
                              {(taxon.confidence * 100).toFixed(0)}%
                            </span>
                          </div>
                        </TableCell>
                        <TableCell>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={(e) => {
                              e.stopPropagation()
                              setSelectedTaxon(taxon)
                            }}
                          >
                            <Eye className="h-4 w-4 mr-1" />
                            View
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
              <div className="flex items-center justify-between mt-4">
                <p className="text-sm text-muted-foreground">
                  Showing {filteredTaxa.length} of {taxaData.length} taxa
                </p>
                <div className="flex items-center space-x-2">
                  <Button variant="outline" size="sm" disabled>
                    Previous
                  </Button>
                  <Button variant="outline" size="sm" disabled>
                    Next
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Enhanced Species Detail Modal/Card */}
      {selectedTaxon && (
        <div className="fixed inset-0 bg-background/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <Card className="w-full max-w-4xl border-border/50 bg-card backdrop-blur-sm max-h-[90vh] overflow-y-auto">
            <CardHeader>
              <div className="flex items-start justify-between">
                <div className="flex items-start space-x-4">
                  <div className="w-12 h-12 bg-primary/20 rounded-lg flex items-center justify-center flex-shrink-0">
                    <Microscope className="h-6 w-6 text-primary" />
                  </div>
                  <div>
                    <CardTitle className="text-2xl">{selectedTaxon.taxonName}</CardTitle>
                    <CardDescription className="text-base mt-2">{selectedTaxon.classification}</CardDescription>
                    <Badge
                      variant={selectedTaxon.novelty === "Novel" ? "destructive" : "secondary"}
                      className={`mt-2 ${selectedTaxon.novelty === "Novel" ? "bg-primary text-primary-foreground" : ""}`}
                    >
                      {selectedTaxon.novelty === "Novel" ? "Potential Novel Species" : "Known Species"}
                    </Badge>
                  </div>
                </div>
                <Button variant="ghost" size="sm" onClick={() => setSelectedTaxon(null)} className="text-xl">
                  ×
                </Button>
              </div>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="bg-muted/30 rounded-lg p-4">
                <h4 className="font-semibold text-foreground mb-2">Species Description</h4>
                <p className="text-muted-foreground leading-relaxed">{selectedTaxon.description}</p>
              </div>

              <div className="grid md:grid-cols-3 gap-6">
                <div className="space-y-4">
                  <div>
                    <h4 className="font-semibold text-foreground mb-2">Abundance</h4>
                    <div className="text-2xl font-bold text-primary">{selectedTaxon.abundance}%</div>
                    <p className="text-sm text-muted-foreground">Relative abundance in sample</p>
                  </div>
                  <div>
                    <h4 className="font-semibold text-foreground mb-2">Confidence Score</h4>
                    <div className="flex items-center space-x-2">
                      <div className="w-24 bg-muted rounded-full h-3">
                        <div
                          className="bg-primary h-3 rounded-full"
                          style={{ width: `${selectedTaxon.confidence * 100}%` }}
                        ></div>
                      </div>
                      <span className="text-sm font-medium">{(selectedTaxon.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <p className="text-xs text-muted-foreground mt-1">AI classification confidence</p>
                  </div>
                </div>
                <div className="space-y-4">
                  <div>
                    <h4 className="font-semibold text-foreground mb-2">Cluster Size</h4>
                    <div className="text-lg font-semibold text-foreground">{selectedTaxon.clusterSize} reads</div>
                    <p className="text-sm text-muted-foreground">Number of sequences in cluster</p>
                  </div>
                  <div>
                    <h4 className="font-semibold text-foreground mb-2">Habitat</h4>
                    <p className="text-muted-foreground">{selectedTaxon.habitat}</p>
                  </div>
                </div>
                <div className="space-y-4">
                  <div>
                    <h4 className="font-semibold text-foreground mb-2">Ecological Role</h4>
                    <p className="text-muted-foreground text-sm">{selectedTaxon.ecologicalRole}</p>
                  </div>
                  <div>
                    <h4 className="font-semibold text-foreground mb-2">Taxonomic Hierarchy</h4>
                    <div className="space-y-1">
                      {selectedTaxon.classification.split(" > ").map((level, index) => (
                        <div key={index} className="text-sm">
                          <span className="text-muted-foreground">
                            {["Kingdom", "Phylum", "Class"][index] || "Order"}:
                          </span>
                          <span className="ml-2 font-medium">{level}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              {selectedTaxon.novelty === "Novel" && (
                <div className="bg-primary/10 border border-primary/20 rounded-lg p-4">
                  <div className="flex items-start space-x-2">
                    <Info className="h-5 w-5 text-primary mt-0.5 flex-shrink-0" />
                    <div>
                      <h4 className="font-semibold text-foreground mb-1">Potential Novel Species</h4>
                      <p className="text-sm text-muted-foreground mb-3">
                        This cluster represents a potentially new species that doesn't match existing database entries.
                        Further taxonomic investigation is recommended.
                      </p>
                      <div className="space-y-2">
                        <h5 className="text-sm font-medium text-foreground">Recommended Next Steps:</h5>
                        <ul className="text-xs text-muted-foreground space-y-1">
                          <li>• Morphological analysis of collected specimens</li>
                          <li>• Extended phylogenetic analysis</li>
                          <li>• Comparison with closely related taxa</li>
                          <li>• Environmental context documentation</li>
                        </ul>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              <div className="flex flex-wrap gap-3 pt-4 border-t border-border/50">
                <Button variant="outline" size="sm">
                  <ExternalLink className="h-4 w-4 mr-2" />
                  View in Database
                </Button>
                <Button variant="outline" size="sm">
                  <Download className="h-4 w-4 mr-2" />
                  Export Sequences
                </Button>
                <Button variant="outline" size="sm">
                  <Info className="h-4 w-4 mr-2" />
                  Research Notes
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}
