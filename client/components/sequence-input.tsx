"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Loader2, Zap, Info, CheckCircle, AlertCircle } from "lucide-react"
import { apiService, utils, SequenceClassificationResponse } from "@/lib/api"

interface SequenceInputProps {
  onResults?: (results: SequenceClassificationResponse) => void;
}

export function SequenceInput({ onResults }: SequenceInputProps) {
  const [sequence, setSequence] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [results, setResults] = useState<SequenceClassificationResponse | null>(null)

  const handleSequenceChange = (value: string) => {
    setSequence(value)
    setError(null)
    setResults(null)
  }

  const handleSubmit = async () => {
    if (!sequence.trim()) {
      setError("Please enter a DNA sequence")
      return
    }

    // Validate sequence
    const validation = utils.validateDnaSequence(sequence)
    if (!validation.isValid) {
      setError(validation.error || "Invalid sequence")
      return
    }

    setIsLoading(true)
    setError(null)

    try {
      const response = await apiService.classifySequence({
        sequence: validation.cleanSequence!
      })
      
      setResults(response)
      onResults?.(response)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Classification failed")
    } finally {
      setIsLoading(false)
    }
  }

  const handleClear = () => {
    setSequence("")
    setError(null)
    setResults(null)
  }

  const handleLoadExample = () => {
    // Example COI sequence from Chordata (fish)
    const exampleSequence = `ACTCTTTACTTAATCTTCGGCGCTTGGGCCGGGATAGTAGGAACAGCCCTTAGCCTGCTCATTCGAGCAGAACTTAGTCAACCCGGCGCCCTGTTGGGGGATGACCAAATTTATAATGTAATTGTTACCGCTCATGCCTTTGTAATAATCTTCTTTATGGTGATGCCAATTATAATCGGAGGTTTTGGAAATTGACTTATCCCCCTTATGATTGGGGCTCCTGACATGGCTTTTCCTCGAATAAATAATATGAGCTTTTGGCTCTTGCCACCCTCTTTTCTGCTCTTGCTAGCTTCGTCAGGTGTTGAGGCTGGGGCAGGGACCGGGTGGACTGTCTACCCTCCCCTTTCTGGAAATTTAGCCCATGCAGGGGGTTCCGTTGATTTAACTATTTTTTCTCTACATTTAGCAGGCATCTCTTCTATTTTAGGAGCAATTAATTTTATTACAACAATTATCAACATGAAGCCCCCTGCTATCTCTCAGTACCAGACCCCTTTGTTCGTGTGGTCTGTGTTAATTACTGCTGTTCTTCTACTTCTTTCACTTCCTGTTCTAGCTGCTGGTATTACTATACTTCTTACGGACCGAAATCTTAACACCACCTTCTTTGATCCTGCAGGAGGGGGGGACCCCATCCTTTACCAACATCTCTT`
    setSequence(exampleSequence)
    setError(null)
    setResults(null)
  }

  // Calculate sequence stats for display
  const sequenceStats = sequence.trim() ? {
    length: sequence.trim().replace(/\s+/g, '').length,
    gcContent: utils.calculateGcContent(sequence),
    composition: utils.getSequenceComposition(sequence)
  } : null

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5" />
            DNA Sequence Classification
          </CardTitle>
          <CardDescription>
            Enter a DNA sequence (18S rRNA or COI) to classify and identify potential taxa
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="sequence">DNA Sequence</Label>
            <Textarea
              id="sequence"
              placeholder="Enter your DNA sequence here (ATCG format)..."
              value={sequence}
              onChange={(e) => handleSequenceChange(e.target.value)}
              className="min-h-[120px] font-mono text-sm"
              disabled={isLoading}
            />
            <div className="flex items-center justify-between text-sm text-muted-foreground">
              <span>
                {sequence.trim() ? `${sequence.trim().replace(/\s+/g, '').length} bp` : "0 bp"}
              </span>
              <Button
                variant="ghost"
                size="sm"
                onClick={handleLoadExample}
                disabled={isLoading}
              >
                Load Example
              </Button>
            </div>
          </div>

          {sequenceStats && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 p-4 bg-muted/50 rounded-lg">
              <div className="text-center">
                <div className="text-lg font-semibold">{sequenceStats.length}</div>
                <div className="text-xs text-muted-foreground">Base pairs</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-semibold">{(sequenceStats.gcContent * 100).toFixed(1)}%</div>
                <div className="text-xs text-muted-foreground">GC content</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-semibold">
                  {sequenceStats.composition.A + sequenceStats.composition.T}
                </div>
                <div className="text-xs text-muted-foreground">A+T bases</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-semibold">
                  {sequenceStats.composition.G + sequenceStats.composition.C}
                </div>
                <div className="text-xs text-muted-foreground">G+C bases</div>
              </div>
            </div>
          )}

          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          <div className="flex gap-2">
            <Button
              onClick={handleSubmit}
              disabled={isLoading || !sequence.trim()}
              className="flex-1"
            >
              {isLoading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Classifying...
                </>
              ) : (
                "Classify Sequence"
              )}
            </Button>
            <Button
              variant="outline"
              onClick={handleClear}
              disabled={isLoading}
            >
              Clear
            </Button>
          </div>
        </CardContent>
      </Card>

      {results && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <CheckCircle className="h-5 w-5 text-green-500" />
              Classification Results
            </CardTitle>
            <CardDescription>
              Analysis ID: {results.analysis_id}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Primary Classification */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <h4 className="font-semibold">Primary Classification</h4>
                <div className="space-y-1">
                  <div className="flex items-center justify-between">
                    <span>Predicted Phylum:</span>
                    <Badge variant="default">{results.classification.predicted_phylum}</Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Confidence:</span>
                    <Badge variant={results.classification.confidence > 0.8 ? "default" : "secondary"}>
                      {(results.classification.confidence * 100).toFixed(1)}%
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Cluster ID:</span>
                    <Badge variant="outline">{results.classification.cluster_id}</Badge>
                  </div>
                </div>
              </div>

              <div className="space-y-2">
                <h4 className="font-semibold">Sequence Information</h4>
                <div className="space-y-1 text-sm">
                  <div className="flex justify-between">
                    <span>Length:</span>
                    <span>{results.sequence_info.length} bp</span>
                  </div>
                  <div className="flex justify-between">
                    <span>GC Content:</span>
                    <span>{results.sequence_info.gc_content}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Similarity Score:</span>
                    <span>{(results.classification.similarity_score * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Taxonomic Hierarchy */}
            <div className="space-y-2">
              <h4 className="font-semibold">Taxonomic Classification</h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
                <div>
                  <span className="text-muted-foreground">Domain:</span>
                  <div className="font-medium">{results.taxonomy.domain}</div>
                </div>
                <div>
                  <span className="text-muted-foreground">Kingdom:</span>
                  <div className="font-medium">{results.taxonomy.kingdom}</div>
                </div>
                <div>
                  <span className="text-muted-foreground">Phylum:</span>
                  <div className="font-medium">{results.taxonomy.phylum}</div>
                </div>
                <div>
                  <span className="text-muted-foreground">Class:</span>
                  <div className="font-medium">{results.taxonomy.class}</div>
                </div>
              </div>
            </div>

            {/* Biodiversity Metrics */}
            <div className="space-y-2">
              <h4 className="font-semibold">Biodiversity Assessment</h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="text-center p-3 bg-muted/50 rounded-lg">
                  <div className="text-lg font-semibold">
                    {(results.biodiversity_metrics.novelty_score * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-muted-foreground">Novelty Score</div>
                </div>
                <div className="text-center p-3 bg-muted/50 rounded-lg">
                  <div className="text-lg font-semibold">
                    {results.biodiversity_metrics.is_potential_novel_taxa ? "Yes" : "No"}
                  </div>
                  <div className="text-xs text-muted-foreground">Novel Taxa</div>
                </div>
                <div className="text-center p-3 bg-muted/50 rounded-lg">
                  <div className="text-lg font-semibold">
                    {(results.biodiversity_metrics.cluster_diversity * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-muted-foreground">Cluster Diversity</div>
                </div>
              </div>
            </div>

            {results.biodiversity_metrics.is_potential_novel_taxa && (
              <Alert>
                <Info className="h-4 w-4" />
                <AlertDescription>
                  This sequence shows characteristics of a potential novel taxa. 
                  Consider further taxonomic investigation and validation.
                </AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  )
}
