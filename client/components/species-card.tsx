"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Info, Microscope } from "lucide-react"

interface SpeciesCardProps {
  taxon: {
    id: number
    taxonName: string
    classification: string
    abundance: number
    novelty: string
    confidence: number
    habitat: string
    clusterSize: number
    description?: string
    ecologicalRole?: string
  }
  onClose?: () => void
  compact?: boolean
}

export function SpeciesCard({ taxon, onClose, compact = false }: SpeciesCardProps) {
  if (compact) {
    return (
      <Card className="border-border/50 bg-card/95 backdrop-blur-sm shadow-lg max-w-sm">
        <CardHeader className="pb-2">
          <div className="flex items-center space-x-2">
            <Microscope className="h-4 w-4 text-primary" />
            <CardTitle className="text-sm">{taxon.taxonName}</CardTitle>
          </div>
          <CardDescription className="text-xs">{taxon.classification}</CardDescription>
        </CardHeader>
        <CardContent className="pt-0">
          <div className="space-y-2">
            <div className="flex justify-between text-xs">
              <span>Abundance:</span>
              <span className="font-semibold">{taxon.abundance}%</span>
            </div>
            <div className="flex justify-between text-xs">
              <span>Confidence:</span>
              <span className="font-semibold">{(taxon.confidence * 100).toFixed(0)}%</span>
            </div>
            <Badge
              variant={taxon.novelty === "Novel" ? "destructive" : "secondary"}
              className={`text-xs ${taxon.novelty === "Novel" ? "bg-primary text-primary-foreground" : ""}`}
            >
              {taxon.novelty}
            </Badge>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="border-border/50 bg-card backdrop-blur-sm">
      <CardHeader>
        <div className="flex items-start justify-between">
          <div className="flex items-start space-x-3">
            <div className="w-10 h-10 bg-primary/20 rounded-lg flex items-center justify-center flex-shrink-0">
              <Microscope className="h-5 w-5 text-primary" />
            </div>
            <div>
              <CardTitle className="text-lg">{taxon.taxonName}</CardTitle>
              <CardDescription className="text-sm mt-1">{taxon.classification}</CardDescription>
            </div>
          </div>
          {onClose && (
            <Button variant="ghost" size="sm" onClick={onClose}>
              ×
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {taxon.description && <p className="text-sm text-muted-foreground">{taxon.description}</p>}

        <div className="grid grid-cols-2 gap-4">
          <div>
            <div className="text-lg font-bold text-primary">{taxon.abundance}%</div>
            <div className="text-xs text-muted-foreground">Abundance</div>
          </div>
          <div>
            <div className="text-lg font-bold text-foreground">{taxon.clusterSize}</div>
            <div className="text-xs text-muted-foreground">Cluster Size</div>
          </div>
        </div>

        <div className="flex items-center justify-between">
          <Badge
            variant={taxon.novelty === "Novel" ? "destructive" : "secondary"}
            className={taxon.novelty === "Novel" ? "bg-primary text-primary-foreground" : ""}
          >
            {taxon.novelty}
          </Badge>
          <div className="flex items-center space-x-2">
            <div className="w-16 bg-muted rounded-full h-2">
              <div className="bg-primary h-2 rounded-full" style={{ width: `${taxon.confidence * 100}%` }}></div>
            </div>
            <span className="text-xs text-muted-foreground">{(taxon.confidence * 100).toFixed(0)}%</span>
          </div>
        </div>

        {taxon.novelty === "Novel" && (
          <div className="bg-primary/10 border border-primary/20 rounded p-3">
            <div className="flex items-start space-x-2">
              <Info className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
              <p className="text-xs text-muted-foreground">
                Potential novel species requiring further taxonomic investigation
              </p>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
