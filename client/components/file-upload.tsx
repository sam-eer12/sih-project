"use client"

import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Progress } from "@/components/ui/progress"
import { 
  Upload, 
  File, 
  FileText, 
  CheckCircle, 
  AlertCircle, 
  Loader2, 
  Download,
  Eye,
  BarChart3
} from "lucide-react"
import { apiService, FileClassificationResponse } from "@/lib/api"

interface FileUploadProps {
  onResults?: (results: FileClassificationResponse) => void;
}

export function FileUpload({ onResults }: FileUploadProps) {
  const [file, setFile] = useState<File | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [results, setResults] = useState<FileClassificationResponse | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const supportedFormats = ['csv', 'fasta', 'fa', 'fas', 'txt']
  const maxFileSize = 16 * 1024 * 1024 // 16MB

  const handleFileSelect = (selectedFile: File) => {
    // Validate file type
    const fileExtension = selectedFile.name.split('.').pop()?.toLowerCase()
    if (!fileExtension || !supportedFormats.includes(fileExtension)) {
      setError(`Unsupported file format. Supported formats: ${supportedFormats.join(', ').toUpperCase()}`)
      return
    }

    // Validate file size
    if (selectedFile.size > maxFileSize) {
      setError(`File too large. Maximum size is ${maxFileSize / (1024 * 1024)}MB`)
      return
    }

    setFile(selectedFile)
    setError(null)
    setResults(null)
    setUploadProgress(0)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    const droppedFile = e.dataTransfer.files[0]
    if (droppedFile) {
      handleFileSelect(droppedFile)
    }
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
  }

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]
    if (selectedFile) {
      handleFileSelect(selectedFile)
    }
  }

  const handleUpload = async () => {
    if (!file) return

    setIsLoading(true)
    setError(null)
    setUploadProgress(0)

    try {
      // Simulate upload progress
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval)
            return 90
          }
          return prev + 10
        })
      }, 200)

      const response = await apiService.classifyFile(file)
      
      clearInterval(progressInterval)
      setUploadProgress(100)
      
      setTimeout(() => {
        setResults(response)
        onResults?.(response)
        setUploadProgress(0)
      }, 500)

    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed")
      setUploadProgress(0)
    } finally {
      setIsLoading(false)
    }
  }

  const handleClear = () => {
    setFile(null)
    setError(null)
    setResults(null)
    setUploadProgress(0)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const getFileIcon = (filename: string) => {
    const extension = filename.split('.').pop()?.toLowerCase()
    switch (extension) {
      case 'csv':
        return <BarChart3 className="h-8 w-8 text-green-500" />
      case 'fasta':
      case 'fa':
        return <FileText className="h-8 w-8 text-blue-500" />
      default:
        return <File className="h-8 w-8 text-gray-500" />
    }
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Upload className="h-5 w-5" />
            File Upload & Batch Processing
          </CardTitle>
          <CardDescription>
            Upload CSV (abundance data) or FASTA files for batch classification and biodiversity analysis
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {!file ? (
            <div
              className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-8 text-center hover:border-muted-foreground/50 transition-colors cursor-pointer"
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onClick={() => fileInputRef.current?.click()}
            >
              <Upload className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
              <h3 className="text-lg font-semibold mb-2">Drop your file here</h3>
              <p className="text-muted-foreground mb-4">
                or click to browse files
              </p>
              <div className="flex flex-wrap justify-center gap-2 mb-4">
                {supportedFormats.map(format => (
                  <Badge key={format} variant="outline">
                    .{format.toUpperCase()}
                  </Badge>
                ))}
              </div>
              <p className="text-sm text-muted-foreground">
                Maximum file size: 16MB
              </p>
              <input
                ref={fileInputRef}
                type="file"
                accept={supportedFormats.map(f => `.${f}`).join(',')}
                onChange={handleFileInputChange}
                className="hidden"
              />
            </div>
          ) : (
            <div className="border rounded-lg p-4 bg-muted/50">
              <div className="flex items-center gap-3">
                {getFileIcon(file.name)}
                <div className="flex-1 min-w-0">
                  <h4 className="font-medium truncate">{file.name}</h4>
                  <p className="text-sm text-muted-foreground">
                    {formatFileSize(file.size)} • {file.type || 'Unknown type'}
                  </p>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleClear}
                  disabled={isLoading}
                >
                  Remove
                </Button>
              </div>
            </div>
          )}

          {uploadProgress > 0 && (
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Processing...</span>
                <span>{uploadProgress}%</span>
              </div>
              <Progress value={uploadProgress} />
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
              onClick={handleUpload}
              disabled={!file || isLoading}
              className="flex-1"
            >
              {isLoading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  <Upload className="mr-2 h-4 w-4" />
                  Upload & Analyze
                </>
              )}
            </Button>
          </div>

          {/* File Format Information */}
          <div className="text-sm text-muted-foreground space-y-2">
            <h4 className="font-medium text-foreground">Supported File Formats:</h4>
            <ul className="space-y-1 ml-4">
              <li><strong>CSV:</strong> Abundance data with ASV and taxonomy columns</li>
              <li><strong>FASTA/FA:</strong> DNA sequences in FASTA format</li>
              <li><strong>FAS:</strong> Plain sequence files (one sequence per line)</li>
              <li><strong>TXT:</strong> Plain text files with sequences</li>
            </ul>
          </div>
        </CardContent>
      </Card>

      {results && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <CheckCircle className="h-5 w-5 text-green-500" />
              Analysis Results
            </CardTitle>
            <CardDescription>
              Analysis ID: {results.analysis_id} • File: {results.file_info.filename}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* File Information */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="text-center p-3 bg-muted/50 rounded-lg">
                <div className="text-lg font-semibold">{results.file_info.file_type}</div>
                <div className="text-xs text-muted-foreground">File Type</div>
              </div>
              <div className="text-center p-3 bg-muted/50 rounded-lg">
                <div className="text-lg font-semibold">
                  {results.file_info.total_asvs || results.file_info.total_sequences || 0}
                </div>
                <div className="text-xs text-muted-foreground">
                  {results.file_info.file_type === 'CSV' ? 'ASVs' : 'Sequences'}
                </div>
              </div>
              <div className="text-center p-3 bg-muted/50 rounded-lg">
                <div className="text-lg font-semibold">
                  {results.file_info.avg_length ? `${Math.round(results.file_info.avg_length)} bp` : 'N/A'}
                </div>
                <div className="text-xs text-muted-foreground">Avg Length</div>
              </div>
            </div>

            {/* Classification Summary for CSV */}
            {results.classification_summary && (
              <div className="space-y-4">
                <h4 className="font-semibold">Classification Summary</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center p-3 bg-green-50 dark:bg-green-950 rounded-lg">
                    <div className="text-lg font-semibold text-green-700 dark:text-green-300">
                      {results.classification_summary.total_classified}
                    </div>
                    <div className="text-xs text-green-600 dark:text-green-400">Classified</div>
                  </div>
                  <div className="text-center p-3 bg-orange-50 dark:bg-orange-950 rounded-lg">
                    <div className="text-lg font-semibold text-orange-700 dark:text-orange-300">
                      {results.classification_summary.unassigned}
                    </div>
                    <div className="text-xs text-orange-600 dark:text-orange-400">Unassigned</div>
                  </div>
                  <div className="text-center p-3 bg-blue-50 dark:bg-blue-950 rounded-lg">
                    <div className="text-lg font-semibold text-blue-700 dark:text-blue-300">
                      {Object.keys(results.classification_summary.phylum_distribution).length}
                    </div>
                    <div className="text-xs text-blue-600 dark:text-blue-400">Phyla</div>
                  </div>
                  <div className="text-center p-3 bg-purple-50 dark:bg-purple-950 rounded-lg">
                    <div className="text-lg font-semibold text-purple-700 dark:text-purple-300">
                      {results.novel_taxa_candidates?.length || 0}
                    </div>
                    <div className="text-xs text-purple-600 dark:text-purple-400">Novel Taxa</div>
                  </div>
                </div>

                {/* Phylum Distribution */}
                <div className="space-y-2">
                  <h5 className="font-medium">Phylum Distribution</h5>
                  <div className="space-y-2">
                    {Object.entries(results.classification_summary.phylum_distribution)
                      .sort(([,a], [,b]) => b - a)
                      .map(([phylum, count]) => (
                        <div key={phylum} className="flex items-center justify-between">
                          <span className="text-sm">{phylum}</span>
                          <div className="flex items-center gap-2">
                            <div className="w-24 bg-muted rounded-full h-2">
                              <div 
                                className="bg-primary h-2 rounded-full" 
                                style={{ 
                                  width: `${(count / Math.max(...Object.values(results.classification_summary!.phylum_distribution))) * 100}%` 
                                }}
                              />
                            </div>
                            <Badge variant="outline">{count}</Badge>
                          </div>
                        </div>
                      ))}
                  </div>
                </div>
              </div>
            )}

            {/* Biodiversity Metrics */}
            {results.biodiversity_metrics && (
              <div className="space-y-2">
                <h4 className="font-semibold">Biodiversity Metrics</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center p-3 bg-muted/50 rounded-lg">
                    <div className="text-lg font-semibold">
                      {results.biodiversity_metrics.shannon_diversity.toFixed(2)}
                    </div>
                    <div className="text-xs text-muted-foreground">Shannon Index</div>
                  </div>
                  <div className="text-center p-3 bg-muted/50 rounded-lg">
                    <div className="text-lg font-semibold">
                      {results.biodiversity_metrics.simpson_diversity.toFixed(2)}
                    </div>
                    <div className="text-xs text-muted-foreground">Simpson Index</div>
                  </div>
                  <div className="text-center p-3 bg-muted/50 rounded-lg">
                    <div className="text-lg font-semibold">
                      {results.biodiversity_metrics.species_richness}
                    </div>
                    <div className="text-xs text-muted-foreground">Species Richness</div>
                  </div>
                  <div className="text-center p-3 bg-muted/50 rounded-lg">
                    <div className="text-lg font-semibold">
                      {results.biodiversity_metrics.evenness.toFixed(2)}
                    </div>
                    <div className="text-xs text-muted-foreground">Evenness</div>
                  </div>
                </div>
              </div>
            )}

            {/* Novel Taxa Candidates */}
            {results.novel_taxa_candidates && results.novel_taxa_candidates.length > 0 && (
              <div className="space-y-2">
                <h4 className="font-semibold">Novel Taxa Candidates</h4>
                <div className="space-y-2">
                  {results.novel_taxa_candidates.map((candidate, index) => (
                    <div key={index} className="flex items-center justify-between p-2 bg-muted/50 rounded">
                      <span className="font-mono text-sm">{candidate.asv_id}</span>
                      <div className="flex items-center gap-2">
                        <Badge variant="outline">{candidate.cluster_id}</Badge>
                        <Badge variant={candidate.novelty_score > 0.8 ? "default" : "secondary"}>
                          {(candidate.novelty_score * 100).toFixed(1)}% novel
                        </Badge>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex gap-2">
              <Button variant="outline" size="sm">
                <Eye className="mr-2 h-4 w-4" />
                View Details
              </Button>
              <Button variant="outline" size="sm">
                <Download className="mr-2 h-4 w-4" />
                Export Results
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
