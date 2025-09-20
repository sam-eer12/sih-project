/**
 * API service functions for eDNA Analysis Pipeline
 * Handles communication with Flask backend
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000';

export interface SequenceClassificationRequest {
  sequence: string;
}

export interface SequenceClassificationResponse {
  analysis_id: string;
  sequence_info: {
    length: number;
    gc_content: number;
    composition: {
      A: number;
      T: number;
      C: number;
      G: number;
      N: number;
    };
  };
  classification: {
    predicted_phylum: string;
    confidence: number;
    cluster_id: string;
    similarity_score: number;
  };
  taxonomy: {
    domain: string;
    kingdom: string;
    phylum: string;
    class: string;
    order: string;
    family: string;
    genus: string;
    species: string;
  };
  biodiversity_metrics: {
    novelty_score: number;
    is_potential_novel_taxa: boolean;
    cluster_diversity: number;
  };
  timestamp: string;
}

export interface FileClassificationResponse {
  analysis_id: string;
  file_info: {
    filename: string;
    file_type: string;
    total_asvs?: number;
    total_sequences?: number;
    sample_columns?: string[];
    avg_length?: number;
  };
  classification_summary?: {
    total_classified: number;
    unassigned: number;
    phylum_distribution: Record<string, number>;
  };
  sequences_processed?: number;
  classification_results?: Array<{
    sequence_id: string;
    length: number;
    predicted_phylum: string;
    confidence: number;
  }>;
  biodiversity_metrics?: {
    shannon_diversity: number;
    simpson_diversity: number;
    species_richness: number;
    evenness: number;
  };
  novel_taxa_candidates?: Array<{
    asv_id: string;
    novelty_score: number;
    cluster_id: string;
  }>;
  timestamp: string;
}

export interface DetailedAnalysisResponse {
  analysis_id: string;
  status: string;
  detailed_classification: {
    clustering_results: {
      kmeans: {
        n_clusters: number;
        silhouette_score: number;
        cluster_sizes: number[];
      };
      dbscan: {
        n_clusters: number;
        noise_points: number;
        core_samples: number;
      };
      hierarchical: {
        n_clusters: number;
        linkage: string;
        cophenetic_correlation: number;
      };
    };
    feature_importance: {
      sequence_features: number;
      abundance_features: number;
      taxonomy_features: number;
    };
    geographic_distribution: Record<string, {
      diversity: number;
      abundance: number;
    }>;
  };
  recommendations: string[];
  export_options: {
    csv_results: string;
    detailed_report: string;
  };
}

export interface SystemStats {
  system_info: {
    version: string;
    supported_markers: string[];
    supported_phyla: string[];
    clustering_algorithms: string[];
    max_file_size: string;
    supported_formats: string[];
  };
  database_stats: {
    '18S_sequences': number;
    'COI_sequences': number;
    total_taxa: number;
    last_updated: string;
  };
}

export interface ApiError {
  error: string;
  details?: string;
  traceback?: string;
}

class ApiService {
  private async handleResponse<T>(response: Response): Promise<T> {
    if (!response.ok) {
      const errorData: ApiError = await response.json().catch(() => ({
        error: `HTTP ${response.status}: ${response.statusText}`
      }));
      throw new Error(errorData.error || 'API request failed');
    }
    return response.json();
  }

  /**
   * Check API health status
   */
  async healthCheck(): Promise<{ status: string; timestamp: string; processor_initialized: boolean }> {
    const response = await fetch(`${API_BASE_URL}/api/health`);
    return this.handleResponse(response);
  }

  /**
   * Classify a single DNA sequence
   */
  async classifySequence(request: SequenceClassificationRequest): Promise<SequenceClassificationResponse> {
    const response = await fetch(`${API_BASE_URL}/api/classify/sequence`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });
    return this.handleResponse(response);
  }

  /**
   * Upload and classify a file
   */
  async classifyFile(file: File): Promise<FileClassificationResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE_URL}/api/classify/file`, {
      method: 'POST',
      body: formData,
    });
    return this.handleResponse(response);
  }

  /**
   * Get detailed analysis results
   */
  async getAnalysisResults(analysisId: string): Promise<DetailedAnalysisResponse> {
    const response = await fetch(`${API_BASE_URL}/api/analysis/${analysisId}`);
    return this.handleResponse(response);
  }

  /**
   * Export analysis results
   */
  async exportResults(analysisId: string, format: 'csv' | 'report'): Promise<Blob> {
    const response = await fetch(`${API_BASE_URL}/api/export/${analysisId}/${format}`);
    if (!response.ok) {
      throw new Error(`Export failed: ${response.statusText}`);
    }
    return response.blob();
  }

  /**
   * Get system statistics
   */
  async getSystemStats(): Promise<SystemStats> {
    const response = await fetch(`${API_BASE_URL}/api/stats`);
    return this.handleResponse(response);
  }

  /**
   * Validate DNA sequence format
   */
  validateDnaSequence(sequence: string): { isValid: boolean; error?: string; cleanSequence?: string } {
    if (!sequence || sequence.trim().length === 0) {
      return { isValid: false, error: 'Empty sequence' };
    }

    // Clean sequence
    const cleanSequence = sequence.trim().toUpperCase().replace(/\s+/g, '').replace(/[\r\n]/g, '');

    // Check for valid DNA characters
    const validChars = /^[ATCGN-]+$/;
    if (!validChars.test(cleanSequence)) {
      const invalidChars = cleanSequence.match(/[^ATCGN-]/g);
      return { 
        isValid: false, 
        error: `Invalid characters found: ${invalidChars?.join(', ')}` 
      };
    }

    // Check length constraints
    if (cleanSequence.length < 50) {
      return { isValid: false, error: 'Sequence too short (minimum 50 bp required)' };
    }

    if (cleanSequence.length > 10000) {
      return { isValid: false, error: 'Sequence too long (maximum 10,000 bp allowed)' };
    }

    return { isValid: true, cleanSequence };
  }

  /**
   * Calculate GC content of a DNA sequence
   */
  calculateGcContent(sequence: string): number {
    if (!sequence) return 0;
    const cleanSeq = sequence.toUpperCase().replace(/[^ATCG]/g, '');
    if (cleanSeq.length === 0) return 0;
    const gcCount = (cleanSeq.match(/[GC]/g) || []).length;
    return gcCount / cleanSeq.length;
  }

  /**
   * Get sequence composition
   */
  getSequenceComposition(sequence: string): { A: number; T: number; C: number; G: number; N: number; other: number } {
    const cleanSeq = sequence.toUpperCase();
    return {
      A: (cleanSeq.match(/A/g) || []).length,
      T: (cleanSeq.match(/T/g) || []).length,
      C: (cleanSeq.match(/C/g) || []).length,
      G: (cleanSeq.match(/G/g) || []).length,
      N: (cleanSeq.match(/N/g) || []).length,
      other: (cleanSeq.match(/[^ATCGN]/g) || []).length,
    };
  }
}

// Export singleton instance
export const apiService = new ApiService();

// Export utility functions
export const utils = {
  validateDnaSequence: apiService.validateDnaSequence.bind(apiService),
  calculateGcContent: apiService.calculateGcContent.bind(apiService),
  getSequenceComposition: apiService.getSequenceComposition.bind(apiService),
};

export default apiService;
