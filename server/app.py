"""
Flask API Wrapper for eDNA Analysis Pipeline
SIH Project - AI-driven Deep-Sea Biodiversity Assessment

This Flask application provides REST API endpoints for:
1. Single DNA sequence classification
2. File upload and batch processing
3. Real-time biodiversity analysis
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
import json
import traceback
from datetime import datetime
import uuid

# Import our eDNA pipeline
from edna_pipeline import eDNAProcessor
from usage_examples import calculate_gc_content

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv', 'fasta', 'fa', 'txt'}

# Global processor instance (initialize once)
processor = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_processor():
    """Initialize the eDNA processor with dataset"""
    global processor
    try:
        dataset_path = os.path.join(os.path.dirname(__file__), '..', 'dataset')
        processor = eDNAProcessor(dataset_path)
        # Load datasets and models
        processor.load_datasets()
        return True
    except Exception as e:
        print(f"Error initializing processor: {str(e)}")
        return False

def validate_dna_sequence(sequence):
    """Validate DNA sequence format"""
    if not sequence:
        return False, "Empty sequence"
    
    # Remove whitespace and convert to uppercase
    sequence = sequence.strip().upper().replace(' ', '').replace('\n', '').replace('\r', '')
    
    # Check for valid DNA characters
    valid_chars = set('ATCGN-')
    if not set(sequence).issubset(valid_chars):
        invalid_chars = set(sequence) - valid_chars
        return False, f"Invalid characters found: {', '.join(invalid_chars)}"
    
    # Check minimum length
    if len(sequence) < 50:
        return False, "Sequence too short (minimum 50 bp required)"
    
    # Check maximum length
    if len(sequence) > 10000:
        return False, "Sequence too long (maximum 10,000 bp allowed)"
    
    return True, sequence

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'processor_initialized': processor is not None
    })

@app.route('/api/classify/sequence', methods=['POST'])
def classify_sequence():
    """Classify a single DNA sequence"""
    try:
        data = request.get_json()
        
        if not data or 'sequence' not in data:
            return jsonify({'error': 'No sequence provided'}), 400
        
        # Validate sequence
        is_valid, result = validate_dna_sequence(data['sequence'])
        if not is_valid:
            return jsonify({'error': result}), 400
        
        sequence = result
        
        # Calculate basic sequence statistics
        gc_content = calculate_gc_content(sequence)
        sequence_length = len(sequence)
        
        # Generate analysis ID
        analysis_id = str(uuid.uuid4())
        
        # For now, return mock classification results
        # In a full implementation, you would use the processor to classify
        mock_results = {
            'analysis_id': analysis_id,
            'sequence_info': {
                'length': sequence_length,
                'gc_content': round(gc_content * 100, 2),
                'composition': {
                    'A': sequence.count('A'),
                    'T': sequence.count('T'),
                    'C': sequence.count('C'),
                    'G': sequence.count('G'),
                    'N': sequence.count('N')
                }
            },
            'classification': {
                'predicted_phylum': 'Chordata',
                'confidence': 0.85,
                'cluster_id': 'cluster_7',
                'similarity_score': 0.78
            },
            'taxonomy': {
                'domain': 'Eukaryota',
                'kingdom': 'Metazoa',
                'phylum': 'Chordata',
                'class': 'Actinopterygii',
                'order': 'Perciformes',
                'family': 'Unknown',
                'genus': 'Unknown',
                'species': 'Unknown'
            },
            'biodiversity_metrics': {
                'novelty_score': 0.23,
                'is_potential_novel_taxa': False,
                'cluster_diversity': 0.67
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(mock_results)
        
    except Exception as e:
        return jsonify({
            'error': 'Internal server error',
            'details': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/classify/file', methods=['POST'])
def classify_file():
    """Process uploaded file for batch classification"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed. Supported: CSV, FASTA, FA, TXT'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Generate analysis ID
        analysis_id = str(uuid.uuid4())
        
        try:
            # Process file based on extension
            file_ext = filename.rsplit('.', 1)[1].lower()
            
            if file_ext == 'csv':
                # Process CSV file
                df = pd.read_csv(file_path)
                
                # Validate CSV structure
                required_columns = ['ASV', 'taxonomy']
                if not all(col in df.columns for col in required_columns):
                    return jsonify({
                        'error': 'Invalid CSV format. Required columns: ASV, taxonomy'
                    }), 400
                
                # Mock processing results
                results = {
                    'analysis_id': analysis_id,
                    'file_info': {
                        'filename': filename,
                        'file_type': 'CSV',
                        'total_asvs': len(df),
                        'sample_columns': [col for col in df.columns if col not in ['ASV', 'taxonomy']]
                    },
                    'classification_summary': {
                        'total_classified': len(df[df['taxonomy'] != 'd__Unassigned']),
                        'unassigned': len(df[df['taxonomy'] == 'd__Unassigned']),
                        'phylum_distribution': {
                            'Chordata': 25,
                            'Mollusca': 18,
                            'Cnidaria': 15,
                            'Arthropoda': 12,
                            'Annelida': 10,
                            'Echinodermata': 8,
                            'Porifera': 5,
                            'Unassigned': 7
                        }
                    },
                    'biodiversity_metrics': {
                        'shannon_diversity': 2.34,
                        'simpson_diversity': 0.78,
                        'species_richness': len(df),
                        'evenness': 0.65
                    },
                    'novel_taxa_candidates': [
                        {
                            'asv_id': 'ASV_001',
                            'novelty_score': 0.89,
                            'cluster_id': 'novel_cluster_1'
                        },
                        {
                            'asv_id': 'ASV_045',
                            'novelty_score': 0.76,
                            'cluster_id': 'novel_cluster_2'
                        }
                    ],
                    'timestamp': datetime.now().isoformat()
                }
                
            elif file_ext in ['fasta', 'fa']:
                # Process FASTA file
                sequences = []
                current_seq = ""
                current_header = ""
                
                with open(file_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('>'):
                            if current_seq:
                                sequences.append({
                                    'header': current_header,
                                    'sequence': current_seq,
                                    'length': len(current_seq)
                                })
                            current_header = line[1:]
                            current_seq = ""
                        else:
                            current_seq += line
                    
                    # Add last sequence
                    if current_seq:
                        sequences.append({
                            'header': current_header,
                            'sequence': current_seq,
                            'length': len(current_seq)
                        })
                
                results = {
                    'analysis_id': analysis_id,
                    'file_info': {
                        'filename': filename,
                        'file_type': 'FASTA',
                        'total_sequences': len(sequences),
                        'avg_length': np.mean([s['length'] for s in sequences]) if sequences else 0
                    },
                    'sequences_processed': min(len(sequences), 10),  # Limit for demo
                    'classification_results': [
                        {
                            'sequence_id': seq['header'][:50],
                            'length': seq['length'],
                            'predicted_phylum': np.random.choice(['Chordata', 'Mollusca', 'Cnidaria', 'Arthropoda']),
                            'confidence': round(np.random.uniform(0.6, 0.95), 2)
                        }
                        for seq in sequences[:10]  # First 10 sequences for demo
                    ],
                    'timestamp': datetime.now().isoformat()
                }
            
            else:
                return jsonify({'error': 'Unsupported file format'}), 400
            
            return jsonify(results)
            
        finally:
            # Clean up uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)
        
    except Exception as e:
        return jsonify({
            'error': 'File processing error',
            'details': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/analysis/<analysis_id>', methods=['GET'])
def get_analysis_results(analysis_id):
    """Get detailed analysis results by ID"""
    try:
        # In a real implementation, you would retrieve results from database
        # For now, return mock detailed results
        detailed_results = {
            'analysis_id': analysis_id,
            'status': 'completed',
            'detailed_classification': {
                'clustering_results': {
                    'kmeans': {
                        'n_clusters': 8,
                        'silhouette_score': 0.72,
                        'cluster_sizes': [45, 38, 29, 25, 18, 12, 8, 5]
                    },
                    'dbscan': {
                        'n_clusters': 6,
                        'noise_points': 12,
                        'core_samples': 168
                    },
                    'hierarchical': {
                        'n_clusters': 7,
                        'linkage': 'ward',
                        'cophenetic_correlation': 0.78
                    }
                },
                'feature_importance': {
                    'sequence_features': 0.45,
                    'abundance_features': 0.35,
                    'taxonomy_features': 0.20
                },
                'geographic_distribution': {
                    'deep_site_1': {'diversity': 2.1, 'abundance': 1250},
                    'deep_site_2': {'diversity': 1.8, 'abundance': 980},
                    'hydrothermal_vent': {'diversity': 2.5, 'abundance': 2100}
                }
            },
            'recommendations': [
                'High novelty sequences detected - consider further taxonomic investigation',
                'Cluster 3 shows unique characteristics - potential new species',
                'Geographic variation suggests environmental adaptation'
            ],
            'export_options': {
                'csv_results': f'/api/export/{analysis_id}/csv',
                'detailed_report': f'/api/export/{analysis_id}/report'
            }
        }
        
        return jsonify(detailed_results)
        
    except Exception as e:
        return jsonify({
            'error': 'Analysis retrieval error',
            'details': str(e)
        }), 500

@app.route('/api/export/<analysis_id>/<format>', methods=['GET'])
def export_results(analysis_id, format):
    """Export analysis results in specified format"""
    try:
        if format == 'csv':
            # Create mock CSV export
            data = {
                'ASV_ID': [f'ASV_{i:03d}' for i in range(1, 21)],
                'Predicted_Phylum': np.random.choice(['Chordata', 'Mollusca', 'Cnidaria', 'Arthropoda'], 20),
                'Confidence': np.random.uniform(0.6, 0.95, 20).round(2),
                'Cluster_ID': [f'cluster_{np.random.randint(1, 9)}' for _ in range(20)],
                'Novelty_Score': np.random.uniform(0.1, 0.9, 20).round(2)
            }
            
            df = pd.DataFrame(data)
            csv_path = os.path.join(tempfile.gettempdir(), f'results_{analysis_id}.csv')
            df.to_csv(csv_path, index=False)
            
            return send_file(csv_path, as_attachment=True, download_name=f'edna_results_{analysis_id}.csv')
        
        elif format == 'report':
            # Create mock detailed report
            report_content = f"""
eDNA Analysis Report
Analysis ID: {analysis_id}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

=== SUMMARY ===
Total sequences analyzed: 180
Successfully classified: 165 (91.7%)
Novel taxa candidates: 15 (8.3%)

=== PHYLUM DISTRIBUTION ===
Chordata: 45 sequences (25.0%)
Mollusca: 32 sequences (17.8%)
Cnidaria: 28 sequences (15.6%)
Arthropoda: 25 sequences (13.9%)
Annelida: 18 sequences (10.0%)
Echinodermata: 12 sequences (6.7%)
Porifera: 5 sequences (2.8%)
Unassigned: 15 sequences (8.3%)

=== BIODIVERSITY METRICS ===
Shannon Diversity Index: 2.34
Simpson Diversity Index: 0.78
Species Richness: 180
Evenness: 0.65

=== NOVEL TAXA DETECTION ===
High-confidence novel candidates: 3
Medium-confidence candidates: 7
Low-confidence candidates: 5

=== RECOMMENDATIONS ===
1. Further investigation recommended for ASV_001, ASV_045, ASV_089
2. Geographic clustering suggests environmental adaptation
3. Consider targeted sequencing for novel taxa validation
            """
            
            report_path = os.path.join(tempfile.gettempdir(), f'report_{analysis_id}.txt')
            with open(report_path, 'w') as f:
                f.write(report_content)
            
            return send_file(report_path, as_attachment=True, download_name=f'edna_report_{analysis_id}.txt')
        
        else:
            return jsonify({'error': 'Unsupported export format'}), 400
            
    except Exception as e:
        return jsonify({
            'error': 'Export error',
            'details': str(e)
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_system_stats():
    """Get system statistics and capabilities"""
    return jsonify({
        'system_info': {
            'version': '1.0.0',
            'supported_markers': ['18S rRNA', 'COI'],
            'supported_phyla': [
                'Annelida', 'Arthropoda', 'Chordata', 
                'Cnidaria', 'Echinodermata', 'Mollusca', 'Porifera'
            ],
            'clustering_algorithms': ['K-means', 'DBSCAN', 'Hierarchical'],
            'max_file_size': '16MB',
            'supported_formats': ['CSV', 'FASTA', 'FA', 'TXT']
        },
        'database_stats': {
            '18S_sequences': 37863,
            'COI_sequences': 13219,
            'total_taxa': 51082,
            'last_updated': '2024-01-15'
        }
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Initializing eDNA Analysis API...")
    
    # Initialize processor (optional for demo)
    if initialize_processor():
        print("✓ eDNA processor initialized successfully")
    else:
        print("⚠ eDNA processor initialization failed - using mock data")
    
    print("Starting Flask server...")
    print("API endpoints available at:")
    print("  - POST /api/classify/sequence - Single sequence classification")
    print("  - POST /api/classify/file - File upload and batch processing")
    print("  - GET /api/analysis/<id> - Detailed analysis results")
    print("  - GET /api/export/<id>/<format> - Export results")
    print("  - GET /api/stats - System statistics")
    print("  - GET /api/health - Health check")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
