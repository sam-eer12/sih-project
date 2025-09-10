"""
eDNA Analysis Pipeline for Deep-Sea Biodiversity Assessment
SIH Project - AI-driven pipeline for eukaryotic taxa classification

This module implements an AI-driven pipeline for processing eDNA data from deep-sea environments,
focusing on unsupervised learning approaches to classify eukaryotic taxa without heavy reliance
on reference databases.
"""

import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import pickle
import warnings
warnings.filterwarnings('ignore')

class eDNAProcessor:
    """
    Main class for processing eDNA sequencing data and performing unsupervised classification
    """
    
    def __init__(self, dataset_path: str):
        """
        Initialize the eDNA processor
        
        Args:
            dataset_path: Path to the dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self.processed_data = {}
        self.sequence_features = None
        self.abundance_features = None
        self.taxonomy_features = None
        self.combined_features = None
        self.models = {}
        
    def load_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load all available datasets including marker gene data and taxonomic group data
        
        Returns:
            Dictionary containing loaded datasets
        """
        datasets = {}
        
        # Load main marker gene datasets
        try:
            datasets['18S'] = pd.read_csv(self.dataset_path / '18S_unfiltered_table_excel_with_taxonomy.csv')
            print(f"Loaded 18S dataset: {datasets['18S'].shape}")
        except Exception as e:
            print(f"Error loading 18S dataset: {e}")
            
        try:
            datasets['COX1'] = pd.read_csv(self.dataset_path / 'cox1_unfiltered_table_excel_with_taxonomy.csv')
            print(f"Loaded COX1 dataset: {datasets['COX1'].shape}")
        except Exception as e:
            print(f"Error loading COX1 dataset: {e}")
        
        # Load taxonomic group datasets
        taxonomic_groups = ['Annelida', 'anthropoda', 'Chordata', 'Cnidaria', 
                           'Echinodermata', 'mollusca', 'Porifera']
        
        for group in taxonomic_groups:
            group_path = self.dataset_path / group
            if group_path.exists():
                try:
                    # Load BOLD data
                    bold_file = group_path / 'bold_data.csv'
                    if bold_file.exists():
                        datasets[f'{group}_bold'] = pd.read_csv(bold_file)
                        print(f"Loaded {group} BOLD data: {datasets[f'{group}_bold'].shape}")
                    
                    # Load FASTA data
                    fasta_file = group_path / 'fasta.fas'
                    if fasta_file.exists():
                        datasets[f'{group}_fasta'] = self._parse_fasta(fasta_file)
                        print(f"Loaded {group} FASTA data: {len(datasets[f'{group}_fasta'])} sequences")
                        
                except Exception as e:
                    print(f"Error loading {group} data: {e}")
        
        self.datasets = datasets
        return datasets
    
    def _parse_fasta(self, fasta_file: Path) -> Dict[str, str]:
        """
        Parse FASTA file and return dictionary of sequences
        
        Args:
            fasta_file: Path to FASTA file
            
        Returns:
            Dictionary with sequence IDs as keys and sequences as values
        """
        sequences = {}
        current_id = None
        current_seq = ""
        
        with open(fasta_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_id is not None:
                        sequences[current_id] = current_seq
                    current_id = line[1:]  # Remove '>'
                    current_seq = ""
                else:
                    current_seq += line
            
            # Add the last sequence
            if current_id is not None:
                sequences[current_id] = current_seq
                
        return sequences
    
    def preprocess_marker_data(self) -> pd.DataFrame:
        """
        Preprocess marker gene data (18S and COX1) for analysis
        
        Returns:
            Processed dataframe with features
        """
        processed_dfs = []
        
        for marker in ['18S', 'COX1']:
            if marker in self.datasets:
                df = self.datasets[marker].copy()
                
                # Extract abundance data (all numeric columns except ASV and taxonomy)
                abundance_cols = [col for col in df.columns if col not in ['ASV', 'taxonomy']]
                abundance_data = df[abundance_cols].fillna(0)
                
                # Calculate abundance statistics
                df['total_abundance'] = abundance_data.sum(axis=1)
                df['sample_presence'] = (abundance_data > 0).sum(axis=1)
                df['max_abundance'] = abundance_data.max(axis=1)
                df['mean_abundance'] = abundance_data.mean(axis=1)
                df['std_abundance'] = abundance_data.std(axis=1).fillna(0)
                
                # Process taxonomy information
                df['taxonomy_processed'] = df['taxonomy'].fillna('d__Unassigned')
                df['domain'] = df['taxonomy_processed'].str.extract(r'd__([^;]+)')
                df['kingdom'] = df['taxonomy_processed'].str.extract(r'k__([^;]+)')
                df['phylum'] = df['taxonomy_processed'].str.extract(r'p__([^;]+)')
                df['class'] = df['taxonomy_processed'].str.extract(r'c__([^;]+)')
                df['order'] = df['taxonomy_processed'].str.extract(r'o__([^;]+)')
                df['family'] = df['taxonomy_processed'].str.extract(r'f__([^;]+)')
                df['genus'] = df['taxonomy_processed'].str.extract(r'g__([^;]+)')
                df['species'] = df['taxonomy_processed'].str.extract(r's__([^;]+)')
                
                # Fill missing taxonomic levels
                for col in ['domain', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']:
                    df[col] = df[col].fillna('Unknown')
                
                # Calculate taxonomic depth (how many levels are assigned)
                df['taxonomic_depth'] = 0
                for col in ['domain', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']:
                    df['taxonomic_depth'] += (df[col] != 'Unknown').astype(int)
                
                # Add marker information
                df['marker'] = marker
                
                # Select relevant columns for processing
                feature_cols = ['ASV', 'marker', 'total_abundance', 'sample_presence', 'max_abundance', 
                               'mean_abundance', 'std_abundance', 'taxonomic_depth', 'domain', 'kingdom', 
                               'phylum', 'class', 'order', 'family', 'genus', 'species']
                
                processed_dfs.append(df[feature_cols])
        
        if processed_dfs:
            combined_df = pd.concat(processed_dfs, ignore_index=True)
            self.processed_data['marker_data'] = combined_df
            return combined_df
        else:
            print("No marker data available for processing")
            return pd.DataFrame()
    
    def preprocess_bold_data(self) -> pd.DataFrame:
        """
        Preprocess BOLD (Barcode of Life) data for analysis
        
        Returns:
            Processed dataframe with features
        """
        processed_dfs = []
        
        for dataset_name in self.datasets:
            if 'bold' in dataset_name:
                df = self.datasets[dataset_name].copy()
                
                # Extract taxonomic information
                taxonomic_cols = [col for col in df.columns if 'taxonomy/' in col or 'taxon/' in col]
                
                # Process sequences if available
                if 'sequences/sequence/nucleotides' in df.columns:
                    df['sequence'] = df['sequences/sequence/nucleotides'].fillna('')
                    df['sequence_length'] = df['sequence'].str.len()
                    df['gc_content'] = df['sequence'].apply(self._calculate_gc_content)
                    df['n_content'] = df['sequence'].apply(self._calculate_n_content)
                else:
                    df['sequence_length'] = 0
                    df['gc_content'] = 0
                    df['n_content'] = 0
                
                # Extract coordinates
                if 'collection_event/coordinates/lat' in df.columns:
                    df['latitude'] = pd.to_numeric(df['collection_event/coordinates/lat'], errors='coerce')
                    df['longitude'] = pd.to_numeric(df['collection_event/coordinates/lon'], errors='coerce')
                else:
                    df['latitude'] = np.nan
                    df['longitude'] = np.nan
                
                # Extract depth information if available (from site names or other fields)
                df['depth_category'] = self._extract_depth_category(df)
                
                # Extract taxonomic group from dataset name
                taxonomic_group = dataset_name.replace('_bold', '')
                df['taxonomic_group'] = taxonomic_group
                
                # Process marker information
                if 'sequences/sequence/markercode' in df.columns:
                    df['marker'] = df['sequences/sequence/markercode'].fillna('Unknown')
                else:
                    df['marker'] = 'Unknown'
                
                # Select relevant columns
                feature_cols = ['record_id', 'taxonomic_group', 'marker', 'sequence_length', 
                               'gc_content', 'n_content', 'latitude', 'longitude', 'depth_category']
                
                # Add taxonomic columns if they exist
                if 'taxonomy/phylum/taxon/name' in df.columns:
                    df['phylum'] = df['taxonomy/phylum/taxon/name'].fillna('Unknown')
                    feature_cols.append('phylum')
                    
                if 'taxonomy/class/taxon/name' in df.columns:
                    df['class'] = df['taxonomy/class/taxon/name'].fillna('Unknown')
                    feature_cols.append('class')
                    
                if 'taxonomy/order/taxon/name' in df.columns:
                    df['order'] = df['taxonomy/order/taxon/name'].fillna('Unknown')
                    feature_cols.append('order')
                    
                if 'taxonomy/family/taxon/name' in df.columns:
                    df['family'] = df['taxonomy/family/taxon/name'].fillna('Unknown')
                    feature_cols.append('family')
                    
                if 'taxonomy/genus/taxon/name' in df.columns:
                    df['genus'] = df['taxonomy/genus/taxon/name'].fillna('Unknown')
                    feature_cols.append('genus')
                    
                if 'taxonomy/species/taxon/name' in df.columns:
                    df['species'] = df['taxonomy/species/taxon/name'].fillna('Unknown')
                    feature_cols.append('species')
                
                processed_dfs.append(df[feature_cols])
        
        if processed_dfs:
            combined_df = pd.concat(processed_dfs, ignore_index=True)
            self.processed_data['bold_data'] = combined_df
            return combined_df
        else:
            print("No BOLD data available for processing")
            return pd.DataFrame()
    
    def _calculate_gc_content(self, sequence: str) -> float:
        """Calculate GC content of a DNA sequence"""
        if not sequence or len(sequence) == 0:
            return 0.0
        sequence = sequence.upper().replace('-', '').replace('N', '')
        if len(sequence) == 0:
            return 0.0
        gc_count = sequence.count('G') + sequence.count('C')
        return gc_count / len(sequence)
    
    def _calculate_n_content(self, sequence: str) -> float:
        """Calculate N content of a DNA sequence"""
        if not sequence or len(sequence) == 0:
            return 0.0
        sequence = sequence.upper()
        n_count = sequence.count('N')
        return n_count / len(sequence)
    
    def _extract_depth_category(self, df: pd.DataFrame) -> pd.Series:
        """Extract depth category from site information"""
        # This is a simplified categorization - can be enhanced based on specific site information
        depth_categories = []
        
        for _, row in df.iterrows():
            site_info = str(row.get('collection_event/exactsite', '')).lower()
            
            if 'deep' in site_info or 'abyssal' in site_info:
                category = 'deep'
            elif 'reef' in site_info or 'shallow' in site_info:
                category = 'shallow'
            elif 'seamount' in site_info:
                category = 'seamount'
            elif 'vent' in site_info or 'hydrothermal' in site_info:
                category = 'hydrothermal'
            else:
                category = 'unknown'
            
            depth_categories.append(category)
        
        return pd.Series(depth_categories)
    
    def create_sequence_features(self) -> np.ndarray:
        """
        Create k-mer based features from DNA sequences for unsupervised learning
        
        Returns:
            Feature matrix for sequences
        """
        sequences = []
        
        # Collect sequences from FASTA data
        for dataset_name in self.datasets:
            if 'fasta' in dataset_name:
                fasta_data = self.datasets[dataset_name]
                sequences.extend(list(fasta_data.values()))
        
        # Collect sequences from BOLD data
        if 'bold_data' in self.processed_data:
            bold_df = self.processed_data['bold_data']
            # Note: BOLD data might not have sequences in the CSV, 
            # they're typically in separate FASTA files
        
        if not sequences:
            print("No sequences found for feature extraction")
            return np.array([])
        
        # Clean sequences
        cleaned_sequences = []
        for seq in sequences:
            # Remove gaps and convert to uppercase
            cleaned_seq = seq.replace('-', '').replace('N', '').upper()
            if len(cleaned_seq) > 50:  # Filter very short sequences
                cleaned_sequences.append(cleaned_seq)
        
        if not cleaned_sequences:
            print("No valid sequences after cleaning")
            return np.array([])
        
        # Create k-mer features (3-mers and 4-mers)
        kmers_3 = self._generate_kmers(cleaned_sequences, k=3)
        kmers_4 = self._generate_kmers(cleaned_sequences, k=4)
        
        # Combine k-mer features
        all_kmers = kmers_3 + kmers_4
        
        # Create count vectorizer for k-mers
        vectorizer = CountVectorizer(vocabulary=all_kmers, binary=True)
        
        # Convert sequences to k-mer representation
        sequence_docs = [self._sequence_to_kmers(seq, [3, 4]) for seq in cleaned_sequences]
        
        feature_matrix = vectorizer.fit_transform(sequence_docs).toarray()
        
        print(f"Created sequence feature matrix: {feature_matrix.shape}")
        self.sequence_features = feature_matrix
        return feature_matrix
    
    def _generate_kmers(self, sequences: List[str], k: int) -> List[str]:
        """Generate all possible k-mers from sequences"""
        kmers = set()
        for seq in sequences:
            for i in range(len(seq) - k + 1):
                kmer = seq[i:i+k]
                if 'N' not in kmer:  # Only include k-mers without ambiguous bases
                    kmers.add(kmer)
        return list(kmers)
    
    def _sequence_to_kmers(self, sequence: str, k_values: List[int]) -> str:
        """Convert sequence to space-separated k-mers"""
        kmers = []
        for k in k_values:
            for i in range(len(sequence) - k + 1):
                kmer = sequence[i:i+k]
                if 'N' not in kmer:
                    kmers.append(kmer)
        return ' '.join(kmers)
    
    def create_abundance_features(self) -> np.ndarray:
        """
        Create abundance-based features from marker gene data
        
        Returns:
            Feature matrix for abundance data
        """
        if 'marker_data' not in self.processed_data:
            print("No marker data available for abundance features")
            return np.array([])
        
        df = self.processed_data['marker_data'].copy()
        
        # Select numerical features
        abundance_cols = ['total_abundance', 'sample_presence', 'max_abundance', 
                         'mean_abundance', 'std_abundance', 'taxonomic_depth']
        
        feature_matrix = df[abundance_cols].fillna(0).values
        
        # Normalize features
        scaler = StandardScaler()
        feature_matrix = scaler.fit_transform(feature_matrix)
        
        print(f"Created abundance feature matrix: {feature_matrix.shape}")
        self.abundance_features = feature_matrix
        return feature_matrix
    
    def create_taxonomy_features(self) -> np.ndarray:
        """
        Create features from taxonomic information using text processing
        
        Returns:
            Feature matrix for taxonomic data
        """
        taxonomy_docs = []
        
        # Process marker data taxonomy
        if 'marker_data' in self.processed_data:
            df = self.processed_data['marker_data']
            for _, row in df.iterrows():
                tax_info = f"{row['domain']} {row['kingdom']} {row['phylum']} {row['class']} {row['order']} {row['family']} {row['genus']} {row['species']}"
                taxonomy_docs.append(tax_info)
        
        # Process BOLD data taxonomy
        if 'bold_data' in self.processed_data:
            df = self.processed_data['bold_data']
            for _, row in df.iterrows():
                tax_info = f"{row.get('phylum', 'Unknown')} {row.get('class', 'Unknown')} {row.get('order', 'Unknown')} {row.get('family', 'Unknown')} {row.get('genus', 'Unknown')} {row.get('species', 'Unknown')}"
                taxonomy_docs.append(tax_info)
        
        if not taxonomy_docs:
            print("No taxonomic information available")
            return np.array([])
        
        # Create TF-IDF features for taxonomic information
        vectorizer = TfidfVectorizer(max_features=500, stop_words=['unknown'], ngram_range=(1, 2))
        feature_matrix = vectorizer.fit_transform(taxonomy_docs).toarray()
        
        print(f"Created taxonomy feature matrix: {feature_matrix.shape}")
        self.taxonomy_features = feature_matrix
        return feature_matrix
    
    def combine_features(self) -> np.ndarray:
        """
        Combine all feature types into a single feature matrix
        
        Returns:
            Combined feature matrix
        """
        features_to_combine = []
        
        if self.sequence_features is not None and self.sequence_features.size > 0:
            features_to_combine.append(self.sequence_features)
            print(f"Including sequence features: {self.sequence_features.shape}")
        
        if self.abundance_features is not None and self.abundance_features.size > 0:
            features_to_combine.append(self.abundance_features)
            print(f"Including abundance features: {self.abundance_features.shape}")
        
        if self.taxonomy_features is not None and self.taxonomy_features.size > 0:
            features_to_combine.append(self.taxonomy_features)
            print(f"Including taxonomy features: {self.taxonomy_features.shape}")
        
        if not features_to_combine:
            print("No features available for combination")
            return np.array([])
        
        # Ensure all feature matrices have the same number of samples
        min_samples = min(f.shape[0] for f in features_to_combine)
        features_to_combine = [f[:min_samples] for f in features_to_combine]
        
        # Combine features horizontally
        combined_features = np.hstack(features_to_combine)
        
        print(f"Combined feature matrix shape: {combined_features.shape}")
        self.combined_features = combined_features
        return combined_features
    
    def train_unsupervised_models(self, features: np.ndarray, n_clusters_range: range = range(2, 21)) -> Dict:
        """
        Train multiple unsupervised learning models for taxa classification
        
        Args:
            features: Feature matrix
            n_clusters_range: Range of cluster numbers to try
            
        Returns:
            Dictionary containing trained models and results
        """
        if features.size == 0:
            print("No features available for training")
            return {}
        
        results = {}
        
        # Dimensionality reduction first (if features are high-dimensional)
        if features.shape[1] > 100:
            print("Applying dimensionality reduction...")
            
            # PCA
            pca = PCA(n_components=min(50, features.shape[0]-1))
            features_pca = pca.fit_transform(features)
            results['pca'] = pca
            results['features_pca'] = features_pca
            
            # t-SNE for visualization
            if features.shape[0] > 50:
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, features.shape[0]-1))
                features_tsne = tsne.fit_transform(features_pca)
                results['tsne'] = features_tsne
        else:
            features_pca = features
            results['features_pca'] = features_pca
        
        print("Training clustering models...")
        
        # K-Means clustering
        best_kmeans = None
        best_kmeans_score = -1
        
        for n_clusters in n_clusters_range:
            if n_clusters >= features_pca.shape[0]:
                break
                
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_pca)
            
            # Calculate silhouette score
            if len(set(cluster_labels)) > 1:
                score = silhouette_score(features_pca, cluster_labels)
                if score > best_kmeans_score:
                    best_kmeans_score = score
                    best_kmeans = kmeans
        
        if best_kmeans is not None:
            results['kmeans'] = best_kmeans
            results['kmeans_labels'] = best_kmeans.predict(features_pca)
            results['kmeans_score'] = best_kmeans_score
            print(f"Best K-Means: {best_kmeans.n_clusters} clusters, silhouette score: {best_kmeans_score:.3f}")
        
        # DBSCAN clustering
        print("Training DBSCAN...")
        eps_values = [0.3, 0.5, 0.7, 1.0, 1.5]
        best_dbscan = None
        best_dbscan_score = -1
        
        for eps in eps_values:
            dbscan = DBSCAN(eps=eps, min_samples=max(2, features_pca.shape[0] // 50))
            cluster_labels = dbscan.fit_predict(features_pca)
            
            # Check if we have meaningful clusters
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            if n_clusters > 1 and n_clusters < features_pca.shape[0] * 0.8:
                score = silhouette_score(features_pca, cluster_labels)
                if score > best_dbscan_score:
                    best_dbscan_score = score
                    best_dbscan = dbscan
        
        if best_dbscan is not None:
            results['dbscan'] = best_dbscan
            results['dbscan_labels'] = best_dbscan.fit_predict(features_pca)
            results['dbscan_score'] = best_dbscan_score
            n_clusters = len(set(results['dbscan_labels'])) - (1 if -1 in results['dbscan_labels'] else 0)
            print(f"Best DBSCAN: {n_clusters} clusters, silhouette score: {best_dbscan_score:.3f}")
        
        # Hierarchical clustering
        print("Training Hierarchical clustering...")
        best_hierarchical = None
        best_hierarchical_score = -1
        
        for n_clusters in n_clusters_range:
            if n_clusters >= features_pca.shape[0]:
                break
                
            hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            cluster_labels = hierarchical.fit_predict(features_pca)
            
            if len(set(cluster_labels)) > 1:
                score = silhouette_score(features_pca, cluster_labels)
                if score > best_hierarchical_score:
                    best_hierarchical_score = score
                    best_hierarchical = hierarchical
        
        if best_hierarchical is not None:
            results['hierarchical'] = best_hierarchical
            results['hierarchical_labels'] = best_hierarchical.fit_predict(features_pca)
            results['hierarchical_score'] = best_hierarchical_score
            print(f"Best Hierarchical: {best_hierarchical.n_clusters} clusters, silhouette score: {best_hierarchical_score:.3f}")
        
        self.models = results
        return results
    
    def estimate_abundance(self, cluster_labels: np.ndarray) -> Dict:
        """
        Estimate abundance of different clusters/taxa
        
        Args:
            cluster_labels: Cluster assignments from unsupervised learning
            
        Returns:
            Dictionary with abundance estimates
        """
        abundance_estimates = {}
        
        if 'marker_data' in self.processed_data:
            df = self.processed_data['marker_data'].copy()
            
            # Ensure we have the same number of samples
            min_samples = min(len(cluster_labels), len(df))
            cluster_labels = cluster_labels[:min_samples]
            df = df.iloc[:min_samples]
            
            df['cluster'] = cluster_labels
            
            # Calculate abundance statistics per cluster
            cluster_stats = df.groupby('cluster').agg({
                'total_abundance': ['mean', 'std', 'sum'],
                'sample_presence': ['mean', 'std'],
                'taxonomic_depth': 'mean'
            }).round(3)
            
            abundance_estimates['cluster_abundance'] = cluster_stats
            
            # Calculate diversity indices per cluster
            diversity_stats = {}
            for cluster_id in df['cluster'].unique():
                cluster_data = df[df['cluster'] == cluster_id]
                
                # Shannon diversity index (simplified)
                abundances = cluster_data['total_abundance'].values
                abundances = abundances[abundances > 0]
                if len(abundances) > 0:
                    proportions = abundances / abundances.sum()
                    shannon_index = -np.sum(proportions * np.log(proportions + 1e-10))
                else:
                    shannon_index = 0
                
                diversity_stats[cluster_id] = {
                    'shannon_diversity': shannon_index,
                    'species_count': len(cluster_data),
                    'total_abundance': cluster_data['total_abundance'].sum()
                }
            
            abundance_estimates['diversity_stats'] = diversity_stats
        
        return abundance_estimates
    
    def annotate_clusters(self, cluster_labels: np.ndarray) -> Dict:
        """
        Annotate clusters with taxonomic information
        
        Args:
            cluster_labels: Cluster assignments
            
        Returns:
            Dictionary with cluster annotations
        """
        annotations = {}
        
        # Combine marker and BOLD data for annotation
        annotation_data = []
        
        if 'marker_data' in self.processed_data:
            df = self.processed_data['marker_data'].copy()
            df['data_source'] = 'marker'
            annotation_data.append(df)
        
        if 'bold_data' in self.processed_data:
            df = self.processed_data['bold_data'].copy()
            df['data_source'] = 'bold'
            annotation_data.append(df)
        
        if not annotation_data:
            return annotations
        
        combined_df = pd.concat(annotation_data, ignore_index=True, sort=False)
        
        # Ensure we have the same number of samples
        min_samples = min(len(cluster_labels), len(combined_df))
        cluster_labels = cluster_labels[:min_samples]
        combined_df = combined_df.iloc[:min_samples].copy()
        combined_df['cluster'] = cluster_labels
        
        # Annotate each cluster
        for cluster_id in combined_df['cluster'].unique():
            cluster_data = combined_df[combined_df['cluster'] == cluster_id]
            
            cluster_annotation = {
                'cluster_id': cluster_id,
                'sample_count': len(cluster_data),
                'data_sources': cluster_data['data_source'].value_counts().to_dict()
            }
            
            # Most common taxonomic assignments
            taxonomic_levels = ['phylum', 'class', 'order', 'family', 'genus', 'species']
            
            for level in taxonomic_levels:
                if level in cluster_data.columns:
                    most_common = cluster_data[level].value_counts().head(3)
                    cluster_annotation[f'most_common_{level}'] = most_common.to_dict()
            
            # Geographic distribution if available
            if 'latitude' in cluster_data.columns and 'longitude' in cluster_data.columns:
                valid_coords = cluster_data.dropna(subset=['latitude', 'longitude'])
                if len(valid_coords) > 0:
                    cluster_annotation['geographic_range'] = {
                        'lat_range': [valid_coords['latitude'].min(), valid_coords['latitude'].max()],
                        'lon_range': [valid_coords['longitude'].min(), valid_coords['longitude'].max()],
                        'sample_locations': len(valid_coords)
                    }
            
            annotations[cluster_id] = cluster_annotation
        
        return annotations
    
    def save_models(self, output_dir: str):
        """
        Save trained models and results
        
        Args:
            output_dir: Directory to save models
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save models
        with open(output_path / 'trained_models.pkl', 'wb') as f:
            pickle.dump(self.models, f)
        
        # Save processed data
        with open(output_path / 'processed_data.pkl', 'wb') as f:
            pickle.dump(self.processed_data, f)
        
        # Save features
        if self.combined_features is not None:
            np.save(output_path / 'combined_features.npy', self.combined_features)
        
        print(f"Models and data saved to {output_path}")
    
    def load_models(self, model_dir: str):
        """
        Load previously trained models
        
        Args:
            model_dir: Directory containing saved models
        """
        model_path = Path(model_dir)
        
        # Load models
        with open(model_path / 'trained_models.pkl', 'rb') as f:
            self.models = pickle.load(f)
        
        # Load processed data
        with open(model_path / 'processed_data.pkl', 'rb') as f:
            self.processed_data = pickle.load(f)
        
        # Load features
        if (model_path / 'combined_features.npy').exists():
            self.combined_features = np.load(model_path / 'combined_features.npy')
        
        print(f"Models and data loaded from {model_path}")

def main():
    """
    Main function to run the eDNA processing pipeline
    """
    # Initialize processor
    dataset_path = r"c:\Users\SAMEER GUPTA\Downloads\sih-project\dataset"
    processor = eDNAProcessor(dataset_path)
    
    print("=== eDNA Processing Pipeline for Deep-Sea Biodiversity ===")
    print("Loading datasets...")
    
    # Load all datasets
    datasets = processor.load_datasets()
    
    if not datasets:
        print("No datasets found. Please check the dataset path.")
        return
    
    print("\nPreprocessing data...")
    
    # Preprocess marker data
    marker_df = processor.preprocess_marker_data()
    
    # Preprocess BOLD data
    bold_df = processor.preprocess_bold_data()
    
    print("\nCreating features...")
    
    # Create different types of features
    sequence_features = processor.create_sequence_features()
    abundance_features = processor.create_abundance_features()
    taxonomy_features = processor.create_taxonomy_features()
    
    # Combine all features
    combined_features = processor.combine_features()
    
    if combined_features.size == 0:
        print("No features created. Cannot proceed with training.")
        return
    
    print("\nTraining unsupervised models...")
    
    # Train models
    models = processor.train_unsupervised_models(combined_features)
    
    if not models:
        print("No models trained successfully.")
        return
    
    print("\nAnalyzing results...")
    
    # Get the best performing model for analysis
    best_model_name = None
    best_score = -1
    
    for model_name in ['kmeans', 'dbscan', 'hierarchical']:
        if f'{model_name}_score' in models:
            score = models[f'{model_name}_score']
            if score > best_score:
                best_score = score
                best_model_name = model_name
    
    if best_model_name:
        print(f"\nBest performing model: {best_model_name} (score: {best_score:.3f})")
        
        # Analyze abundance and annotations
        cluster_labels = models[f'{best_model_name}_labels']
        
        abundance_estimates = processor.estimate_abundance(cluster_labels)
        print("\nAbundance analysis completed.")
        
        cluster_annotations = processor.annotate_clusters(cluster_labels)
        print(f"Annotated {len(cluster_annotations)} clusters.")
        
        # Print summary
        print(f"\n=== RESULTS SUMMARY ===")
        print(f"Total samples processed: {len(cluster_labels)}")
        print(f"Number of clusters identified: {len(set(cluster_labels))}")
        print(f"Feature matrix shape: {combined_features.shape}")
        
        if abundance_estimates and 'diversity_stats' in abundance_estimates:
            avg_diversity = np.mean([stats['shannon_diversity'] 
                                   for stats in abundance_estimates['diversity_stats'].values()])
            print(f"Average Shannon diversity: {avg_diversity:.3f}")
    
    # Save results
    print("\nSaving models and results...")
    processor.save_models("./models")
    
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
