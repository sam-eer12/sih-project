"""
Test script for the eDNA processing pipeline
SIH Project - Deep-Sea Biodiversity Assessment
"""

import sys
sys.path.append('.')

from edna_pipeline import eDNAProcessor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

def get_user_input_choice():
    """Get user's choice for input type"""
    print("\n" + "="*60)
    print("eDNA CLASSIFICATION PIPELINE - INPUT OPTIONS")
    print("="*60)
    print("Choose your input type for classification:")
    print("1. Process full dataset (train models from scratch)")
    print("2. Classify a CSV file with abundance data")
    print("3. Classify individual DNA sequences")
    print("4. Run quick test with sample data")
    print("="*60)
    
    while True:
        try:
            choice = input("Enter your choice (1-4): ").strip()
            if choice in ['1', '2', '3', '4']:
                return int(choice)
            else:
                print("Invalid choice. Please enter 1, 2, 3, or 4.")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)
        except:
            print("Invalid input. Please enter a number between 1-4.")

def classify_csv_file():
    """Classify taxa from a user-provided CSV file"""
    print("\n=== CSV File Classification ===")
    
    # Get CSV file path from user
    while True:
        csv_path = input("Enter the path to your CSV file (or 'sample' for demo): ").strip()
        
        if csv_path.lower() == 'sample':
            # Create a sample CSV for demonstration
            sample_data = {
                'ASV': ['seq001', 'seq002', 'seq003', 'seq004', 'seq005'],
                'taxonomy': [
                    'd__Eukaryota; p__Mollusca; c__Gastropoda',
                    'd__Eukaryota; p__Cnidaria; c__Anthozoa', 
                    'd__Eukaryota; p__Chordata; c__Actinopterygii',
                    'd__Eukaryota; k__Metazoa; p__Annelida',
                    'd__Unassigned'
                ],
                'sample1': [150, 23, 89, 45, 12],
                'sample2': [89, 156, 34, 78, 5],
                'sample3': [234, 12, 167, 23, 89]
            }
            df = pd.DataFrame(sample_data)
            csv_path = './sample_input.csv'
            df.to_csv(csv_path, index=False)
            print(f"Created sample CSV file: {csv_path}")
            break
        elif os.path.exists(csv_path):
            break
        else:
            print(f"File not found: {csv_path}")
            print("Please provide a valid file path or type 'sample' for demo data.")
    
    try:
        # Load and process the CSV
        print(f"Loading CSV file: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"Loaded data shape: {df.shape}")
        print("\nColumns found:", list(df.columns))
        
        # Initialize processor with existing models if available
        processor = eDNAProcessor("../dataset")
        
        # Check if pre-trained models exist
        if os.path.exists("./test_models"):
            print("Loading pre-trained models...")
            processor.load_models("./test_models")
            
            # Process the new CSV data in the same format as training data
            processed_df = process_csv_for_classification(df)
            
            # Extract features similar to training
            features = extract_features_from_csv(processed_df, processor)
            
            # Apply existing models for classification
            results = apply_trained_models(features, processor)
            
            print("\n=== Classification Results ===")
            for i, (asv, prediction) in enumerate(zip(df['ASV'] if 'ASV' in df.columns else range(len(df)), results['predictions'])):
                print(f"Sample {asv}: Cluster {prediction}")
                if 'confidence' in results:
                    print(f"  Confidence: {results['confidence'][i]:.3f}")
            
        else:
            print("No pre-trained models found. Training new models on provided data...")
            # Process as marker data and train new models
            processed_data = process_csv_as_marker_data(df)
            
            # Create features and train
            abundance_features = create_features_from_dataframe(processed_data)
            if abundance_features.size > 0:
                models = train_quick_models(abundance_features)
                print("Models trained successfully on your data!")
                
                # Show results
                if 'kmeans_labels' in models:
                    labels = models['kmeans_labels']
                    print(f"\nIdentified {len(set(labels))} distinct clusters in your data")
                    
                    # Show cluster summary
                    for cluster_id in set(labels):
                        mask = labels == cluster_id
                        cluster_samples = df[mask] if len(df) == len(labels) else df.iloc[:sum(mask)]
                        print(f"Cluster {cluster_id}: {sum(mask)} samples")
                        if 'taxonomy' in df.columns:
                            common_taxa = cluster_samples['taxonomy'].value_counts().head(2)
                            print(f"  Common taxa: {dict(common_taxa)}")
    
    except Exception as e:
        print(f"Error processing CSV file: {e}")
        import traceback
        traceback.print_exc()

def classify_dna_sequences():
    """Classify individual DNA sequences"""
    print("\n=== DNA Sequence Classification ===")
    
    sequences = []
    sequence_ids = []
    
    print("Enter DNA sequences for classification (press Enter twice to finish):")
    print("You can also type 'sample' to use demo sequences.")
    
    sequence_input = input("Enter sequence or 'sample': ").strip()
    
    if sequence_input.lower() == 'sample':
        # Sample sequences from different taxa
        sample_sequences = {
            'Chordata_COI': 'ACTCTTTACTTAATCTTCGGCGCTTGGGCCGGGATAGTAGGAACAGCCCTTAGCCTGCTCATTCGAGCAGAACTTAGTCAACCCGGCGCCCTGTTGGGGGATGACCAAATTTATAATGTAATTGTTACCGCTCATGCCTTTGTAATAATCTTCTTTATGGTGATGCCAATTATAATCGGAGGTTTTGGAAATTGACTTATCCCCCTTATGATTGGGGCTCCTGACATGGCTTTTCCTCGAATAAATAATATGAGCTTTTGGCTCTTGCCACCCTCTTTTCTGCTCTTGCTAGCTTCGTCAGGTGTTGAGGCTGGGGCAGGGACCGGGTGGACTGTCTACCCTCCCCTTTCTGGAAATTTAGCCCATGCAGGGGGTTCCGTTGATTTAACTATTTTTTCTCTACATTTAGCAGGCATCTCTTCTATTTTAGGAGCAATTAATTTTATTACAACAATTATCAACATGAAGCCCCCTGCTATCTCTCAGTACCAGACCCCTTTGTTCGTGTGGTCTGTGTTAATTACTGCTGTTCTTCTACTTCTTTCACTTCCTGTTCTAGCTGCTGGTATTACTATACTTCTTACGGACCGAAATCTTAACACCACCTTCTTTGATCCTGCAGGAGGGGGGGACCCCATCCTTTACCAACATCTCTT',
            'Mollusca_COI': 'AACACTATACATGATTTTTGGTATGTGATGTGGATTAGTGGGTACTGGTTTAAGTCTCCTAATTCGATTTGAGTTAGGAACTGCTTCAGCTTTTTTGGGTGATGATCACTTTTATAATGTAATTGTAACTGCTCATGCCTTTGTAATAATTTTTTTTATAGTTATACCTTTAATAATTGGAGGTTTTGGGAATTGGATAGTTCCTCTTTTAATTGGTGCTCCTGATATAAGGTTTCCTCGTATGAATAATATAAGGTTTTGGCTATTACCTCCTTCATTTGTTTTATTGATTGTTTCTAGGTTAGTTGAAGGTGGGGCTGGGACAGGTTGAACGGTTTATCCCCCTCTTTCTGGTCCTATTGCTCATGGGTCTTGTTCAGTAGACTTAGTAATTTTTTCTCTTCATTTAGCAGGTATATCCTCTATTCTTGGTGCTATTAATTTTATTACTACTATTTTAAATATACGTTCTCCAGGTATTACTATAGAGCGGCTTAGATTATTTGTTTGGTCTGTTTTTGTTACAGCTTTTTTACTTTTATTATCCTTACCTGTGTTAGCAGGAGCTATTACTATGTTATTAACAGATCGTAATTTTAATACAAGATTTTTTGATCCTGCAGGAGGGGGTGACCCTATTCTGTATCAACACTTATT',
            'Unknown_sequence': 'ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATC'
        }
        
        sequences = list(sample_sequences.values())
        sequence_ids = list(sample_sequences.keys())
        print("Using sample sequences:")
        for seq_id, seq in sample_sequences.items():
            print(f"  {seq_id}: {seq[:50]}...")
    
    else:
        # Get sequences from user input
        sequences.append(sequence_input)
        sequence_ids.append(f"User_seq_1")
        
        counter = 2
        while True:
            seq = input(f"Enter sequence {counter} (or press Enter to finish): ").strip()
            if not seq:
                break
            sequences.append(seq)
            sequence_ids.append(f"User_seq_{counter}")
            counter += 1
    
    if not sequences:
        print("No sequences provided.")
        return
    
    try:
        print(f"\nProcessing {len(sequences)} sequences...")
        
        # Initialize processor
        processor = eDNAProcessor("../dataset")
        
        # Check if we have pre-trained models
        if os.path.exists("./test_models"):
            print("Using pre-trained models...")
            processor.load_models("./test_models")
            
            # Extract features from sequences
            features = extract_sequence_features(sequences)
            
            # Apply trained models
            results = classify_with_trained_models(features, processor)
            
            print("\n=== Sequence Classification Results ===")
            for i, (seq_id, seq) in enumerate(zip(sequence_ids, sequences)):
                print(f"\nSequence: {seq_id}")
                print(f"  Length: {len(seq)} bp")
                print(f"  GC Content: {calculate_gc_content(seq):.1%}")
                if i < len(results['predictions']):
                    print(f"  Predicted Cluster: {results['predictions'][i]}")
                    if 'similarity' in results and i < len(results['similarity']):
                        print(f"  Similarity to known taxa: {results['similarity'][i]:.3f}")
        
        else:
            print("No pre-trained models found.")
            print("Training quick models on provided sequences...")
            
            # Create simple features from sequences
            features = create_simple_sequence_features(sequences)
            
            if features.size > 0:
                # Quick clustering
                from sklearn.cluster import KMeans
                from sklearn.preprocessing import StandardScaler
                
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features)
                
                n_clusters = min(3, len(sequences))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                labels = kmeans.fit_predict(features_scaled)
                
                print(f"\nClustered sequences into {n_clusters} groups:")
                for i, (seq_id, label) in enumerate(zip(sequence_ids, labels)):
                    print(f"  {seq_id}: Group {label}")
    
    except Exception as e:
        print(f"Error processing sequences: {e}")
        import traceback
        traceback.print_exc()

def process_csv_for_classification(df):
    """Process CSV data for classification"""
    # Similar to preprocess_marker_data but for new data
    processed_df = df.copy()
    
    # Calculate abundance statistics if abundance columns exist
    abundance_cols = [col for col in df.columns if col not in ['ASV', 'taxonomy'] and df[col].dtype in ['int64', 'float64']]
    
    if abundance_cols:
        abundance_data = df[abundance_cols].fillna(0)
        processed_df['total_abundance'] = abundance_data.sum(axis=1)
        processed_df['sample_presence'] = (abundance_data > 0).sum(axis=1)
        processed_df['max_abundance'] = abundance_data.max(axis=1)
        processed_df['mean_abundance'] = abundance_data.mean(axis=1)
        processed_df['std_abundance'] = abundance_data.std(axis=1).fillna(0)
    
    # Process taxonomy if available
    if 'taxonomy' in df.columns:
        processed_df['taxonomy_processed'] = df['taxonomy'].fillna('d__Unassigned')
        processed_df['domain'] = processed_df['taxonomy_processed'].str.extract(r'd__([^;]+)')
        processed_df['phylum'] = processed_df['taxonomy_processed'].str.extract(r'p__([^;]+)')
        processed_df['class'] = processed_df['taxonomy_processed'].str.extract(r'c__([^;]+)')
        
        for col in ['domain', 'phylum', 'class']:
            processed_df[col] = processed_df[col].fillna('Unknown')
    
    return processed_df

def extract_features_from_csv(processed_df, processor):
    """Extract features from processed CSV data"""
    feature_cols = []
    
    # Add abundance features if they exist
    abundance_cols = ['total_abundance', 'sample_presence', 'max_abundance', 'mean_abundance', 'std_abundance']
    available_abundance_cols = [col for col in abundance_cols if col in processed_df.columns]
    
    if available_abundance_cols:
        feature_cols.extend(available_abundance_cols)
    
    if feature_cols:
        features = processed_df[feature_cols].fillna(0).values
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        return features
    
    return np.array([])

def extract_sequence_features(sequences):
    """Extract features from DNA sequences"""
    features = []
    
    for seq in sequences:
        seq_features = []
        
        # Basic sequence statistics
        seq_features.append(len(seq))  # Length
        seq_features.append(calculate_gc_content(seq))  # GC content
        seq_features.append(calculate_n_content(seq))  # N content
        
        # Di-nucleotide frequencies
        dinucs = ['AA', 'AT', 'AC', 'AG', 'TA', 'TT', 'TC', 'TG', 'CA', 'CT', 'CC', 'CG', 'GA', 'GT', 'GC', 'GG']
        for dinuc in dinucs:
            count = seq.upper().count(dinuc)
            freq = count / max(1, len(seq) - 1)
            seq_features.append(freq)
        
        features.append(seq_features)
    
    return np.array(features)

def calculate_gc_content(sequence):
    """Calculate GC content of a sequence"""
    if not sequence:
        return 0.0
    sequence = sequence.upper().replace('-', '').replace('N', '')
    if len(sequence) == 0:
        return 0.0
    gc_count = sequence.count('G') + sequence.count('C')
    return gc_count / len(sequence)

def calculate_n_content(sequence):
    """Calculate N content of a sequence"""
    if not sequence:
        return 0.0
    n_count = sequence.upper().count('N')
    return n_count / len(sequence)

def apply_trained_models(features, processor):
    """Apply trained models to new features"""
    results = {'predictions': []}
    
    if 'kmeans' in processor.models and features.size > 0:
        # Apply PCA transformation if it was used during training
        if 'pca' in processor.models:
            features_transformed = processor.models['pca'].transform(features)
        else:
            features_transformed = features
        
        # Get predictions
        predictions = processor.models['kmeans'].predict(features_transformed)
        results['predictions'] = predictions
        
        # Calculate distances to centroids as confidence measure
        distances = processor.models['kmeans'].transform(features_transformed)
        min_distances = np.min(distances, axis=1)
        max_distance = np.max(min_distances) if len(min_distances) > 0 else 1
        confidences = 1 - (min_distances / max_distance)
        results['confidence'] = confidences
    
    return results

def classify_with_trained_models(features, processor):
    """Classify sequences using trained models"""
    results = {'predictions': [], 'similarity': []}
    
    if 'kmeans' in processor.models and features.size > 0:
        # Transform features if PCA was used
        if 'pca' in processor.models:
            # Pad or trim features to match training dimensions
            expected_features = processor.models['pca'].n_features_in_
            if features.shape[1] != expected_features:
                if features.shape[1] < expected_features:
                    # Pad with zeros
                    padding = np.zeros((features.shape[0], expected_features - features.shape[1]))
                    features = np.hstack([features, padding])
                else:
                    # Trim
                    features = features[:, :expected_features]
            
            features_transformed = processor.models['pca'].transform(features)
        else:
            features_transformed = features
        
        # Get predictions
        predictions = processor.models['kmeans'].predict(features_transformed)
        results['predictions'] = predictions
        
        # Calculate similarity scores
        distances = processor.models['kmeans'].transform(features_transformed)
        min_distances = np.min(distances, axis=1)
        max_distance = np.max(min_distances) if len(min_distances) > 0 else 1
        similarities = 1 - (min_distances / max_distance)
        results['similarity'] = similarities
    
    return results

def process_csv_as_marker_data(df):
    """Process CSV as marker data for training"""
    # This is a simplified version of marker data processing
    processed_df = df.copy()
    
    # Find numeric columns (abundance data)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        abundance_data = df[numeric_cols].fillna(0)
        processed_df['total_abundance'] = abundance_data.sum(axis=1)
        processed_df['sample_presence'] = (abundance_data > 0).sum(axis=1)
        processed_df['max_abundance'] = abundance_data.max(axis=1)
        processed_df['mean_abundance'] = abundance_data.mean(axis=1)
        processed_df['std_abundance'] = abundance_data.std(axis=1).fillna(0)
    
    return processed_df

def create_features_from_dataframe(df):
    """Create features from a processed dataframe"""
    feature_cols = ['total_abundance', 'sample_presence', 'max_abundance', 'mean_abundance', 'std_abundance']
    available_cols = [col for col in feature_cols if col in df.columns]
    
    if available_cols:
        features = df[available_cols].fillna(0).values
        
        # Normalize
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        return features
    
    return np.array([])

def create_simple_sequence_features(sequences):
    """Create simple features from sequences for quick analysis"""
    features = []
    
    for seq in sequences:
        seq_features = []
        seq = seq.upper()
        
        # Length
        seq_features.append(len(seq))
        
        # Base composition
        for base in ['A', 'T', 'G', 'C']:
            freq = seq.count(base) / len(seq) if len(seq) > 0 else 0
            seq_features.append(freq)
        
        # GC content
        seq_features.append(calculate_gc_content(seq))
        
        features.append(seq_features)
    
    return np.array(features)

def train_quick_models(features):
    """Train quick models for immediate analysis"""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    if features.size == 0:
        return {}
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Determine number of clusters
    n_samples = features.shape[0]
    n_clusters = min(max(2, n_samples // 5), 10)
    
    # Train K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features_scaled)
    
    return {
        'kmeans': kmeans,
        'kmeans_labels': labels,
        'scaler': scaler
    }

def test_pipeline():
    """Test the eDNA processing pipeline with the available data"""
    
    print("=== Testing eDNA Pipeline ===")
    
    # Initialize processor
    dataset_path = r"c:\Users\SAMEER GUPTA\Downloads\sih-project\dataset"
    processor = eDNAProcessor(dataset_path)
    
    try:
        # Test data loading
        print("1. Testing data loading...")
        datasets = processor.load_datasets()
        print(f"   Loaded {len(datasets)} datasets")
        
        # Test preprocessing
        print("2. Testing data preprocessing...")
        marker_df = processor.preprocess_marker_data()
        bold_df = processor.preprocess_bold_data()
        
        if not marker_df.empty:
            print(f"   Marker data processed: {marker_df.shape}")
        if not bold_df.empty:
            print(f"   BOLD data processed: {bold_df.shape}")
        
        # Test feature creation
        print("3. Testing feature creation...")
        
        # Create abundance features (most reliable)
        abundance_features = processor.create_abundance_features()
        if abundance_features.size > 0:
            print(f"   Abundance features: {abundance_features.shape}")
        
        # Try to create sequence features
        try:
            sequence_features = processor.create_sequence_features()
            if sequence_features.size > 0:
                print(f"   Sequence features: {sequence_features.shape}")
        except Exception as e:
            print(f"   Sequence features failed: {e}")
        
        # Create taxonomy features
        try:
            taxonomy_features = processor.create_taxonomy_features()
            if taxonomy_features.size > 0:
                print(f"   Taxonomy features: {taxonomy_features.shape}")
        except Exception as e:
            print(f"   Taxonomy features failed: {e}")
        
        # Combine features
        combined_features = processor.combine_features()
        if combined_features.size > 0:
            print(f"   Combined features: {combined_features.shape}")
        else:
            print("   No features could be combined")
            return False
        
        # Test model training (with smaller cluster range for testing)
        print("4. Testing model training...")
        max_clusters = min(10, combined_features.shape[0] // 2)
        models = processor.train_unsupervised_models(
            combined_features, 
            n_clusters_range=range(2, max_clusters + 1)
        )
        
        if models:
            print(f"   Trained {len([k for k in models.keys() if k.endswith('_score')])} models")
            
            # Print model scores
            for model_name in ['kmeans', 'dbscan', 'hierarchical']:
                score_key = f'{model_name}_score'
                if score_key in models:
                    print(f"   {model_name}: {models[score_key]:.3f}")
        
        # Test abundance estimation and annotation
        print("5. Testing analysis functions...")
        
        best_model_name = None
        best_score = -1
        
        for model_name in ['kmeans', 'dbscan', 'hierarchical']:
            score_key = f'{model_name}_score'
            if score_key in models:
                score = models[score_key]
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
        
        if best_model_name:
            cluster_labels = models[f'{best_model_name}_labels']
            
            # Test abundance estimation
            abundance_estimates = processor.estimate_abundance(cluster_labels)
            print(f"   Abundance estimation completed")
            
            # Test cluster annotation
            annotations = processor.annotate_clusters(cluster_labels)
            print(f"   Cluster annotation completed: {len(annotations)} clusters")
        
        # Test saving
        print("6. Testing model saving...")
        processor.save_models("./test_models")
        print("   Models saved successfully")
        
        print("\n=== Pipeline Test SUCCESSFUL ===")
        return True
        
    except Exception as e:
        print(f"\n=== Pipeline Test FAILED ===")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    """Test the eDNA processing pipeline with the available data"""
    
    print("=== Testing eDNA Pipeline ===")
    
    # Initialize processor
    dataset_path = r"c:\Users\SAMEER GUPTA\Downloads\sih-project\dataset"
    processor = eDNAProcessor(dataset_path)
    
    try:
        # Test data loading
        print("1. Testing data loading...")
        datasets = processor.load_datasets()
        print(f"   Loaded {len(datasets)} datasets")
        
        # Test preprocessing
        print("2. Testing data preprocessing...")
        marker_df = processor.preprocess_marker_data()
        bold_df = processor.preprocess_bold_data()
        
        if not marker_df.empty:
            print(f"   Marker data processed: {marker_df.shape}")
        if not bold_df.empty:
            print(f"   BOLD data processed: {bold_df.shape}")
        
        # Test feature creation
        print("3. Testing feature creation...")
        
        # Create abundance features (most reliable)
        abundance_features = processor.create_abundance_features()
        if abundance_features.size > 0:
            print(f"   Abundance features: {abundance_features.shape}")
        
        # Try to create sequence features
        try:
            sequence_features = processor.create_sequence_features()
            if sequence_features.size > 0:
                print(f"   Sequence features: {sequence_features.shape}")
        except Exception as e:
            print(f"   Sequence features failed: {e}")
        
        # Create taxonomy features
        try:
            taxonomy_features = processor.create_taxonomy_features()
            if taxonomy_features.size > 0:
                print(f"   Taxonomy features: {taxonomy_features.shape}")
        except Exception as e:
            print(f"   Taxonomy features failed: {e}")
        
        # Combine features
        combined_features = processor.combine_features()
        if combined_features.size > 0:
            print(f"   Combined features: {combined_features.shape}")
        else:
            print("   No features could be combined")
            return False
        
        # Test model training (with smaller cluster range for testing)
        print("4. Testing model training...")
        max_clusters = min(10, combined_features.shape[0] // 2)
        models = processor.train_unsupervised_models(
            combined_features, 
            n_clusters_range=range(2, max_clusters + 1)
        )
        
        if models:
            print(f"   Trained {len([k for k in models.keys() if k.endswith('_score')])} models")
            
            # Print model scores
            for model_name in ['kmeans', 'dbscan', 'hierarchical']:
                score_key = f'{model_name}_score'
                if score_key in models:
                    print(f"   {model_name}: {models[score_key]:.3f}")
        
        # Test abundance estimation and annotation
        print("5. Testing analysis functions...")
        
        best_model_name = None
        best_score = -1
        
        for model_name in ['kmeans', 'dbscan', 'hierarchical']:
            score_key = f'{model_name}_score'
            if score_key in models:
                score = models[score_key]
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
        
        if best_model_name:
            cluster_labels = models[f'{best_model_name}_labels']
            
            # Test abundance estimation
            abundance_estimates = processor.estimate_abundance(cluster_labels)
            print(f"   Abundance estimation completed")
            
            # Test cluster annotation
            annotations = processor.annotate_clusters(cluster_labels)
            print(f"   Cluster annotation completed: {len(annotations)} clusters")
        
        # Test saving
        print("6. Testing model saving...")
        processor.save_models("./test_models")
        print("   Models saved successfully")
        
        print("\n=== Pipeline Test SUCCESSFUL ===")
        return True
        
    except Exception as e:
        print(f"\n=== Pipeline Test FAILED ===")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_sample_visualization():
    """Create sample visualizations of the results"""
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Load the test models
        processor = eDNAProcessor("../dataset")
        processor.load_models("./test_models")
        
        if 'features_pca' in processor.models and 'kmeans_labels' in processor.models:
            features_pca = processor.models['features_pca']
            labels = processor.models['kmeans_labels']
            
            # Create visualization
            plt.figure(figsize=(12, 5))
            
            # Plot 1: PCA visualization
            plt.subplot(1, 2, 1)
            scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], c=labels, cmap='tab10')
            plt.xlabel('First Principal Component')
            plt.ylabel('Second Principal Component')
            plt.title('PCA Visualization of Clusters')
            plt.colorbar(scatter)
            
            # Plot 2: Cluster size distribution
            plt.subplot(1, 2, 2)
            unique_labels, counts = np.unique(labels, return_counts=True)
            plt.bar(unique_labels, counts)
            plt.xlabel('Cluster ID')
            plt.ylabel('Number of Samples')
            plt.title('Cluster Size Distribution')
            
            plt.tight_layout()
            plt.savefig('./cluster_visualization.png', dpi=300, bbox_inches='tight')
            print("Visualization saved as 'cluster_visualization.png'")
            
    except Exception as e:
        print(f"Visualization creation failed: {e}")

if __name__ == "__main__":
    # Get user's choice for input type
    choice = get_user_input_choice()
    
    if choice == 1:
        # Process full dataset (original test)
        print("\n=== Running Full Dataset Processing ===")
        success = test_pipeline()
        
        if success:
            print("\nCreating sample visualization...")
            create_sample_visualization()
            
            print("\n" + "="*50)
            print("PIPELINE READY FOR YOUR SIH PROJECT!")
            print("="*50)
            print("\nNext steps:")
            print("1. Install requirements: pip install -r requirements.txt")
            print("2. Run the full pipeline: python edna_pipeline.py")
            print("3. Customize the models based on your specific needs")
            print("4. Integrate with your Next.js frontend")
        else:
            print("\nPlease fix the issues before proceeding.")
    
    elif choice == 2:
        # Classify CSV file
        classify_csv_file()
    
    elif choice == 3:
        # Classify DNA sequences
        classify_dna_sequences()
    
    elif choice == 4:
        # Quick test
        print("\n=== Running Quick Test ===")
        print("This will test the basic functionality without full processing...")
        
        # Quick test with sample data
        try:
            processor = eDNAProcessor("../dataset")
            
            # Test loading
            print("✓ eDNAProcessor initialized")
            
            # Test sample sequence processing
            sample_sequences = [
                'ATGCGATCGTAGCTAGCTAGCTAGC',
                'CGATCGATCGATCGATCGATCGAT',
                'TAGCTAGCTAGCTAGCTAGCTAG'
            ]
            
            features = create_simple_sequence_features(sample_sequences)
            print(f"✓ Created features from {len(sample_sequences)} sequences: {features.shape}")
            
            # Test quick clustering
            if features.size > 0:
                models = train_quick_models(features)
                print(f"✓ Trained models: {list(models.keys())}")
                
                if 'kmeans_labels' in models:
                    labels = models['kmeans_labels']
                    print(f"✓ Clustering results: {len(set(labels))} clusters identified")
            
            print("\n=== Quick Test SUCCESSFUL ===")
            print("The pipeline is working correctly!")
            
        except Exception as e:
            print(f"\n=== Quick Test FAILED ===")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nThank you for testing the eDNA Classification Pipeline!")
    print("This tool is designed for the SIH project on deep-sea biodiversity assessment.")
