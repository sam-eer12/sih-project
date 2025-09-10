# eDNA Deep-Sea Biodiversity Analysis Pipeline

## SIH Project - AI-driven Pipeline for Eukaryotic Taxa Classification

This pipeline addresses the challenge of analyzing environmental DNA (eDNA) from deep-sea environments where traditional reference databases are inadequate. It uses unsupervised machine learning to classify eukaryotic taxa and estimate biodiversity directly from raw eDNA data.

## 🎯 Problem Statement

Deep-sea ecosystems harbor significant biodiversity but are poorly represented in reference databases (SILVA, PR2, NCBI). Traditional bioinformatic pipelines fail with novel deep-sea taxa, leading to misclassifications and underestimated biodiversity.

## 🔬 Solution

An AI-driven pipeline that:
- **Classifies sequences** without heavy reliance on reference databases
- **Annotates taxa** using deep learning/unsupervised learning
- **Estimates abundance** from raw eDNA reads
- **Discovers novel taxa** in deep-sea environments
- **Reduces computational time** compared to traditional methods

## 📊 Data Sources

### Marker Gene Data
- **18S rRNA**: Eukaryotic diversity marker
- **COX1**: Mitochondrial marker for metazoans

### Taxonomic Groups
- Annelida (segmented worms)
- Arthropoda (crustaceans, etc.)
- Chordata (fish, vertebrates)
- Cnidaria (jellyfish, corals)
- Echinodermata (sea stars, urchins)
- Mollusca (shells, gastropods)
- Porifera (sponges)

## 🚀 Features

### 1. Data Processing
- **Multi-format support**: CSV abundance tables, FASTA sequences
- **Abundance analysis**: Sample presence, total abundance, diversity metrics
- **Taxonomic parsing**: Hierarchical taxonomy extraction
- **Geographic integration**: Coordinate-based analysis

### 2. Feature Engineering
- **K-mer features**: 3-mer and 4-mer sequence patterns
- **Abundance features**: Statistical abundance measures
- **Taxonomic features**: TF-IDF of taxonomic information
- **Combined features**: Integrated multi-modal features

### 3. Unsupervised Learning
- **K-Means**: Centroid-based clustering
- **DBSCAN**: Density-based clustering for novel taxa discovery
- **Hierarchical**: Agglomerative clustering for taxonomic structure
- **Dimensionality reduction**: PCA, t-SNE for visualization

### 4. Analysis & Annotation
- **Abundance estimation**: Shannon diversity, species richness
- **Cluster annotation**: Taxonomic consensus, geographic distribution
- **Model evaluation**: Silhouette scores, cluster validation

## 📁 Project Structure

```
server/
├── edna_pipeline.py      # Main pipeline implementation
├── test_pipeline.py      # Testing and validation script
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── models/              # Saved trained models (generated)
```

## ⚙️ Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Optional advanced packages**:
```bash
# For UMAP dimensionality reduction
pip install umap-learn

# For HDBSCAN clustering
pip install hdbscan

# For deep learning extensions
pip install tensorflow torch
```

## 🎮 Usage

### Quick Start
```bash
# Test the pipeline
python test_pipeline.py

# Run full pipeline
python edna_pipeline.py
```

### Programmatic Usage
```python
from edna_pipeline import eDNAProcessor

# Initialize processor
processor = eDNAProcessor("path/to/dataset")

# Load and process data
datasets = processor.load_datasets()
processor.preprocess_marker_data()
processor.preprocess_bold_data()

# Create features
abundance_features = processor.create_abundance_features()
sequence_features = processor.create_sequence_features()
combined_features = processor.combine_features()

# Train models
models = processor.train_unsupervised_models(combined_features)

# Analyze results
cluster_labels = models['kmeans_labels']
abundance_estimates = processor.estimate_abundance(cluster_labels)
annotations = processor.annotate_clusters(cluster_labels)

# Save results
processor.save_models("./models")
```

## 🔍 Key Classes and Methods

### `eDNAProcessor`
Main pipeline class for processing eDNA data.

#### Key Methods:
- `load_datasets()`: Load all CSV and FASTA files
- `preprocess_marker_data()`: Process 18S/COX1 abundance data
- `preprocess_bold_data()`: Process BOLD taxonomic data
- `create_sequence_features()`: K-mer based sequence features
- `create_abundance_features()`: Statistical abundance features
- `create_taxonomy_features()`: TF-IDF taxonomic features
- `train_unsupervised_models()`: Train clustering models
- `estimate_abundance()`: Calculate diversity metrics
- `annotate_clusters()`: Assign taxonomic labels to clusters

## 📈 Output and Results

### Model Performance
- **Silhouette scores**: Cluster quality metrics
- **Cluster validation**: Internal clustering metrics
- **Feature importance**: Contribution analysis

### Biodiversity Analysis
- **Species richness**: Number of distinct taxa
- **Shannon diversity**: Diversity index per cluster
- **Abundance patterns**: Distribution across samples
- **Geographic patterns**: Spatial distribution analysis

### Taxonomic Annotation
- **Consensus taxonomy**: Most likely taxonomic assignment
- **Confidence scores**: Assignment reliability
- **Novel taxa detection**: Unassigned/divergent sequences

## 🔧 Customization

### Adding New Features
```python
def custom_feature_extractor(sequences):
    # Your custom feature extraction logic
    return feature_matrix

# Integrate into pipeline
processor.custom_features = custom_feature_extractor(sequences)
```

### Custom Clustering
```python
from sklearn.cluster import SpectralClustering

# Add custom clustering algorithm
spectral = SpectralClustering(n_clusters=10)
custom_labels = spectral.fit_predict(features)
```

## 🎯 SIH Project Integration

### For the Hackathon:
1. **Demo Pipeline**: Use `test_pipeline.py` for quick demonstration
2. **Web Interface**: Integrate with Next.js frontend
3. **API Endpoints**: Create REST API for model predictions
4. **Visualization**: Generate plots for biodiversity patterns
5. **Real-time Analysis**: Process new eDNA samples

### Expected Deliverables:
- ✅ Unsupervised classification pipeline
- ✅ Abundance estimation algorithms
- ✅ Novel taxa discovery capability
- ✅ Reduced computational requirements
- ✅ Database-independent operation

## 🌊 Deep-Sea Specific Features

### Environmental Context
- **Depth categorization**: Abyssal, seamount, hydrothermal
- **Geographic clustering**: Regional biodiversity patterns
- **Sample type analysis**: Water vs. sediment eDNA

### Novel Taxa Discovery
- **Outlier detection**: DBSCAN for divergent sequences
- **Sequence divergence**: K-mer based novelty detection
- **Taxonomic gaps**: Unassigned sequence identification

## 📊 Performance Metrics

### Computational Efficiency
- **Processing time**: Optimized for large datasets
- **Memory usage**: Efficient feature representation
- **Scalability**: Handles growing sample sizes

### Biological Relevance
- **Taxonomic accuracy**: Validated against known taxa
- **Diversity estimation**: Compared to traditional methods
- **Geographic patterns**: Consistent with ecological expectations

## 🔮 Future Enhancements

### Deep Learning Integration
- **Autoencoder features**: Learned sequence representations
- **CNN models**: Convolutional features for sequences
- **Attention mechanisms**: Sequence pattern discovery

### Advanced Analytics
- **Phylogenetic clustering**: Evolution-aware grouping
- **Temporal analysis**: Time-series biodiversity
- **Ecological modeling**: Environment-biodiversity relationships

## 🤝 Contributing

This is a SIH project. To contribute:
1. Fork the repository
2. Create feature branch
3. Test with your data
4. Submit pull request

## 📝 Citation

If you use this pipeline in your research:

```
eDNA Deep-Sea Biodiversity Analysis Pipeline
SIH Project 2025 - Centre for Marine Living Resources and Ecology (CMLRE)
AI-driven pipeline for eukaryotic taxa classification in deep-sea environments
```

## 📞 Support

For questions about the SIH project implementation:
- Check the test outputs in `test_pipeline.py`
- Review error logs for debugging
- Ensure all dependencies are installed
- Verify dataset file paths and formats

---

**Built for Smart India Hackathon 2025**  
*Advancing Deep-Sea Biodiversity Discovery through AI*
