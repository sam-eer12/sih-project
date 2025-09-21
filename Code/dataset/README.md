# eDNA Biodiversity Assessment Dataset

## Overview

This dataset contains environmental DNA (eDNA) sequences and taxonomic data for deep-sea biodiversity assessment, specifically designed for the Smart India Hackathon (SIH) AI-driven biodiversity classification project. The dataset supports the development of machine learning models for identifying and classifying marine eukaryotic taxa from raw eDNA reads, addressing the challenge of poor database representation for deep-sea organisms.

## Dataset Structure

```
dataset/
├── README.md                                          # This documentation file
├── 18S_unfiltered_table_excel_with_taxonomy.csv      # 18S rRNA marker gene data (5.8 MB)
├── cox1_unfiltered_table_excel_with_taxonomy.csv     # COI marker gene data (1.7 MB)
├── Annelida/                                          # Annelida phylum sequences
│   ├── bold_data.csv                                  # BOLD database metadata (155 KB)
│   └── fasta.fas                                      # FASTA sequences (109 KB)
├── Arthropoda/                                        # Arthropoda phylum sequences
│   ├── bold_data.csv                                  # BOLD database metadata (147 KB)
│   └── fasta.fas                                      # FASTA sequences (92 KB)
├── Chordata/                                          # Chordata phylum sequences
│   ├── bold_data.csv                                  # BOLD database metadata (98 KB)
│   └── fasta.fas                                      # FASTA sequences (66 KB)
├── Cnidaria/                                          # Cnidaria phylum sequences
│   ├── bold_data.csv                                  # BOLD database metadata (1.0 MB)
│   └── fasta.fas                                      # FASTA sequences (720 KB)
├── Echinodermata/                                     # Echinodermata phylum sequences
│   ├── bold_data.csv                                  # BOLD database metadata (342 KB)
│   └── fasta.fas                                      # FASTA sequences (237 KB)
├── Mollusca/                                          # Mollusca phylum sequences
│   ├── bold_data.csv                                  # BOLD database metadata (169 KB)
│   └── fasta.fas                                      # FASTA sequences (105 KB)
└── Porifera/                                          # Porifera phylum sequences
    ├── bold_data.csv                                  # BOLD database metadata (215 KB)
    └── fasta.fas                                      # FASTA sequences (143 KB)
```

## Data Sources

### Primary Marker Genes
- **18S rRNA**: Small subunit ribosomal RNA gene, widely used for eukaryotic phylogenetic studies
- **COI (cox1)**: Cytochrome c oxidase subunit I gene, standard DNA barcode for animal species identification

### Reference Database
- **BOLD Systems**: Barcode of Life Data Systems, providing comprehensive taxonomic and sequence data
- **Antarctic/Deep-sea Focus**: Sequences primarily from Antarctic and deep-sea environments

## File Descriptions

### Main Dataset Files

#### `18S_unfiltered_table_excel_with_taxonomy.csv`
- **Size**: 5.8 MB (37,863 rows)
- **Format**: ASV (Amplicon Sequence Variant) abundance table
- **Columns**: 
  - `ASV`: Unique sequence identifier (MD5 hash)
  - `taxonomy`: Taxonomic classification string
  - Sample columns: Abundance counts across multiple sampling sites
- **Sampling Sites**: RB1903 expedition samples from Sedna and Darya locations
- **Sample Types**: 1L Sterivex filters, 10L Prefilters, control samples
- **Taxonomic Coverage**: Eukaryota, Bacteria, Archaea, Fungi

#### `cox1_unfiltered_table_excel_with_taxonomy.csv`
- **Size**: 1.7 MB (13,219 rows)
- **Format**: ASV abundance table for COI marker
- **Similar structure** to 18S dataset but focused on metazoan taxa
- **Higher proportion of unassigned sequences** indicating novel deep-sea diversity

### Taxonomic Reference Data

Each phylum directory contains:

#### `bold_data.csv`
Comprehensive metadata from BOLD database including:
- **record_id**: BOLD record identifier
- **processid**: BOLD process ID
- **bin_uri**: Barcode Index Number URI
- **specimen_identifiers**: Sample and field numbers
- **taxonomy**: Complete taxonomic hierarchy (phylum, class, order, family, genus)
- **collection_event**: Geographic coordinates, collection date, collectors
- **sequences**: Sequence ID, marker code, nucleotide sequences
- **genbank_accession**: GenBank accession numbers where available

#### `fasta.fas`
FASTA-formatted sequences with headers containing:
- BOLD process ID
- Taxonomic family
- Marker code (COI-5P)
- GenBank accession (when available)

## Taxonomic Coverage

### Target Phyla (7 major groups)
1. **Annelida** - Segmented worms (177 records)
2. **Arthropoda** - Arthropods (note: directory named "anthropoda") 
3. **Chordata** - Vertebrates and relatives
4. **Cnidaria** - Jellyfish, corals, sea anemones (largest dataset)
5. **Echinodermata** - Sea stars, sea urchins, sea cucumbers
6. **Mollusca** - Mollusks (snails, clams, octopuses)
7. **Porifera** - Sponges

### Geographic Focus
- **Antarctic waters**: Davis Station region, Prince Elizabeth Land
- **Deep-sea environments**: Specialized sampling from research vessels
- **Coordinates**: Primarily around -68.5°S, 77-78°E

## Data Quality and Processing

### Sequence Quality
- **Unfiltered data**: Raw ASV tables without quality filtering
- **Length variation**: Sequences of varying lengths depending on marker
- **Taxonomic assignment**: Hierarchical classification with confidence levels

### Sampling Strategy
- **Multi-filter approach**: 1L and 10L volume filtering
- **Control samples**: Pre-cruise, post-cruise, and laboratory extraction controls
- **Replication**: Multiple sampling sites and dates

### Data Limitations
- **Reference database bias**: Limited deep-sea representation in reference databases
- **Unassigned sequences**: High proportion in COI dataset indicating novel diversity
- **Geographic scope**: Primarily Antarctic, may not represent global deep-sea diversity

## Usage Guidelines

### For Machine Learning Applications
1. **Feature extraction**: Use k-mer analysis, GC content, sequence length
2. **Clustering**: Apply unsupervised learning (K-means, DBSCAN, hierarchical)
3. **Classification**: Train models on known taxonomic assignments
4. **Novel taxa detection**: Identify sequences with low similarity to references

### Data Processing Recommendations
- **Sequence validation**: Check for valid nucleotide characters (A, T, G, C)
- **Length filtering**: Consider minimum/maximum sequence length thresholds
- **Abundance filtering**: Remove low-abundance ASVs that may represent sequencing errors
- **Taxonomic validation**: Verify taxonomic strings follow standard nomenclature

## Technical Specifications

### File Formats
- **CSV**: Comma-separated values with UTF-8 encoding
- **FASTA**: Standard nucleotide sequence format
- **Taxonomic strings**: Semicolon-delimited hierarchical classification

### Computational Requirements
- **Storage**: ~8.5 MB total dataset size
- **Memory**: Moderate requirements for in-memory processing
- **Processing**: Suitable for both local development and cloud deployment

## Integration with SIH Project

This dataset directly supports the project's objectives:
- **AI-driven classification**: Training data for machine learning models
- **Deep-sea focus**: Addresses database representation gaps
- **Multiple markers**: Comprehensive taxonomic coverage
- **Novel taxa discovery**: Unassigned sequences for algorithm development

## Citation and Acknowledgments

- **BOLD Systems**: Barcode of Life Data Systems (www.boldsystems.org)
- **Antarctic Division**: Australian Antarctic Division sampling efforts
- **Research Vessel**: RB1903 expedition data
- **SIH Project**: Smart India Hackathon biodiversity assessment initiative

## Data Updates and Versioning

- **Current Version**: v1.0 (Initial dataset compilation)
- **Last Updated**: 2024
- **Update Frequency**: Static reference dataset for project development

## Contact Information

For questions regarding dataset usage or technical issues, please refer to the main project documentation or contact the SIH project team.

---

*This dataset is part of the Smart India Hackathon project for AI-driven deep-sea biodiversity assessment using environmental DNA.*