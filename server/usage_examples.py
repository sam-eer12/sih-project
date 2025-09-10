"""
Simple usage examples for the eDNA Classification Pipeline
SIH Project - Deep-Sea Biodiversity Assessment
"""

# Example 1: Classify a single DNA sequence
def classify_single_sequence():
    print("=== Example 1: Single Sequence Classification ===")
    
    # Sample COI sequence from Chordata
    sequence = "ACTCTTTACTTAATCTTCGGCGCTTGGGCCGGGATAGTAGGAACAGCCCTTAGCCTGCTCATTCGAGCAGAACTTAGTCAACCCGGCGCCCTGTTGGGGGATGACCAAATTTATAATGTAATTGTTACCGCTCATGCCTTTGTAATAATCTTCTTTATGGTGATGCCAATTATAATCGGAGGTTTTGGAAATTGACTTATCCCCCTTATGATTGGGGCTCCTGACATGGCTTTTCCTCGAATAAATAATATGAGCTTTTGGCTCTTGCCACCCTCTTTTCTGCTCTTGCTAGCTTCGTCAGGTGTTGAGGCTGGGGCAGGGACCGGGTGGACTGTCTACCCTCCCCTTTCTGGAAATTTAGCCCATGCAGGGGGTTCCGTTGATTTAACTATTTTTTCTCTACATTTAGCAGGCATCTCTTCTATTTTAGGAGCAATTAATTTTATTACAACAATTATCAACATGAAGCCCCCTGCTATCTCTCAGTACCAGACCCCTTTGTTCGTGTGGTCTGTGTTAATTACTGCTGTTCTTCTACTTCTTTCACTTCCTGTTCTAGCTGCTGGTATTACTATACTTCTTACGGACCGAAATCTTAACACCACCTTCTTTGATCCTGCAGGAGGGGGGGACCCCATCCTTTACCAACATCTCTT"
    
    print(f"Sequence length: {len(sequence)} bp")
    print(f"GC content: {calculate_gc_content(sequence):.2%}")
    print("Expected: Chordata (fish/vertebrate)")
    print("\nTo classify this sequence, run: python test_pipeline.py")
    print("Then select option 3 and paste this sequence when prompted.")

# Example 2: Create sample CSV for testing
def create_sample_csv():
    print("\n=== Example 2: Sample CSV Creation ===")
    
    import pandas as pd
    
    # Create sample abundance data
    sample_data = {
        'ASV': ['ASV_001', 'ASV_002', 'ASV_003', 'ASV_004', 'ASV_005'],
        'taxonomy': [
            'd__Eukaryota; p__Mollusca; c__Gastropoda; f__Philinidae',
            'd__Eukaryota; p__Cnidaria; c__Anthozoa; f__Scleractinia', 
            'd__Eukaryota; p__Chordata; c__Actinopterygii; f__Nototheniidae',
            'd__Eukaryota; k__Metazoa; p__Annelida; c__Polychaeta',
            'd__Unassigned'
        ],
        'Deep_Site_1': [150, 23, 89, 45, 12],
        'Deep_Site_2': [89, 156, 34, 78, 5],
        'Deep_Site_3': [234, 12, 167, 23, 89],
        'Hydrothermal_Vent': [45, 234, 12, 156, 34]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('sample_edna_data.csv', index=False)
    
    print("Created sample_edna_data.csv with:")
    print(f"- {len(df)} ASVs (taxa)")
    print(f"- {len(df.columns)-2} sampling sites")
    print("- Mixed taxonomic assignments including 'Unassigned'")
    print("\nThis represents typical eDNA metabarcoding results where:")
    print("- Some sequences match known taxa")
    print("- Some remain unassigned (novel deep-sea species)")
    print("- Abundance varies across sampling sites")
    print("\nTo analyze this file, run: python test_pipeline.py")
    print("Then select option 2 and enter 'sample_edna_data.csv'")

# Example 3: Expected outputs
def show_expected_outputs():
    print("\n=== Example 3: Expected Pipeline Outputs ===")
    
    print("When you run the pipeline, you should expect:")
    print("\n1. DATA LOADING:")
    print("   ✓ Loaded 18S dataset: (37863, X) - where X is number of samples")
    print("   ✓ Loaded COX1 dataset: (13219, X)")
    print("   ✓ Loaded taxonomic group data for 7 phyla")
    
    print("\n2. FEATURE CREATION:")
    print("   ✓ Abundance features: Statistical measures of read counts")
    print("   ✓ Sequence features: K-mer patterns from DNA sequences")
    print("   ✓ Taxonomy features: TF-IDF of taxonomic assignments")
    
    print("\n3. UNSUPERVISED CLASSIFICATION:")
    print("   ✓ K-Means clustering: 5-15 distinct clusters")
    print("   ✓ DBSCAN clustering: Density-based novel taxa detection")
    print("   ✓ Hierarchical clustering: Taxonomic structure preservation")
    
    print("\n4. BIODIVERSITY ANALYSIS:")
    print("   ✓ Shannon diversity indices per cluster")
    print("   ✓ Abundance estimates for each taxonomic group")
    print("   ✓ Geographic distribution patterns")
    
    print("\n5. NOVEL TAXA DISCOVERY:")
    print("   ✓ Identification of 'Unassigned' sequences")
    print("   ✓ Clustering of divergent sequences")
    print("   ✓ Potential new species candidates")

def calculate_gc_content(sequence):
    """Calculate GC content of a DNA sequence"""
    if not sequence:
        return 0.0
    sequence = sequence.upper().replace('-', '').replace('N', '')
    if len(sequence) == 0:
        return 0.0
    gc_count = sequence.count('G') + sequence.count('C')
    return gc_count / len(sequence)

# Main function to run examples
def main():
    print("eDNA CLASSIFICATION PIPELINE - USAGE EXAMPLES")
    print("=" * 60)
    print("SIH Project: AI-driven Deep-Sea Biodiversity Assessment")
    print("=" * 60)
    
    classify_single_sequence()
    create_sample_csv()
    show_expected_outputs()
    
    print("\n" + "=" * 60)
    print("GETTING STARTED:")
    print("=" * 60)
    print("1. Install requirements: pip install -r requirements.txt")
    print("2. Run the test: python test_pipeline.py")
    print("3. Choose your input type:")
    print("   - Option 1: Full dataset processing (research mode)")
    print("   - Option 2: CSV file classification (practical mode)")
    print("   - Option 3: DNA sequence classification (query mode)")
    print("   - Option 4: Quick functionality test")
    print("\n4. For web interface integration:")
    print("   - Use the eDNAProcessor class in your API endpoints")
    print("   - Load pre-trained models for real-time classification")
    print("   - Implement file upload for CSV/FASTA processing")
    
    print("\n" + "=" * 60)
    print("EXAMPLE USE CASES FOR SIH DEMO:")
    print("=" * 60)
    print("🔬 Research Scenario:")
    print("   'We collected water samples from 3000m depth in the Arabian Sea'")
    print("   → Upload CSV with abundance data → Get taxonomic clusters")
    
    print("\n🧬 Query Scenario:")
    print("   'We found this DNA sequence - what organism could it be?'")
    print("   → Input sequence → Get cluster assignment + similarity score")
    
    print("\n📊 Monitoring Scenario:")
    print("   'We want to track biodiversity changes over time'")
    print("   → Process multiple CSV files → Compare diversity indices")
    
    print("\n🌊 Discovery Scenario:")
    print("   'Are there novel species in our deep-sea samples?'")
    print("   → Run full pipeline → Identify unassigned clusters")

if __name__ == "__main__":
    main()
