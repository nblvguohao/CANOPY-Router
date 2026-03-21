#!/usr/bin/env python3
"""
W5: Routing weight interpretability analysis for known driver genes.

Extracts routing weights for well-known cancer driver genes across different
cancer types to demonstrate biological interpretability of the routing mechanism.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path('/data/lgh/CANOPYNet-main')
sys.path.insert(0, str(ROOT / 'src'))

# Known driver genes with cancer-type associations
KNOWN_DRIVERS = {
    'TP53': ['BRCA', 'COAD', 'LIHC', 'LUAD', 'STAD', 'UCEC'],  # pan-cancer
    'KRAS': ['COAD', 'LUAD'],  # colon, lung
    'PIK3CA': ['BRCA', 'UCEC'],  # breast, endometrial
    'EGFR': ['LUAD'],  # lung
    'BRAF': ['COAD'],  # colon
    'APC': ['COAD'],  # colon
    'PTEN': ['BRCA', 'UCEC'],  # breast, endometrial
    'CTNNB1': ['LIHC'],  # liver
    'IDH1': ['LIHC'],  # liver
}

CHANNEL_NAMES = [
    'Graph Structure',
    'Raw Omics',
    'Pathogenicity',
    'Cancer Context',
    'Gene Sets',
    'Hypergraph Pathway'
]

def load_routing_weights(cancer, data_type='hetero'):
    """Load routing weights from saved TSV files."""
    pred_dir = ROOT / 'runs' / 'revision_v2' / 'fusion_ablation' / data_type / cancer / 'router'
    routing_file = pred_dir / 'routing_weights.tsv'
    
    if not routing_file.exists():
        print(f"Warning: {routing_file} not found")
        return None, None
    
    df = pd.read_csv(routing_file, sep='\t')
    gene_names = df['gene'].values
    # Extract weight columns (skip 'gene' column)
    routing_weights = df.iloc[:, 1:].values  # [N, 6]
    
    return gene_names, routing_weights


def analyze_driver_routing():
    """Analyze routing patterns for known driver genes."""
    results = []
    
    for gene, relevant_cancers in KNOWN_DRIVERS.items():
        for cancer in relevant_cancers:
            gene_names, routing_weights = load_routing_weights(cancer, 'hetero')
            if gene_names is None:
                continue
            
            # Find gene index
            gene_idx = None
            for i, gn in enumerate(gene_names):
                if gn == gene:
                    gene_idx = i
                    break
            
            if gene_idx is None:
                print(f"  {gene} not found in {cancer}")
                continue
            
            weights = routing_weights[gene_idx]
            
            results.append({
                'gene': gene,
                'cancer': cancer,
                **{CHANNEL_NAMES[i]: weights[i] for i in range(6)}
            })
            
            print(f"{gene} in {cancer}: " + 
                  ", ".join([f"{CHANNEL_NAMES[i]}={weights[i]:.3f}" for i in range(6)]))
    
    df = pd.DataFrame(results)
    return df


def plot_routing_heatmap(df, output_path):
    """Create heatmap of routing weights for known drivers."""
    # Pivot to gene x channel matrix, averaged across cancers
    pivot = df.groupby('gene')[CHANNEL_NAMES].mean()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', 
                vmin=0, vmax=0.4, cbar_kws={'label': 'Routing Weight'},
                ax=ax)
    ax.set_xlabel('Evidence Channel')
    ax.set_ylabel('Known Driver Gene')
    ax.set_title('Routing Weight Distribution for Known Cancer Drivers\n(Averaged Across Relevant Cancer Types)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to {output_path}")


def plot_cancer_specific_routing(df, output_path):
    """Plot cancer-specific routing patterns for TP53."""
    tp53_data = df[df['gene'] == 'TP53'].copy()
    if len(tp53_data) == 0:
        print("No TP53 data found")
        return
    
    tp53_data = tp53_data.set_index('cancer')[CHANNEL_NAMES]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    tp53_data.T.plot(kind='bar', ax=ax, width=0.8)
    ax.set_xlabel('Evidence Channel')
    ax.set_ylabel('Routing Weight')
    ax.set_title('Cancer-Type-Specific Routing Patterns for TP53')
    ax.legend(title='Cancer Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved TP53 cancer-specific plot to {output_path}")


def compute_channel_entropy(df):
    """Compute entropy of routing distribution per gene."""
    entropies = []
    for gene in df['gene'].unique():
        gene_data = df[df['gene'] == gene]
        avg_weights = gene_data[CHANNEL_NAMES].mean().values
        # Normalize to sum to 1
        avg_weights = avg_weights / (avg_weights.sum() + 1e-8)
        entropy = -np.sum(avg_weights * np.log(avg_weights + 1e-8))
        entropies.append({'gene': gene, 'entropy': entropy})
    
    ent_df = pd.DataFrame(entropies).sort_values('entropy')
    print("\nRouting Entropy (lower = more selective):")
    for _, row in ent_df.iterrows():
        print(f"  {row['gene']}: {row['entropy']:.3f}")
    
    return ent_df


def main():
    output_dir = ROOT / 'runs' / 'revision_v2' / 'interpretability'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Routing Weight Interpretability Analysis")
    print("="*60)
    
    print("\nExtracting routing weights for known driver genes...")
    df = analyze_driver_routing()
    
    if len(df) == 0:
        print("ERROR: No routing data found. Run fusion ablation experiments first.")
        return
    
    # Save raw data
    csv_path = output_dir / 'driver_routing_weights.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nSaved routing data to {csv_path}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_routing_heatmap(df, output_dir / 'routing_heatmap.png')
    plot_cancer_specific_routing(df, output_dir / 'tp53_cancer_specific.png')
    
    # Compute entropy
    ent_df = compute_channel_entropy(df)
    ent_df.to_csv(output_dir / 'routing_entropy.csv', index=False)
    
    # Generate summary statistics
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    print("\nAverage routing weights across all known drivers:")
    avg_weights = df[CHANNEL_NAMES].mean()
    for ch, w in avg_weights.items():
        print(f"  {ch}: {w:.3f}")
    
    print("\nMost selective genes (lowest entropy):")
    for _, row in ent_df.head(3).iterrows():
        gene_data = df[df['gene'] == row['gene']]
        top_channel = gene_data[CHANNEL_NAMES].mean().idxmax()
        top_weight = gene_data[CHANNEL_NAMES].mean().max()
        print(f"  {row['gene']}: entropy={row['entropy']:.3f}, "
              f"top channel={top_channel} ({top_weight:.3f})")
    
    print("\n" + "="*60)
    print(f"Results saved to {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
