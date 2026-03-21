#!/usr/bin/env python3
"""
External validation against IntOGen and OncoKB driver gene catalogs.
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

# Hardcoded external gene lists
INTOGEN = {
    'BRCA': ['TP53', 'PIK3CA', 'GATA3', 'CDH1', 'PTEN', 'AKT1', 'MAP3K1', 'CDKN2A', 'RB1', 'FOXA1'],
    'COAD': ['APC', 'TP53', 'KRAS', 'PIK3CA', 'SMAD4', 'FBXW7', 'TCF7L2', 'NRAS', 'BRAF', 'ACVR2A'],
    'LIHC': ['TP53', 'CTNNB1', 'AXIN1', 'TERT', 'ALB', 'ARID1A', 'RPS6KA3', 'VEGFA', 'MET', 'CCNE1'],
    'LUAD': ['TP53', 'KRAS', 'EGFR', 'STK11', 'KEAP1', 'NF1', 'BRAF', 'PIK3CA', 'MET', 'RB1'],
    'STAD': ['TP53', 'CDH1', 'ARID1A', 'PIK3CA', 'RHOA', 'KRAS', 'SMAD4', 'CTNNB1', 'RNF43', 'ERBB2'],
    'UCEC': ['PTEN', 'PIK3CA', 'TP53', 'FBXW7', 'ARID1A', 'KRAS', 'CTNNB1', 'PPP2R1A', 'TP53', 'SPOP']
}

ONCOKB = {
    'BRCA': ['TP53', 'PIK3CA', 'ERBB2', 'AKT1', 'PTEN', 'CDH1', 'GATA3', 'FOXA1', 'ESR1', 'BRCA1'],
    'COAD': ['KRAS', 'BRAF', 'NRAS', 'PIK3CA', 'TP53', 'SMAD4', 'APC', 'FBXW7', 'ACVR2A', 'MSI2'],
    'LIHC': ['TP53', 'CTNNB1', 'TERT', 'AXIN1', 'ARID1A', 'RPS6KA3', 'VEGFA', 'MET', 'FGF19', 'CCND1'],
    'LUAD': ['EGFR', 'KRAS', 'ALK', 'ROS1', 'BRAF', 'MET', 'RET', 'HER2', 'TP53', 'STK11'],
    'STAD': ['ERBB2', 'FGFR2', 'MET', 'KRAS', 'PIK3CA', 'TP53', 'CDH1', 'ARID1A', 'RHOA', 'SMAD4'],
    'UCEC': ['PTEN', 'PIK3CA', 'TP53', 'FBXW7', 'ARID1A', 'KRAS', 'CTNNB1', 'PPP2R1A', 'SPOP', 'MSH2']
}

# Druggable and CGC gene lists
DRUGGABLE = ['EGFR', 'ALK', 'ROS1', 'RET', 'MET', 'FGFR2', 'ERBB2', 'PIK3CA', 'AKT1', 'PTEN', 
             'BRAF', 'KRAS', 'NRAS', 'MEK1', 'MEK2', 'CDK4', 'CDK6', 'CCND1', 'CCNE1', 'MDM2']

CGC = ['TP53', 'KRAS', 'BRAF', 'NRAS', 'PIK3CA', 'PTEN', 'APC', 'SMAD4', 'FBXW7', 'CTNNB1',
        'ARID1A', 'CDH1', 'EGFR', 'MET', 'ERBB2', 'ALK', 'ROS1', 'RET', 'FGFR2', 'VEGFA']

def load_predictions(pred_path):
    """Load predictions from TSV file."""
    predictions = {}
    with open(pred_path, 'r') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                gene, score = parts
                try:
                    predictions[gene] = float(score)
                except ValueError:
                    continue
    return predictions

def load_training_genes(h5_path):
    """Load training positive genes from H5 file."""
    try:
        import h5py
        with h5py.File(h5_path, 'r') as f:
            train_mask = f['mask_train'][:].astype(bool)
            val_mask = f['mask_val'][:].astype(bool)
            y_train = f['y_train'][:].reshape(-1)
            y_val = f['y_val'][:].reshape(-1)
            
            # Get gene names
            if 'gene_names' in f:
                gene_names = [name.decode() if isinstance(name, bytes) else str(name) 
                            for name in f['gene_names'][:]]
            else:
                gene_names = [f"gene_{i}" for i in range(len(y_train))]
            
            # Get training positive genes
            train_pos_mask = train_mask & (y_train == 1)
            val_pos_mask = val_mask & (y_val == 1)
            train_pos_genes = set()
            
            for i, is_pos in enumerate(train_pos_mask):
                if is_pos:
                    train_pos_genes.add(gene_names[i])
            
            for i, is_pos in enumerate(val_pos_mask):
                if is_pos:
                    train_pos_genes.add(gene_names[i])
            
            return gene_names, train_pos_genes
    except Exception as e:
        print(f"Warning: Could not load training genes from {h5_path}: {e}")
        return [], set()

def compute_auc_aupr(labels, scores):
    """Compute AUC and AUPR."""
    if len(np.unique(labels)) < 2:
        return None, None
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))

def evaluate_external(predictions, external_genes, train_pos_genes):
    """Evaluate predictions against external gene set."""
    # Align predictions with external labels
    aligned_genes = []
    aligned_scores = []
    aligned_labels = []
    
    for gene, score in predictions.items():
        if gene in external_genes and gene not in train_pos_genes:
            aligned_genes.append(gene)
            aligned_scores.append(score)
            aligned_labels.append(1)  # External positives
    
    # Add some negatives (genes not in external set)
    for gene, score in predictions.items():
        if gene not in external_genes and gene not in train_pos_genes and len(aligned_labels) < 1000:
            aligned_genes.append(gene)
            aligned_scores.append(score)
            aligned_labels.append(0)  # Negatives
    
    if len(np.unique(aligned_labels)) < 2:
        return None, None, []
    
    auc, aupr = compute_auc_aupr(aligned_labels, aligned_scores)
    
    return auc, aupr, aligned_genes

def evaluate_novel_only(predictions, external_genes, train_pos_genes):
    """Evaluate on novel genes only (exclude training positives)."""
    novel_predictions = {}
    for gene, score in predictions.items():
        if gene not in train_pos_genes:
            novel_predictions[gene] = score
    
    return evaluate_external(novel_predictions, external_genes, train_pos_genes)

def compute_enrichment(predictions, target_genes, top_k=100):
    """Compute enrichment of target genes in top-k predictions."""
    # Sort predictions by score
    sorted_genes = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    
    # Get top-k genes
    top_genes = [gene for gene, score in sorted_genes[:top_k]]
    top_set = set(top_genes)
    
    # Compute enrichment
    target_set = set(target_genes)
    overlap = len(top_set & target_set)
    enrichment = overlap / top_k
    
    return enrichment, top_genes

def main():
    parser = argparse.ArgumentParser(description='External validation against IntOGen and OncoKB')
    
    parser.add_argument('--predictions_root', type=str, required=True, help='Root directory with predictions')
    parser.add_argument('--tree_data_root', type=str, required=True, help='Root directory for TREE data')
    parser.add_argument('--output_file', type=str, help='Output JSON file')
    
    args = parser.parse_args()
    
    # Cancer types
    cancer_types = ['BRCA', 'COAD', 'LIHC', 'LUAD', 'STAD', 'UCEC']
    
    # Results storage
    results = []
    
    print("Performing external validation...")
    
    for cancer_type in cancer_types:
        print(f"\nProcessing {cancer_type}...")
        
        # Find prediction directories
        pred_root = Path(args.predictions_root)
        
        # Look for sharp v2 and no-routing runs
        sharp_dirs = list(pred_root.glob(f"*/{cancer_type}_multiomics_seed*_router_on_sharp*"))
        off_dirs = list(pred_root.glob(f"*/{cancer_type}_multiomics_seed*_router_off*"))
        
        if not sharp_dirs or not off_dirs:
            print(f"Warning: Missing runs for {cancer_type}")
            continue
        
        # Get training genes
        h5_path = f"{args.tree_data_root}/{cancer_type}_multiomics.h5"
        gene_names, train_pos_genes = load_training_genes(h5_path)
        
        # Process each seed
        for seed in [1234, 2345, 3456]:
            # Find corresponding runs
            sharp_run = None
            off_run = None
            
            for sharp_dir in sharp_dirs:
                if f"seed{seed}" in sharp_dir.name:
                    sharp_run = sharp_dir
                    break
            
            for off_dir in off_dirs:
                if f"seed{seed}" in off_dir.name:
                    off_run = off_dir
                    break
            
            if sharp_run is None or off_run is None:
                print(f"Warning: Missing seed {seed} runs for {cancer_type}")
                continue
            
            # Load predictions
            sharp_preds = load_predictions(sharp_run / "preds_all.tsv")
            off_preds = load_predictions(off_run / "preds_all.tsv")
            
            if not sharp_preds or not off_preds:
                print(f"Warning: Could not load predictions for {cancer_type} seed {seed}")
                continue
            
            # External validation
            intogen_genes = INTOGEN.get(cancer_type, [])
            oncokb_genes = ONCOKB.get(cancer_type, [])
            
            # Full evaluation
            sharp_intogen_auc, sharp_intogen_aupr, _ = evaluate_external(sharp_preds, intogen_genes, train_pos_genes)
            sharp_oncokb_auc, sharp_oncokb_aupr, _ = evaluate_external(sharp_preds, oncokb_genes, train_pos_genes)
            
            off_intogen_auc, off_intogen_aupr, _ = evaluate_external(off_preds, intogen_genes, train_pos_genes)
            off_oncokb_auc, off_oncokb_aupr, _ = evaluate_external(off_preds, oncokb_genes, train_pos_genes)
            
            # Novel-only evaluation
            sharp_novel_auc, sharp_novel_aupr, _ = evaluate_novel_only(sharp_preds, intogen_genes, train_pos_genes)
            sharp_novel_oncokb_auc, sharp_novel_oncokb_aupr, _ = evaluate_novel_only(sharp_preds, oncokb_genes, train_pos_genes)
            
            off_novel_auc, off_novel_aupr, _ = evaluate_novel_only(off_preds, intogen_genes, train_pos_genes)
            off_novel_oncokb_auc, off_novel_oncokb_aupr, _ = evaluate_novel_only(off_preds, oncokb_genes, train_pos_genes)
            
            # Enrichment analysis
            top_k = max(20, int(len(sharp_preds) * 0.02))
            
            sharp_drug_rate, _ = compute_enrichment(sharp_preds, DRUGGABLE, top_k)
            sharp_cgc_rate, _ = compute_enrichment(sharp_preds, CGC, top_k)
            sharp_oncokb_rate, _ = compute_enrichment(sharp_preds, oncokb_genes, top_k)
            
            off_drug_rate, _ = compute_enrichment(off_preds, DRUGGABLE, top_k)
            off_cgc_rate, _ = compute_enrichment(off_preds, CGC, top_k)
            off_oncokb_rate, _ = compute_enrichment(off_preds, oncokb_genes, top_k)
            
            # Store results
            result = {
                'cancer': cancer_type,
                'seed': seed,
                'method': 'sharp_v2',
                'intogen_auc': sharp_intogen_auc,
                'intogen_aupr': sharp_intogen_aupr,
                'novel_intogen_auc': sharp_novel_auc,
                'novel_intogen_aupr': sharp_novel_aupr,
                'oncokb_auc': sharp_oncokb_auc,
                'oncokb_aupr': sharp_oncokb_aupr,
                'novel_oncokb_auc': sharp_novel_oncokb_auc,
                'novel_oncokb_aupr': sharp_novel_oncokb_aupr,
                'top_drug_rate': sharp_drug_rate,
                'top_cgc_rate': sharp_cgc_rate,
                'top_oncokb_rate': sharp_oncokb_rate,
            }
            results.append(result)
            
            result = {
                'cancer': cancer_type,
                'seed': seed,
                'method': 'no_routing',
                'intogen_auc': off_intogen_auc,
                'intogen_aupr': off_intogen_aupr,
                'novel_intogen_auc': off_novel_auc,
                'novel_intogen_aupr': off_novel_aupr,
                'oncokb_auc': off_oncokb_auc,
                'oncokb_aupr': off_oncokb_aupr,
                'novel_oncokb_auc': off_novel_oncokb_auc,
                'novel_oncokb_aupr': off_novel_oncokb_aupr,
                'top_drug_rate': off_drug_rate,
                'top_cgc_rate': off_cgc_rate,
                'top_oncokb_rate': off_oncokb_rate,
            }
            results.append(result)
    
    # Compute macro averages
    sharp_results = [r for r in results if r['method'] == 'sharp_v2']
    off_results = [r for r in results if r['method'] == 'no_routing']
    
    if sharp_results and off_results:
        macro_sharp = {
            'intogen_auc': np.mean([r['intogen_auc'] for r in sharp_results if r['intogen_auc'] is not None]),
            'intogen_aupr': np.mean([r['intogen_aupr'] for r in sharp_results if r['intogen_aupr'] is not None]),
            'oncokb_auc': np.mean([r['oncokb_auc'] for r in sharp_results if r['oncokb_auc'] is not None]),
            'oncokb_aupr': np.mean([r['oncokb_aupr'] for r in sharp_results if r['oncokb_aupr'] is not None]),
        }
        
        macro_off = {
            'intogen_auc': np.mean([r['intogen_auc'] for r in off_results if r['intogen_auc'] is not None]),
            'intogen_aupr': np.mean([r['intogen_aupr'] for r in off_results if r['intogen_aupr'] is not None]),
            'oncokb_auc': np.mean([r['oncokb_auc'] for r in off_results if r['oncokb_auc'] is not None]),
            'oncokb_aupr': np.mean([r['oncokb_aupr'] for r in off_results if r['oncokb_aupr'] is not None]),
        }
        
        print(f"\n{'='*60}")
        print(f"MACRO AVERAGED RESULTS")
        print(f"{'='*60}")
        print(f"Sharp v2:")
        print(f"  IntOGen AUC: {macro_sharp['intogen_auc']:.4f}")
        print(f"  IntOGen AUPR: {macro_sharp['intogen_aupr']:.4f}")
        print(f"  OncoKB AUC: {macro_sharp['oncokb_auc']:.4f}")
        print(f"  OncoKB AUPR: {macro_sharp['oncokb_aupr']:.4f}")
        print(f"No-routing:")
        print(f"  IntOGen AUC: {macro_off['intogen_auc']:.4f}")
        print(f"  IntOGen AUPR: {macro_off['intogen_aupr']:.4f}")
        print(f"  OncoKB AUC: {macro_off['oncokb_auc']:.4f}")
        print(f"  OncoKB AUPR: {macro_off['oncokb_aupr']:.4f}")
    
    # Save results
    if args.output_file:
        output_data = {
            'results': results,
            'macro_sharp': macro_sharp if sharp_results else {},
            'macro_off': macro_off if off_results else {},
        }
        
        with open(args.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to {args.output_file}")
    
    print(f"\n🎉 External validation completed!")

if __name__ == '__main__':
    main()
