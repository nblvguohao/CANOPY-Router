#!/usr/bin/env python3
"""
Statistical significance testing and druggable genome enrichment analysis.
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np
from scipy import stats
from scipy.stats import wilcoxon, bootstrap
import warnings
warnings.filterwarnings('ignore')

# Hardcoded gene lists
DRUGGABLE_KINASES = ['EGFR', 'ALK', 'ROS1', 'RET', 'MET', 'FGFR2', 'ERBB2', 'PIK3CA', 'AKT1', 'PTEN', 
                      'BRAF', 'KRAS', 'NRAS', 'MEK1', 'MEK2', 'CDK4', 'CDK6', 'CCND1', 'CCNE1', 'MDM2',
                      'ATM', 'ATR', 'CHEK1', 'CHEK2', 'WEE1', 'PLK1', 'AURKA', 'AURKB', 'CDK1', 'CDK2']

CGC_TIER1 = ['TP53', 'KRAS', 'BRAF', 'NRAS', 'PIK3CA', 'PTEN', 'APC', 'SMAD4', 'FBXW7', 'CTNNB1',
             'ARID1A', 'CDH1', 'EGFR', 'MET', 'ERBB2', 'ALK', 'ROS1', 'RET', 'FGFR2', 'VEGFA',
             'ATM', 'BRCA1', 'BRCA2', 'CHEK2', 'PALB2', 'RAD51', 'MLH1', 'MSH2', 'MSH6', 'PMS2']

def load_metrics_summary(run_dir):
    """Load metrics summary from run directory."""
    summary_file = run_dir / "metrics_summary.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            return json.load(f)
    return None

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

def compute_paired_comparison(sharp_runs, off_runs, metric='test_aupr'):
    """Compute paired statistical comparison between sharp and off runs."""
    # Collect paired metrics
    sharp_values = []
    off_values = []
    
    for (cancer, seed), sharp_dir in sharp_runs.items():
        off_dir = off_runs.get((cancer, seed))
        if off_dir is None:
            continue
        
        sharp_metrics = load_metrics_summary(sharp_dir)
        off_metrics = load_metrics_summary(off_dir)
        
        if sharp_metrics and off_metrics and metric in sharp_metrics and metric in off_metrics:
            sharp_values.append(sharp_metrics[metric])
            off_values.append(off_metrics[metric])
    
    if len(sharp_values) < 2:
        return None
    
    sharp_values = np.array(sharp_values)
    off_values = np.array(off_values)
    
    # Compute differences
    deltas = sharp_values - off_values
    
    # Sign test
    sign_test_p = stats.binom_test(sum(deltas > 0), len(deltas), p=0.5, alternative='greater')
    
    # Wilcoxon signed-rank test
    try:
        wilcoxon_stat, wilcoxon_p = wilcoxon(sharp_values, off_values, alternative='greater')
    except ValueError:
        wilcoxon_stat, wilcoxon_p = 0, 1.0
    
    # Bootstrap confidence interval for mean difference
    def stat_func(x):
        return np.mean(x)
    
    try:
        bootstrap_result = bootstrap((deltas,), stat_func, n_resamples=1000, 
                                   confidence_level=0.95, random_state=42)
        ci_lower, ci_upper = bootstrap_result.confidence_interval
    except:
        ci_lower, ci_upper = np.mean(deltas) - np.std(deltas), np.mean(deltas) + np.std(deltas)
    
    # Effect size (rank-biserial correlation)
    n = len(deltas)
    rank_biserial = 1 - (2 * wilcoxon_stat) / (n * (n + 1))
    
    return {
        'n_pairs': len(deltas),
        'sharp_mean': np.mean(sharp_values),
        'off_mean': np.mean(off_values),
        'mean_delta': np.mean(deltas),
        'std_delta': np.std(deltas),
        'sign_test_p': sign_test_p,
        'wilcoxon_stat': wilcoxon_stat,
        'wilcoxon_p': wilcoxon_p,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'rank_biserial': rank_biserial,
        'wins': sum(deltas > 0),
        'losses': sum(deltas < 0),
        'ties': sum(deltas == 0)
    }

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
    
    # Statistical significance (hypergeometric test)
    from scipy.stats import hypergeom
    N = len(predictions)  # Total number of genes
    K = len(target_set)  # Number of target genes
    n = top_k  # Number of draws
    k = overlap  # Number of successes
    
    if K > 0 and N > 0 and n > 0:
        p_value = hypergeom.sf(k - 1, N, K, n)
    else:
        p_value = 1.0
    
    return enrichment, p_value, top_genes

def analyze_druggable_enrichment(sharp_runs, off_runs):
    """Analyze druggable genome enrichment."""
    results = []
    
    for (cancer, seed), sharp_dir in sharp_runs.items():
        off_dir = off_runs.get((cancer, seed))
        if off_dir is None:
            continue
        
        # Load predictions
        sharp_preds = load_predictions(sharp_dir / "preds_all.tsv")
        off_preds = load_predictions(off_dir / "preds_all.tsv")
        
        if not sharp_preds or not off_preds:
            continue
        
        # Compute enrichment for sharp
        sharp_drug_enrich, sharp_drug_p, sharp_drug_top = compute_enrichment(sharp_preds, DRUGGABLE_KINASES)
        sharp_cgc_enrich, sharp_cgc_p, sharp_cgc_top = compute_enrichment(sharp_preds, CGC_TIER1)
        
        # Compute enrichment for off
        off_drug_enrich, off_drug_p, off_drug_top = compute_enrichment(off_preds, DRUGGABLE_KINASES)
        off_cgc_enrich, off_cgc_p, off_cgc_top = compute_enrichment(off_preds, CGC_TIER1)
        
        result = {
            'cancer': cancer,
            'seed': seed,
            'sharp_drug_enrichment': sharp_drug_enrich,
            'sharp_drug_p': sharp_drug_p,
            'sharp_cgc_enrichment': sharp_cgc_enrich,
            'sharp_cgc_p': sharp_cgc_p,
            'off_drug_enrichment': off_drug_enrich,
            'off_drug_p': off_drug_p,
            'off_cgc_enrichment': off_cgc_enrich,
            'off_cgc_p': off_cgc_p,
        }
        
        results.append(result)
    
    return results

def generate_latex_tables(comparisons, enrichment_results):
    """Generate LaTeX tables for results."""
    latex_lines = []
    
    # Statistical comparison table
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Statistical comparison between sharp-v2 and no-routing variants.}")
    latex_lines.append("\\begin{tabular}{lcccccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("Metric & Sharp Mean & Off Mean & Delta & 95\\% CI & Wilcoxon $p$ \\\\")
    latex_lines.append("\\midrule")
    
    for metric, comp in comparisons.items():
        if comp is None:
            continue
        
        latex_lines.append(f"{metric.upper()} & {comp['sharp_mean']:.4f} & {comp['off_mean']:.4f} & "
                          f"+{comp['mean_delta']:.4f} & [{comp['ci_lower']:.4f}, {comp['ci_upper']:.4f}] & "
                          f"{comp['wilcoxon_p']:.2e} \\\\")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    # Enrichment table
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Druggable genome enrichment analysis.}")
    latex_lines.append("\\begin{tabular}{lcccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("Cancer & Sharp Drug & Off Drug & Sharp CGC & Off CGC \\\\")
    latex_lines.append("\\midrule")
    
    for result in enrichment_results:
        cancer = result['cancer']
        latex_lines.append(f"{cancer} & {result['sharp_drug_enrichment']:.3f} & {result['off_drug_enrichment']:.3f} & "
                          f"{result['sharp_cgc_enrichment']:.3f} & {result['off_cgc_enrichment']:.3f} \\\\")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    return "\n".join(latex_lines)

def main():
    parser = argparse.ArgumentParser(description='Statistical significance and druggability analysis')
    
    parser.add_argument('--predictions_root', type=str, required=True, help='Root directory with predictions')
    parser.add_argument('--output_file', type=str, help='Output JSON file')
    parser.add_argument('--latex_file', type=str, help='Output LaTeX file')
    
    args = parser.parse_args()
    
    # Find all runs
    pred_root = Path(args.predictions_root)
    
    # Find sharp v2 runs
    sharp_runs = {}
    for sharp_dir in pred_root.glob(f"*/*_multiomics_seed*_router_on_sharp*"):
        # Extract cancer type and seed
        parts = sharp_dir.name.split('_')
        if len(parts) >= 4:
            cancer = parts[0]
            seed = int(parts[2].replace('seed', ''))
            sharp_runs[(cancer, seed)] = sharp_dir
    
    # Find no-routing runs
    off_runs = {}
    for off_dir in pred_root.glob(f"*/*_multiomics_seed*_router_off*"):
        # Extract cancer type and seed
        parts = off_dir.name.split('_')
        if len(parts) >= 4:
            cancer = parts[0]
            seed = int(parts[2].replace('seed', ''))
            off_runs[(cancer, seed)] = off_dir
    
    print(f"Found {len(sharp_runs)} sharp runs and {len(off_runs)} no-routing runs")
    
    # Paired comparisons
    metrics = ['test_auc', 'test_aupr', 'test_f1']
    comparisons = {}
    
    for metric in metrics:
        print(f"\nComputing paired comparison for {metric}...")
        comp = compute_paired_comparison(sharp_runs, off_runs, metric)
        comparisons[metric] = comp
        
        if comp:
            print(f"  Pairs: {comp['n_pairs']}")
            print(f"  Sharp mean: {comp['sharp_mean']:.4f}")
            print(f"  Off mean: {comp['off_mean']:.4f}")
            print(f"  Mean delta: +{comp['mean_delta']:.4f}")
            print(f"  95% CI: [{comp['ci_lower']:.4f}, {comp['ci_upper']:.4f}]")
            print(f"  Wilcoxon p: {comp['wilcoxon_p']:.2e}")
            print(f"  Wins/Losses: {comp['wins']}/{comp['losses']}")
    
    # Druggable enrichment analysis
    print(f"\nAnalyzing druggable genome enrichment...")
    enrichment_results = analyze_druggable_enrichment(sharp_runs, off_runs)
    
    # Summary statistics
    if enrichment_results:
        sharp_drug_mean = np.mean([r['sharp_drug_enrichment'] for r in enrichment_results])
        off_drug_mean = np.mean([r['off_drug_enrichment'] for r in enrichment_results])
        sharp_cgc_mean = np.mean([r['sharp_cgc_enrichment'] for r in enrichment_results])
        off_cgc_mean = np.mean([r['off_cgc_enrichment'] for r in enrichment_results])
        
        print(f"  Sharp drug enrichment: {sharp_drug_mean:.3f}")
        print(f"  Off drug enrichment: {off_drug_mean:.3f}")
        print(f"  Sharp CGC enrichment: {sharp_cgc_mean:.3f}")
        print(f"  Off CGC enrichment: {off_cgc_mean:.3f}")
    
    # Generate LaTeX tables
    latex_content = generate_latex_tables(comparisons, enrichment_results)
    
    # Save results
    if args.output_file:
        output_data = {
            'comparisons': {k: v.__dict__ if v is not None else None for k, v in comparisons.items()},
            'enrichment': enrichment_results,
            'latex_tables': latex_content
        }
        
        with open(args.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to {args.output_file}")
    
    if args.latex_file:
        with open(args.latex_file, 'w') as f:
            f.write(latex_content)
        
        print(f"LaTeX tables saved to {args.latex_file}")
    
    print(f"\n🎉 Statistical analysis completed!")

if __name__ == '__main__':
    main()
