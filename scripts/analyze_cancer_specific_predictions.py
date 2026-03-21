#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

INTOGEN = {
    'BRCA': ['TP53', 'PIK3CA', 'GATA3', 'CDH1', 'PTEN', 'AKT1', 'MAP3K1', 'CDKN2A', 'RB1', 'FOXA1'],
    'COAD': ['APC', 'TP53', 'KRAS', 'PIK3CA', 'SMAD4', 'FBXW7', 'TCF7L2', 'NRAS', 'BRAF', 'ACVR2A'],
    'LIHC': ['TP53', 'CTNNB1', 'AXIN1', 'TERT', 'ALB', 'ARID1A', 'RPS6KA3', 'VEGFA', 'MET', 'CCNE1'],
    'LUAD': ['TP53', 'KRAS', 'EGFR', 'STK11', 'KEAP1', 'NF1', 'BRAF', 'PIK3CA', 'MET', 'RB1'],
    'STAD': ['TP53', 'CDH1', 'ARID1A', 'PIK3CA', 'RHOA', 'KRAS', 'SMAD4', 'CTNNB1', 'RNF43', 'ERBB2'],
    'UCEC': ['PTEN', 'PIK3CA', 'TP53', 'FBXW7', 'ARID1A', 'KRAS', 'CTNNB1', 'PPP2R1A', 'SPOP']
}

ONCOKB = {
    'BRCA': ['TP53', 'PIK3CA', 'ERBB2', 'AKT1', 'PTEN', 'CDH1', 'GATA3', 'FOXA1', 'ESR1', 'BRCA1'],
    'COAD': ['KRAS', 'BRAF', 'NRAS', 'PIK3CA', 'TP53', 'SMAD4', 'APC', 'FBXW7', 'ACVR2A', 'MSI2'],
    'LIHC': ['TP53', 'CTNNB1', 'TERT', 'AXIN1', 'ARID1A', 'RPS6KA3', 'VEGFA', 'MET', 'FGF19', 'CCND1'],
    'LUAD': ['EGFR', 'KRAS', 'ALK', 'ROS1', 'BRAF', 'MET', 'RET', 'HER2', 'TP53', 'STK11'],
    'STAD': ['ERBB2', 'FGFR2', 'MET', 'KRAS', 'PIK3CA', 'TP53', 'CDH1', 'ARID1A', 'RHOA', 'SMAD4'],
    'UCEC': ['PTEN', 'PIK3CA', 'TP53', 'FBXW7', 'ARID1A', 'KRAS', 'CTNNB1', 'PPP2R1A', 'SPOP', 'MSH2']
}

DRUG = {'EGFR', 'ALK', 'ROS1', 'RET', 'MET', 'FGFR2', 'ERBB2', 'PIK3CA', 'AKT1', 'PTEN', 'BRAF', 'KRAS', 'NRAS', 'MAP2K1', 'MAP2K2', 'CDK4', 'CDK6', 'CCND1', 'CCNE1', 'MDM2', 'PARP1', 'SRC', 'RAF1', 'PIK3CD'}
CGC = {'TP53', 'KRAS', 'BRAF', 'NRAS', 'PIK3CA', 'PTEN', 'APC', 'SMAD4', 'FBXW7', 'CTNNB1', 'ARID1A', 'CDH1', 'EGFR', 'MET', 'ERBB2', 'ALK', 'ROS1', 'RET', 'FGFR2', 'VEGFA', 'BRCA1', 'BRCA2', 'ATM', 'MLH1', 'MSH2', 'MSH6', 'PMS2', 'RAC1'}


def parse_gene_name(item):
    if isinstance(item, np.ndarray) or isinstance(item, (list, tuple)):
        item = item[-1]
    return item.decode() if isinstance(item, bytes) else str(item)


def load_train_pos(h5_path):
    with h5py.File(h5_path, 'r') as f:
        genes = [parse_gene_name(x) for x in f['gene_names'][:]]
        train_mask = np.asarray(f['mask_train'][:]).astype(bool)
        val_mask = np.asarray(f['mask_val'][:]).astype(bool)
        y_train = np.asarray(f['y_train'][:]).reshape(-1).astype(bool)
        y_val = np.asarray(f['y_val'][:]).reshape(-1).astype(bool)
    seen = set()
    for i, g in enumerate(genes):
        if (train_mask[i] and y_train[i]) or (val_mask[i] and y_val[i]):
            seen.add(g)
    return genes, seen


def load_preds(path):
    preds = {}
    with open(path) as f:
        next(f, None)
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) == 2:
                try:
                    preds[parts[0]] = float(parts[1])
                except ValueError:
                    pass
    return preds


def load_top_source(path):
    rows = {}
    with open(path) as f:
        next(f, None)
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) >= 3:
                try:
                    rows[parts[0]] = {'source': parts[1], 'weight': float(parts[2])}
                except ValueError:
                    rows[parts[0]] = {'source': parts[1], 'weight': None}
    return rows


def load_entropy(path):
    rows = {}
    with open(path) as f:
        next(f, None)
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) >= 2:
                try:
                    rows[parts[0]] = float(parts[1])
                except ValueError:
                    pass
    return rows


def summarize_cancer(cancer, run_dir, h5_root):
    h5_path = Path(h5_root) / f'{cancer}_multiomics.h5'
    preds = load_preds(run_dir / 'preds_all.tsv')
    top_source = load_top_source(run_dir / 'routing_top_source.tsv')
    entropy = load_entropy(run_dir / 'routing_entropy.tsv')
    genes, seen = load_train_pos(h5_path)

    ranked = sorted(preds.items(), key=lambda kv: kv[1], reverse=True)
    novel_ranked = [(g, s) for g, s in ranked if g not in seen]
    top_k = max(20, int(len(ranked) * 0.02))
    top_ranked = ranked[:top_k]
    top_genes = [g for g, _ in top_ranked]

    enrich = {
        'top_k': top_k,
        'intogen_rate': len(set(top_genes) & set(INTOGEN.get(cancer, []))) / top_k,
        'oncokb_rate': len(set(top_genes) & set(ONCOKB.get(cancer, []))) / top_k,
        'drug_rate': len(set(top_genes) & DRUG) / top_k,
        'cgc_rate': len(set(top_genes) & CGC) / top_k,
    }

    case_rows = []
    for gene, score in novel_ranked[:50]:
        src = top_source.get(gene, {})
        case_rows.append({
            'gene': gene,
            'score': score,
            'source': src.get('source', 'NA'),
            'weight': src.get('weight'),
            'entropy': entropy.get(gene),
            'intogen': gene in INTOGEN.get(cancer, []),
            'oncokb': gene in ONCOKB.get(cancer, []),
            'drug': gene in DRUG,
            'cgc': gene in CGC,
        })

    supported_first = sorted(
        case_rows,
        key=lambda r: (
            -(int(r['intogen']) + int(r['oncokb']) + int(r['drug']) + int(r['cgc'])),
            -r['score']
        )
    )

    return {
        'cancer': cancer,
        'n_genes': len(genes),
        'n_seen_train_val': len(seen),
        'n_novel': len(novel_ranked),
        'enrichment': enrich,
        'novel_only_top20': supported_first[:20],
        'case_study_top5': supported_first[:5],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs_root', required=True)
    parser.add_argument('--h5_root', required=True)
    parser.add_argument('--output_json', required=True)
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    out = {'runs_root': str(runs_root), 'results': []}

    for run_dir in sorted(runs_root.glob('*_multiomics_seed1234_*')):
        cancer = run_dir.name.split('_')[0]
        required = ['preds_all.tsv', 'routing_top_source.tsv', 'routing_entropy.tsv']
        if not all((run_dir / x).exists() for x in required):
            continue
        out['results'].append(summarize_cancer(cancer, run_dir, args.h5_root))

    if out['results']:
        top_case = []
        for rec in out['results']:
            top_case.extend([dict(cancer=rec['cancer'], **row) for row in rec['case_study_top5']])
        out['global_case_studies'] = sorted(
            top_case,
            key=lambda r: (
                -(int(r['intogen']) + int(r['oncokb']) + int(r['drug']) + int(r['cgc'])),
                -r['score']
            )
        )[:12]

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(out, f, indent=2)

    print(json.dumps({'n_results': len(out['results']), 'output_json': args.output_json}, indent=2))


if __name__ == '__main__':
    main()
