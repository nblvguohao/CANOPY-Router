#!/usr/bin/env python3
"""
Validate CANOPY-Router data files.
"""

import argparse
import numpy as np
import torch
import pandas as pd
from pathlib import Path
import sys

def validate_gene_sets(c2_path, c5_path, gene_list_path):
    """Validate gene set files."""
    print("🔍 验证基因集文件...")
    
    results = {}
    
    # Validate C2 gene sets
    try:
        c2_data = np.load(c2_path)
        results['c2'] = {
            'shape': c2_data['matrix'].shape,
            'n_genes': len(c2_data['gene_names']),
            'n_gene_sets': len(c2_data['gene_set_names']),
            'sparsity': (c2_data['matrix'] == 0).sum() / c2_data['matrix'].size,
            'sample_gene_sets': c2_data['gene_set_names'][:5].tolist()
        }
        print(f"✅ C2基因集: {results['c2']['shape']}")
        print(f"   - 基因数: {results['c2']['n_genes']}")
        print(f"   - 基因集数: {results['c2']['n_gene_sets']}")
        print(f"   - 稀疏度: {results['c2']['sparsity']:.2%}")
    except Exception as e:
        print(f"❌ C2基因集验证失败: {e}")
        results['c2'] = {'error': str(e)}
    
    # Validate C5 gene sets
    try:
        c5_data = np.load(c5_path)
        results['c5'] = {
            'shape': c5_data['matrix'].shape,
            'n_genes': len(c5_data['gene_names']),
            'n_gene_sets': len(c5_data['gene_set_names']),
            'sparsity': (c5_data['matrix'] == 0).sum() / c5_data['matrix'].size,
            'sample_gene_sets': c5_data['gene_set_names'][:5].tolist()
        }
        print(f"✅ C5基因集: {results['c5']['shape']}")
        print(f"   - 基因数: {results['c5']['n_genes']}")
        print(f"   - GO集合数: {results['c5']['n_gene_sets']}")
        print(f"   - 稀疏度: {results['c5']['sparsity']:.2%}")
    except Exception as e:
        print(f"❌ C5基因集验证失败: {e}")
        results['c5'] = {'error': str(e)}
    
    # Validate gene list
    try:
        gene_df = pd.read_csv(gene_list_path)
        results['gene_list'] = {
            'shape': gene_df.shape,
            'columns': gene_df.columns.tolist(),
            'sample_genes': gene_df['gene_symbol'].head(10).tolist()
        }
        print(f"✅ 基因列表: {results['gene_list']['shape']}")
        print(f"   - 列: {results['gene_list']['columns']}")
        print(f"   - 前10个基因: {results['gene_list']['sample_genes']}")
    except Exception as e:
        print(f"❌ 基因列表验证失败: {e}")
        results['gene_list'] = {'error': str(e)}
    
    return results

def validate_alphamissense(alpha_path):
    """Validate AlphaMissense scores."""
    print("\n🔍 验证AlphaMissense分数...")
    
    try:
        alpha_scores = torch.load(alpha_path)
        results = {
            'shape': alpha_scores.shape,
            'dtype': str(alpha_scores.dtype),
            'mean': float(alpha_scores.mean()),
            'std': float(alpha_scores.std()),
            'min': float(alpha_scores.min()),
            'max': float(alpha_scores.max()),
            'sample_scores': alpha_scores[:10].tolist()
        }
        
        print(f"✅ AlphaMissense分数: {results['shape']}")
        print(f"   - 数据类型: {results['dtype']}")
        print(f"   - 均值±标准差: {results['mean']:.4f}±{results['std']:.4f}")
        print(f"   - 范围: [{results['min']:.4f}, {results['max']:.4f}]")
        print(f"   - 前10个分数: {[f'{s:.3f}' for s in results['sample_scores']]}")
        
        return results
    except Exception as e:
        print(f"❌ AlphaMissense验证失败: {e}")
        return {'error': str(e)}

def check_file_sizes():
    """Check file sizes."""
    print("\n📊 文件大小信息:")
    
    files = [
        'data/msigdb/c2_GenesetsMatrix.npz',
        'data/msigdb/c5_GenesetsMatrix.npz',
        'data/msigdb/geneList.csv',
        'inputs/alphamissense_scores.pt'
    ]
    
    size_info = {}
    for file_path in files:
        if Path(file_path).exists():
            size_mb = Path(file_path).stat().st_size / (1024 * 1024)
            size_info[file_path] = f"{size_mb:.2f} MB"
            print(f"  - {file_path}: {size_mb:.2f} MB")
        else:
            size_info[file_path] = "Missing"
            print(f"  - {file_path}: ❌ 缺失")
    
    return size_info

def main():
    parser = argparse.ArgumentParser(description='Validate CANOPY-Router data files')
    
    parser.add_argument('--c2_path', type=str, default='data/msigdb/c2_GenesetsMatrix.npz')
    parser.add_argument('--c5_path', type=str, default='data/msigdb/c5_GenesetsMatrix.npz')
    parser.add_argument('--gene_list_path', type=str, default='data/msigdb/geneList.csv')
    parser.add_argument('--alpha_path', type=str, default='inputs/alphamissense_scores.pt')
    parser.add_argument('--output_file', type=str, help='Output JSON file')
    
    args = parser.parse_args()
    
    print("🧪 CANOPY-Router 数据文件验证")
    print("=" * 50)
    
    # Validate all files
    validation_results = {}
    
    # Check if all files exist
    missing_files = []
    for file_path in [args.c2_path, args.c5_path, args.gene_list_path, args.alpha_path]:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ 以下文件缺失: {missing_files}")
        return
    
    # Validate gene sets
    validation_results['gene_sets'] = validate_gene_sets(
        args.c2_path, args.c5_path, args.gene_list_path
    )
    
    # Validate AlphaMissense
    validation_results['alphamissense'] = validate_alphamissense(args.alpha_path)
    
    # Check file sizes
    validation_results['file_sizes'] = check_file_sizes()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 验证总结:")
    
    all_valid = True
    for category, results in validation_results.items():
        if category == 'file_sizes':
            continue
        
        if isinstance(results, dict) and 'error' in results:
            print(f"❌ {category}: {results['error']}")
            all_valid = False
        else:
            print(f"✅ {category}: 验证通过")
    
    if all_valid:
        print("\n🎉 所有数据文件验证通过！")
        print("🚀 现在可以运行CANOPY-Router实验了！")
    else:
        print("\n⚠️  部分文件验证失败，请检查上述错误信息。")
    
    # Save results
    if args.output_file:
        import json
        with open(args.output_file, 'w') as f:
            json.dump(validation_results, f, indent=2)
        print(f"\n💾 验证结果已保存到: {args.output_file}")

if __name__ == '__main__':
    main()
