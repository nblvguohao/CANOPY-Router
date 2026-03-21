#!/usr/bin/env python3
"""
Run pan-cancer experiments for all networks and seeds.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def run_experiment(network, seed, args):
    """Run single experiment."""
    # Construct H5 file path
    h5_file = f"{args.tree_data_root}/{network}_multiomics.h5"
    
    if not os.path.exists(h5_file):
        print(f"Warning: H5 file not found: {h5_file}")
        return None
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{network}_multiomics_seed{seed}_{args.tag}"
    output_dir = f"{args.out_root}/{timestamp}/{run_name}"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct command
    cmd = [
        sys.executable, "src/run.py",
        "--tree_h5", h5_file,
        "--output_dir", output_dir,
        "--run_name", run_name,
        "--device", str(args.device),
        "--seed", str(seed),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        "--hidden_dim", str(args.hidden_dim),
        "--num_heads", str(args.num_heads),
        "--num_layers", str(args.num_layers),
        "--top_k", str(args.top_k),
        "--mvga_max_edges", str(args.mvga_max_edges),
        "--temperature", str(args.temperature),
        "--entropy_reg", str(args.entropy_reg),
        "--chunk_rows", str(args.chunk_rows),
        "--amp", args.amp,
        "--tree_model", args.tree_model,
        "--batch_size", str(args.batch_size)
    ]
    
    # Add feature flags
    if args.use_raw_features:
        cmd.append("--use_raw_features")
    
    if args.disable_evidence_routing:
        cmd.append("--disable_evidence_routing")
    
    # Add module flags
    if args.enable_supcl:
        cmd.append("--enable_supcl")
    
    if args.enable_ctam:
        cmd.append("--enable_ctam")
    
    if args.enable_gske:
        cmd.append("--enable_gske")
        cmd.extend(["--gske_npz", args.gske_npz])
        cmd.extend(["--gske_c5_npz", args.gske_c5_npz])
        cmd.extend(["--gske_gene_list", args.gske_gene_list])
    
    if args.enable_mvga:
        cmd.append("--enable_mvga")
    
    if args.enable_pgem:
        cmd.append("--enable_pgem")
        cmd.extend(["--alphamissense_path", args.alphamissense_path])
    
    if args.enable_hyperconv:
        cmd.append("--enable_hyperconv")
        cmd.extend(["--hyperconv_dim", str(args.hyperconv_dim)])
    
    if args.enable_adaptive_patho:
        cmd.append("--enable_adaptive_patho")
    
    if args.save_gate_stats:
        cmd.append("--save_gate_stats")
    
    print(f"Running: {network} seed {seed}")
    print(f"Command: {' '.join(cmd)}")
    
    # Run command
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
        
        if result.returncode == 0:
            print(f"✅ {network} seed {seed} completed successfully")
            return output_dir
        else:
            print(f"❌ {network} seed {seed} failed with return code {result.returncode}")
            print(f"Error: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {network} seed {seed} timed out after {args.timeout} seconds")
        return None
    except Exception as e:
        print(f"💥 {network} seed {seed} crashed with exception: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Run pan-cancer experiments')
    
    # Data arguments
    parser.add_argument('--tree_data_root', type=str, required=True, help='Root directory for TREE data')
    parser.add_argument('--out_root', type=str, required=True, help='Root output directory')
    
    # Networks and seeds
    networks = ['CPDB', 'IREF', 'MULTINET', 'PCNET', 'STRINGdb']
    parser.add_argument('--networks', type=str, nargs='+', default=networks, help='Networks to run')
    parser.add_argument('--seeds', type=str, default='1234,2345,3456', help='Comma-separated seeds')
    
    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of transformer layers')
    parser.add_argument('--top_k', type=int, default=256, help='Top-k for graph construction')
    parser.add_argument('--mvga_max_edges', type=int, default=5000000, help='Maximum edges for MVGA')
    parser.add_argument('--temperature', type=float, default=1.0, help='Routing temperature')
    parser.add_argument('--entropy_reg', type=float, default=0.01, help='Routing entropy regularization')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--chunk_rows', type=int, default=256, help='Chunk size for large graphs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--timeout', type=int, default=3600*4, help='Timeout per experiment (seconds)')
    
    # Hardware arguments
    parser.add_argument('--device', type=int, default=0, help='GPU device')
    parser.add_argument('--amp', type=str, default='bf16', choices=['none', 'fp16', 'bf16'], help='Mixed precision')
    parser.add_argument('--max_parallel', type=int, default=1, help='Maximum parallel experiments')
    
    # Feature flags
    parser.add_argument('--tree_model', type=str, default='treenet', choices=['treenet', 'gcn', 'gat'], help='Base model')
    parser.add_argument('--use_raw_features', action='store_true', help='Use raw multi-omics features')
    parser.add_argument('--disable_evidence_routing', action='store_true', help='Disable evidence routing')
    
    # Module flags
    parser.add_argument('--enable_supcl', action='store_true', help='Enable supervised contrastive learning')
    parser.add_argument('--enable_ctam', action='store_true', help='Enable cancer-type aware masking')
    parser.add_argument('--enable_gske', action='store_true', help='Enable gene-set knowledge embedding')
    parser.add_argument('--gske_npz', type=str, help='Path to gene-set NPZ file')
    parser.add_argument('--gske_c5_npz', type=str, help='Path to C5 gene-set NPZ file')
    parser.add_argument('--gske_gene_list', type=str, help='Path to gene list file')
    parser.add_argument('--enable_mvga', action='store_true', help='Enable multi-view graph attention')
    parser.add_argument('--enable_pgem', action='store_true', help='Enable pathogenicity-gated embedding')
    parser.add_argument('--alphamissense_path', type=str, help='Path to AlphaMissense scores')
    parser.add_argument('--enable_hyperconv', action='store_true', help='Enable hypergraph convolution')
    parser.add_argument('--hyperconv_dim', type=int, default=32, help='Hypergraph convolution dimension')
    parser.add_argument('--enable_adaptive_patho', action='store_true', help='Enable adaptive pathogenicity')
    parser.add_argument('--save_gate_stats', action='store_true', help='Save gate statistics')
    
    # Other arguments
    parser.add_argument('--tag', type=str, default='router_pc_run', help='Tag for run identification')
    
    args = parser.parse_args()
    
    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    
    # Create output directory
    os.makedirs(args.out_root, exist_ok=True)
    
    print(f"Starting pan-cancer experiments...")
    print(f"Networks: {args.networks}")
    print(f"Seeds: {seeds}")
    print(f"Output root: {args.out_root}")
    print(f"Device: {args.device}")
    print(f"Max parallel: {args.max_parallel}")
    
    # Track results
    successful_runs = []
    failed_runs = []
    
    # Run experiments
    total_experiments = len(args.networks) * len(seeds)
    completed = 0
    
    for i, network in enumerate(args.networks):
        for j, seed in enumerate(seeds):
            completed += 1
            print(f"\n[{completed}/{total_experiments}] Running {network} seed {seed}")
            
            output_dir = run_experiment(network, seed, args)
            
            if output_dir:
                successful_runs.append((network, seed, output_dir))
            else:
                failed_runs.append((network, seed))
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Total experiments: {total_experiments}")
    print(f"Successful: {len(successful_runs)}")
    print(f"Failed: {len(failed_runs)}")
    
    if successful_runs:
        print(f"\n✅ Successful runs:")
        for network, seed, output_dir in successful_runs:
            print(f"  {network} seed {seed}: {output_dir}")
    
    if failed_runs:
        print(f"\n❌ Failed runs:")
        for network, seed in failed_runs:
            print(f"  {network} seed {seed}")
    
    print(f"\n🎉 Pan-cancer experiments completed!")

if __name__ == '__main__':
    main()
