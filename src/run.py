#!/usr/bin/env python3
"""
Main script for running CANOPY-Router experiments on TREE benchmark.
"""

import argparse
import json
import os
import sys
import copy
import time
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import h5py
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(str(Path(__file__).parent))

from model import CANOPYRouter
from graft_adapter import GRAFTAdapter
from utils import load_tree_data, compute_metrics, setup_logging, build_bpcl_membership


def _split_indices(data, split):
    if split == 'train':
        return data.train_indices
    if split == 'val':
        return data.val_indices
    return data.test_indices


def _collect_predictions(model, data, device, split='test'):
    model.eval()

    with torch.no_grad():
        indices = _split_indices(data, split)
        if indices is None:
            return None

        outputs, aux = model(
            data.features.to(device),
            data.edge_index.to(device) if data.edge_index is not None else None,
            data.edge_attr.to(device) if data.edge_attr is not None else None,
            raw_features=data.raw_features.to(device) if data.raw_features is not None else None,
            pathogenicity_scores=data.pathogenicity_scores.to(device) if data.pathogenicity_scores is not None else None,
            cancer_context=data.cancer_context.to(device) if data.cancer_context is not None else None,
            gene_set_features=data.gene_set_features.to(device) if data.gene_set_features is not None else None,
            hypergraph_features=data.hypergraph_features.to(device) if data.hypergraph_features is not None else None,
            return_aux=True
        )

        split_predictions = outputs[indices].detach().cpu().numpy().flatten()
        split_labels = data.labels[indices].detach().cpu().numpy().flatten()
        metrics = compute_metrics(split_labels, split_predictions)
        return {
            'indices': indices.detach().cpu().numpy().tolist(),
            'labels': split_labels.tolist(),
            'predictions': split_predictions.tolist(),
            'metrics': metrics,
            'aux': aux,
        }

class TreeDataset(Dataset):
    """TREE dataset for cancer driver gene prioritization."""
    
    def __init__(self, features, labels, edge_index=None, edge_attr=None, 
                 raw_features=None, pathogenicity_scores=None, cancer_context=None,
                 gene_set_features=None, hypergraph_features=None, 
                 train_mask=None, val_mask=None, test_mask=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.edge_index = torch.LongTensor(edge_index) if edge_index is not None else None
        self.edge_attr = torch.FloatTensor(edge_attr) if edge_attr is not None else None
        self.raw_features = torch.FloatTensor(raw_features) if raw_features is not None else None
        self.pathogenicity_scores = torch.FloatTensor(pathogenicity_scores) if pathogenicity_scores is not None else None
        self.cancer_context = torch.FloatTensor(cancer_context) if cancer_context is not None else None
        self.gene_set_features = torch.FloatTensor(gene_set_features) if gene_set_features is not None else None
        self.hypergraph_features = torch.FloatTensor(hypergraph_features) if hypergraph_features is not None else None
        
        self.train_mask = torch.BoolTensor(train_mask) if train_mask is not None else None
        self.val_mask = torch.BoolTensor(val_mask) if val_mask is not None else None
        self.test_mask = torch.BoolTensor(test_mask) if test_mask is not None else None
        
        # Create indices for each split
        self.train_indices = torch.where(self.train_mask)[0] if self.train_mask is not None else None
        self.val_indices = torch.where(self.val_mask)[0] if self.val_mask is not None else None
        self.test_indices = torch.where(self.test_mask)[0] if self.test_mask is not None else None
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'labels': self.labels[idx],
            'raw_features': self.raw_features[idx] if self.raw_features is not None else None,
            'pathogenicity_scores': self.pathogenicity_scores[idx] if self.pathogenicity_scores is not None else None,
            'cancer_context': self.cancer_context[idx] if self.cancer_context is not None else None,
            'gene_set_features': self.gene_set_features[idx] if self.gene_set_features is not None else None,
            'hypergraph_features': self.hypergraph_features[idx] if self.hypergraph_features is not None else None,
            'train_mask': self.train_mask[idx] if self.train_mask is not None else False,
            'val_mask': self.val_mask[idx] if self.val_mask is not None else False,
            'test_mask': self.test_mask[idx] if self.test_mask is not None else False,
        }

def collate_fn(batch):
    """Custom collate function for graph data."""
    # For now, assume single graph processing
    # In a more complex setup, this would handle batching of multiple graphs
    batch_size = len(batch)
    
    # Stack all features
    features = torch.stack([item['features'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    # Handle optional features
    raw_features = None
    if batch[0]['raw_features'] is not None:
        raw_features = torch.stack([item['raw_features'] for item in batch])
    
    pathogenicity_scores = None
    if batch[0]['pathogenicity_scores'] is not None:
        pathogenicity_scores = torch.stack([item['pathogenicity_scores'] for item in batch])
    
    cancer_context = None
    if batch[0]['cancer_context'] is not None:
        cancer_context = torch.stack([item['cancer_context'] for item in batch])
    
    gene_set_features = None
    if batch[0]['gene_set_features'] is not None:
        gene_set_features = torch.stack([item['gene_set_features'] for item in batch])
    
    hypergraph_features = None
    if batch[0]['hypergraph_features'] is not None:
        hypergraph_features = torch.stack([item['hypergraph_features'] for item in batch])
    
    # Masks
    train_mask = torch.stack([item['train_mask'] for item in batch])
    val_mask = torch.stack([item['val_mask'] for item in batch])
    test_mask = torch.stack([item['test_mask'] for item in batch])
    
    return {
        'features': features,
        'labels': labels,
        'raw_features': raw_features,
        'pathogenicity_scores': pathogenicity_scores,
        'cancer_context': cancer_context,
        'gene_set_features': gene_set_features,
        'hypergraph_features': hypergraph_features,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
    }

def evaluate_model(model, data, device, split='test'):
    """Evaluate model on specified split."""
    collected = _collect_predictions(model, data, device, split=split)
    if collected is None:
        return {}

    metrics = dict(collected['metrics'])
    aux = collected['aux']
    metrics['routing_weights'] = aux['routing_weights'].detach().cpu().numpy()
    metrics['routing_entropy'] = aux['routing_entropy'].detach().cpu().numpy().flatten()
    metrics['routing_source_names'] = aux['routing_source_names']

    if 'pathogenicity_gate_alpha' in aux:
        metrics['pathogenicity_gate_alpha'] = aux['pathogenicity_gate_alpha'].detach().cpu().numpy().flatten()

    # EUAR uncertainty outputs
    if 'evidential_uncertainty' in aux:
        metrics['evidential_uncertainty'] = aux['evidential_uncertainty'].detach().cpu().numpy().flatten()
    if 'evidential_alpha' in aux:
        metrics['evidential_alpha'] = aux['evidential_alpha'].detach().cpu().numpy()

    return metrics


def build_model(args, input_dim):
    if args.model_type == 'graft':
        return GRAFTAdapter(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.graft_dropout,
        )

    return CANOPYRouter(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        top_k=args.top_k,
        max_edges=args.mvga_max_edges,
        temperature=args.temperature,
        entropy_reg=args.entropy_reg,
        fusion_mode=args.fusion_mode,
    )


def _get_optimizer(model, args):
    if args.optimizer == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

def train_model(model, data, args, device):
    """Train the model."""
    optimizer = _get_optimizer(model, args)

    # Cosine-annealing LR scheduler (disabled for homogeneous data)
    use_cosine_lr = not getattr(args, '_disable_cosine_lr', False)
    if use_cosine_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    else:
        scheduler = None
    
    # Get training data
    train_indices = data.train_indices
    val_indices = data.val_indices
    
    if train_indices is None or val_indices is None:
        raise ValueError("Training or validation masks not found")

    # --- Class-balanced BCE weight ---
    train_labels_np = data.labels[train_indices].cpu().numpy().reshape(-1)
    n_pos = float(train_labels_np.sum())
    n_neg = float(len(train_labels_np) - n_pos)
    if n_pos > 0:
        raw_pos_weight = n_neg / n_pos
        factor = getattr(args, 'pos_weight_factor', 1.0)
        pos_weight_val = raw_pos_weight * factor
    else:
        pos_weight_val = 1.0
    pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
    logging.info(f"Class-balanced loss: n_pos={n_pos:.0f}, n_neg={n_neg:.0f}, "
                 f"pos_weight={pos_weight_val:.2f}")

    grad_clip = getattr(args, 'grad_clip', 1.0)
    
    best_val_aupr = 0.0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        model.train()
        
        # Training step
        optimizer.zero_grad()
        
        train_features = data.features
        train_labels = data.labels[train_indices]
        
        if data.raw_features is not None:
            train_raw_features = data.raw_features
        else:
            train_raw_features = None
            
        if data.pathogenicity_scores is not None:
            train_pathogenicity_scores = data.pathogenicity_scores
        else:
            train_pathogenicity_scores = None
            
        if data.cancer_context is not None:
            train_cancer_context = data.cancer_context
        else:
            train_cancer_context = None
            
        if data.gene_set_features is not None:
            train_gene_set_features = data.gene_set_features
        else:
            train_gene_set_features = None
            
        if data.hypergraph_features is not None:
            train_hypergraph_features = data.hypergraph_features
        else:
            train_hypergraph_features = None
        
        # Forward pass
        if data.edge_index is not None:
            edge_index = data.edge_index
            edge_attr = data.edge_attr
        else:
            edge_index = None
            edge_attr = None
        
        outputs, aux = model(
            train_features.to(device),
            edge_index.to(device) if edge_index is not None else None,
            edge_attr.to(device) if edge_attr is not None else None,
            raw_features=train_raw_features.to(device) if train_raw_features is not None else None,
            pathogenicity_scores=train_pathogenicity_scores.to(device) if train_pathogenicity_scores is not None else None,
            cancer_context=train_cancer_context.to(device) if train_cancer_context is not None else None,
            gene_set_features=train_gene_set_features.to(device) if train_gene_set_features is not None else None,
            hypergraph_features=train_hypergraph_features.to(device) if train_hypergraph_features is not None else None,
            return_aux=True
        )
        
        # Compute class-balanced loss
        loss = F.binary_cross_entropy(
            outputs[train_indices],
            train_labels.to(device),
            weight=torch.where(
                train_labels.to(device) > 0.5,
                pos_weight_tensor,
                torch.ones(1, device=device),
            ).expand_as(train_labels.to(device)),
        )
        
        # Add routing entropy regularization (CANOPYRouter only)
        if getattr(model, 'fusion_mode', None) == 'router' and not getattr(model, 'disable_routing', True) and 'routing_entropy' in aux:
            entropy_loss = torch.mean(aux['routing_entropy'])
            loss = loss - model.router.entropy_reg * entropy_loss
        
        # EUAR: evidential KL regularisation
        if getattr(model, 'fusion_mode', None) == 'evidential' and not getattr(model, 'disable_routing', True):
            euar_kl = model.evidential_router.evidential_kl_loss(epoch=epoch)
            loss = loss + getattr(args, 'euar_kl_weight', 0.05) * euar_kl
        
        # BPCL: pathway consistency loss
        if (getattr(model, 'fusion_mode', None) in ('evidential', 'router')
                and not getattr(args, 'disable_bpcl', False)
                and 'routing_weights' in aux
                and model.bpcl.pathway_gene_lists):
            bpcl_loss = model.bpcl(aux['routing_weights'])
            loss = loss + getattr(args, 'bpcl_weight', 0.01) * bpcl_loss
        
        # SupCL: supervised contrastive loss
        if getattr(args, 'enable_supcl', False) and 'hidden_features' in aux:
            from src.supcl_module import SupervisedContrastiveLoss, SupCLProjectionHead
            if not hasattr(model, 'supcl_loss'):
                model.supcl_loss = SupervisedContrastiveLoss(
                    temperature=getattr(args, 'supcl_temperature', 0.07)
                ).to(device)
                model.supcl_proj = SupCLProjectionHead(
                    input_dim=aux['hidden_features'].shape[1],
                    hidden_dim=128,
                    output_dim=64
                ).to(device)
            
            # Project features and compute contrastive loss (only on training nodes)
            train_hidden = aux['hidden_features'][train_indices]
            projected = model.supcl_proj(train_hidden)
            supcl_loss = model.supcl_loss(projected, train_labels.long().to(device))
            loss = loss + getattr(args, 'supcl_weight', 0.1) * supcl_loss
        
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        # Validation
        if epoch % args.eval_every == 0:
            val_metrics = evaluate_model(model, data, device, split='val')
            val_aupr = val_metrics.get('aupr', 0.0)
            
            logging.info(f"Epoch {epoch}: Loss = {loss.item():.4f}, Val AUPR = {val_aupr:.4f}, "
                         f"LR = {optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping
            if val_aupr > best_val_aupr:
                best_val_aupr = val_aupr
                patience_counter = 0
                
                # Save best model
                torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    logging.info(f"Early stopping at epoch {epoch}")
                    break
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pth')))

    return model

def save_predictions(model, data, device, args):
    """Save predictions and auxiliary information."""
    model.eval()

    with torch.no_grad():
        # Get all data
        features = data.features.to(device)
        labels = data.labels.to(device)

        if data.raw_features is not None:
            raw_features = data.raw_features.to(device)
        else:
            raw_features = None

        if data.pathogenicity_scores is not None:
            pathogenicity_scores = data.pathogenicity_scores.to(device)
        else:
            pathogenicity_scores = None

        if data.cancer_context is not None:
            cancer_context = data.cancer_context.to(device)
        else:
            cancer_context = None

        if data.gene_set_features is not None:
            gene_set_features = data.gene_set_features.to(device)
        else:
            gene_set_features = None

        if data.hypergraph_features is not None:
            hypergraph_features = data.hypergraph_features.to(device)
        else:
            hypergraph_features = None

        # Forward pass
        if data.edge_index is not None:
            edge_index = data.edge_index.to(device)
            edge_attr = data.edge_attr.to(device) if data.edge_attr is not None else None
        else:
            edge_index = None
            edge_attr = None

        outputs, aux = model(
            features,
            edge_index,
            edge_attr,
            raw_features=raw_features,
            pathogenicity_scores=pathogenicity_scores,
            cancer_context=cancer_context,
            gene_set_features=gene_set_features,
            hypergraph_features=hypergraph_features,
            return_aux=True
        )

        predictions = outputs.cpu().numpy().flatten()
        gene_names = getattr(data, 'gene_names', [f"gene_{i}" for i in range(len(predictions))])

        # Save predictions
        with open(os.path.join(args.output_dir, 'preds_all.tsv'), 'w') as f:
            f.write("gene\tscore\n")
            for gene_name, score in zip(gene_names, predictions):
                f.write(f"{gene_name}\t{score:.6f}\n")

        np.save(os.path.join(args.output_dir, 'yhat_all.npy'), predictions)

        # Save routing information
        if not getattr(model, 'disable_routing', True):
            routing_weights = aux['routing_weights'].cpu().numpy()
            routing_entropy = aux['routing_entropy'].cpu().numpy().flatten()
            source_names = aux['routing_source_names']

            with open(os.path.join(args.output_dir, 'routing_weights.tsv'), 'w') as f:
                f.write("gene\t" + "\t".join(source_names) + "\n")
                for gene_name, weights in zip(gene_names, routing_weights):
                    weight_str = "\t".join([f"{w:.6f}" for w in weights])
                    f.write(f"{gene_name}\t{weight_str}\n")

            with open(os.path.join(args.output_dir, 'routing_entropy.tsv'), 'w') as f:
                f.write("gene\tentropy\n")
                for gene_name, entropy in zip(gene_names, routing_entropy):
                    f.write(f"{gene_name}\t{entropy:.6f}\n")

            with open(os.path.join(args.output_dir, 'routing_top_source.tsv'), 'w') as f:
                f.write("gene\ttop_source\tweight\n")
                for gene_name, weights in zip(gene_names, routing_weights):
                    top_idx = np.argmax(weights)
                    top_source = source_names[top_idx]
                    top_weight = weights[top_idx]
                    f.write(f"{gene_name}\t{top_source}\t{top_weight:.6f}\n")

        # Save pathogenicity gate coefficients
        if 'pathogenicity_gate_alpha' in aux:
            gate_alpha = aux['pathogenicity_gate_alpha'].cpu().numpy().flatten()
            with open(os.path.join(args.output_dir, 'pathogenicity_gate_alpha.tsv'), 'w') as f:
                f.write("gene\talpha\n")
                for gene_name, alpha in zip(gene_names, gate_alpha):
                    f.write(f"{gene_name}\t{alpha:.6f}\n")

def maybe_move_dataset_to_device(data, device):
    data.features = data.features.to(device)
    data.labels = data.labels.to(device)
    if data.edge_index is not None:
        data.edge_index = data.edge_index.to(device)
    if data.edge_attr is not None:
        data.edge_attr = data.edge_attr.to(device)
    if data.raw_features is not None:
        data.raw_features = data.raw_features.to(device)
    if data.pathogenicity_scores is not None:
        data.pathogenicity_scores = data.pathogenicity_scores.to(device)
    if data.cancer_context is not None:
        data.cancer_context = data.cancer_context.to(device)
    if data.gene_set_features is not None:
        data.gene_set_features = data.gene_set_features.to(device)
    if data.hypergraph_features is not None:
        data.hypergraph_features = data.hypergraph_features.to(device)
    if data.train_mask is not None:
        data.train_mask = data.train_mask.to(device)
    if data.val_mask is not None:
        data.val_mask = data.val_mask.to(device)
    if data.test_mask is not None:
        data.test_mask = data.test_mask.to(device)
    if data.train_indices is not None:
        data.train_indices = data.train_indices.to(device)
    if data.val_indices is not None:
        data.val_indices = data.val_indices.to(device)
    if data.test_indices is not None:
        data.test_indices = data.test_indices.to(device)
    return data

def clone_dataset_to_device(data, device):
    return maybe_move_dataset_to_device(copy.deepcopy(data), device)


def parse_int_grid(grid_text):
    values = []
    for token in str(grid_text).split(','):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    return values

def main():
    parser = argparse.ArgumentParser(description='Run CANOPY-Router on TREE benchmark')

    # Data arguments
    parser.add_argument('--tree_h5', type=str, required=True, help='Path to TREE H5 file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--run_name', type=str, default='canopy_router', help='Run name')

    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of transformer layers')
    parser.add_argument('--model_type', type=str, default='canopy', choices=['canopy', 'graft'], help='Model backbone to train under identical TREE splits')
    parser.add_argument('--graft_dropout', type=float, default=0.1, help='Dropout used by GRAFT adapter')
    parser.add_argument('--graft_grid_layers', type=str, default='2,3', help='Grid candidates for GRAFT layers')
    parser.add_argument('--graft_grid_heads', type=str, default='2,4', help='Grid candidates for GRAFT attention heads')
    parser.add_argument('--run_hparam_grid', action='store_true', help='Enable fair hyperparameter grid search budget (used for GRAFT)')
    parser.add_argument('--top_k', type=int, default=256, help='Top-k for graph construction')
    parser.add_argument('--mvga_max_edges', type=int, default=5000000, help='Maximum edges for MVGA')
    parser.add_argument('--temperature', type=float, default=1.0, help='Routing temperature')
    parser.add_argument('--entropy_reg', type=float, default=0.01, help='Routing entropy regularization')
    parser.add_argument('--fusion_mode', type=str, default='evidential', choices=['evidential', 'router', 'attn', 'cond_attn', 'concat'], help='Evidence fusion mode')
    parser.add_argument('--external_context_mode', type=str, default='both', choices=['both', 'patho_only', 'cancer_only', 'none'], help='External context ablation: both/patho_only/cancer_only/none')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw'], help='Optimizer used for training')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--eval_every', type=int, default=5, help='Evaluation frequency')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--chunk_rows', type=int, default=256, help='Chunk size for large graphs')
    
    # Hardware arguments
    parser.add_argument('--device', type=int, default=0, help='GPU device')
    parser.add_argument('--amp', type=str, default='none', choices=['none', 'fp16', 'bf16'], help='Mixed precision')
    
    # Feature flags
    parser.add_argument('--tree_model', type=str, default='treenet', choices=['treenet', 'gcn', 'gat'], help='Base model')
    parser.add_argument('--use_raw_features', action='store_true', help='Use raw multi-omics features')
    parser.add_argument('--disable_evidence_routing', action='store_true', help='Disable evidence routing')
    parser.add_argument('--export_only', action='store_true', help='Skip training and export predictions from checkpoint')
    parser.add_argument('--checkpoint_path', type=str, help='Path to checkpoint used for export_only or explicit model loading')
    parser.add_argument('--eval_on_cpu', action='store_true', help='Run final evaluation on CPU to reduce GPU memory pressure')
    parser.add_argument('--export_on_cpu', action='store_true', help='Run final prediction export on CPU to reduce GPU memory pressure')

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
    parser.add_argument(
        '--pathogenicity_source',
        type=str,
        default='alphamissense',
        choices=['alphamissense', 'feature_l2', 'phylop', 'constant', 'random'],
        help='Source used to build pathogenicity scores when not provided in H5'
    )
    parser.add_argument('--phylop_path', type=str, help='Optional gene-level PhyloP score file (.tsv/.csv/.npy/.pt)')
    parser.add_argument('--disable_pathogenicity_source', action='store_true', help='Hard-mask pathogenicity input branch in forward pass')
    parser.add_argument('--enable_hyperconv', action='store_true', help='Enable hypergraph convolution')
    parser.add_argument('--hyperconv_dim', type=int, default=32, help='Hypergraph convolution dimension')
    parser.add_argument('--disable_hyperconv', action='store_true', help='Ablation: zero out HyperConv channel (Ch.6)')
    parser.add_argument('--disable_gene_set', action='store_true', help='Ablation: zero out Gene-Set channel (Ch.5)')
    parser.add_argument('--enable_adaptive_patho', action='store_true', help='Enable adaptive pathogenicity')
    parser.add_argument('--save_gate_stats', action='store_true', help='Save gate statistics')
    parser.add_argument('--save_routing', action='store_true', help='Save per-gene routing weights to split_predictions.json for interpretability analysis')
    
    # EUAR / BPCL flags
    parser.add_argument('--euar_kl_weight', type=float, default=0.05, help='Weight for EUAR KL-divergence loss')
    parser.add_argument('--euar_kl_annealing', type=int, default=10, help='Epochs to linearly anneal EUAR KL weight')
    parser.add_argument('--bpcl_weight', type=float, default=0.01, help='Weight for Biological Pathway Consistency Loss')
    parser.add_argument('--disable_bpcl', action='store_true', help='Disable BPCL even in evidential mode')
    
    # SupCL additional flags
    parser.add_argument('--supcl_weight', type=float, default=0.1, help='Weight for SupCL loss')
    parser.add_argument('--supcl_temperature', type=float, default=0.07, help='Temperature for SupCL')
    
    # Heterogeneous data enhancements
    parser.add_argument('--knn_k', type=int, default=15, help='k for k-NN graph augmentation on sparse heterogeneous graphs (0=disable)')
    parser.add_argument('--pos_weight_factor', type=float, default=1.0, help='Multiplier for class-balanced pos_weight')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping max norm (0=disable)')
    parser.add_argument('--cancer_name', type=str, default=None, help='Cancer type name for heterogeneous evidence construction')

    # Fine-grained architecture control (override auto-detection)
    parser.add_argument('--homo_adapt', type=str, default='auto', choices=['auto','on','off'],
                        help='Homogeneous adaptation: auto=detect by input_dim, on=force, off=disable')
    parser.add_argument('--use_residual_gat', type=str, default='auto', choices=['auto','true','false'],
                        help='MVGA residual connections: auto=follow homo_adapt, true/false=force')
    parser.add_argument('--use_cosine_lr', type=str, default='auto', choices=['auto','true','false'],
                        help='Cosine annealing LR: auto=follow homo_adapt, true/false=force')
    parser.add_argument('--skip_cross_evidence', type=str, default='true', choices=['auto','true','false'],
                        help='Skip cross-evidence attention: true=default (strict attribution, +0.003 AUPR), false=enable CE as optional enhancement, auto=follow homo_adapt')

    # Reproducibility
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup logging
    setup_logging(args.output_dir)
    logging.info(f"Starting run: {args.run_name}")
    logging.info(f"Arguments: {args}")
    
    # Setup device
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Load data
    logging.info(f"Loading data from {args.tree_h5}")
    data = load_tree_data(args.tree_h5, args)
    logging.info(f"Loaded data with {len(data)} genes")
    
    # Create model
    input_dim = data.features.shape[1]
    model = build_model(args, input_dim)
    
    # Auto-detect homogeneous data and adjust model behaviour
    is_homo = (input_dim <= 4)
    homo_adapt = getattr(args, 'homo_adapt', 'auto')
    if homo_adapt == 'auto':
        apply_homo = is_homo
    elif homo_adapt == 'on':
        apply_homo = True
    else:
        apply_homo = False

    if args.model_type == 'canopy':
        # Resolve skip_cross_evidence
        sce = getattr(args, 'skip_cross_evidence', 'auto')
        if sce == 'auto':
            model.skip_cross_evidence = apply_homo
        else:
            model.skip_cross_evidence = (sce == 'true')

        # Resolve use_residual_gat
        urg = getattr(args, 'use_residual_gat', 'auto')
        if urg == 'auto':
            model.mvga.use_residual = not apply_homo
        else:
            model.mvga.use_residual = (urg == 'true')

        # Resolve cosine LR
        ucl = getattr(args, 'use_cosine_lr', 'auto')
        if ucl == 'auto':
            args._disable_cosine_lr = apply_homo
        else:
            args._disable_cosine_lr = (ucl == 'false')

        # Only cap pos_weight if homo_adapt is explicitly 'on' or 'auto' with homo data
        # Do NOT cap when individual overrides are used
        if apply_homo and homo_adapt == 'auto':
            args.pos_weight_factor = min(getattr(args, 'pos_weight_factor', 1.0), 0.3)

        logging.info(f"Architecture config: skip_cross_evidence={model.skip_cross_evidence}, "
                     f"use_residual={model.mvga.use_residual}, "
                     f"cosine_lr={not args._disable_cosine_lr}, "
                     f"pos_weight_factor={args.pos_weight_factor}")

    # Register BPCL pathways (must happen before .to(device))
    if (args.model_type == 'canopy'
            and args.fusion_mode in ('evidential', 'router')
            and not getattr(args, 'disable_bpcl', False)):
        bpcl_mat = build_bpcl_membership(data.gene_names)
        if bpcl_mat is not None:
            model.bpcl.register_pathways(bpcl_mat)
            logging.info(f"BPCL: registered {len(model.bpcl.pathway_gene_lists)} pathways")
        else:
            logging.info("BPCL: no gene-set data available, skipping")

    # Set EUAR KL annealing from args
    if args.model_type == 'canopy' and hasattr(model, 'evidential_router'):
        model.evidential_router.kl_annealing_epochs = getattr(args, 'euar_kl_annealing', 10)

    # Disable routing if requested (CANOPYRouter only)
    if args.model_type == 'canopy' and args.disable_evidence_routing:
        model.disable_routing = True
        logging.info("Evidence routing disabled")
    if args.model_type == 'canopy' and args.disable_pathogenicity_source:
        model.disable_pathogenicity_source = True
        logging.info("Pathogenicity source branch hard-masked")
    if args.model_type == 'canopy' and getattr(args, 'disable_hyperconv', False):
        model.disable_hypergraph_channel = True
        logging.info("HyperConv channel (Ch.6) ablated")
    if args.model_type == 'canopy' and getattr(args, 'disable_gene_set', False):
        model.disable_gene_set_channel = True
        logging.info("Gene-Set channel (Ch.5) ablated")
    if args.model_type == 'canopy' and hasattr(args, 'external_context_mode'):
        model.external_context_mode = args.external_context_mode
        if args.external_context_mode != 'both':
            logging.info(f"External context ablation mode: {args.external_context_mode}")
    model = model.to(device)

    # Setup mixed precision
    if args.amp == 'fp16':
        scaler = torch.cuda.amp.GradScaler()
    elif args.amp == 'bf16':
        scaler = torch.cuda.amp.GradScaler()
        model = model.bfloat16()
    else:
        scaler = None

    checkpoint_path = args.checkpoint_path or os.path.join(args.output_dir, 'best_model.pth')

    if args.export_only:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found for export_only: {checkpoint_path}")
        logging.info(f"Loading checkpoint for export-only mode: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        if args.model_type == 'graft' and args.run_hparam_grid:
            layer_grid = parse_int_grid(args.graft_grid_layers)
            head_grid = parse_int_grid(args.graft_grid_heads)
            if not layer_grid or not head_grid:
                raise ValueError('GRAFT grid search requires non-empty graft_grid_layers and graft_grid_heads')

            best_trial = None
            best_val_aupr = -1.0
            logging.info(f"Starting GRAFT fair grid search: layers={layer_grid}, heads={head_grid}")

            for n_layers in layer_grid:
                for n_heads in head_grid:
                    trial_args = copy.deepcopy(args)
                    trial_args.num_layers = n_layers
                    trial_args.num_heads = n_heads
                    trial_dir = os.path.join(args.output_dir, f"grid_layers{n_layers}_heads{n_heads}")
                    os.makedirs(trial_dir, exist_ok=True)
                    trial_args.output_dir = trial_dir

                    trial_model = build_model(trial_args, input_dim).to(device)
                    logging.info(f"Grid trial start: layers={n_layers}, heads={n_heads}")
                    trial_model = train_model(trial_model, data, trial_args, device)
                    trial_val_metrics = evaluate_model(trial_model, data, device, split='val')
                    trial_val_aupr = float(trial_val_metrics.get('aupr', 0.0))
                    logging.info(f"Grid trial done: layers={n_layers}, heads={n_heads}, val_aupr={trial_val_aupr:.4f}")

                    if trial_val_aupr > best_val_aupr:
                        best_val_aupr = trial_val_aupr
                        best_trial = {
                            'num_layers': n_layers,
                            'num_heads': n_heads,
                            'checkpoint': os.path.join(trial_dir, 'best_model.pth'),
                        }

            if best_trial is None:
                raise RuntimeError('No valid GRAFT grid trial completed.')

            logging.info(f"Best GRAFT trial: {best_trial}")
            args.num_layers = best_trial['num_layers']
            args.num_heads = best_trial['num_heads']
            model = build_model(args, input_dim).to(device)
            model.load_state_dict(torch.load(best_trial['checkpoint'], map_location=device))
            torch.save(model.state_dict(), checkpoint_path)
            with open(os.path.join(args.output_dir, 'graft_grid_search.json'), 'w') as f:
                json.dump({'best': best_trial, 'val_aupr': best_val_aupr}, f, indent=2)
        else:
            logging.info("Starting training...")
            model = train_model(model, data, args, device)

    eval_model_ref = model
    eval_data_ref = data
    eval_device = device

    if args.eval_on_cpu:
        logging.info("Switching final evaluation to CPU")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        eval_device = torch.device('cpu')
        eval_model_ref = copy.deepcopy(model).to(eval_device)
        eval_data_ref = clone_dataset_to_device(data, eval_device)

    logging.info("Evaluating on train/val/test splits...")
    train_metrics = evaluate_model(eval_model_ref, eval_data_ref, eval_device, split='train')
    val_metrics = evaluate_model(eval_model_ref, eval_data_ref, eval_device, split='val')
    test_metrics = evaluate_model(eval_model_ref, eval_data_ref, eval_device, split='test')

    # Save results
    results = {
        'train_auc': train_metrics.get('auc'),
        'train_aupr': train_metrics.get('aupr'),
        'train_f1': train_metrics.get('f1'),
        'train_f1_max': train_metrics.get('f1_max', train_metrics.get('f1')),
        'val_auc': val_metrics.get('auc'),
        'val_aupr': val_metrics.get('aupr'),
        'val_f1': val_metrics.get('f1'),
        'val_f1_max': val_metrics.get('f1_max', val_metrics.get('f1')),
        'test_auc': test_metrics['auc'],
        'test_aupr': test_metrics['aupr'],
        'test_f1': test_metrics['f1'],
        'test_f1_max': test_metrics.get('f1_max', test_metrics.get('f1')),
        'fusion_mode': args.fusion_mode,
        'args': vars(args),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(args.output_dir, 'metrics_summary.json'), 'w') as f:
        json.dump(results, f, indent=2)
    logging.info(f"Test AUC: {test_metrics['auc']:.4f}")
    logging.info(f"Test AUPR: {test_metrics['aupr']:.4f}")
    logging.info(f"Test F1: {test_metrics['f1']:.4f}")

    export_model_ref = model
    export_data_ref = data
    export_device = device

    if args.export_on_cpu:
        logging.info("Switching prediction export to CPU")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        export_device = torch.device('cpu')
        export_model_ref = copy.deepcopy(model).to(export_device)
        export_data_ref = clone_dataset_to_device(data, export_device)

    save_predictions(export_model_ref, export_data_ref, export_device, args)

    split_predictions = {}
    for split_name in ('train', 'val', 'test'):
        split_payload = _collect_predictions(export_model_ref, export_data_ref, export_device, split=split_name)
        if split_payload is None:
            continue
        entry = {
            'indices': split_payload['indices'],
            'labels': split_payload['labels'],
            'predictions': split_payload['predictions'],
            'metrics': split_payload['metrics'],
        }
        if getattr(args, 'save_routing', False):
            rw = split_payload['aux'].get('routing_weights')
            if rw is not None:
                entry['routing_weights'] = rw.detach().cpu().numpy().tolist()
        split_predictions[split_name] = entry

    with open(os.path.join(args.output_dir, 'split_predictions.json'), 'w') as f:
        json.dump(split_predictions, f, indent=2)

    logging.info("Run completed successfully!")

if __name__ == '__main__':
    main()
