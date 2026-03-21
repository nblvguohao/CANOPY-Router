"""
Utility functions for CANOPY-Router.
"""

import logging
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve
from sklearn.decomposition import TruncatedSVD

def setup_logging(output_dir):
    """Setup logging configuration."""
    log_file = os.path.join(output_dir, 'run.log')
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def build_sparse_gene_set_matrix(npz_path):
    gene_sets = np.load(npz_path, allow_pickle=True)
    fmt = gene_sets['format']
    if isinstance(fmt, np.ndarray):
        fmt = fmt.item()
    if isinstance(fmt, bytes):
        fmt = fmt.decode()
    shape = tuple(int(x) for x in gene_sets['shape'])
    matrix = sparse.csc_matrix((gene_sets['data'], gene_sets['indices'], gene_sets['indptr']), shape=shape)
    return matrix

def build_gene_set_features(gene_names, args, input_dim):
    if not getattr(args, 'enable_gske', False):
        return None
    if not getattr(args, 'gske_npz', None) or not getattr(args, 'gske_gene_list', None):
        return None
    if not os.path.exists(args.gske_npz) or not os.path.exists(args.gske_gene_list):
        return None

    with open(args.gske_gene_list, 'r') as f:
        ref_genes = [line.strip().split(',')[0] for line in f if line.strip()]
    ref_index = {gene: idx for idx, gene in enumerate(ref_genes)}

    matrices = [build_sparse_gene_set_matrix(args.gske_npz)]
    if getattr(args, 'gske_c5_npz', None) and os.path.exists(args.gske_c5_npz):
        matrices.append(build_sparse_gene_set_matrix(args.gske_c5_npz))

    aligned_blocks = []
    for matrix in matrices:
        rows = []
        missing = []
        for gene in gene_names:
            idx = ref_index.get(gene)
            if idx is None or idx >= matrix.shape[0]:
                missing.append(True)
                rows.append(sparse.csr_matrix((1, matrix.shape[1]), dtype=np.float32))
            else:
                missing.append(False)
                rows.append(matrix.getrow(idx).astype(np.float32))
        aligned = sparse.vstack(rows, format='csr')
        if aligned.shape[1] > input_dim:
            svd = TruncatedSVD(n_components=input_dim, random_state=42)
            dense = svd.fit_transform(aligned)
        else:
            dense = aligned.toarray()
            if dense.shape[1] < input_dim:
                dense = np.pad(dense, ((0, 0), (0, input_dim - dense.shape[1])))
        aligned_blocks.append(dense.astype(np.float32))

    if not aligned_blocks:
        return None

    combined = np.mean(np.stack(aligned_blocks, axis=0), axis=0).astype(np.float32)
    return combined

def build_pathogenicity_scores(gene_names, args):
    if not getattr(args, 'enable_pgem', False):
        return None
    source = getattr(args, 'pathogenicity_source', 'alphamissense')

    if source == 'feature_l2':
        return None

    if not getattr(args, 'gske_gene_list', None) or not os.path.exists(args.gske_gene_list):
        return None
    with open(args.gske_gene_list, 'r') as f:
        ref_genes = [line.strip().split(',')[0] for line in f if line.strip()]

    raw_values = None
    if source == 'phylop':
        phylop_path = getattr(args, 'phylop_path', None)
        if not phylop_path or not os.path.exists(phylop_path):
            return None
        path_obj = Path(phylop_path)
        suffix = path_obj.suffix.lower()
        if suffix in {'.pt', '.pth'}:
            raw_values = torch.load(path_obj, map_location='cpu')
            if isinstance(raw_values, torch.Tensor):
                raw_values = raw_values.detach().cpu().numpy()
        elif suffix == '.npy':
            raw_values = np.load(path_obj, allow_pickle=True)
        elif suffix in {'.csv', '.tsv'}:
            delimiter = ',' if suffix == '.csv' else '\t'
            table = np.genfromtxt(path_obj, delimiter=delimiter, names=True, dtype=None, encoding='utf-8')
            if getattr(table, 'dtype', None) is None or len(table.dtype.names or []) == 0:
                return None
            col_names = [name.lower() for name in table.dtype.names]
            gene_col = None
            score_col = None
            for idx, name in enumerate(col_names):
                if gene_col is None and name in {'gene', 'symbol', 'gene_name'}:
                    gene_col = table.dtype.names[idx]
                if score_col is None and ('phylop' in name or 'score' in name):
                    score_col = table.dtype.names[idx]
            if gene_col is None:
                gene_col = table.dtype.names[0]
            if score_col is None:
                score_col = table.dtype.names[-1]
            mapping = {str(row[gene_col]): float(row[score_col]) for row in table}
            scores = np.zeros((len(gene_names), 1), dtype=np.float32)
            for i, gene in enumerate(gene_names):
                scores[i, 0] = mapping.get(gene, 0.0)
            return scores
        else:
            return None
    else:
        if not getattr(args, 'alphamissense_path', None) or not os.path.exists(args.alphamissense_path):
            return None
        raw_values = torch.load(args.alphamissense_path, map_location='cpu')
        if isinstance(raw_values, torch.Tensor):
            raw_values = raw_values.detach().cpu().numpy()

    if raw_values is None:
        return None

    raw_values = np.asarray(raw_values, dtype=np.float32).reshape(-1)
    ref_index = {gene: idx for idx, gene in enumerate(ref_genes[:len(raw_values)])}
    scores = np.zeros((len(gene_names), 1), dtype=np.float32)
    for i, gene in enumerate(gene_names):
        idx = ref_index.get(gene)
        if idx is not None and idx < len(raw_values):
            scores[i, 0] = raw_values[idx]
    return scores

def limit_edge_count(edge_index, edge_attr, max_edges):
    if edge_index is None or max_edges is None:
        return edge_index, edge_attr

    max_edges = int(max_edges)
    if max_edges <= 0:
        return edge_index, edge_attr

    num_edges = edge_index.shape[1]
    if num_edges <= max_edges:
        return edge_index, edge_attr

    if edge_attr is not None and len(edge_attr) == num_edges:
        edge_scores = np.asarray(edge_attr)
        if edge_scores.ndim > 1:
            edge_scores = edge_scores[:, 0]
        edge_scores = np.abs(edge_scores.reshape(-1))
        keep_idx = np.argpartition(edge_scores, -max_edges)[-max_edges:]
    else:
        keep_idx = np.linspace(0, num_edges - 1, num=max_edges, dtype=np.int64)

    keep_idx = np.sort(keep_idx)
    limited_edge_index = edge_index[:, keep_idx]
    limited_edge_attr = edge_attr[keep_idx] if edge_attr is not None else None
    return limited_edge_index, limited_edge_attr

def load_tree_data(h5_path, args):
    """Load TREE benchmark data from H5 file."""
    with h5py.File(h5_path, 'r') as f:
        # Load basic features and labels
        features = f['features'][:]
        if 'labels' in f:
            labels = f['labels'][:]
        else:
            y_train = f['y_train'][:] if 'y_train' in f else np.zeros((len(features), 1), dtype=np.float32)
            y_val = f['y_val'][:] if 'y_val' in f else np.zeros((len(features), 1), dtype=np.float32)
            y_test = f['y_test'][:] if 'y_test' in f else np.zeros((len(features), 1), dtype=np.float32)
            labels = y_train + y_val + y_test
        
        # Load masks
        train_mask = f['mask_train'][:].astype(bool)
        val_mask = f['mask_val'][:].astype(bool)
        test_mask = f['mask_test'][:].astype(bool)
        
        # Load edge index if available
        edge_index = f['edge_index'][:] if 'edge_index' in f else None
        edge_attr = f['edge_attr'][:] if 'edge_attr' in f else None
        if edge_index is None and 'network' in f:
            network = f['network'][:]
            if network.ndim != 2 or network.shape[0] != network.shape[1]:
                raise ValueError(f"Unsupported network shape: {network.shape}")

            num_nodes = network.shape[0]
            density = float(np.count_nonzero(network)) / float(network.size)
            max_neighbors = getattr(args, 'top_k', 256)

            if density > 0.1:
                src_list = []
                dst_list = []
                attr_list = []
                for node_idx in range(num_nodes):
                    row = network[node_idx]
                    neighbors = np.flatnonzero(row)
                    neighbors = neighbors[neighbors != node_idx]
                    if neighbors.size == 0:
                        continue

                    if neighbors.size > max_neighbors:
                        weights = row[neighbors]
                        top_idx = np.argsort(weights)[-max_neighbors:]
                        neighbors = neighbors[top_idx]

                    src_list.append(np.full(neighbors.shape[0], node_idx, dtype=np.int64))
                    dst_list.append(neighbors.astype(np.int64, copy=False))
                    attr_list.append(row[neighbors].astype(np.float32, copy=False))

                if src_list:
                    src = np.concatenate(src_list)
                    dst = np.concatenate(dst_list)
                    edge_index = np.vstack([src, dst])
                    edge_attr = np.concatenate(attr_list).reshape(-1, 1)
                else:
                    edge_index = np.zeros((2, 0), dtype=np.int64)
                    edge_attr = np.zeros((0, 1), dtype=np.float32)
            else:
                src, dst = np.nonzero(network)
                keep = src != dst
                src = src[keep]
                dst = dst[keep]
                edge_index = np.vstack([src, dst])
                edge_attr = network[src, dst].reshape(-1, 1)
        
        edge_index, edge_attr = limit_edge_count(
            edge_index,
            edge_attr,
            getattr(args, 'mvga_max_edges', None)
        )

        # Load optional features
        raw_features = None
        if 'raw_features' in f:
            raw_features = f['raw_features'][:]
        elif 'features_raw' in f:
            raw_features = f['features_raw'][:]
        pathogenicity_scores = f['pathogenicity_scores'][:] if 'pathogenicity_scores' in f else None
        cancer_context = f['cancer_context'][:] if 'cancer_context' in f else None
        gene_set_features = f['gene_set_features'][:] if 'gene_set_features' in f else None
        hypergraph_features = f['hypergraph_features'][:] if 'hypergraph_features' in f else None
        
        # Load gene names if available
        if 'gene_names' in f:
            gene_name_array = f['gene_names'][:]
            gene_names = []
            for name in gene_name_array:
                if isinstance(name, np.ndarray) or isinstance(name, (list, tuple)):
                    candidate = name[-1]
                else:
                    candidate = name
                gene_names.append(candidate.decode() if isinstance(candidate, bytes) else str(candidate))
        else:
            gene_names = [f"gene_{i}" for i in range(len(features))]
        
        # ------------------------------------------------------------------
        # Auto-detect heterogeneous data and build evidence channels
        # ------------------------------------------------------------------
        feature_names_list = None
        if 'feature_names' in f:
            fn_raw = f['feature_names'][:]
            feature_names_list = [
                x.decode() if isinstance(x, bytes) else str(x) for x in fn_raw
            ]

        is_hetero = False
        if feature_names_list and features.shape[1] > 4:
            groups = detect_feature_groups(feature_names_list)
            hetero_prefixes = {'MF', 'CNA', 'GE', 'METH'}
            if len(hetero_prefixes & set(groups.keys())) >= 2:
                is_hetero = True

        labels = labels.astype(np.float32).reshape(-1, 1)

        if is_hetero:
            # --- Heterogeneous path: build ALL evidence channels ---
            logging.info('Detected heterogeneous multi-omics features '
                         f'(dim={features.shape[1]}, groups={list(groups.keys())}). '
                         'Auto-building evidence channels.')
            # Infer cancer name from H5 filename (e.g. BRCA_multiomics.h5)
            cancer_name = getattr(args, 'cancer_name', None)
            if cancer_name is None:
                stem = Path(h5_path).stem  # e.g. 'BRCA_multiomics'
                cancer_name = stem.split('_')[0]

            evidence = build_hetero_evidence(
                features, feature_names_list, gene_names,
                train_mask, labels,
                cancer_name=cancer_name,
                input_dim=features.shape[1],
            )
            raw_features = evidence['raw_features']
            pathogenicity_scores = evidence['pathogenicity_scores']
            # Override pathogenicity scores for shock therapy experiments
            patho_src = getattr(args, 'pathogenicity_source', 'alphamissense')
            if patho_src == 'constant':
                pathogenicity_scores = np.full((features.shape[0], 1), 0.5, dtype=np.float32)
                logging.info('Shock therapy: using CONSTANT (0.5) pathogenicity scores')
            elif patho_src == 'random':
                rng = np.random.RandomState(getattr(args, 'seed', 42))
                pathogenicity_scores = rng.uniform(0, 1, size=(features.shape[0], 1)).astype(np.float32)
                logging.info('Shock therapy: using RANDOM pathogenicity scores')
            if evidence['cancer_context'] is not None:
                cancer_context = evidence['cancer_context']
            if evidence['gene_set_features'] is not None:
                gene_set_features = evidence['gene_set_features']
            if evidence['hypergraph_features'] is not None:
                hypergraph_features = evidence['hypergraph_features']

            # Augment sparse graph with k-NN feature-similarity edges
            if edge_index is not None:
                density = float(edge_index.shape[1]) / float(features.shape[0] ** 2)
            else:
                density = 0.0
            knn_k = getattr(args, 'knn_k', 15)
            if density < 0.01 and knn_k > 0:
                logging.info(f'Sparse graph (density={density:.6f}). '
                             f'Augmenting with k-NN edges (k={knn_k}).')
                edge_index, edge_attr = augment_sparse_graph_knn(
                    features, edge_index, k=knn_k,
                    max_total_edges=getattr(args, 'mvga_max_edges', None),
                )
        else:
            # --- Original homogeneous path ---
            if raw_features is None and getattr(args, 'use_raw_features', False):
                raw_features = features.copy()
            if getattr(args, 'enable_pgem', False) and getattr(args, 'pathogenicity_source', 'alphamissense') == 'feature_l2':
                feature_norm = np.linalg.norm(features, axis=1, keepdims=True)
                scale = np.percentile(feature_norm, 95) + 1e-8
                pathogenicity_scores = (feature_norm / scale).astype(np.float32)
                pathogenicity_scores = np.clip(pathogenicity_scores, 0.0, 1.0)
            if pathogenicity_scores is None:
                pathogenicity_scores = build_pathogenicity_scores(gene_names, args)
            if gene_set_features is None:
                gene_set_features = build_gene_set_features(gene_names, args, features.shape[1])
            if hypergraph_features is None and gene_set_features is not None:
                hypergraph_features = gene_set_features.copy()
    
    # Create dataset object
    from run import TreeDataset
    
    dataset = TreeDataset(
        features=features,
        labels=labels,
        edge_index=edge_index,
        edge_attr=edge_attr,
        raw_features=raw_features,
        pathogenicity_scores=pathogenicity_scores,
        cancer_context=cancer_context,
        gene_set_features=gene_set_features,
        hypergraph_features=hypergraph_features,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )
    
    # Store gene names
    dataset.gene_names = gene_names
    
    return dataset

def compute_metrics(labels, predictions, thresholds=None):
    """Compute evaluation metrics."""
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1)
    
    metrics = {}
    
    # AUC and AUPR
    try:
        metrics['auc'] = roc_auc_score(labels, predictions)
    except ValueError:
        metrics['auc'] = 0.5
        
    try:
        metrics['aupr'] = average_precision_score(labels, predictions)
    except ValueError:
        metrics['aupr'] = 0.0
    
    # F1-max from PR curve (threshold-optimized and comparable across baselines)
    try:
        precision, recall, pr_thresholds = precision_recall_curve(labels, predictions)
        denom = precision + recall
        f1_curve = np.divide(2 * precision * recall, denom, out=np.zeros_like(precision), where=denom > 0)
        best_idx = int(np.argmax(f1_curve)) if len(f1_curve) else 0
        best_f1 = float(f1_curve[best_idx]) if len(f1_curve) else 0.0

        # precision_recall_curve returns thresholds with len = len(precision)-1
        if len(pr_thresholds) > 0:
            best_thresh = float(pr_thresholds[min(best_idx, len(pr_thresholds) - 1)])
        else:
            best_thresh = 0.5

        metrics['f1_max'] = best_f1
        metrics['f1_max_threshold'] = best_thresh

        # Backward-compatible keys used elsewhere in current pipeline
        metrics['f1'] = best_f1
        metrics['f1_threshold'] = best_thresh
    except ValueError:
        metrics['f1_max'] = 0.0
        metrics['f1_max_threshold'] = 0.5
        metrics['f1'] = 0.0
        metrics['f1_threshold'] = 0.5

    # Keep coarse threshold sweep for debugging/legacy analysis only
    try:
        best_f1 = 0.0
        best_thresh = 0.5
        for thresh in thresholds:
            pred_binary = (predictions >= thresh).astype(int)
            f1 = f1_score(labels, pred_binary)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        metrics['f1_grid'] = best_f1
        metrics['f1_grid_threshold'] = best_thresh
    except ValueError:
        metrics['f1_grid'] = 0.0
        metrics['f1_grid_threshold'] = 0.5
    
    return metrics

# ---------------------------------------------------------------------------
# Heterogeneous evidence construction helpers
# ---------------------------------------------------------------------------

_DISHYPER_DATA = Path(__file__).resolve().parents[1] / 'external' / 'research_repos_clean' / 'research_repos' / 'GRAFT' / 'baseline' / 'DISFusion' / 'Data'


def detect_feature_groups(feature_names):
    """Detect semantic groups in heterogeneous feature names.

    Returns a dict mapping prefix ('MF', 'CNA', 'GE', 'METH', ...)
    to list of column indices.
    """
    groups = {}
    for idx, name in enumerate(feature_names):
        if ':' in name:
            prefix = name.split(':')[0].strip()
        else:
            prefix = 'other'
        groups.setdefault(prefix, []).append(idx)
    return groups


def _load_global_genesets(data_dir=None):
    """Load c2+c5 gene-set matrices used by DISHyper."""
    import pandas as pd
    data_dir = Path(data_dir) if data_dir else _DISHYPER_DATA
    gene_list_file = data_dir / 'geneList.txt'
    c2_file = data_dir / 'c2_GenesetsMatrix.npz'
    c5_file = data_dir / 'c5_GenesetsMatrix.npz'
    if not gene_list_file.exists() or not c2_file.exists():
        return None, None
    gene_df = pd.read_csv(gene_list_file, header=None)
    symbols = gene_df.iloc[:, 1].astype(str).tolist()
    c2 = sparse.load_npz(str(c2_file))
    mats = [c2]
    if c5_file.exists():
        mats.append(sparse.load_npz(str(c5_file)))
    mat = sparse.hstack(mats).tocsr()
    return symbols, mat


def _build_geneset_features_from_matrix(gene_names, global_symbols, global_mat, target_dim):
    """Align gene-set membership matrix to current gene list and reduce dim."""
    symbol_to_idx = {g: i for i, g in enumerate(global_symbols)}
    rows_idx = [symbol_to_idx.get(g) for g in gene_names]
    valid = np.array([r is not None for r in rows_idx], dtype=bool)
    mapped = [r for r in rows_idx if r is not None]
    if not mapped:
        return None
    sub = global_mat[mapped].toarray().astype(np.float32)
    full = np.zeros((len(gene_names), sub.shape[1]), dtype=np.float32)
    full[valid] = sub
    # Dimensionality reduction to target_dim via truncated SVD
    if full.shape[1] > target_dim:
        svd = TruncatedSVD(n_components=target_dim, random_state=42)
        reduced = svd.fit_transform(sparse.csr_matrix(full))
    else:
        reduced = full
        if reduced.shape[1] < target_dim:
            reduced = np.pad(reduced, ((0, 0), (0, target_dim - reduced.shape[1])))
    return reduced.astype(np.float32)


def _build_hypergraph_propagated_features(gene_names, features, train_mask, labels,
                                           global_symbols, global_mat, target_dim):
    """Build features via normalised hypergraph Laplacian propagation (DISHyper-style)."""
    symbol_to_idx = {g: i for i, g in enumerate(global_symbols)}
    rows_idx = [symbol_to_idx.get(g) for g in gene_names]
    valid = np.array([r is not None for r in rows_idx], dtype=bool)
    mapped = [r for r in rows_idx if r is not None]
    if not mapped:
        return None

    sub = global_mat[mapped].toarray().astype(np.float32)
    h = np.zeros((len(gene_names), sub.shape[1]), dtype=np.float32)
    h[valid] = sub

    # Select informative hyperedges using training positives
    train_pos = np.where(train_mask & (labels.reshape(-1) > 0.5))[0]
    if len(train_pos) > 0:
        pos_sum = h[train_pos].sum(axis=0)
        selected = np.where(pos_sum >= 2)[0]
        if selected.size == 0:
            selected = np.argsort(pos_sum)[-min(256, h.shape[1]):]
    else:
        selected = np.arange(min(256, h.shape[1]))
    h = h[:, selected].copy()
    if h.shape[1] == 0:
        return None

    # Compute edge weights
    edge_weight = h[train_pos].sum(axis=0) if len(train_pos) else h.sum(axis=0)
    edge_weight = np.asarray(edge_weight, dtype=np.float32)
    denom = h.sum(axis=0)
    denom[denom == 0] = 1.0
    edge_weight = edge_weight / denom

    # Build normalised hypergraph Laplacian: Dv^{-1/2} H W De^{-1} H^T Dv^{-1/2}
    dv = np.sum(h * edge_weight, axis=1)
    import random as _random
    for i in range(dv.shape[0]):
        if dv[i] == 0:
            t = _random.randint(0, h.shape[1] - 1)
            h[i, t] = 1e-4
    dv = np.sum(h * edge_weight, axis=1)
    de = np.sum(h, axis=0)
    de[de == 0] = 1.0
    dv[dv == 0] = 1.0

    inv_de = np.diag(1.0 / de)
    dv_inv_sqrt = np.diag(np.power(dv, -0.5))
    wmat = np.diag(edge_weight)
    g_matrix = dv_inv_sqrt @ h @ wmat @ inv_de @ h.T @ dv_inv_sqrt  # (N, N)

    # Propagate features through hypergraph
    propagated = g_matrix @ features  # (N, feat_dim)
    # Reduce to target_dim
    if propagated.shape[1] > target_dim:
        svd = TruncatedSVD(n_components=target_dim, random_state=42)
        propagated = svd.fit_transform(sparse.csr_matrix(propagated))
    elif propagated.shape[1] < target_dim:
        propagated = np.pad(propagated, ((0, 0), (0, target_dim - propagated.shape[1])))
    return propagated.astype(np.float32)


def build_hetero_evidence(features, feature_names, gene_names, train_mask, labels,
                          cancer_name=None, input_dim=None):
    """Construct all evidence channels for heterogeneous data.

    Returns dict with keys: raw_features, pathogenicity_scores, cancer_context,
    gene_set_features, hypergraph_features.  Any channel that cannot be built
    is set to None.
    """
    n_nodes = features.shape[0]
    feat_dim = features.shape[1]
    target_dim = input_dim or feat_dim
    groups = detect_feature_groups(feature_names)
    result = {}

    # --- raw_features: full feature matrix (always available) ---
    result['raw_features'] = features.copy()

    # --- cancer_context: target-cancer-specific columns + cross-cancer statistics ---
    if cancer_name and any(k in groups for k in ('MF', 'CNA', 'GE', 'METH')):
        context_cols = []
        for prefix in ('MF', 'CNA', 'GE', 'METH'):
            if prefix not in groups:
                continue
            for idx in groups[prefix]:
                col_name = feature_names[idx]
                # e.g. 'MF: BRCA' → cancer suffix is 'BRCA'
                suffix = col_name.split(':')[-1].strip()
                if suffix.upper() == cancer_name.upper():
                    context_cols.append(idx)
        if context_cols:
            # Target cancer cols + cross-cancer mean/std per omics type
            parts = [features[:, context_cols]]
            for prefix in ('MF', 'CNA', 'GE', 'METH'):
                if prefix in groups:
                    block = features[:, groups[prefix]]
                    parts.append(block.mean(axis=1, keepdims=True))
                    parts.append(block.std(axis=1, keepdims=True))
            cancer_ctx = np.hstack(parts).astype(np.float32)
            # Pad or reduce to target_dim
            if cancer_ctx.shape[1] < target_dim:
                cancer_ctx = np.pad(cancer_ctx, ((0, 0), (0, target_dim - cancer_ctx.shape[1])))
            elif cancer_ctx.shape[1] > target_dim:
                cancer_ctx = cancer_ctx[:, :target_dim]
            result['cancer_context'] = cancer_ctx
        else:
            result['cancer_context'] = None
    else:
        result['cancer_context'] = None

    # --- pathogenicity_scores: L2 norm proxy ---
    feat_norm = np.linalg.norm(features, axis=1, keepdims=True)
    scale = np.percentile(feat_norm, 95) + 1e-8
    result['pathogenicity_scores'] = np.clip(feat_norm / scale, 0.0, 1.0).astype(np.float32)

    # --- gene_set_features & hypergraph_features from gene-set matrices ---
    global_symbols, global_mat = _load_global_genesets()
    if global_symbols is not None and global_mat is not None:
        result['gene_set_features'] = _build_geneset_features_from_matrix(
            gene_names, global_symbols, global_mat, target_dim)
        result['hypergraph_features'] = _build_hypergraph_propagated_features(
            gene_names, features, train_mask, labels,
            global_symbols, global_mat, target_dim)
    else:
        result['gene_set_features'] = None
        result['hypergraph_features'] = None

    return result


def augment_sparse_graph_knn(features, edge_index, k=10, max_total_edges=None):
    """Add k-NN feature-similarity edges to supplement a sparse biological graph."""
    from sklearn.neighbors import NearestNeighbors
    n_nodes = features.shape[0]
    nn = NearestNeighbors(n_neighbors=min(k + 1, n_nodes), metric='cosine', algorithm='brute')
    nn.fit(features)
    distances, indices = nn.kneighbors(features)

    src_list, dst_list = [], []
    for i in range(n_nodes):
        for j_pos in range(1, indices.shape[1]):  # skip self
            j = indices[i, j_pos]
            src_list.append(i)
            dst_list.append(j)
    knn_src = np.array(src_list, dtype=np.int64)
    knn_dst = np.array(dst_list, dtype=np.int64)
    knn_edge_index = np.vstack([knn_src, knn_dst])

    if edge_index is not None and edge_index.shape[1] > 0:
        combined = np.hstack([edge_index, knn_edge_index])
        # Deduplicate
        edge_set = set()
        dedup_src, dedup_dst = [], []
        for e in range(combined.shape[1]):
            pair = (combined[0, e], combined[1, e])
            if pair not in edge_set:
                edge_set.add(pair)
                dedup_src.append(pair[0])
                dedup_dst.append(pair[1])
        combined = np.vstack([dedup_src, dedup_dst]).astype(np.int64)
    else:
        combined = knn_edge_index

    if max_total_edges and combined.shape[1] > max_total_edges:
        keep = np.random.choice(combined.shape[1], max_total_edges, replace=False)
        combined = combined[:, np.sort(keep)]

    edge_attr = np.ones((combined.shape[1], 1), dtype=np.float32)
    return combined, edge_attr


def build_bpcl_membership(gene_names):
    """Build a binary [N_genes, M_pathways] matrix for BPCL registration.

    Uses the same c2+c5 gene-set matrices as the evidence channels but
    returns the *raw* binary membership (no SVD reduction) so that the
    PathwayConsistencyLoss can identify co-pathway gene sets.
    """
    global_symbols, global_mat = _load_global_genesets()
    if global_symbols is None or global_mat is None:
        return None
    symbol_to_idx = {g: i for i, g in enumerate(global_symbols)}
    rows_idx = [symbol_to_idx.get(g) for g in gene_names]
    valid = np.array([r is not None for r in rows_idx], dtype=bool)
    mapped = [r for r in rows_idx if r is not None]
    if not mapped:
        return None
    sub = global_mat[mapped]                         # sparse [M_mapped, P]
    full = sparse.lil_matrix((len(gene_names), sub.shape[1]), dtype=np.float32)
    full[np.where(valid)[0]] = sub
    return full.tocsr()


def load_gene_sets(npz_path, gene_list_path):
    """Load gene sets from NPZ file."""
    try:
        gene_sets = np.load(npz_path, allow_pickle=True)
        
        # Load gene list
        with open(gene_list_path, 'r') as f:
            gene_list = [line.strip() for line in f]
        
        return gene_sets, gene_list
    except Exception as e:
        print(f"Warning: Could not load gene sets from {npz_path}: {e}")
        return None, None

def load_alphamissense_scores(alphamissense_path):
    """Load AlphaMissense scores."""
    try:
        scores = np.load(alphamissense_path)
        return scores
    except Exception as e:
        print(f"Warning: Could not load AlphaMissense scores from {alphamissense_path}: {e}")
        return None

def create_edge_index_from_features(features, top_k=32, metric='cosine'):
    """Create edge index from feature similarity."""
    from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
    
    n_nodes = features.shape[0]
    
    if metric == 'cosine':
        similarities = cosine_similarity(features)
    elif metric == 'euclidean':
        similarities = -euclidean_distances(features)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Set diagonal to -inf to exclude self-loops
    np.fill_diagonal(similarities, -np.inf)
    
    # Get top-k neighbors for each node
    top_k_indices = np.argsort(similarities, axis=1)[:, -top_k:]
    
    # Create edge list
    edges = []
    for i in range(n_nodes):
        for j in top_k_indices[i]:
            if i != j:  # Exclude self-loops
                edges.append([i, j])
    
    edge_index = np.array(edges).T
    
    return edge_index

def normalize_features(features, method='z-score'):
    """Normalize features."""
    if method == 'z-score':
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True)
        features = (features - mean) / (std + 1e-8)
    elif method == 'min-max':
        min_val = np.min(features, axis=0, keepdims=True)
        max_val = np.max(features, axis=0, keepdims=True)
        features = (features - min_val) / (max_val - min_val + 1e-8)
    elif method == 'l2':
        features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return features

def split_data(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """Split data into train/val/test sets."""
    np.random.seed(random_seed)
    
    n_samples = len(data['labels'])
    indices = np.random.permutation(n_samples)
    
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create masks
    train_mask = np.zeros(n_samples, dtype=bool)
    val_mask = np.zeros(n_samples, dtype=bool)
    test_mask = np.zeros(n_samples, dtype=bool)
    
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    
    return train_mask, val_mask, test_mask

def save_predictions(predictions, gene_names, output_path):
    """Save predictions to TSV file."""
    with open(output_path, 'w') as f:
        f.write("gene\tscore\n")
        for gene, score in zip(gene_names, predictions):
            f.write(f"{gene}\t{score:.6f}\n")

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

def compute_external_metrics(predictions, external_labels, gene_names):
    """Compute external validation metrics."""
    # Align predictions with external labels
    aligned_preds = []
    aligned_labels = []
    
    for gene in gene_names:
        if gene in predictions and gene in external_labels:
            aligned_preds.append(predictions[gene])
            aligned_labels.append(external_labels[gene])
    
    if len(aligned_labels) == 0:
        return {}
    
    aligned_preds = np.array(aligned_preds)
    aligned_labels = np.array(aligned_labels)
    
    metrics = compute_metrics(aligned_labels, aligned_preds)
    
    return metrics

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_device(device_id=0):
    """Get device for computation."""
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{device_id}')
        print(f"Using GPU: {device}")
        print(f"GPU name: {torch.cuda.get_device_name(device_id)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(device_id).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device

def count_parameters(model):
    """Count number of parameters in model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return total_params, trainable_params

def create_directory(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)

def get_timestamp():
    """Get current timestamp."""
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def format_time(seconds):
    """Format time in human readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"
