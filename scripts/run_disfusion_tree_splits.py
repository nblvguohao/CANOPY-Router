#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from torch.nn.parameter import Parameter
from torch_geometric.nn import ChebConv


class HGNNConv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super().__init__()
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, g):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        return g.matmul(x)


class HypergraphHGNN(nn.Module):
    def __init__(self, in_ch, n_hid, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.fc = nn.Linear(in_ch, n_hid)
        self.hgc1 = HGNNConv(n_hid, n_hid)
        self.hgc2 = HGNNConv(n_hid, n_hid)
        self.hgc3 = HGNNConv(n_hid, n_hid)

    def forward(self, x, g):
        x1 = F.relu(self.fc(x))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.relu(self.hgc1(x1, g) + x1)
        x2 = F.dropout(x2, self.dropout, training=self.training)
        x3 = F.relu(self.hgc2(x2, g) + x2)
        x3 = F.dropout(x3, self.dropout, training=self.training)
        x4 = F.relu(self.hgc3(x3, g) + x3)
        return x4


class GraphChebNet(nn.Module):
    def __init__(self, in_dim, hdim=256, dropout=0.5):
        super().__init__()
        self.conv1 = ChebConv(in_dim, hdim, K=2)
        self.conv2 = ChebConv(hdim, hdim, K=2)
        self.conv3 = ChebConv(hdim, hdim, K=2)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.relu(self.conv1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.relu(self.conv2(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return x


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes, dropout):
        super().__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        self.dropout = dropout
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, seq):
        seq = F.dropout(seq, self.dropout, training=self.training)
        ret = self.fc(seq)
        return F.log_softmax(ret, dim=1)


class DISFusion(nn.Module):
    def __init__(self, input_dim, lambdinter, attention, dropout=0.5):
        super().__init__()
        self.lambdinter = lambdinter
        self.attention = attention
        self.dropout = dropout
        self.w_list = nn.ModuleList([nn.Linear(input_dim, input_dim, bias=True) for _ in range(2)])
        self.y_list = nn.ModuleList([nn.Linear(input_dim, 1) for _ in range(2)])
        self.att_act1 = nn.Tanh()
        self.att_act2 = nn.Softmax(dim=-1)
        self.logistic = LogReg(input_dim, 2, self.dropout)
        self.concat_fc = nn.Linear(input_dim * 2, input_dim)

    def combine_att(self, input1, input2):
        h_list = [input1, input2]
        h_combine_list = []
        for i, h in enumerate(h_list):
            h = self.w_list[i](h)
            h = self.y_list[i](h)
            h_combine_list.append(h)
        score = torch.cat(h_combine_list, -1)
        score = self.att_act1(score)
        score = self.att_act2(score)
        score = torch.unsqueeze(score, -1)
        h = torch.stack(h_list, dim=1)
        h = score * h
        return torch.sum(h, dim=1)

    def combine_concat(self, input1, input2):
        x = torch.cat([input1, input2], 1)
        return F.relu(self.concat_fc(x))

    def forward(self, input1, input2):
        if self.attention:
            h_fusion = self.combine_att(input1, input2)
        else:
            h_fusion = self.combine_concat(input1, input2)
        semi = self.logistic(h_fusion)
        eps = 1e-15
        batch_size = input1.size(0)
        input1n = (input1 - input1.mean(dim=0)) / (input1.std(dim=0) + eps)
        input2n = (input2 - input2.mean(dim=0)) / (input2.std(dim=0) + eps)
        inter_c = input1n.T @ input2n / batch_size
        on_diag = torch.diagonal(inter_c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(inter_c).pow_(2).sum()
        loss_inter = on_diag + self.lambdinter * off_diag
        return loss_inter, semi


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_global_genesets(data_dir):
    gene_df = pd.read_csv(Path(data_dir) / 'geneList.txt', header=None)
    symbols = gene_df.iloc[:, 1].astype(str).tolist()
    c2 = sp.load_npz(Path(data_dir) / 'c2_GenesetsMatrix.npz')
    c5 = sp.load_npz(Path(data_dir) / 'c5_GenesetsMatrix.npz')
    mat = sp.hstack([c2, c5]).tocsr()
    return symbols, mat


def load_h5_dataset(h5_path):
    with h5py.File(h5_path, 'r') as f:
        features = f['features'][:].astype(np.float32)
        network = f['network'][:].astype(np.float32)
        gene_names = [n.decode() if isinstance(n, bytes) else str(n) for n in f['gene_names'][:]]
        y = np.zeros((features.shape[0],), dtype=np.float32)
        if 'y_train' in f:
            y += f['y_train'][:].reshape(-1).astype(np.float32)
        if 'y_val' in f:
            y += f['y_val'][:].reshape(-1).astype(np.float32)
        if 'y_test' in f:
            y += f['y_test'][:].reshape(-1).astype(np.float32)
        train_mask = f['mask_train'][:].astype(bool)
        val_mask = f['mask_val'][:].astype(bool)
        test_mask = f['mask_test'][:].astype(bool)
    return features, network, gene_names, y, train_mask, val_mask, test_mask


def build_edge_index(network, top_k=256):
    density = float(np.count_nonzero(network)) / float(network.size)
    if density > 0.1:
        src_list = []
        dst_list = []
        num_nodes = network.shape[0]
        for node_idx in range(num_nodes):
            row = network[node_idx]
            neighbors = np.flatnonzero(row)
            neighbors = neighbors[neighbors != node_idx]
            if neighbors.size == 0:
                continue
            if neighbors.size > top_k:
                weights = row[neighbors]
                top_idx = np.argsort(weights)[-top_k:]
                neighbors = neighbors[top_idx]
            src_list.append(np.full(neighbors.shape[0], node_idx, dtype=np.int64))
            dst_list.append(neighbors.astype(np.int64, copy=False))
        if not src_list:
            return np.zeros((2, 0), dtype=np.int64)
        src = np.concatenate(src_list)
        dst = np.concatenate(dst_list)
        return np.vstack([src, dst]).astype(np.int64)
    src, dst = np.nonzero(network)
    keep = src != dst
    src = src[keep]
    dst = dst[keep]
    return np.vstack([src, dst]).astype(np.int64)


def build_hypergraph_for_genes(gene_names, train_mask, labels, global_symbols, global_mat):
    symbol_to_idx = {g: i for i, g in enumerate(global_symbols)}
    rows = []
    valid_mask = []
    for g in gene_names:
        idx = symbol_to_idx.get(g)
        if idx is None:
            rows.append(None)
            valid_mask.append(False)
        else:
            rows.append(idx)
            valid_mask.append(True)
    valid_mask = np.array(valid_mask, dtype=bool)
    mapped_rows = [r for r in rows if r is not None]
    if not mapped_rows:
        raise RuntimeError('No gene overlap with DISFusion gene-set universe.')
    sub = global_mat[mapped_rows].toarray().astype(np.float32)
    full = np.zeros((len(gene_names), sub.shape[1]), dtype=np.float32)
    full[valid_mask] = sub
    train_pos = np.where(train_mask & (labels > 0.5))[0]
    if len(train_pos) == 0:
        selected = np.arange(full.shape[1])
    else:
        pos_sum = full[train_pos].sum(axis=0)
        selected = np.where(pos_sum >= 3)[0]
        if selected.size == 0:
            selected = np.argsort(pos_sum)[-min(128, full.shape[1]):]
    h = full[:, selected].copy()
    if h.shape[1] == 0:
        h = full[:, :min(128, full.shape[1])].copy()
    edge_weight = h[train_pos].sum(axis=0) if len(train_pos) else h.sum(axis=0)
    edge_weight = np.asarray(edge_weight, dtype=np.float32)
    denom = h.sum(axis=0)
    denom[denom == 0] = 1.0
    edge_weight = edge_weight / denom
    dv = np.sum(h * edge_weight, axis=1)
    if h.shape[1] == 0:
        raise RuntimeError('Empty hypergraph after selection.')
    for i in range(dv.shape[0]):
        if dv[i] == 0:
            t = random.randint(0, h.shape[1] - 1)
            h[i, t] = 1e-4
    return generate_g_from_h_weight(h, edge_weight)


def generate_g_from_h_weight(h, w):
    dv = np.sum(h * w, axis=1)
    de = np.sum(h, axis=0)
    de[de == 0] = 1.0
    dv[dv == 0] = 1.0
    inv_de = np.diag(1.0 / de)
    dv2 = np.diag(np.power(dv, -0.5))
    wmat = np.diag(w)
    return dv2 @ h @ wmat @ inv_de @ h.T @ dv2


def compute_metrics_from_log_probs(log_probs, labels):
    probs = torch.exp(log_probs)[:, 1].detach().cpu().numpy()
    labels = labels.detach().cpu().numpy().astype(int)
    auroc = roc_auc_score(labels, probs)
    aupr = average_precision_score(labels, probs)
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(precision), where=denom > 0)
    f1_max = float(np.max(f1)) if len(f1) else 0.0
    return auroc, aupr, f1_max


def train_one_split(h5_path, global_symbols, global_mat, seed, args):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    features, network, gene_names, labels_np, train_mask_np, val_mask_np, test_mask_np = load_h5_dataset(h5_path)
    edge_index = torch.from_numpy(build_edge_index(network)).long().to(device)
    hyper_g = torch.tensor(build_hypergraph_for_genes(gene_names, train_mask_np, labels_np, global_symbols, global_mat), dtype=torch.float32, device=device)
    x_graph = torch.tensor(features, dtype=torch.float32, device=device)
    x_hyper = torch.eye(features.shape[0], dtype=torch.float32, device=device)
    y = torch.tensor(labels_np, dtype=torch.long, device=device)
    train_mask = torch.tensor(train_mask_np, dtype=torch.bool, device=device)
    val_mask = torch.tensor(val_mask_np, dtype=torch.bool, device=device)
    test_mask = torch.tensor(test_mask_np, dtype=torch.bool, device=device)

    graph_model = GraphChebNet(in_dim=features.shape[1], hdim=args.hidden_dim, dropout=args.dropout).to(device)
    hyper_model = HypergraphHGNN(in_ch=features.shape[0], n_hid=args.hidden_dim, dropout=args.dropout).to(device)
    fusion_model = DISFusion(input_dim=args.hidden_dim, lambdinter=args.lambdinter, attention=args.attention, dropout=args.dropout).to(device)

    params = list(graph_model.parameters()) + list(hyper_model.parameters()) + list(fusion_model.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    class_counts = np.bincount(labels_np[train_mask_np].astype(int), minlength=2)
    pos_weight = float(class_counts[0] / max(class_counts[1], 1))
    weight = torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device)

    best = None
    best_state = None
    patience = 0
    for epoch in range(args.epochs):
        graph_model.train()
        hyper_model.train()
        fusion_model.train()
        optimizer.zero_grad()
        out_hyper = hyper_model(x_hyper, hyper_g)
        out_graph = graph_model(x_graph, edge_index)
        loss_inter, logits = fusion_model(out_hyper, out_graph)
        loss_cls = F.nll_loss(logits[train_mask], y[train_mask], weight=weight)
        loss = loss_cls + args.alpha_inter * loss_inter
        loss.backward()
        optimizer.step()

        graph_model.eval()
        hyper_model.eval()
        fusion_model.eval()
        with torch.no_grad():
            out_hyper = hyper_model(x_hyper, hyper_g)
            out_graph = graph_model(x_graph, edge_index)
            _, logits = fusion_model(out_hyper, out_graph)
            _, val_aupr, _ = compute_metrics_from_log_probs(logits[val_mask], y[val_mask])
        if best is None or val_aupr > best:
            best = val_aupr
            best_state = {
                'graph': {k: v.detach().cpu() for k, v in graph_model.state_dict().items()},
                'hyper': {k: v.detach().cpu() for k, v in hyper_model.state_dict().items()},
                'fusion': {k: v.detach().cpu() for k, v in fusion_model.state_dict().items()},
            }
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break

    graph_model.load_state_dict(best_state['graph'])
    hyper_model.load_state_dict(best_state['hyper'])
    fusion_model.load_state_dict(best_state['fusion'])
    graph_model.to(device)
    hyper_model.to(device)
    fusion_model.to(device)
    graph_model.eval()
    hyper_model.eval()
    fusion_model.eval()
    with torch.no_grad():
        out_hyper = hyper_model(x_hyper, hyper_g)
        out_graph = graph_model(x_graph, edge_index)
        _, logits = fusion_model(out_hyper, out_graph)
        auroc, aupr, f1_max = compute_metrics_from_log_probs(logits[test_mask], y[test_mask])
    return {'auroc': auroc, 'aupr': aupr, 'f1_max': f1_max}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='/data/lgh/CANOPYNet-main/data/cancer_specific/homogeneous')
    parser.add_argument('--disfusion-data', default='/data/lgh/CANOPYNet-main/external/research_repos_clean/research_repos/GRAFT/baseline/DISFusion/Data')
    parser.add_argument('--cancers', nargs='+', default=['BRCA', 'COAD', 'LIHC', 'LUAD', 'STAD', 'UCEC'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[1234, 2345, 3456])
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--lambdinter', type=float, default=1e-4)
    parser.add_argument('--alpha-inter', type=float, default=0.1)
    parser.add_argument('--attention', type=int, default=0)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--output-json', default='/data/lgh/CANOPYNet-main/paper/disfusion_tree_results.json')
    args = parser.parse_args()

    global_symbols, global_mat = load_global_genesets(args.disfusion_data)
    results = {}
    macro = {'auroc': [], 'aupr': [], 'f1_max': []}
    for cancer in args.cancers:
        h5_path = str(Path(args.data_root) / f'{cancer}_multiomics.h5')
        cancer_runs = []
        for seed in args.seeds:
            metrics = train_one_split(h5_path, global_symbols, global_mat, seed, args)
            metrics['seed'] = seed
            cancer_runs.append(metrics)
            print(cancer, seed, json.dumps(metrics))
        results[cancer] = cancer_runs
        for key in macro:
            macro[key].append(float(np.mean([r[key] for r in cancer_runs])))
    results['macro_average'] = {k: float(np.mean(v)) for k, v in macro.items()}
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print('MACRO', json.dumps(results['macro_average']))


if __name__ == '__main__':
    main()
