import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class EdgeAttentionBias(nn.Module):
    """Edge-aware attention bias encoder adapted from GRAFT EdgeImportanceEncoder.

    This variant uses a single homogeneous graph and scalar edge scores.
    """

    def __init__(self, num_heads: int):
        super().__init__()
        self.linear = nn.Linear(3, num_heads)

    def forward(self, edge_index: torch.Tensor, edge_score: torch.Tensor, num_nodes: int) -> torch.Tensor:
        device = edge_index.device
        row, col = edge_index[0], edge_index[1]
        features = torch.stack([row.float(), col.float(), edge_score.float()], dim=-1)
        bias_per_edge = torch.sigmoid(self.linear(features))  # [E, H]

        # [H, N, N], dense bias matrix used as additive attention mask per head.
        edge_bias = torch.zeros((self.linear.out_features, num_nodes, num_nodes), device=device)
        for h in range(self.linear.out_features):
            edge_bias[h, row, col] = edge_bias[h, row, col] + bias_per_edge[:, h]
        return edge_bias


class GraphAwareFusionTransformerLayer(nn.Module):
    """Single GRAFT-style transformer layer with edge-aware bias."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor = None) -> torch.Tensor:
        residual = x
        attn_output, _ = self.attn(x, x, x, attn_mask=attn_bias)
        x = self.norm1(residual + self.dropout(attn_output))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class GRAFTAdapter(nn.Module):
    """GRAFT core module adapted to CANOPYNet homogeneous TREE pipeline.

    Forward API is aligned with pipeline usage while supporting a minimal
    clean interface: forward(edge_index, node_features).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
        self.fusion_mode = "graft"
        self.disable_routing = True

        # Graph encoder adapted from GRAFT GNN backbone to single homogeneous graph.
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)

        self.edge_encoder = EdgeAttentionBias(num_heads=num_heads)
        self.input_proj = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [GraphAwareFusionTransformerLayer(hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.out_proj = nn.Linear(hidden_dim, 1)

    def _forward_impl(self, edge_index: torch.Tensor, node_features: torch.Tensor):
        if edge_index is None:
            raise ValueError("GRAFTAdapter requires edge_index for homogeneous PPI graph.")

        num_nodes = node_features.size(0)
        edge_score = torch.ones(edge_index.size(1), device=node_features.device, dtype=node_features.dtype)

        gnn_hidden = F.relu(self.gcn1(node_features, edge_index))
        gnn_feat = self.gcn2(gnn_hidden, edge_index)

        x = torch.cat([node_features, gnn_feat], dim=-1)
        x = self.input_proj(x).unsqueeze(0)  # [1, N, D]

        edge_bias = self.edge_encoder(edge_index=edge_index, edge_score=edge_score, num_nodes=num_nodes)
        # MultiheadAttention expects [B*H, T, S] for 3D attn_mask when batch_first=True.
        attn_bias = edge_bias.unsqueeze(1)  # [H, 1, N, N]
        attn_bias = attn_bias.reshape(edge_bias.size(0), num_nodes, num_nodes)

        for layer in self.layers:
            x = layer(x, attn_bias=attn_bias)

        logits = self.out_proj(x.squeeze(0))
        probs = torch.sigmoid(logits)

        aux = {
            "routing_weights": torch.zeros(num_nodes, 1, device=node_features.device, dtype=node_features.dtype),
            "routing_entropy": torch.zeros(num_nodes, 1, device=node_features.device, dtype=node_features.dtype),
            "routing_source_names": ["graft_single_graph"],
        }
        return probs, aux

    def forward(self, *args, edge_attr=None, return_aux=False, **kwargs):
        """Supports both signatures:
        1) forward(edge_index, node_features)
        2) pipeline call: forward(node_features, edge_index, ...)
        """
        if len(args) < 2:
            raise ValueError("GRAFTAdapter.forward expects at least two positional args")

        first, second = args[0], args[1]
        if first.dim() == 2 and first.size(0) == 2:
            edge_index, node_features = first, second
        else:
            node_features, edge_index = first, second

        probs, aux = self._forward_impl(edge_index=edge_index, node_features=node_features)
        if return_aux:
            return probs, aux
        return probs


__all__ = ["GRAFTAdapter"]
