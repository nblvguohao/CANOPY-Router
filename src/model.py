import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.utils import add_self_loops
import math

class EvidenceRouter(nn.Module):
    """Context-conditioned evidence routing module.

    Routing logits are conditioned on an *external* context vector that is
    **independent** of the evidence-channel representations.  The external
    context is built from raw pathogenicity scores and cancer-type features
    (see ``CANOPYRouter.forward``), ensuring that the router has access to
    orthogonal biological signals when scoring each channel.
    """
    
    def __init__(self, hidden_dim, num_sources=6, temperature=1.0, entropy_reg=0.01,
                 external_dim=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_sources = num_sources
        self.temperature = temperature
        self.entropy_reg = entropy_reg
        
        # External context projection (pathogenicity + cancer-type signals)
        ext_in = external_dim if external_dim is not None else hidden_dim
        self.external_proj = nn.Sequential(
            nn.Linear(ext_in, hidden_dim),
            nn.Tanh(),
        )
        self.source_projs = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_sources)
        ])
        self.source_scorers = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_sources)
        ])
        self.source_bias = nn.Parameter(torch.zeros(num_sources))
        
        # Gated fusion: routing weights gate each channel, then concat+project
        # This eliminates the rank-1 bottleneck of weighted-average while
        # preserving per-source interpretable routing weights.
        self.gated_proj = nn.Sequential(
            nn.Linear(hidden_dim * num_sources, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
    def forward(self, evidence_features, external_context, pathogenicity_gate=None):
        """
        Args:
            evidence_features: List of tensors [N, hidden_dim] per source
            external_context: Tensor [N, ext_dim] — external biological
                signals (pathogenicity + cancer-type), *not* derived from
                evidence channels.
            pathogenicity_gate: unused, kept for API compat
        Returns:
            routed_feature: Tensor [N, hidden_dim]
            routing_weights: Tensor [N, num_sources]
            routing_entropy: Tensor [N, 1]
        """
        # Project external context (independent of channel representations)
        context_hidden = self.external_proj(external_context)

        routing_logits = []
        for idx, feature in enumerate(evidence_features):
            source_hidden = torch.tanh(self.source_projs[idx](feature) + context_hidden)
            source_logit = self.source_scorers[idx](source_hidden).squeeze(-1) + self.source_bias[idx]
            routing_logits.append(source_logit)

        routing_logits = torch.stack(routing_logits, dim=-1)
        routing_weights = F.softmax(routing_logits / self.temperature, dim=-1)
        
        evidence_stack = torch.stack(evidence_features, dim=1)  # [N, num_sources, hidden_dim]
        # Gated fusion: use routing weights as per-channel gates, then concat+project.
        # Each channel is scaled by its routing weight (preserving interpretability),
        # but the subsequent linear projection allows full-rank cross-channel interaction
        # — strictly more expressive than the old weighted-average (rank-1) approach.
        gated = evidence_stack * routing_weights.unsqueeze(-1)   # [N, S, D]
        gated_concat = gated.reshape(gated.size(0), -1)          # [N, S*D]
        routed_feature = self.gated_proj(gated_concat)            # [N, D]
        
        routing_entropy = -torch.sum(routing_weights * torch.log(routing_weights + 1e-8), dim=-1, keepdim=True)
        
        return routed_feature, routing_weights, routing_entropy


class EvidentialRouter(nn.Module):
    """Evidential Uncertainty-Aware Routing (EUAR).

    Replaces softmax routing with evidential deep learning.  Each channel
    scoring network outputs non-negative *evidence* values; Dirichlet
    parameters are alpha_k = evidence_k + 1.  Routing weights are the
    expected probabilities E[p_k] = alpha_k / S  (S = sum of alpha_k).
    Prediction uncertainty is u = K / S.

    Benefits over softmax routing:
      1. Calibrated per-gene uncertainty ("how much does the model know
         about which channel to trust?").
      2. Theoretically motivated from Subjective Logic / Dempster-Shafer
         theory of evidence (Sensoy et al., NeurIPS 2018).
      3. KL regularisation naturally prevents overconfident routing
         without a separate entropy penalty.
    """

    def __init__(self, hidden_dim, num_sources=6, entropy_reg=0.01,
                 external_dim=None, kl_annealing_epochs=10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_sources = num_sources
        self.entropy_reg = entropy_reg
        self.kl_annealing_epochs = kl_annealing_epochs

        ext_in = external_dim if external_dim is not None else hidden_dim
        self.external_proj = nn.Sequential(
            nn.Linear(ext_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )
        # Per-source evidence networks  (output >= 0 via Softplus)
        self.source_evidence_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
                nn.Softplus(),
            )
            for _ in range(num_sources)
        ])

    def forward(self, evidence_features, external_context,
                pathogenicity_gate=None):
        """
        Returns:
            routed_feature  : [N, hidden_dim]
            routing_weights : [N, num_sources]  (Dirichlet expectation)
            routing_entropy : [N, 1]
        Side-effect: stores self._last_alpha, self._last_uncertainty
        for loss computation and logging.
        """
        context_hidden = self.external_proj(external_context)

        evidences = []
        for idx, feature in enumerate(evidence_features):
            combined = torch.cat([feature, context_hidden], dim=-1)
            evidence = self.source_evidence_nets[idx](combined).squeeze(-1)
            evidences.append(evidence)

        evidence_tensor = torch.stack(evidences, dim=-1)       # [N, K]
        alpha = evidence_tensor + 1.0                          # [N, K]
        S = alpha.sum(dim=-1, keepdim=True)                    # [N, 1]

        routing_weights = alpha / S                             # [N, K]
        uncertainty = float(self.num_sources) / S               # [N, 1]

        # Store for external loss computation
        self._last_alpha = alpha
        self._last_uncertainty = uncertainty

        evidence_stack = torch.stack(evidence_features, dim=1)  # [N,K,D]
        routed_feature = torch.sum(
            evidence_stack * routing_weights.unsqueeze(-1), dim=1)

        routing_entropy = -torch.sum(
            routing_weights * torch.log(routing_weights + 1e-8),
            dim=-1, keepdim=True)

        return routed_feature, routing_weights, routing_entropy

    # ------ Evidential KL regularisation ------
    def evidential_kl_loss(self, epoch=0):
        """KL(Dir(alpha) || Dir(1,...,1)) with linear annealing.

        Encourages the routing Dirichlet to stay close to uniform when
        evidence is weak, preventing overconfident routing.
        """
        alpha = self._last_alpha                      # [N, K]
        K = alpha.shape[-1]
        ones = torch.ones_like(alpha)
        S = alpha.sum(dim=-1, keepdim=True)

        kl = (
            torch.lgamma(S.squeeze(-1))
            - torch.lgamma(torch.tensor(float(K), device=alpha.device))
            - torch.sum(torch.lgamma(alpha), dim=-1)
            + torch.sum(torch.lgamma(ones), dim=-1)
            + torch.sum((alpha - ones)
                        * (torch.digamma(alpha) - torch.digamma(S)),
                        dim=-1)
        )

        annealing = min(1.0, epoch / max(self.kl_annealing_epochs, 1))
        return annealing * kl.mean()


class PathwayConsistencyLoss(nn.Module):
    """Biological Pathway Consistency Loss (BPCL).

    Regularises routing weights so that genes sharing the same biological
    pathway exhibit similar routing distributions.

    Loss = (1/|P|) sum_{p in P}  (1/|p|) sum_{i in p} KL(omega_i || mu_p)

    where mu_p is the stop-gradient mean routing distribution over pathway p.
    This provides a biologically-grounded inductive bias: co-pathway genes
    *should* rely on the same evidence types.
    """

    def __init__(self, min_genes_per_pathway=5, max_pathways=500,
                 sample_per_step=100):
        super().__init__()
        self.min_genes = min_genes_per_pathway
        self.max_pathways = max_pathways
        self.sample_per_step = sample_per_step
        # Populated by register_pathways()
        self.pathway_gene_lists = []

    def register_pathways(self, membership_matrix, gene_names=None):
        """Build internal pathway index from a binary membership matrix.

        Args:
            membership_matrix: array-like [N_genes, M_pathways], binary.
        """
        import numpy as np
        if hasattr(membership_matrix, 'toarray'):
            membership_matrix = membership_matrix.toarray()
        membership_matrix = np.asarray(membership_matrix)
        pathways = []
        for j in range(membership_matrix.shape[1]):
            genes = np.where(membership_matrix[:, j] > 0)[0]
            if len(genes) >= self.min_genes:
                pathways.append(torch.LongTensor(genes))
        pathways.sort(key=len, reverse=True)
        if len(pathways) > self.max_pathways:
            pathways = pathways[:self.max_pathways]
        self.pathway_gene_lists = pathways

    def forward(self, routing_weights):
        """Compute BPCL loss.

        Args:
            routing_weights: [N, K]
        Returns:
            scalar loss
        """
        if not self.pathway_gene_lists:
            return torch.tensor(0.0, device=routing_weights.device,
                                requires_grad=True)

        device = routing_weights.device
        N = routing_weights.shape[0]

        # Subsample pathways each step for efficiency
        if len(self.pathway_gene_lists) > self.sample_per_step:
            import random
            sampled = random.sample(self.pathway_gene_lists,
                                   self.sample_per_step)
        else:
            sampled = self.pathway_gene_lists

        total_kl = torch.tensor(0.0, device=device)
        count = 0

        for gene_idx in sampled:
            gene_idx = gene_idx.to(device)
            valid = gene_idx < N
            gene_idx = gene_idx[valid]
            if len(gene_idx) < self.min_genes:
                continue

            pw = routing_weights[gene_idx]                   # [M, K]
            mu_p = pw.mean(dim=0, keepdim=True).detach()     # [1, K]

            # KL(omega_i || mu_p) for each gene
            log_pw = torch.log(pw + 1e-8)
            log_mu = torch.log(mu_p + 1e-8)
            kl = torch.sum(pw * (log_pw - log_mu), dim=-1)   # [M]
            total_kl = total_kl + kl.mean()
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return total_kl / count


class AttentionFusion(nn.Module):
    """Shared implementation for self-attention and conditional-attention baselines."""

    def __init__(self, hidden_dim, num_sources=6, temperature=1.0, conditional=False):
        super().__init__()
        self.temperature = temperature
        self.conditional = conditional
        self.query = nn.Parameter(torch.randn(hidden_dim))
        self.query_proj = nn.Linear(hidden_dim * 2, hidden_dim) if conditional else None
        self.pathogenicity_proj = nn.Linear(1, hidden_dim) if conditional else None
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.source_bias = nn.Parameter(torch.zeros(num_sources))

    def forward(self, evidence_features, gene_context, pathogenicity_gate=None):
        evidence_stack = torch.stack(evidence_features, dim=1)
        keys = self.key_proj(evidence_stack)
        values = self.value_proj(evidence_stack)

        if self.conditional:
            # Use mean + max pooling across sources for richer query conditioning
            # (previously both halves were identical mean-pooling — a bug)
            mean_pool = torch.mean(evidence_stack, dim=1)   # [N, D]
            max_pool = torch.max(evidence_stack, dim=1)[0]  # [N, D]
            conditional_input = torch.cat([mean_pool, max_pool], dim=-1)
            query = self.query_proj(conditional_input)
            if pathogenicity_gate is not None:
                query = query + self.pathogenicity_proj(pathogenicity_gate)
        else:
            query = self.query.unsqueeze(0).expand(gene_context.size(0), -1)

        scores = torch.sum(keys * query.unsqueeze(1), dim=-1) + self.source_bias.unsqueeze(0)
        weights = F.softmax(scores / self.temperature, dim=-1)
        fused_feature = torch.sum(values * weights.unsqueeze(-1), dim=1)
        entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=-1, keepdim=True)
        return fused_feature, weights, entropy

class MultiViewGraphAttention(nn.Module):
    """Multi-view graph attention module for structural evidence.

    Enhanced with residual connections, LayerNorm and dropout for
    better gradient flow on sparse heterogeneous graphs.
    """
    
    def __init__(self, input_dim, hidden_dim, num_heads=8, top_k=256, max_edges=1000000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.top_k = top_k
        self.max_edges = max_edges
        self.use_residual = True  # set False for dense homogeneous graphs
        
        # Input projection for residual when input_dim != hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        
        # Graph attention layers
        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, concat=False)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x, edge_index, edge_attr=None):
        if self.use_residual:
            # Enhanced path: residual + LayerNorm (for sparse hetero graphs)
            residual = self.input_proj(x)
            h = self.gat1(x, edge_index, edge_attr)
            h = F.elu(h)
            h = self.dropout(h)
            h = self.norm1(h + residual)
            residual2 = h
            h = self.gat2(h, edge_index, edge_attr)
            h = F.elu(h)
            h = self.dropout(h)
            h = self.norm2(h + residual2)
        else:
            # Classic path: simple GAT (for dense homogeneous graphs)
            h = self.gat1(x, edge_index, edge_attr)
            h = F.elu(h)
            h = self.gat2(h, edge_index, edge_attr)
            h = F.elu(h)
        
        # Output projection
        h = self.output_proj(h)
        
        return h


class CrossEvidenceAttention(nn.Module):
    """Cross-attention between evidence sources before routing.

    Lets each evidence channel attend to all others, enriching
    representations before the router makes its decision.  Inspired by
    multi-view cross-attention in HGT (Hu et al., WWW 2020) and
    SeHGNN (Yang et al., KDD 2023).
    """

    def __init__(self, hidden_dim, num_sources=6, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_sources = num_sources
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, evidence_list):
        """evidence_list: list of [N, D] tensors, one per source."""
        # Stack → [N, S, D]  then treat S as sequence length
        x = torch.stack(evidence_list, dim=1)
        N, S, D = x.shape
        # Self-attention across sources for every node
        residual = x
        x_flat = x.reshape(N, S, D)
        attn_out, _ = self.mha(x_flat, x_flat, x_flat)
        x = self.norm(attn_out + residual)
        residual2 = x
        x = self.norm2(self.ffn(x) + residual2)
        # Unstack back to list
        return [x[:, i, :] for i in range(S)]

class PathogenicityGate(nn.Module):
    """Pathogenicity-modulated feature gating."""
    
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(input_dim + 1, hidden_dim)
        self.gate_output = nn.Linear(hidden_dim, 1)
        
    def forward(self, features, pathogenicity_scores):
        """
        Args:
            features: Tensor [batch_size, input_dim]
            pathogenicity_scores: Tensor [batch_size, 1]
        Returns:
            gated_features: Tensor [batch_size, input_dim]
            gate_coefficients: Tensor [batch_size, 1]
        """
        # Concatenate features with pathogenicity scores
        gate_input = torch.cat([features, pathogenicity_scores], dim=-1)
        
        # Compute gate coefficients
        gate_hidden = F.relu(self.gate_proj(gate_input))
        gate_coefficients = torch.sigmoid(self.gate_output(gate_hidden))
        
        # Apply gating
        gated_features = features * gate_coefficients
        
        return gated_features, gate_coefficients

class CANOPYRouter(nn.Module):
    """Complete CANOPY-Router model with evidence routing."""
    
    def __init__(self, input_dim, hidden_dim=256, num_heads=8, top_k=256, max_edges=1000000, 
                 num_sources=6, temperature=1.0, entropy_reg=0.01, num_layers=3,
                 fusion_mode='router'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_sources = num_sources
        self.num_layers = num_layers
        self.fusion_mode = fusion_mode
        self.skip_cross_evidence = False  # set True for homogeneous data
        
        # Input projections for different evidence sources
        self.structure_proj = nn.Identity()
        self.raw_proj = nn.Linear(input_dim, hidden_dim)
        self.pathogenicity_proj = nn.Linear(input_dim, hidden_dim)
        self.cancer_context_proj = nn.Linear(input_dim, hidden_dim)
        self.gene_set_proj = nn.Linear(input_dim, hidden_dim)
        self.hypergraph_proj = nn.Linear(input_dim, hidden_dim)
        
        # Evidence processing modules
        self.mvga = MultiViewGraphAttention(input_dim, hidden_dim, num_heads, top_k, max_edges)
        self.pathogenicity_gate = PathogenicityGate(input_dim, hidden_dim)
        
        # External context projection for router conditioning.
        # Maps raw pathogenicity (1-d) + cancer-context (input_dim) to hidden_dim.
        # These signals are *independent* of the five evidence channels.
        self.external_context_proj = nn.Linear(1 + input_dim, hidden_dim)
        
        # Cross-evidence attention (enriches source representations before routing)
        self.cross_evidence_attn = CrossEvidenceAttention(hidden_dim, num_sources, num_heads=4)
        
        # Evidence fusion modules
        self.router = EvidenceRouter(hidden_dim, num_sources, temperature, entropy_reg,
                                     external_dim=hidden_dim)
        self.evidential_router = EvidentialRouter(
            hidden_dim, num_sources, entropy_reg,
            external_dim=hidden_dim, kl_annealing_epochs=10)
        self.attention_fusion = AttentionFusion(hidden_dim, num_sources, temperature, conditional=False)
        self.conditional_attention_fusion = AttentionFusion(hidden_dim, num_sources, temperature, conditional=True)
        
        # Biological Pathway Consistency Loss module
        self.bpcl = PathwayConsistencyLoss()
        
        # Transformer layers for processing routed features
        # Stored as a ModuleList so we can apply gradient checkpointing per-layer,
        # which cuts peak memory from O(N²×L) to O(N²×1) – critical for BRCA (N=7962).
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        
        # Output layers
        self.no_routing_proj = nn.Linear(hidden_dim * num_sources, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.final_classifier = nn.Linear(hidden_dim, 1)
        
        # Disable routing flag
        self.disable_routing = False
        self.disable_pathogenicity_source = False
        self.disable_hypergraph_channel = False    # ablation: zero out Ch.6 HyperConv
        self.disable_gene_set_channel = False       # ablation: zero out Ch.5 Gene-Set
        # External context ablation mode: 'both', 'patho_only', 'cancer_only', 'none'
        self.external_context_mode = 'both'
        
    def forward(self, x, edge_index, edge_attr=None, batch=None, 
                raw_features=None, pathogenicity_scores=None, cancer_context=None,
                gene_set_features=None, hypergraph_features=None, 
                return_aux=False):
        """
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_dim] (optional)
            batch: Batch indices [num_nodes] (optional)
            raw_features: Raw multi-omics features [num_nodes, input_dim]
            pathogenicity_scores: Pathogenicity scores [num_nodes, 1]
            cancer_context: Cancer-specific context [num_nodes, input_dim]
            gene_set_features: Gene-set embeddings [num_nodes, input_dim]
            hypergraph_features: Hypergraph features [num_nodes, input_dim]
            return_aux: Whether to return auxiliary information
        Returns:
            predictions: Tensor [num_nodes, 1] or [batch_size, 1]
            aux: Dictionary with auxiliary information if return_aux=True
        """
        aux = {}
        num_nodes = x.size(0)
        
        # ---- Build external context (independent of evidence channels) ----
        # Pathogenicity scores: raw scalar per gene
        if pathogenicity_scores is not None:
            patho_raw = pathogenicity_scores  # [N, 1]
        else:
            patho_raw = torch.zeros(num_nodes, 1, device=x.device)
        # Cancer-context: raw cancer-type features
        if cancer_context is not None:
            ctx_raw = cancer_context  # [N, input_dim]
        else:
            ctx_raw = torch.zeros(num_nodes, self.input_dim, device=x.device)
        # External context ablation: zero-out components per mode
        if self.external_context_mode == 'patho_only':
            ctx_raw = torch.zeros_like(ctx_raw)
        elif self.external_context_mode == 'cancer_only':
            patho_raw = torch.zeros_like(patho_raw)
        elif self.external_context_mode == 'none':
            patho_raw = torch.zeros_like(patho_raw)
            ctx_raw = torch.zeros_like(ctx_raw)
        # Concatenate and project → external conditioning signal for the router
        external_ctx = self.external_context_proj(
            torch.cat([patho_raw, ctx_raw], dim=-1))  # [N, hidden_dim]
        
        # Process structural evidence
        structure_features = self.mvga(x, edge_index, edge_attr)
        
        # Process raw features
        if raw_features is not None:
            raw_features_h = self.raw_proj(raw_features)
        else:
            raw_features_h = torch.zeros_like(structure_features)
            
        # Process pathogenicity-modulated features
        if (not self.disable_pathogenicity_source) and pathogenicity_scores is not None:
            pathogenicity_features, gate_alpha = self.pathogenicity_gate(x, pathogenicity_scores)
            pathogenicity_features = self.pathogenicity_proj(pathogenicity_features)
            aux['pathogenicity_gate_alpha'] = gate_alpha
        else:
            pathogenicity_features = torch.zeros_like(structure_features)
            aux['pathogenicity_gate_alpha'] = torch.zeros(num_nodes, 1, device=x.device)
            
        # Process cancer context (projected into channel space)
        if cancer_context is not None:
            cancer_context_h = self.cancer_context_proj(cancer_context)
        else:
            cancer_context_h = torch.zeros_like(structure_features)
            
        # Process gene-set features
        if gene_set_features is not None and not self.disable_gene_set_channel:
            gene_set_features_h = self.gene_set_proj(gene_set_features)
        else:
            gene_set_features_h = torch.zeros_like(structure_features)
            
        # Process hypergraph features
        if hypergraph_features is not None and not self.disable_hypergraph_channel:
            hypergraph_features_h = self.hypergraph_proj(hypergraph_features)
        else:
            hypergraph_features_h = torch.zeros_like(structure_features)
        
        # Collect all evidence sources
        evidence_features = [
            structure_features,
            raw_features_h,
            pathogenicity_features,
            cancer_context_h,
            gene_set_features_h,
            hypergraph_features_h
        ]
        
        # Cross-evidence attention: let sources interact before routing
        # Skipped when most channels are zero (homogeneous data) to prevent routing collapse
        if not self.skip_cross_evidence:
            evidence_features = self.cross_evidence_attn(evidence_features)
        
        # gene_context for attention baselines (channel-derived, used only by attn/cond_attn)
        gene_context = torch.mean(torch.stack(evidence_features, dim=0), dim=0)
        
        # Apply evidence fusion
        if self.fusion_mode == 'evidential' and not self.disable_routing:
            # EUAR: evidential uncertainty-aware routing
            routed_feature, routing_weights, routing_entropy = self.evidential_router(
                evidence_features, external_ctx, None
            )
            aux['routing_weights'] = routing_weights
            aux['routing_entropy'] = routing_entropy
            aux['evidential_alpha'] = self.evidential_router._last_alpha
            aux['evidential_uncertainty'] = self.evidential_router._last_uncertainty
        elif self.fusion_mode == 'router' and not self.disable_routing:
            # Original softmax router
            routed_feature, routing_weights, routing_entropy = self.router(
                evidence_features, external_ctx, None
            )
            aux['routing_weights'] = routing_weights
            aux['routing_entropy'] = routing_entropy
        elif self.fusion_mode == 'attn':
            routed_feature, routing_weights, routing_entropy = self.attention_fusion(
                evidence_features, gene_context, None
            )
            aux['routing_weights'] = routing_weights
            aux['routing_entropy'] = routing_entropy
        elif self.fusion_mode == 'cond_attn':
            routed_feature, routing_weights, routing_entropy = self.conditional_attention_fusion(
                evidence_features, gene_context, aux['pathogenicity_gate_alpha']
            )
            aux['routing_weights'] = routing_weights
            aux['routing_entropy'] = routing_entropy
        else:
            routed_feature = torch.cat(evidence_features, dim=-1)
            routed_feature = self.no_routing_proj(routed_feature)
            aux['routing_weights'] = torch.zeros(num_nodes, self.num_sources, device=x.device)
            aux['routing_entropy'] = torch.zeros(num_nodes, 1, device=x.device)
        
        # Process through transformer layers with per-layer gradient checkpointing.
        # batch_first=True: input shape is [1, num_nodes, hidden_dim].
        # Checkpointing recomputes each layer's activations during backward instead of
        # storing them, reducing peak memory from O(N²×L) to O(N²) – this is the fix
        # that allows BRCA (N=7962) to run without CUDA out-of-memory errors.
        x_t = routed_feature.unsqueeze(0)  # [1, num_nodes, hidden_dim]
        use_ckpt = self.training
        for layer in self.transformer_layers:
            if use_ckpt:
                x_t = checkpoint(layer, x_t, use_reentrant=False)
            else:
                x_t = layer(x_t)
        transformer_output = x_t.squeeze(0)  # [num_nodes, hidden_dim]
        
        # Final projection and classification
        output_features = self.output_proj(transformer_output)
        predictions = torch.sigmoid(self.final_classifier(output_features))
        
        # Store routing source names for interpretability
        aux['routing_source_names'] = [
            'structure', 'raw', 'pathogenicity', 'cancer_context', 'gene_set', 'hypergraph'
        ]
        
        # Store hidden features for SupCL
        aux['hidden_features'] = output_features
        
        if return_aux:
            return predictions, aux
        else:
            return predictions
