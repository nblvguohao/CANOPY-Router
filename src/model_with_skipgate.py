import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.utils import add_self_loops
import math

class EvidenceRouter(nn.Module):
    """Reliability-driven evidence routing module."""
    
    def __init__(self, hidden_dim, num_sources=6, temperature=1.0, entropy_reg=0.01):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_sources = num_sources
        self.temperature = temperature
        self.entropy_reg = entropy_reg
        
        # Source-specific embeddings
        self.source_embeddings = nn.Embedding(num_sources, hidden_dim)
        
        # Routing projection layers
        self.routing_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.routing_output = nn.Linear(hidden_dim, num_sources)
        
        # Structural prior (optional)
        self.structural_prior = nn.Parameter(torch.ones(num_sources) / num_sources)
        
    def forward(self, evidence_features, gene_context, pathogenicity_gate=None):
        """
        Args:
            evidence_features: List of tensors [batch_size, hidden_dim] for each evidence source
            gene_context: Tensor [batch_size, hidden_dim] 
            pathogenicity_gate: Optional tensor [batch_size, 1] for pathogenicity modulation
        Returns:
            routed_feature: Tensor [batch_size, hidden_dim]
            routing_weights: Tensor [batch_size, num_sources]
            routing_entropy: Tensor [batch_size, 1]
        """
        batch_size = gene_context.size(0)
        
        # Compute routing logits
        routing_input = torch.cat([gene_context, gene_context], dim=-1)
        routing_hidden = F.relu(self.routing_proj(routing_input))
        routing_logits = self.routing_output(routing_hidden)
        
        # Add source embeddings
        source_ids = torch.arange(self.num_sources, device=gene_context.device)
        source_emb = self.source_embeddings(source_ids).unsqueeze(0)  # [1, num_sources, hidden_dim]
        
        # Compute attention between gene context and source embeddings
        gene_expanded = gene_context.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        attention_scores = torch.matmul(gene_expanded, source_emb.transpose(-2, -1)).squeeze(1)  # [batch_size, num_sources]
        
        # Combine routing logits with attention and structural prior
        routing_logits = routing_logits + attention_scores + self.structural_prior.unsqueeze(0)
        
        # Modulate by pathogenicity if provided
        if pathogenicity_gate is not None:
            routing_logits = routing_logits * pathogenicity_gate
        
        # Apply temperature scaling and softmax
        routing_weights = F.softmax(routing_logits / self.temperature, dim=-1)
        
        # Compute routed feature as weighted sum
        evidence_stack = torch.stack(evidence_features, dim=1)  # [batch_size, num_sources, hidden_dim]
        routed_feature = torch.sum(evidence_stack * routing_weights.unsqueeze(-1), dim=1)
        
        # Compute routing entropy for regularization
        routing_entropy = -torch.sum(routing_weights * torch.log(routing_weights + 1e-8), dim=-1, keepdim=True)
        
        return routed_feature, routing_weights, routing_entropy

class MultiViewGraphAttention(nn.Module):
    """Multi-view graph attention module for structural evidence."""
    
    def __init__(self, input_dim, hidden_dim, num_heads=8, top_k=256, max_edges=1000000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.top_k = top_k
        self.max_edges = max_edges
        
        # Graph attention layers
        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, concat=False)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x, edge_index, edge_attr=None):
        # First GAT layer
        x = self.gat1(x, edge_index, edge_attr)
        x = F.elu(x)
        
        # Second GAT layer
        x = self.gat2(x, edge_index, edge_attr)
        x = F.elu(x)
        
        # Output projection
        x = self.output_proj(x)
        
        return x

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
                 num_sources=6, temperature=1.0, entropy_reg=0.01, num_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_sources = num_sources
        self.num_layers = num_layers
        
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
        
        # Evidence router
        self.router = EvidenceRouter(hidden_dim, num_sources, temperature, entropy_reg)
        
        # Learned source-context aggregation
        self.context_weight_proj = nn.Linear(hidden_dim * num_sources, num_sources)
        
        # Routing skip gate
        self.routing_skip_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Transformer layers for processing routed features
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layers
        self.no_routing_proj = nn.Linear(hidden_dim * num_sources, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.final_classifier = nn.Linear(hidden_dim, 1)
        
        # Disable routing flag
        self.disable_routing = False
        
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
        
        # Process structural evidence
        structure_features = self.mvga(x, edge_index, edge_attr)
        
        # Process raw features
        if raw_features is not None:
            raw_features = self.raw_proj(raw_features)
        else:
            raw_features = torch.zeros_like(structure_features)
            
        # Process pathogenicity-modulated features
        if pathogenicity_scores is not None:
            pathogenicity_features, gate_alpha = self.pathogenicity_gate(x, pathogenicity_scores)
            pathogenicity_features = self.pathogenicity_proj(pathogenicity_features)
            aux['pathogenicity_gate_alpha'] = gate_alpha
        else:
            pathogenicity_features = torch.zeros_like(structure_features)
            aux['pathogenicity_gate_alpha'] = torch.zeros(structure_features.size(0), 1, device=structure_features.device)
            
        # Process cancer context
        if cancer_context is not None:
            cancer_context = self.cancer_context_proj(cancer_context)
        else:
            cancer_context = torch.zeros_like(structure_features)
            
        # Process gene-set features
        if gene_set_features is not None:
            gene_set_features = self.gene_set_proj(gene_set_features)
        else:
            gene_set_features = torch.zeros_like(structure_features)
            
        # Process hypergraph features
        if hypergraph_features is not None:
            hypergraph_features = self.hypergraph_proj(hypergraph_features)
        else:
            hypergraph_features = torch.zeros_like(structure_features)
        
        # Collect all evidence sources
        evidence_features = [
            structure_features,
            raw_features,
            pathogenicity_features,
            cancer_context,
            gene_set_features,
            hypergraph_features
        ]
        
        # Compute gene context using learned source reweighting
        concat_feature = torch.cat(evidence_features, dim=-1)
        context_logits = self.context_weight_proj(concat_feature)
        context_weights = F.softmax(context_logits, dim=-1)
        evidence_stack = torch.stack(evidence_features, dim=1)
        gene_context = torch.sum(evidence_stack * context_weights.unsqueeze(-1), dim=1)
        aux['context_weights'] = context_weights
        
        # Apply evidence routing
        if not self.disable_routing:
            routed_feature, routing_weights, routing_entropy = self.router(
                evidence_features, gene_context, aux['pathogenicity_gate_alpha']
            )
            fallback_feature = self.no_routing_proj(concat_feature)
            skip_gate = torch.sigmoid(
                self.routing_skip_gate(torch.cat([routed_feature, fallback_feature], dim=-1))
            )
            routed_feature = skip_gate * routed_feature + (1.0 - skip_gate) * fallback_feature
            aux['routing_weights'] = routing_weights
            aux['routing_entropy'] = routing_entropy
            aux['routing_skip_gate'] = skip_gate
        else:
            # Simple concatenation without routing
            routed_feature = self.no_routing_proj(concat_feature)
            aux['routing_weights'] = torch.zeros(gene_context.size(0), self.num_sources, device=gene_context.device)
            aux['routing_entropy'] = torch.zeros(gene_context.size(0), 1, device=gene_context.device)
            aux['routing_skip_gate'] = torch.zeros(gene_context.size(0), 1, device=gene_context.device)
        
        # Process through transformer layers
        # Reshape for transformer: [sequence_length, batch_size, hidden_dim]
        if batch is not None:
            # Graph batching case
            routed_feature = routed_feature.unsqueeze(0)  # [1, num_nodes, hidden_dim]
            transformer_output = self.transformer(routed_feature)
            transformer_output = transformer_output.squeeze(0)  # [num_nodes, hidden_dim]
        else:
            # Single graph case
            routed_feature = routed_feature.unsqueeze(0)  # [1, num_nodes, hidden_dim]
            transformer_output = self.transformer(routed_feature)
            transformer_output = transformer_output.squeeze(0)  # [num_nodes, hidden_dim]
        
        # Final projection and classification
        output_features = self.output_proj(transformer_output)
        predictions = torch.sigmoid(self.final_classifier(output_features))
        
        # Store routing source names for interpretability
        aux['routing_source_names'] = [
            'structure', 'raw', 'pathogenicity', 'cancer_context', 'gene_set', 'hypergraph'
        ]
        
        if return_aux:
            return predictions, aux
        else:
            return predictions
