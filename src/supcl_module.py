"""Supervised Contrastive Learning module for multi-omics fusion.

Based on SupCon (Khosla et al., NeurIPS 2020) adapted for cancer driver gene
prediction. Encourages similar representations for genes with the same label
(driver/non-driver) while pushing apart different-label genes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    """Supervised Contrastive Learning Loss.
    
    For each anchor gene, positive samples are other genes with the same label
    (driver/non-driver), and negative samples are genes with different labels.
    
    Args:
        temperature: Temperature scaling parameter (default: 0.07)
        base_temperature: Base temperature for normalization (default: 0.07)
        contrast_mode: 'all' or 'one' - whether to contrast all positives or one
    """
    
    def __init__(self, temperature=0.07, base_temperature=0.07, contrast_mode='all'):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.contrast_mode = contrast_mode
        
    def forward(self, features, labels, mask=None):
        """Compute supervised contrastive loss.
        
        Args:
            features: [N, D] normalized feature vectors
            labels: [N] binary labels (0=non-driver, 1=driver)
            mask: Optional [N, N] mask for valid pairs
            
        Returns:
            Scalar loss
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T)  # [N, N]
        
        # Create label mask: positive pairs have same label
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        
        # Mask for positive pairs (same label, excluding self)
        label_mask = torch.eq(labels, labels.T).float().to(device)  # [N, N]
        
        # Mask out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(label_mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        
        # Combine masks
        mask = label_mask * logits_mask
        
        # Compute log_prob
        exp_logits = torch.exp(similarity_matrix / self.temperature) * logits_mask
        log_prob = similarity_matrix / self.temperature - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Compute mean of log-likelihood over positive pairs
        # Only compute for samples that have at least one positive pair
        mask_sum = mask.sum(1)
        valid_samples = mask_sum > 0
        
        if valid_samples.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        mean_log_prob_pos = (mask * log_prob).sum(1)[valid_samples] / mask_sum[valid_samples]
        
        # Loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        
        return loss


class MultiViewContrastiveLoss(nn.Module):
    """Multi-view contrastive loss for different evidence channels.
    
    Encourages agreement between different evidence channels (e.g., graph structure,
    raw omics, gene sets) for the same gene.
    """
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, view1, view2, labels=None):
        """Compute contrastive loss between two views.
        
        Args:
            view1: [N, D] features from first view
            view2: [N, D] features from second view
            labels: Optional [N] labels for supervised variant
            
        Returns:
            Scalar loss
        """
        device = view1.device
        batch_size = view1.shape[0]
        
        # Normalize
        view1 = F.normalize(view1, dim=1)
        view2 = F.normalize(view2, dim=1)
        
        # Concatenate views
        features = torch.cat([view1, view2], dim=0)  # [2N, D]
        
        # Compute similarity
        similarity_matrix = torch.matmul(features, features.T)  # [2N, 2N]
        
        # Create positive pair mask: (i, i+N) and (i+N, i) are positive pairs
        mask = torch.zeros((2 * batch_size, 2 * batch_size), device=device)
        for i in range(batch_size):
            mask[i, i + batch_size] = 1
            mask[i + batch_size, i] = 1
        
        # Mask out self-contrast
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(2 * batch_size).view(-1, 1).to(device),
            0
        )
        
        # Compute log prob
        exp_logits = torch.exp(similarity_matrix / self.temperature) * logits_mask
        log_prob = similarity_matrix / self.temperature - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
        loss = -mean_log_prob_pos.mean()
        
        return loss


class SupCLProjectionHead(nn.Module):
    """Projection head for contrastive learning.
    
    Maps hidden representations to a lower-dimensional space where
    contrastive loss is computed.
    """
    
    def __init__(self, input_dim, hidden_dim=128, output_dim=64):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.projection(x)
