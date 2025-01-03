import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class SelfAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, 
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 bias: bool = True):
        """
        Args:
            embed_dim: Dimension of input embeddings
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias in projections
        """
        super().__init__()
        
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"Embedding dimension {embed_dim} must be divisible by number of heads {num_heads}"
            )
            
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        """Initialize parameters"""
        # Initialize weights with Xavier uniform
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        # Initialize biases if present
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.)
            nn.init.constant_(self.k_proj.bias, 0.)
            nn.init.constant_(self.v_proj.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
            
    def forward(self, 
                query: torch.Tensor,
                key: Optional[torch.Tensor] = None,
                value: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
                need_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: Query tensor of shape (batch_size, target_len, embed_dim)
            key: Key tensor of shape (batch_size, source_len, embed_dim)
            value: Value tensor of shape (batch_size, source_len, embed_dim)
            key_padding_mask: Mask for padded elements in key
            attn_mask: Mask to prevent attention to certain positions
            need_weights: Whether to return attention weights
            
        Returns:
            output: Attention output
            attention_weights: Optional attention weights if need_weights is True
        """
        batch_size, target_len, embed_dim = query.size()
        
        scaling = float(self.head_dim) ** -0.5
        
        # Set key and value to query if not provided (self-attention)
        if key is None:
            key = query
        if value is None:
            value = query
            
        # Linear projections and reshape for multi-head attention
        q = self.q_proj(query).view(batch_size, target_len, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # (batch_size, num_heads, target_len, head_dim)
        k = k.transpose(1, 2)  # (batch_size, num_heads, source_len, head_dim)
        v = v.transpose(1, 2)  # (batch_size, num_heads, source_len, head_dim)
        
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scaling
        
        # Apply attention masks if provided
        if attn_mask is not None:
            attn_weights = attn_weights.masked_fill(
                attn_mask.unsqueeze(1).unsqueeze(0) == 0,
                float('-inf')
            )
            
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
            
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        
        # Compute output
        output = torch.matmul(attn_weights, v)
        
        # Reshape and apply output projection
        output = output.transpose(1, 2).contiguous().view(
            batch_size, target_len, embed_dim
        )
        output = self.out_proj(output)
        
        if need_weights:
            return output, attn_weights
        return output, None
        
    def extra_repr(self) -> str:
        """String representation of module"""
        return f'embed_dim={self.embed_dim}, num_heads={self.num_heads}, dropout={self.dropout}'