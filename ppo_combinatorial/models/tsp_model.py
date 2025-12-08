"""
Transformer-based Policy and Value Networks for TSP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention mechanism."""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)
        
        return self.out_proj(attn_output)


class FeedForward(nn.Module):
    """Feed-Forward Network for Transformer."""
    
    def __init__(self, embed_dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * embed_dim
        
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        return self.fc2(x)


class TransformerEncoderLayer(nn.Module):
    """Single Transformer Encoder Layer."""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, ffn_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, mask)
        x = residual + self.dropout(x)
        
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + self.dropout(x)
        
        return x


class TSPPolicyNetwork(nn.Module):
    """Transformer-based Policy Network for TSP."""
    
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.node_embed = nn.Linear(2, embed_dim)
        
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.context_proj = nn.Linear(3 * embed_dim, embed_dim)
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** -0.5
    
    def encode(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Encode city coordinates using Transformer."""
        x = self.node_embed(coordinates)
        for layer in self.encoder_layers:
            x = layer(x)
        return x
    
    def forward(self, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute action probabilities given state."""
        coordinates = state['coordinates']
        action_mask = state['action_mask']
        current_city = state['current_city']
        start_city = state['start_city']
        
        batch_size, num_cities, _ = coordinates.shape
        
        embeddings = self.encode(coordinates)
        graph_embedding = embeddings.mean(dim=1)
        
        batch_idx = torch.arange(batch_size, device=coordinates.device)
        current_embedding = embeddings[batch_idx, current_city]
        start_embedding = embeddings[batch_idx, start_city]
        
        context = torch.cat([graph_embedding, current_embedding, start_embedding], dim=-1)
        context = self.context_proj(context)
        
        query = self.query_proj(context)
        keys = self.key_proj(embeddings)
        
        logits = torch.bmm(keys, query.unsqueeze(-1)).squeeze(-1) * self.scale
        logits = logits.masked_fill(~action_mask, float('-inf'))
        
        action_probs = F.softmax(logits, dim=-1)
        
        return action_probs, logits
    
    def sample_action(self, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        action_probs, logits = self.forward(state)
        
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy
    
    def evaluate(self, states: Dict[str, torch.Tensor], actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log probability and entropy for state-action pairs."""
        batch_size, seq_len = actions.shape
        
        log_probs_list = []
        entropy_list = []
        
        for t in range(seq_len):
            state_t = {
                'coordinates': states['coordinates'][:, t] if states['coordinates'].dim() == 4 else states['coordinates'],
                'action_mask': states['action_mask'][:, t],
                'current_city': states['current_city'][:, t],
                'start_city': states['start_city'][:, t] if states['start_city'].dim() == 2 else states['start_city'],
            }
            
            action_probs, _ = self.forward(state_t)
            dist = torch.distributions.Categorical(action_probs)
            
            log_probs_list.append(dist.log_prob(actions[:, t]))
            entropy_list.append(dist.entropy())
        
        return torch.stack(log_probs_list, dim=1), torch.stack(entropy_list, dim=1)


class TSPValueNetwork(nn.Module):
    """Value Network for TSP."""
    
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        shared_encoder: Optional[nn.Module] = None
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.shared_encoder = shared_encoder
        
        if shared_encoder is None:
            self.node_embed = nn.Linear(2, embed_dim)
            self.encoder_layers = nn.ModuleList([
                TransformerEncoderLayer(embed_dim, num_heads, dropout=dropout)
                for _ in range(num_layers)
            ])
        
        input_dim = 2 * embed_dim + 2
        
        self.value_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def encode(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Encode coordinates."""
        if self.shared_encoder is not None:
            return self.shared_encoder.encode(coordinates)
        
        x = self.node_embed(coordinates)
        for layer in self.encoder_layers:
            x = layer(x)
        return x
    
    def forward(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute state value."""
        coordinates = state['coordinates']
        current_city = state['current_city']
        step = state.get('step', torch.zeros(1))
        
        batch_size, num_cities = coordinates.shape[:2]
        device = coordinates.device
        
        embeddings = self.encode(coordinates)
        graph_embedding = embeddings.mean(dim=1)
        
        batch_idx = torch.arange(batch_size, device=device)
        current_embedding = embeddings[batch_idx, current_city]
        
        if isinstance(step, int):
            step_norm = torch.full((batch_size, 1), step / num_cities, device=device)
        else:
            step_norm = (step.float() / num_cities).unsqueeze(-1).expand(batch_size, 1)
        
        partial_length_norm = torch.zeros(batch_size, 1, device=device)
        
        features = torch.cat([graph_embedding, current_embedding, step_norm, partial_length_norm], dim=-1)
        
        return self.value_mlp(features)
