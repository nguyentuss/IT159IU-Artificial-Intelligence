"""
GNN-based Policy and Value Networks for Graph Coloring Problem.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class MessagePassingLayer(nn.Module):
    """Graph Neural Network Message Passing Layer."""
    
    def __init__(self, node_dim: int, hidden_dim: Optional[int] = None, aggregation: str = 'mean'):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = node_dim
        
        self.aggregation = aggregation
        
        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim),
        )
        
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim),
        )
        
        self.layer_norm = nn.LayerNorm(node_dim)
    
    def forward(self, node_embeddings: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """Forward pass of message passing."""
        batch_size, num_nodes, node_dim = node_embeddings.shape
        
        messages = self.message_mlp(node_embeddings)
        aggregated = torch.bmm(adjacency, messages)
        
        if self.aggregation == 'mean':
            degrees = adjacency.sum(dim=-1, keepdim=True).clamp(min=1)
            aggregated = aggregated / degrees
        
        combined = torch.cat([node_embeddings, aggregated], dim=-1)
        update = self.update_mlp(combined)
        
        return self.layer_norm(node_embeddings + update)


class GraphColoringPolicyNetwork(nn.Module):
    """GNN-based Policy Network for Graph Coloring."""
    
    def __init__(
        self,
        num_colors: int = 5,
        embed_dim: int = 64,
        num_gnn_layers: int = 3,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        self.num_colors = num_colors
        self.embed_dim = embed_dim
        
        node_feature_dim = 1 + (num_colors + 1) + 1 + num_colors
        
        self.node_embed = nn.Linear(node_feature_dim, embed_dim)
        
        self.gnn_layers = nn.ModuleList([
            MessagePassingLayer(embed_dim, hidden_dim, aggregation='mean')
            for _ in range(num_gnn_layers)
        ])
        
        self.color_embed = nn.Embedding(num_colors + 1, embed_dim)
        self.query_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute action probabilities given state."""
        adjacency = state['adjacency']
        degrees = state['degrees']
        color_onehot = state['color_onehot']
        is_current = state['is_current_node']
        valid_actions = state['valid_actions']
        current_node = state['current_node']
        blocked_colors = state['blocked_colors']
        
        batch_size, num_nodes = degrees.shape
        device = adjacency.device
        
        blocked_all = torch.zeros(batch_size, num_nodes, self.num_colors, device=device)
        batch_idx = torch.arange(batch_size, device=device)
        blocked_all[batch_idx, current_node] = blocked_colors.float()
        
        node_features = torch.cat([
            degrees.unsqueeze(-1) / num_nodes,
            color_onehot,
            is_current.unsqueeze(-1),
            blocked_all,
        ], dim=-1)
        
        node_embeddings = self.node_embed(node_features)
        
        for gnn_layer in self.gnn_layers:
            node_embeddings = gnn_layer(node_embeddings, adjacency)
        
        current_embedding = node_embeddings[batch_idx, current_node]
        query = self.query_proj(current_embedding)
        
        color_indices = torch.arange(1, self.num_colors + 1, device=device)
        color_embeddings = self.color_embed(color_indices)
        
        logits = torch.matmul(query, color_embeddings.T)
        masked_logits = logits.masked_fill(~valid_actions, float('-inf'))
        
        action_probs = F.softmax(masked_logits, dim=-1)
        
        return action_probs, logits
    
    def sample_action(self, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        action_probs, _ = self.forward(state)
        
        dist = torch.distributions.Categorical(action_probs)
        action_idx = dist.sample()
        action = action_idx + 1  # Convert to 1-indexed
        log_prob = dist.log_prob(action_idx)
        entropy = dist.entropy()
        
        return action, log_prob, entropy
    
    def evaluate(self, states: Dict[str, torch.Tensor], actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log probability and entropy for state-action pairs."""
        batch_size, seq_len = actions.shape
        
        log_probs_list = []
        entropy_list = []
        
        for t in range(seq_len):
            state_t = {
                'adjacency': states['adjacency'],
                'degrees': states['degrees'],
                'color_onehot': states['color_onehot'][:, t],
                'is_current_node': states['is_current_node'][:, t],
                'blocked_colors': states['blocked_colors'][:, t],
                'valid_actions': states['valid_actions'][:, t],
                'current_node': states['current_node'][:, t],
            }
            
            action_probs, _ = self.forward(state_t)
            dist = torch.distributions.Categorical(action_probs)
            
            action_idx = actions[:, t] - 1
            
            log_probs_list.append(dist.log_prob(action_idx))
            entropy_list.append(dist.entropy())
        
        return torch.stack(log_probs_list, dim=1), torch.stack(entropy_list, dim=1)


class GraphColoringValueNetwork(nn.Module):
    """Value Network for Graph Coloring Problem."""
    
    def __init__(
        self,
        num_colors: int = 5,
        embed_dim: int = 64,
        num_gnn_layers: int = 3,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        self.num_colors = num_colors
        self.embed_dim = embed_dim
        
        node_feature_dim = 1 + (num_colors + 1) + 1 + num_colors
        
        self.node_embed = nn.Linear(node_feature_dim, embed_dim)
        
        self.gnn_layers = nn.ModuleList([
            MessagePassingLayer(embed_dim, hidden_dim, aggregation='mean')
            for _ in range(num_gnn_layers)
        ])
        
        self.value_mlp = nn.Sequential(
            nn.Linear(embed_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute state value estimate."""
        adjacency = state['adjacency']
        degrees = state['degrees']
        color_onehot = state['color_onehot']
        is_current = state['is_current_node']
        conflicts = state['conflicts']
        colors_used = state['colors_used']
        step = state['step']
        
        batch_size, num_nodes = degrees.shape
        device = adjacency.device
        
        blocked_placeholder = torch.zeros(batch_size, num_nodes, self.num_colors, device=device)
        
        node_features = torch.cat([
            degrees.unsqueeze(-1) / num_nodes,
            color_onehot,
            is_current.unsqueeze(-1),
            blocked_placeholder,
        ], dim=-1)
        
        node_embeddings = self.node_embed(node_features)
        
        for gnn_layer in self.gnn_layers:
            node_embeddings = gnn_layer(node_embeddings, adjacency)
        
        graph_embedding = node_embeddings.mean(dim=1)
        
        num_edges = adjacency.sum(dim=(1, 2)) / 2
        
        if isinstance(step, int):
            t = step
        else:
            t = step.item() if step.numel() == 1 else step[0].item()
        
        progress = torch.full((batch_size, 1), t / num_nodes, device=device)
        conflict_ratio = (conflicts / (num_edges + 1e-8)).unsqueeze(-1)
        color_ratio = (colors_used / self.num_colors).unsqueeze(-1)
        
        features = torch.cat([graph_embedding, progress, conflict_ratio, color_ratio], dim=-1)
        
        return self.value_mlp(features)
