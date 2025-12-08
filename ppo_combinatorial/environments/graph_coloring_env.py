"""
Graph Coloring Problem Environment for PPO.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict


class GraphColoringEnvironment:
    """
    Graph Coloring Environment for reinforcement learning.
    
    State: (adjacency, color_assignments, current_step)
    Action: Assign color k in {1, ..., K} to current node
    Reward: Negative conflicts minus penalty for new colors
    """
    
    def __init__(
        self,
        num_nodes: int = 50,
        num_colors: int = 5,
        edge_probability: float = 0.3,
        batch_size: int = 128,
        device: str = 'cpu'
    ):
        """
        Initialize Graph Coloring Environment.
        
        Args:
            num_nodes: Number of nodes N
            num_colors: Number of available colors K
            edge_probability: Edge probability for random graphs
            batch_size: Number of parallel environments
            device: Device for tensors
        """
        self.num_nodes = num_nodes
        self.num_colors = num_colors
        self.edge_probability = edge_probability
        self.batch_size = batch_size
        self.device = device
        
        self.adjacency = None
        self.degrees = None
        self.num_edges = None
        self.colors = None
        self.step_count = None
        
    def reset(
        self,
        adjacency: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Reset environment to initial state."""
        if adjacency is None:
            self.adjacency = self._generate_erdos_renyi()
        else:
            self.adjacency = adjacency.to(self.device)
            self.batch_size = adjacency.shape[0]
        
        self.degrees = self.adjacency.sum(dim=-1)
        self.num_edges = self.adjacency.sum(dim=(1, 2)) / 2
        
        self.colors = torch.zeros(
            self.batch_size, self.num_nodes,
            dtype=torch.long, device=self.device
        )
        
        self.step_count = 0
        
        return self._get_state()
    
    def _generate_erdos_renyi(self) -> torch.Tensor:
        """Generate random Erdos-Renyi graph."""
        n = self.num_nodes
        p = self.edge_probability
        
        edges = torch.rand(self.batch_size, n, n, device=self.device) < p
        mask = 1 - torch.eye(n, device=self.device)
        edges = edges * mask.unsqueeze(0)
        
        upper = torch.triu(edges, diagonal=1)
        adjacency = upper + upper.transpose(-2, -1)
        
        return adjacency.float()
    
    def _get_neighbors(self, node_idx: int) -> torch.Tensor:
        """Get neighbor mask for a node."""
        return self.adjacency[:, node_idx, :] > 0
    
    def _get_blocked_colors(self, node_idx: int) -> torch.Tensor:
        """Get colors blocked by neighbors."""
        neighbor_mask = self._get_neighbors(node_idx)
        neighbor_colors = self.colors * neighbor_mask
        
        blocked = torch.zeros(
            self.batch_size, self.num_colors,
            dtype=torch.bool, device=self.device
        )
        
        for k in range(1, self.num_colors + 1):
            blocked[:, k-1] = (neighbor_colors == k).any(dim=-1)
        
        return blocked
    
    def _get_state(self) -> Dict[str, torch.Tensor]:
        """Get current state representation."""
        t = self.step_count
        t_safe = min(t, self.num_nodes - 1)
        
        color_onehot = F.one_hot(
            self.colors, num_classes=self.num_colors + 1
        ).float()
        
        is_current = torch.zeros(
            self.batch_size, self.num_nodes,
            device=self.device
        )
        is_current[:, t_safe] = 1.0
        
        blocked_colors = self._get_blocked_colors(t_safe)
        valid_actions = ~blocked_colors
        conflicts = self._count_conflicts()
        colors_used = self._count_colors_used()
        
        return {
            'adjacency': self.adjacency,
            'colors': self.colors,
            'color_onehot': color_onehot,
            'degrees': self.degrees,
            'is_current_node': is_current,
            'current_node': torch.full((self.batch_size,), t_safe, dtype=torch.long, device=self.device),
            'blocked_colors': blocked_colors,
            'valid_actions': valid_actions,
            'conflicts': conflicts,
            'colors_used': colors_used,
            'step': torch.tensor(t, device=self.device),
        }

    
    def _count_conflicts(self) -> torch.Tensor:
        """Count coloring conflicts."""
        c_i = self.colors.unsqueeze(-1)
        c_j = self.colors.unsqueeze(-2)
        same_color = (c_i == c_j) & (c_i != 0)
        conflicts = (same_color * self.adjacency).sum(dim=(1, 2)) / 2
        return conflicts
    
    def _count_colors_used(self) -> torch.Tensor:
        """Count distinct colors used."""
        colors_used = torch.zeros(self.batch_size, device=self.device)
        for k in range(1, self.num_colors + 1):
            has_color_k = (self.colors == k).any(dim=-1)
            colors_used += has_color_k.float()
        return colors_used
    
    def step(
        self,
        action: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict]:
        """
        Execute action and transition to next state.
        
        Reward: r_t = -alpha * conflicts_created - beta * is_new_color
        """
        action = action.to(self.device)
        t = self.step_count
        
        neighbor_mask = self._get_neighbors(t)
        neighbor_colors = self.colors * neighbor_mask.long()
        
        # Count conflicts created
        conflicts_created = (neighbor_colors == action.unsqueeze(-1)).sum(dim=-1).float()
        
        # Check if new color
        colors_used_before = self.colors[:, :t]
        if t > 0:
            is_new_color = ~(colors_used_before == action.unsqueeze(-1)).any(dim=-1)
        else:
            is_new_color = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
        
        # Reward
        alpha = 1.0
        beta = 0.1
        reward = -alpha * conflicts_created - beta * is_new_color.float()
        
        # Update state
        self.colors[:, t] = action
        self.step_count += 1
        
        done = self.step_count >= self.num_nodes
        done_tensor = torch.full(
            (self.batch_size,), done,
            dtype=torch.bool, device=self.device
        )
        
        info = {}
        if done:
            info['total_conflicts'] = self._count_conflicts()
            info['colors_used'] = self._count_colors_used()
            info['is_valid_coloring'] = (info['total_conflicts'] == 0)
        
        return self._get_state(), reward, done_tensor, info
    
    def get_valid_actions(self) -> torch.Tensor:
        """Get mask of valid actions."""
        t = self.step_count
        return ~self._get_blocked_colors(t)


def generate_graph_instances(
    num_instances: int,
    num_nodes: int,
    edge_probability: float = 0.3,
    device: str = 'cpu'
) -> torch.Tensor:
    """Generate random Erdos-Renyi graphs."""
    n = num_nodes
    p = edge_probability
    
    edges = torch.rand(num_instances, n, n, device=device) < p
    mask = 1 - torch.eye(n, device=device)
    edges = edges * mask.unsqueeze(0)
    
    upper = torch.triu(edges, diagonal=1)
    adjacency = upper + upper.transpose(-2, -1)
    
    return adjacency.float()
