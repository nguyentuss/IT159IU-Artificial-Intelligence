"""
Knapsack Problem Environment for PPO.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict


class KnapsackEnvironment:
    """
    Knapsack Environment for reinforcement learning.
    
    State: (item_status, current_weight, current_value, step)
    Action: Accept (1) or reject (0) current item
    Reward: Value gained minus overflow penalty
    """
    
    def __init__(
        self,
        num_items: int = 50,
        batch_size: int = 128,
        capacity_ratio: float = 0.5,
        device: str = 'cpu'
    ):
        """
        Initialize Knapsack Environment.
        
        Args:
            num_items: Number of items N
            batch_size: Number of parallel environments
            capacity_ratio: Capacity as ratio of total weight
            device: Device for tensors
        """
        self.num_items = num_items
        self.batch_size = batch_size
        self.capacity_ratio = capacity_ratio
        self.device = device
        
        self.values = None
        self.weights = None
        self.capacity = None
        self.item_status = None
        self.current_weight = None
        self.current_value = None
        self.step_count = None
        self.total_value = None
        self.total_weight = None
        
    def reset(
        self,
        values: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Reset environment to initial state."""
        if values is None:
            self.values = 1 + 9 * torch.rand(
                self.batch_size, self.num_items, device=self.device
            )
        else:
            self.values = values.to(self.device)
            self.batch_size = values.shape[0]
        
        if weights is None:
            self.weights = 1 + 9 * torch.rand(
                self.batch_size, self.num_items, device=self.device
            )
        else:
            self.weights = weights.to(self.device)
        
        self.total_weight = self.weights.sum(dim=-1)
        self.capacity = self.capacity_ratio * self.total_weight
        self.total_value = self.values.sum(dim=-1)
        
        self.item_status = torch.zeros(
            self.batch_size, self.num_items,
            dtype=torch.float32, device=self.device
        )
        
        self.current_weight = torch.zeros(self.batch_size, device=self.device)
        self.current_value = torch.zeros(self.batch_size, device=self.device)
        self.step_count = 0
        
        return self._get_state()
    
    def _get_state(self) -> Dict[str, torch.Tensor]:
        """Get current state representation."""
        mean_value = self.values.mean(dim=-1, keepdim=True)
        mean_weight = self.weights.mean(dim=-1, keepdim=True)
        value_weight_ratio = self.values / (self.weights + 1e-8)
        mean_ratio = value_weight_ratio.mean(dim=-1, keepdim=True)
        
        item_features = torch.stack([
            self.values / (mean_value + 1e-8),
            self.weights / (mean_weight + 1e-8),
            value_weight_ratio / (mean_ratio + 1e-8),
            self.weights / self.capacity.unsqueeze(-1),
        ], dim=-1)
        
        t = self.step_count
        current_item_idx = torch.full(
            (self.batch_size,), t,
            dtype=torch.long, device=self.device
        )
        
        remaining_capacity = self.capacity - self.current_weight
        
        if t < self.num_items:
            current_item_weight = self.weights[:, t]
            current_item_value = self.values[:, t]
        else:
            current_item_weight = torch.zeros(self.batch_size, device=self.device)
            current_item_value = torch.zeros(self.batch_size, device=self.device)
        
        context_features = torch.stack([
            self.current_weight / self.capacity,
            remaining_capacity / self.capacity,
            self.current_value / (self.total_value + 1e-8),
            torch.full((self.batch_size,), t / self.num_items, device=self.device),
            remaining_capacity / (current_item_weight + 1e-8),
        ], dim=-1)
        
        can_add = remaining_capacity >= current_item_weight
        
        return {
            'item_features': item_features,
            'context_features': context_features,
            'current_item_idx': current_item_idx,
            'current_item_value': current_item_value,
            'current_item_weight': current_item_weight,
            'can_add_item': can_add,
            'values': self.values,
            'weights': self.weights,
            'capacity': self.capacity,
            'current_weight': self.current_weight,
            'current_value': self.current_value,
            'remaining_capacity': remaining_capacity,
            'step': torch.tensor(t, device=self.device),
        }
    
    def step(
        self,
        action: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict]:
        """
        Execute action and transition to next state.
        
        Reward: r_t = action * (alpha * v_t/V_max - beta * overflow/C_max)
        """
        action = action.to(self.device).float()
        t = self.step_count
        
        current_weight = self.weights[:, t]
        current_value = self.values[:, t]
        
        # Reward coefficients
        alpha = 1.0
        beta = 2.0
        
        normalized_value = current_value / (self.total_value + 1e-8)
        new_weight = self.current_weight + current_weight
        overflow = torch.clamp(new_weight - self.capacity, min=0)
        normalized_overflow = overflow / (self.capacity + 1e-8)
        
        reward = action * (alpha * normalized_value - beta * normalized_overflow)
        
        # Update state
        self.item_status[:, t] = action * 2 - 1
        self.current_weight = self.current_weight + action * current_weight
        self.current_value = self.current_value + action * current_value
        self.step_count += 1
        
        done = self.step_count >= self.num_items
        done_tensor = torch.full(
            (self.batch_size,), done,
            dtype=torch.bool, device=self.device
        )
        
        info = {}
        if done:
            info['total_value'] = self.current_value.clone()
            info['total_weight'] = self.current_weight.clone()
            info['feasible'] = (self.current_weight <= self.capacity)
            info['utilization'] = self.current_weight / self.capacity
        
        return self._get_state(), reward, done_tensor, info
    
    def get_feasibility_mask(self) -> torch.Tensor:
        """Get feasibility mask for current action."""
        t = self.step_count
        if t >= self.num_items:
            return torch.ones(self.batch_size, 2, device=self.device)
        
        current_weight = self.weights[:, t]
        remaining = self.capacity - self.current_weight
        can_add = remaining >= current_weight
        
        mask = torch.stack([
            torch.ones(self.batch_size, device=self.device),
            can_add.float()
        ], dim=-1)
        
        return mask


def generate_knapsack_instances(
    num_instances: int,
    num_items: int,
    capacity_ratio: float = 0.5,
    correlation: str = 'uncorrelated',
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate random Knapsack instances."""
    weights = 1 + 9 * torch.rand(num_instances, num_items, device=device)
    
    if correlation == 'uncorrelated':
        values = 1 + 9 * torch.rand(num_instances, num_items, device=device)
    elif correlation == 'weakly_correlated':
        noise = 4 * torch.rand(num_instances, num_items, device=device) - 2
        values = torch.clamp(weights + noise, min=1)
    elif correlation == 'strongly_correlated':
        values = weights + 5
    else:
        raise ValueError(f"Unknown correlation type: {correlation}")
    
    return values, weights
