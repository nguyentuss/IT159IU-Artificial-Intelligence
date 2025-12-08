"""
Traveling Salesman Problem (TSP) Environment for PPO.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict


class TSPEnvironment:
    """
    TSP Environment for reinforcement learning.
    
    State: (coordinates, visited_mask, current_city, start_city)
    Action: Select next unvisited city
    Reward: Negative normalized distance
    """
    
    def __init__(
        self,
        num_cities: int = 20,
        batch_size: int = 128,
        device: str = 'cpu'
    ):
        """
        Initialize TSP Environment.
        
        Args:
            num_cities: Number of cities N in each instance
            batch_size: Number of parallel environments
            device: Device for tensors
        """
        self.num_cities = num_cities
        self.batch_size = batch_size
        self.device = device
        
        self.coordinates = None
        self.visited = None
        self.current_city = None
        self.start_city = None
        self.step_count = None
        self.avg_distance = None
        
    def reset(
        self,
        coordinates: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Reset environment to initial state."""
        if coordinates is None:
            self.coordinates = torch.rand(
                self.batch_size, self.num_cities, 2,
                device=self.device
            )
        else:
            self.coordinates = coordinates.to(self.device)
            self.batch_size = coordinates.shape[0]
        
        self.visited = torch.zeros(
            self.batch_size, self.num_cities,
            dtype=torch.bool, device=self.device
        )
        self.visited[:, 0] = True
        
        self.current_city = torch.zeros(
            self.batch_size, dtype=torch.long, device=self.device
        )
        
        self.start_city = torch.zeros(
            self.batch_size, dtype=torch.long, device=self.device
        )
        
        self.step_count = 0
        self.avg_distance = self._compute_avg_distance()
        
        return self._get_state()
    
    def _compute_avg_distance(self) -> torch.Tensor:
        """Compute average pairwise distance for reward normalization."""
        diff = self.coordinates.unsqueeze(2) - self.coordinates.unsqueeze(1)
        dist_matrix = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)
        
        n = self.num_cities
        num_pairs = n * (n - 1) / 2
        mask = torch.triu(torch.ones(n, n, device=self.device), diagonal=1)
        total_dist = (dist_matrix * mask.unsqueeze(0)).sum(dim=(1, 2))
        
        return total_dist / num_pairs
    
    def _get_distance(
        self,
        city1_idx: torch.Tensor,
        city2_idx: torch.Tensor
    ) -> torch.Tensor:
        """Compute Euclidean distance between two cities."""
        batch_idx = torch.arange(self.batch_size, device=self.device)
        p1 = self.coordinates[batch_idx, city1_idx]
        p2 = self.coordinates[batch_idx, city2_idx]
        return torch.sqrt(((p1 - p2) ** 2).sum(dim=-1) + 1e-8)
    
    def _get_state(self) -> Dict[str, torch.Tensor]:
        """Get current state representation."""
        action_mask = ~self.visited
        
        return {
            'coordinates': self.coordinates,
            'visited': self.visited,
            'current_city': self.current_city,
            'start_city': self.start_city,
            'action_mask': action_mask,
            'step': torch.tensor(self.step_count, device=self.device),
        }
    
    def step(
        self,
        action: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict]:
        """
        Execute action and transition to next state.
        
        Reward: r_t = -distance(current_city, action) / avg_distance
        """
        action = action.to(self.device)
        
        # Compute step reward
        distance = self._get_distance(self.current_city, action)
        reward = -distance / self.avg_distance
        
        # Update visited mask
        batch_idx = torch.arange(self.batch_size, device=self.device)
        self.visited[batch_idx, action] = True
        
        # Update current city
        self.current_city = action
        self.step_count += 1
        
        # Check if done
        done = self.step_count >= (self.num_cities - 1)
        
        # Add return trip distance at terminal
        if done:
            return_distance = self._get_distance(self.current_city, self.start_city)
            reward = reward - return_distance / self.avg_distance
        
        done_tensor = torch.full(
            (self.batch_size,), done,
            dtype=torch.bool, device=self.device
        )
        
        info = {}
        if done:
            info['tour_length'] = self._compute_tour_length()
        
        return self._get_state(), reward, done_tensor, info
    
    def _compute_tour_length(self) -> torch.Tensor:
        """Compute total tour length."""
        return torch.zeros(self.batch_size, device=self.device)
    
    def get_valid_actions(self) -> torch.Tensor:
        """Get mask of valid actions (unvisited cities)."""
        return ~self.visited
    
    def sample_random_action(self) -> torch.Tensor:
        """Sample a random valid action."""
        valid_mask = self.get_valid_actions().float()
        probs = valid_mask / valid_mask.sum(dim=-1, keepdim=True)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)


def generate_tsp_instances(
    num_instances: int,
    num_cities: int,
    device: str = 'cpu'
) -> torch.Tensor:
    """Generate random TSP instances."""
    return torch.rand(num_instances, num_cities, 2, device=device)
