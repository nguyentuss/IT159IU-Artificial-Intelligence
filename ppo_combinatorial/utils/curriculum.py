"""
Curriculum Learning utilities for combinatorial optimization.

Provides strategies for training on progressively harder problem instances.
"""

import torch
import numpy as np
from typing import List, Dict, Callable, Optional, Tuple


class CurriculumLearning:
    """
    Curriculum Learning Manager for Combinatorial Optimization.
    
    Supports discrete levels, continuous (self-paced), and mixed strategies.
    """
    
    def __init__(
        self,
        levels: List[Dict],
        advancement_threshold: float = 0.9,
        min_episodes_per_level: int = 100,
        method: str = 'discrete',
        mixing_probs: Tuple[float, float, float] = (0.2, 0.6, 0.2)
    ):
        """
        Initialize Curriculum Learning Manager.
        
        Args:
            levels: List of level configurations, each a dict with problem parameters
            advancement_threshold: Success rate threshold to advance (for discrete)
            min_episodes_per_level: Minimum episodes before considering advancement
            method: Curriculum method - 'discrete', 'continuous', or 'mixed'
            mixing_probs: (prev_level, current_level, next_level) sampling probs for mixed
        """
        self.levels = levels
        self.num_levels = len(levels)
        self.advancement_threshold = advancement_threshold
        self.min_episodes_per_level = min_episodes_per_level
        self.method = method
        self.mixing_probs = mixing_probs
        
        # Current state
        self.current_level = 0
        self.episodes_at_level = 0
        self.recent_successes: List[float] = []
        self.level_history: List[int] = []
    
    def get_current_params(self) -> Dict:
        """Get parameters for current curriculum level."""
        if self.method == 'mixed':
            return self._sample_mixed_level()
        return self.levels[self.current_level]
    
    def _sample_mixed_level(self) -> Dict:
        """Sample level using mixed curriculum strategy."""
        prob_prev, prob_curr, prob_next = self.mixing_probs
        
        levels_to_sample = [self.current_level]
        probs = [prob_curr]
        
        if self.current_level > 0:
            levels_to_sample.append(self.current_level - 1)
            probs.append(prob_prev)
        
        if self.current_level < self.num_levels - 1:
            levels_to_sample.append(self.current_level + 1)
            probs.append(prob_next)
        
        probs = np.array(probs)
        probs = probs / probs.sum()
        
        sampled_level = np.random.choice(levels_to_sample, p=probs)
        
        return self.levels[sampled_level]
    
    def update(self, success: float) -> bool:
        """
        Update curriculum state with episode result.
        
        Args:
            success: Success metric for episode (e.g., 1 if valid solution, 0 otherwise)
        
        Returns:
            advanced: Whether level was advanced
        """
        self.episodes_at_level += 1
        self.recent_successes.append(success)
        self.level_history.append(self.current_level)
        
        window_size = max(self.min_episodes_per_level, 100)
        if len(self.recent_successes) > window_size:
            self.recent_successes = self.recent_successes[-window_size:]
        
        if self.method == 'discrete' or self.method == 'mixed':
            if self.episodes_at_level >= self.min_episodes_per_level:
                success_rate = np.mean(self.recent_successes)
                
                if success_rate >= self.advancement_threshold:
                    return self._advance_level()
        
        return False
    
    def _advance_level(self) -> bool:
        """Advance to next curriculum level."""
        if self.current_level < self.num_levels - 1:
            self.current_level += 1
            self.episodes_at_level = 0
            self.recent_successes = []
            return True
        return False
    
    def get_difficulty_score(self) -> float:
        """Get current difficulty score (0 to 1)."""
        if self.num_levels <= 1:
            return 1.0
        return self.current_level / (self.num_levels - 1)
    
    def get_stats(self) -> Dict:
        """Get curriculum statistics."""
        return {
            'current_level': self.current_level,
            'difficulty': self.get_difficulty_score(),
            'episodes_at_level': self.episodes_at_level,
            'success_rate': np.mean(self.recent_successes) if self.recent_successes else 0.0,
        }


def create_tsp_curriculum(
    city_counts: List[int] = [20, 30, 50, 75, 100],
    **kwargs
) -> CurriculumLearning:
    """Create TSP curriculum with increasing city counts."""
    levels = [{'num_cities': n} for n in city_counts]
    return CurriculumLearning(levels, **kwargs)


def create_knapsack_curriculum(
    item_counts: List[int] = [10, 20, 50, 100],
    capacity_ratios: List[float] = [0.7, 0.5, 0.3],
    **kwargs
) -> CurriculumLearning:
    """Create Knapsack curriculum with item counts and capacity ratios."""
    levels = []
    for n in item_counts:
        for ratio in capacity_ratios:
            levels.append({
                'num_items': n,
                'capacity_ratio': ratio,
            })
    
    return CurriculumLearning(levels, **kwargs)


def create_graph_coloring_curriculum(
    node_counts: List[int] = [20, 50, 100],
    edge_densities: List[float] = [0.1, 0.2, 0.3],
    color_buffers: List[int] = [2, 1, 0],
    **kwargs
) -> CurriculumLearning:
    """Create Graph Coloring curriculum with node counts and edge densities."""
    levels = []
    for n in node_counts:
        for p in edge_densities:
            for buffer in color_buffers:
                levels.append({
                    'num_nodes': n,
                    'edge_probability': p,
                    'color_buffer': buffer,
                })
    
    return CurriculumLearning(levels, **kwargs)
