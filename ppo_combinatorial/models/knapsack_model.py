"""
MLP-based Policy and Value Networks for Knapsack Problem.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class KnapsackPolicyNetwork(nn.Module):
    """MLP-based Policy Network for Knapsack."""
    
    def __init__(
        self,
        item_feature_dim: int = 4,
        context_feature_dim: int = 5,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        use_feasibility_mask: bool = True
    ):
        super().__init__()
        
        self.use_feasibility_mask = use_feasibility_mask
        
        self.item_embed = nn.Sequential(
            nn.Linear(item_feature_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )
        
        self.context_embed = nn.Sequential(
            nn.Linear(context_feature_dim, embed_dim),
            nn.ReLU(),
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute action probabilities given state."""
        item_features = state['item_features']
        context_features = state['context_features']
        current_idx = state['current_item_idx']
        can_add = state.get('can_add_item', None)
        
        batch_size = item_features.shape[0]
        device = item_features.device
        
        batch_idx = torch.arange(batch_size, device=device)
        num_items = item_features.shape[1]
        safe_idx = torch.clamp(current_idx, 0, num_items - 1)
        
        current_item_features = item_features[batch_idx, safe_idx]
        
        item_embedding = self.item_embed(current_item_features)
        context_embedding = self.context_embed(context_features)
        
        combined = torch.cat([item_embedding, context_embedding], dim=-1)
        logit = self.policy_head(combined)
        
        accept_prob = torch.sigmoid(logit).squeeze(-1)
        
        if self.use_feasibility_mask and can_add is not None:
            accept_prob = accept_prob * can_add.float()
        
        reject_prob = 1 - accept_prob
        action_probs = torch.stack([reject_prob, accept_prob], dim=-1)
        action_probs = action_probs / (action_probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        return action_probs, logit
    
    def sample_action(self, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        action_probs, _ = self.forward(state)
        
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
                'item_features': states['item_features'],
                'context_features': states['context_features'][:, t],
                'current_item_idx': states['current_item_idx'][:, t],
                'can_add_item': states.get('can_add_item', {}).get(t) if isinstance(states.get('can_add_item'), dict) else (states['can_add_item'][:, t] if 'can_add_item' in states else None),
            }
            
            action_probs, _ = self.forward(state_t)
            dist = torch.distributions.Categorical(action_probs)
            
            log_probs_list.append(dist.log_prob(actions[:, t]))
            entropy_list.append(dist.entropy())
        
        return torch.stack(log_probs_list, dim=1), torch.stack(entropy_list, dim=1)


class KnapsackValueNetwork(nn.Module):
    """Value Network for Knapsack Problem."""
    
    def __init__(self, feature_dim: int = 5, hidden_dim: int = 128):
        super().__init__()
        
        self.value_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute state value estimate."""
        values = state['values']
        current_value = state['current_value']
        current_weight = state['current_weight']
        capacity = state['capacity']
        step = state['step']
        
        batch_size, num_items = values.shape
        device = values.device
        
        total_value = values.sum(dim=-1)
        
        if isinstance(step, int):
            t = step
        else:
            t = step.item() if step.numel() == 1 else step[0].item()
        
        remaining_value = values[:, t+1:].sum(dim=-1) if t < num_items - 1 else torch.zeros(batch_size, device=device)
        
        features = torch.stack([
            current_value / (total_value + 1e-8),
            current_weight / (capacity + 1e-8),
            (capacity - current_weight) / (capacity + 1e-8),
            torch.full((batch_size,), (num_items - t) / num_items, device=device),
            remaining_value / (total_value + 1e-8),
        ], dim=-1)
        
        return self.value_mlp(features)
