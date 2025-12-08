"""
Core PPO (Proximal Policy Optimization) implementation for combinatorial optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional, List


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lambda_gae: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    GAE formula: A_t = sum_{l=0}^{T-t} (gamma * lambda)^l * delta_{t+l}
    where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t) is the TD residual.
    
    Args:
        rewards: Tensor of shape [batch_size, seq_len] - rewards at each timestep
        values: Tensor of shape [batch_size, seq_len] - value estimates V(s_t)
        next_values: Tensor of shape [batch_size, seq_len] - value estimates V(s_{t+1})
        dones: Tensor of shape [batch_size, seq_len] - episode termination flags
        gamma: Discount factor, controls importance of future rewards
        lambda_gae: GAE parameter, controls bias-variance trade-off
    
    Returns:
        advantages: Tensor of shape [batch_size, seq_len] - GAE advantages
        returns: Tensor of shape [batch_size, seq_len] - return targets
    """
    batch_size, seq_len = rewards.shape
    device = rewards.device
    
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros(batch_size, device=device)
    
    # Backward pass: A_{t-1} = delta_{t-1} + gamma * lambda * A_t
    for t in reversed(range(seq_len)):
        mask = 1.0 - dones[:, t]
        
        # TD residual: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        delta = rewards[:, t] + gamma * next_values[:, t] * mask - values[:, t]
        
        # GAE recursive: A_t = delta_t + gamma * lambda * A_{t+1}
        last_advantage = delta + gamma * lambda_gae * mask * last_advantage
        advantages[:, t] = last_advantage
    
    # Return targets: R_t = A_t + V(s_t)
    returns = advantages + values
    
    return advantages, returns


def compute_ppo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    entropy: torch.Tensor,
    clip_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    value_clip: Optional[float] = None,
    old_values: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the total PPO loss with clipped surrogate objective.
    
    PPO loss = -L_CLIP + c1 * L_VF - c2 * H
    
    Where:
    - L_CLIP = min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
    - L_VF = (V(s) - R)^2  (value function loss)
    - H = entropy bonus for exploration
    
    Args:
        log_probs: Log probabilities under current policy
        old_log_probs: Log probabilities under old policy
        advantages: GAE advantages (should be normalized)
        values: Current value predictions
        returns: Target returns
        entropy: Policy entropy
        clip_epsilon: Clipping parameter (typically 0.1-0.2)
        value_coef: Value loss coefficient (typically 0.5)
        entropy_coef: Entropy bonus coefficient (typically 0.01)
        value_clip: Optional value clipping parameter
        old_values: Old value predictions (needed if value_clip is used)
    
    Returns:
        total_loss: Scalar total PPO loss to minimize
        info: Dictionary with individual loss components
    """
    # Probability ratio: ratio = pi(a|s) / pi_old(a|s)
    ratio = torch.exp(log_probs - old_log_probs)
    
    # Clipped surrogate objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # Value function loss
    if value_clip is not None and old_values is not None:
        value_clipped = old_values + torch.clamp(
            values - old_values, -value_clip, value_clip
        )
        value_loss1 = (values - returns) ** 2
        value_loss2 = (value_clipped - returns) ** 2
        value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
    else:
        value_loss = 0.5 * ((values - returns) ** 2).mean()
    
    # Entropy bonus (negative because we maximize entropy)
    entropy_loss = -entropy.mean()
    
    # Total loss
    total_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
    
    # Compute metrics
    with torch.no_grad():
        approx_kl = ((ratio - 1) - (log_probs - old_log_probs)).mean().item()
        clip_fraction = (
            (ratio < 1 - clip_epsilon) | (ratio > 1 + clip_epsilon)
        ).float().mean().item()
    
    info = {
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'entropy': entropy.mean().item(),
        'approx_kl': approx_kl,
        'clip_fraction': clip_fraction,
        'ratio_mean': ratio.mean().item(),
    }
    
    return total_loss, info


class RolloutBuffer:
    """Buffer to store trajectory data collected during rollouts."""
    
    def __init__(self):
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.dones: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.masks: List[torch.Tensor] = []
    
    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ):
        """Add a transition to the buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        if mask is not None:
            self.masks.append(mask)
    
    def get(self) -> Dict[str, torch.Tensor]:
        """Get all data as stacked tensors."""
        return {
            'states': torch.stack(self.states, dim=1),
            'actions': torch.stack(self.actions, dim=1),
            'rewards': torch.stack(self.rewards, dim=1),
            'dones': torch.stack(self.dones, dim=1),
            'log_probs': torch.stack(self.log_probs, dim=1),
            'values': torch.stack(self.values, dim=1),
        }
    
    def clear(self):
        """Clear the buffer."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
        self.masks.clear()


class PPOAgent:
    """PPO Agent for combinatorial optimization."""
    
    def __init__(
        self,
        policy_network: nn.Module,
        value_network: nn.Module,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        lambda_gae: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        num_epochs: int = 4,
        device: str = 'cpu'
    ):
        """
        Initialize PPO Agent.
        
        Args:
            policy_network: Neural network that outputs action probabilities
            value_network: Neural network that outputs state values
            learning_rate: Adam optimizer learning rate
            gamma: Discount factor
            lambda_gae: GAE parameter
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            num_epochs: Number of PPO epochs per batch
            device: Device to use
        """
        self.policy_network = policy_network.to(device)
        self.value_network = value_network.to(device)
        self.device = device
        
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        
        self.optimizer = torch.optim.Adam(
            list(policy_network.parameters()) + list(value_network.parameters()),
            lr=learning_rate
        )
        
        self.buffer = RolloutBuffer()
    
    def normalize_advantages(
        self,
        advantages: torch.Tensor,
        epsilon: float = 1e-8
    ) -> torch.Tensor:
        """Normalize advantages to have zero mean and unit variance."""
        return (advantages - advantages.mean()) / (advantages.std() + epsilon)
    
    def update(self, data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform PPO update on collected trajectories."""
        states = data['states']
        actions = data['actions']
        old_log_probs = data['log_probs']
        returns = data['returns']
        advantages = data['advantages']
        old_values = data['values']
        
        advantages = self.normalize_advantages(advantages)
        
        batch_size, seq_len = actions.shape[:2]
        
        all_info = []
        
        for epoch in range(self.num_epochs):
            log_probs, entropy = self.policy_network.evaluate(states, actions)
            values = self.value_network(states).squeeze(-1)
            
            loss, info = compute_ppo_loss(
                log_probs=log_probs.view(-1),
                old_log_probs=old_log_probs.view(-1),
                advantages=advantages.view(-1),
                values=values.view(-1),
                returns=returns.view(-1),
                entropy=entropy.view(-1),
                clip_epsilon=self.clip_epsilon,
                value_coef=self.value_coef,
                entropy_coef=self.entropy_coef,
                old_values=old_values.view(-1)
            )
            
            self.optimizer.zero_grad()
            loss.backward()
            
            nn.utils.clip_grad_norm_(
                list(self.policy_network.parameters()) + 
                list(self.value_network.parameters()),
                self.max_grad_norm
            )
            
            self.optimizer.step()
            all_info.append(info)
        
        avg_info = {
            key: np.mean([info[key] for info in all_info])
            for key in all_info[0].keys()
        }
        
        return avg_info
    
    def train_step(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor,
        **data
    ) -> Dict[str, float]:
        """Complete training step: compute advantages + update."""
        advantages, returns = compute_gae(
            rewards=rewards,
            values=values,
            next_values=next_values,
            dones=dones,
            gamma=self.gamma,
            lambda_gae=self.lambda_gae
        )
        
        data['advantages'] = advantages
        data['returns'] = returns
        data['values'] = values
        
        return self.update(data)
