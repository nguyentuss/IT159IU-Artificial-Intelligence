"""
Training Script for Knapsack Problem with PPO.

Supports both random generation and static dataset loading.
"""

import argparse
import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import ast
from tqdm import tqdm

from ppo_combinatorial.environments.knapsack_env import KnapsackEnvironment, generate_knapsack_instances
from ppo_combinatorial.models.knapsack_model import KnapsackPolicyNetwork, KnapsackValueNetwork
from ppo_combinatorial.core.ppo import compute_gae, compute_ppo_loss


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train PPO for Knapsack')
    
    # Data source
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Path to CSV dataset (if None, use random generation)')
    
    # Problem parameters (used when dataset_path is None)
    parser.add_argument('--num_items', type=int, default=50,
                        help='Number of items N (or max items for curriculum)')
    parser.add_argument('--capacity_ratio', type=float, default=0.5,
                        help='Capacity ratio = C_max / total_weight (or min for curriculum)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Number of parallel environments')
    
    # Curriculum learning
    parser.add_argument('--curriculum', action='store_true',
                        help='Enable curriculum learning (start easy, increase difficulty)')
    parser.add_argument('--curriculum_start_items', type=int, default=5,
                        help='Starting number of items for curriculum')
    parser.add_argument('--curriculum_end_items', type=int, default=200,
                        help='Final number of items for curriculum')
    parser.add_argument('--curriculum_start_ratio', type=float, default=0.7,
                        help='Starting capacity ratio (easier)')
    parser.add_argument('--curriculum_end_ratio', type=float, default=0.3,
                        help='Final capacity ratio (harder)')
    
    # PPO hyperparameters
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='Discount factor')
    parser.add_argument('--lambda_gae', type=float, default=0.95,
                        help='GAE parameter')
    parser.add_argument('--clip_epsilon', type=float, default=0.2,
                        help='PPO clipping parameter')
    parser.add_argument('--value_coef', type=float, default=0.5,
                        help='Value loss coefficient')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                        help='Entropy coefficient')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help='Gradient clipping')
    parser.add_argument('--num_epochs', type=int, default=4,
                        help='PPO epochs')
    
    # Training parameters
    parser.add_argument('--epochs', '--max_iterations', type=int, default=5000,
                        dest='max_iterations',
                        help='Number of training epochs/iterations')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log every N iterations')
    parser.add_argument('--save_interval', type=int, default=100,
                        help='Save every N iterations')
    
    # Model parameters
    parser.add_argument('--embed_dim', type=int, default=64,
                        help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='MLP hidden dimension')
    
    # Device
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/knapsack')
    
    # Experiment name
    parser.add_argument('--exp_name', type=str, default='knapsack',
                        help='Experiment name for checkpoint files')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    
    return parser.parse_args()


def load_dataset(csv_path, device='cpu'):
    """Load knapsack dataset from CSV file."""
    df = pd.read_csv(csv_path)
    
    def parse_array(s):
        s = s.strip().replace('[', '').replace(']', '')
        return np.array([float(x) for x in s.split(',')])
    
    all_data = []
    
    for _, row in df.iterrows():
        weights = parse_array(str(row['Weights']))
        values = parse_array(str(row['Values']))
        capacity = float(row['Capacity'])
        optimal = float(row['Optimal_Value'])
        n = int(row['N'])
        
        all_data.append({
            'n': n,
            'weights': torch.tensor(weights, dtype=torch.float32, device=device),
            'values': torch.tensor(values, dtype=torch.float32, device=device),
            'capacity': torch.tensor([capacity], dtype=torch.float32, device=device),
            'optimal': optimal,
        })
    
    return all_data


class DatasetEnvironment:
    """Environment that loads instances from a static dataset."""
    
    def __init__(self, dataset, max_items, device='cpu'):
        self.dataset = dataset
        self.max_items = max_items
        self.device = device
        self.current_batch = None
        self.num_items = max_items
        self.batch_size = 1
        
        # State variables
        self.values = None
        self.weights = None
        self.capacity = None
        self.item_status = None
        self.current_weight = None
        self.current_value = None
        self.step_count = None
        self.total_value = None
        self.total_weight = None
    
    def reset(self, batch_indices=None):
        """Reset with instances from dataset."""
        if batch_indices is None:
            batch_indices = np.random.choice(len(self.dataset), size=1, replace=True)
        
        self.batch_size = len(batch_indices)
        
        # Pad instances to max_items
        values_list = []
        weights_list = []
        capacity_list = []
        
        for idx in batch_indices:
            inst = self.dataset[idx]
            n = inst['n']
            
            # Pad to max_items
            padded_values = torch.zeros(self.max_items, device=self.device)
            padded_weights = torch.zeros(self.max_items, device=self.device)
            
            padded_values[:n] = inst['values']
            padded_weights[:n] = inst['weights']
            
            values_list.append(padded_values)
            weights_list.append(padded_weights)
            capacity_list.append(inst['capacity'])
        
        self.values = torch.stack(values_list)
        self.weights = torch.stack(weights_list)
        self.capacity = torch.cat(capacity_list)
        
        self.total_weight = self.weights.sum(dim=-1)
        self.total_value = self.values.sum(dim=-1)
        
        self.item_status = torch.zeros(
            self.batch_size, self.max_items,
            dtype=torch.float32, device=self.device
        )
        
        self.current_weight = torch.zeros(self.batch_size, device=self.device)
        self.current_value = torch.zeros(self.batch_size, device=self.device)
        self.step_count = 0
        
        return self._get_state()
    
    def _get_state(self):
        """Get current state representation."""
        mean_value = self.values.mean(dim=-1, keepdim=True).clamp(min=1e-8)
        mean_weight = self.weights.mean(dim=-1, keepdim=True).clamp(min=1e-8)
        value_weight_ratio = self.values / (self.weights + 1e-8)
        mean_ratio = value_weight_ratio.mean(dim=-1, keepdim=True).clamp(min=1e-8)
        
        item_features = torch.stack([
            self.values / mean_value,
            self.weights / mean_weight,
            value_weight_ratio / mean_ratio,
            self.weights / self.capacity.unsqueeze(-1).clamp(min=1e-8),
        ], dim=-1)
        
        t = self.step_count
        remaining_capacity = self.capacity - self.current_weight
        
        if t < self.num_items:
            current_item_weight = self.weights[:, t]
            current_item_value = self.values[:, t]
        else:
            current_item_weight = torch.zeros(self.batch_size, device=self.device)
            current_item_value = torch.zeros(self.batch_size, device=self.device)
        
        context_features = torch.stack([
            self.current_weight / self.capacity.clamp(min=1e-8),
            remaining_capacity / self.capacity.clamp(min=1e-8),
            self.current_value / self.total_value.clamp(min=1e-8),
            torch.full((self.batch_size,), t / self.num_items, device=self.device),
            remaining_capacity / current_item_weight.clamp(min=1e-8),
        ], dim=-1)
        
        can_add = remaining_capacity >= current_item_weight
        
        return {
            'item_features': item_features,
            'context_features': context_features,
            'current_item_idx': torch.full((self.batch_size,), t, dtype=torch.long, device=self.device),
            'current_item_value': current_item_value,
            'current_item_weight': current_item_weight,
            'can_add_item': can_add,
            'values': self.values,
            'weights': self.weights,
            'capacity': self.capacity,
            'current_weight': self.current_weight,
            'current_value': self.current_value,
            'remaining_capacity': remaining_capacity,
            'step': t,
        }
    
    def step(self, action):
        """Execute action."""
        action = action.to(self.device).float()
        t = self.step_count
        
        current_weight = self.weights[:, t]
        current_value = self.values[:, t]
        
        # Dense reward
        alpha = 1.0
        beta = 2.0
        
        normalized_value = current_value / self.total_value.clamp(min=1e-8)
        new_weight = self.current_weight + action * current_weight
        overflow = torch.clamp(new_weight - self.capacity, min=0)
        normalized_overflow = overflow / self.capacity.clamp(min=1e-8)
        
        reward = action * (alpha * normalized_value - beta * normalized_overflow)
        
        # Update state
        self.item_status[:, t] = action * 2 - 1
        self.current_weight = self.current_weight + action * current_weight
        self.current_value = self.current_value + action * current_value
        self.step_count += 1
        
        done = self.step_count >= self.num_items
        done_tensor = torch.full((self.batch_size,), done, dtype=torch.bool, device=self.device)
        
        info = {}
        if done:
            info['total_value'] = self.current_value.clone()
            info['total_weight'] = self.current_weight.clone()
            info['feasible'] = (self.current_weight <= self.capacity)
            info['utilization'] = self.current_weight / self.capacity.clamp(min=1e-8)
        
        return self._get_state(), reward, done_tensor, info


def collect_trajectories(env, policy, value_net, device):
    """Collect trajectories for Knapsack problem."""
    states_list = []
    actions_list = []
    rewards_list = []
    log_probs_list = []
    values_list = []
    dones_list = []
    
    state = env.reset()
    
    for t in range(env.num_items):
        states_list.append({
            'item_features': state['item_features'].clone(),
            'context_features': state['context_features'].clone(),
            'current_item_idx': state['current_item_idx'].clone(),
            'can_add_item': state['can_add_item'].clone(),
            'values': state['values'].clone(),
            'weights': state['weights'].clone(),
            'capacity': state['capacity'].clone(),
            'current_weight': state['current_weight'].clone(),
            'current_value': state['current_value'].clone(),
            'step': t,
        })
        
        with torch.no_grad():
            value = value_net(state).squeeze(-1)
        values_list.append(value)
        
        with torch.no_grad():
            action, log_prob, _ = policy.sample_action(state)
        
        actions_list.append(action)
        log_probs_list.append(log_prob)
        
        next_state, reward, done, info = env.step(action)
        
        rewards_list.append(reward)
        dones_list.append(done.float())
        
        state = next_state
    
    with torch.no_grad():
        final_value = value_net(state).squeeze(-1)
    
    actions = torch.stack(actions_list, dim=1)
    rewards = torch.stack(rewards_list, dim=1)
    log_probs = torch.stack(log_probs_list, dim=1)
    values = torch.stack(values_list, dim=1)
    dones = torch.stack(dones_list, dim=1)
    
    next_values = torch.cat([values[:, 1:], final_value.unsqueeze(-1)], dim=1)
    
    return {
        'states': states_list,
        'actions': actions,
        'rewards': rewards,
        'log_probs': log_probs,
        'values': values,
        'next_values': next_values,
        'dones': dones,
        'info': info,
    }


def train_step(policy, value_net, optimizer, data, args):
    """PPO update step for Knapsack."""
    states_list = data['states']
    actions = data['actions']
    rewards = data['rewards']
    old_log_probs = data['log_probs']
    old_values = data['values']
    next_values = data['next_values']
    dones = data['dones']
    
    advantages, returns = compute_gae(
        rewards=rewards,
        values=old_values,
        next_values=next_values,
        dones=dones,
        gamma=args.gamma,
        lambda_gae=args.lambda_gae
    )
    
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    all_metrics = []
    
    for epoch in range(args.num_epochs):
        epoch_log_probs = []
        epoch_entropies = []
        epoch_values = []
        
        for t, state in enumerate(states_list):
            action_probs, _ = policy(state)
            dist = torch.distributions.Categorical(action_probs)
            
            log_prob = dist.log_prob(actions[:, t])
            entropy = dist.entropy()
            
            epoch_log_probs.append(log_prob)
            epoch_entropies.append(entropy)
            
            value = value_net(state).squeeze(-1)
            epoch_values.append(value)
        
        log_probs = torch.stack(epoch_log_probs, dim=1)
        entropies = torch.stack(epoch_entropies, dim=1)
        values = torch.stack(epoch_values, dim=1)
        
        loss, metrics = compute_ppo_loss(
            log_probs=log_probs.view(-1),
            old_log_probs=old_log_probs.view(-1),
            advantages=advantages.view(-1),
            values=values.view(-1),
            returns=returns.view(-1),
            entropy=entropies.view(-1),
            clip_epsilon=args.clip_epsilon,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            old_values=old_values.view(-1)
        )
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(policy.parameters()) + list(value_net.parameters()),
            args.max_grad_norm
        )
        optimizer.step()
        
        all_metrics.append(metrics)
    
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }
    
    return avg_metrics


def train(args):
    """Main training loop for Knapsack."""
    
    # Determine data source
    if args.dataset_path:
        print(f"Training Knapsack with static dataset: {args.dataset_path}")
        dataset = load_dataset(args.dataset_path, args.device)
        
        # Find max items in dataset
        max_items = max(inst['n'] for inst in dataset)
        args.num_items = max_items
        
        env = DatasetEnvironment(dataset, max_items, args.device)
        env.batch_size = args.batch_size
        
        print(f"  Loaded {len(dataset)} instances, max items: {max_items}")
    else:
        print(f"Training Knapsack with random data: {args.num_items} items")
        env = KnapsackEnvironment(
            num_items=args.num_items,
            capacity_ratio=args.capacity_ratio,
            batch_size=args.batch_size,
            device=args.device
        )
    
    print(f"Device: {args.device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize networks
    policy = KnapsackPolicyNetwork(
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        use_feasibility_mask=True,
    ).to(args.device)
    
    value_net = KnapsackValueNetwork(
        hidden_dim=args.hidden_dim,
    ).to(args.device)
    
    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(value_net.parameters()),
        lr=args.learning_rate
    )
    
    # Resume from checkpoint if specified
    start_iteration = 1
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        policy.load_state_dict(checkpoint['policy_state_dict'])
        value_net.load_state_dict(checkpoint['value_net_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iteration = checkpoint['iteration'] + 1
        print(f"  Resuming from iteration {start_iteration}")
    
    # Training loop
    pbar = tqdm(range(start_iteration, args.max_iterations + 1), desc='Training')
    
    for iteration in pbar:
        # Curriculum learning: adjust difficulty based on progress
        if args.curriculum and not args.dataset_path:
            progress = (iteration - 1) / args.max_iterations  # 0 to 1
            
            # Interpolate number of items
            curr_items = int(args.curriculum_start_items + 
                           progress * (args.curriculum_end_items - args.curriculum_start_items))
            
            # Interpolate capacity ratio (higher = easier)
            curr_ratio = args.curriculum_start_ratio + \
                        progress * (args.curriculum_end_ratio - args.curriculum_start_ratio)
            
            # Update environment
            env.num_items = curr_items
            env.capacity_ratio = curr_ratio
            
            # Display curriculum info periodically
            if iteration % args.log_interval == 1:
                pbar.set_description(f'Training [N={curr_items}, ratio={curr_ratio:.2f}]')
        
        # For dataset mode, sample a batch of instances
        if args.dataset_path:
            batch_indices = np.random.choice(len(dataset), size=args.batch_size, replace=True)
            env.reset(batch_indices)
        
        data = collect_trajectories(env, policy, value_net, args.device)
        metrics = train_step(policy, value_net, optimizer, data, args)
        
        info = data['info']
        avg_value = info['total_value'].mean().item()
        avg_weight = info['total_weight'].mean().item()
        feasible_ratio = info['feasible'].float().mean().item()
        
        pbar.set_postfix({
            'value': f"{avg_value:.2f}",
            'feasible': f"{feasible_ratio:.2%}",
            'entropy': f"{metrics['entropy']:.4f}",
        })
        
        if iteration % args.log_interval == 0:
            print(f"\nIteration {iteration}:")
            print(f"  Total Value: {avg_value:.2f}")
            print(f"  Total Weight: {avg_weight:.2f}")
            print(f"  Feasible Rate: {feasible_ratio:.2%}")
            print(f"  Policy Loss: {metrics['policy_loss']:.6f}")
            print(f"  Value Loss: {metrics['value_loss']:.6f}")
        
        if iteration % args.save_interval == 0:
            checkpoint = {
                'iteration': iteration,
                'policy_state_dict': policy.state_dict(),
                'value_net_state_dict': value_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': vars(args),
            }
            path = os.path.join(args.output_dir, f'{args.exp_name}_epoch{iteration}.pt')
            torch.save(checkpoint, path)
    
    print("\nTraining completed!")
    
    final_path = os.path.join(args.output_dir, f'{args.exp_name}_final.pt')
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'value_net_state_dict': value_net.state_dict(),
        'args': vars(args),
    }, final_path)
    print(f"Saved final model to {final_path}")


if __name__ == '__main__':
    args = parse_args()
    train(args)
