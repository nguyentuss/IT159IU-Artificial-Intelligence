"""
Training Script for Knapsack Problem with PPO.

Uses static datasets (p01-p08) for curriculum learning.
"""

import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from ppo_combinatorial.models.knapsack_model import KnapsackPolicyNetwork, KnapsackValueNetwork
from ppo_combinatorial.core.ppo import compute_gae, compute_ppo_loss


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train PPO for Knapsack')
    
    # Dataset
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset folder (e.g., data/knapsack/p01)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Number of parallel environments (copies of same instance)')
    
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
                        help='PPO epochs per iteration')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log every N epochs')
    parser.add_argument('--save_interval', type=int, default=100,
                        help='Save every N epochs')
    
    # Model parameters
    parser.add_argument('--embed_dim', type=int, default=64,
                        help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='MLP hidden dimension')
    
    # Device and output
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/knapsack')
    parser.add_argument('--exp_name', type=str, default='knapsack',
                        help='Experiment name for checkpoint files')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    
    return parser.parse_args()


def load_knapsack_dataset(data_dir, device='cpu'):
    """
    Load knapsack dataset from folder.
    
    Expected files:
    - pXX_c.txt: capacity (single number)
    - pXX_w.txt: weights (one per line)
    - pXX_p.txt: profits/values (one per line)
    - pXX_s.txt: optimal solution (one 0/1 per line)
    """
    # Find the prefix (p01, p02, etc.)
    prefix = os.path.basename(data_dir)
    
    # Read capacity
    with open(os.path.join(data_dir, f'{prefix}_c.txt'), 'r') as f:
        capacity = int(f.read().strip())
    
    # Read weights
    with open(os.path.join(data_dir, f'{prefix}_w.txt'), 'r') as f:
        weights = [int(line.strip()) for line in f if line.strip()]
    
    # Read values/profits
    with open(os.path.join(data_dir, f'{prefix}_p.txt'), 'r') as f:
        values = [int(line.strip()) for line in f if line.strip()]
    
    # Read optimal solution
    with open(os.path.join(data_dir, f'{prefix}_s.txt'), 'r') as f:
        optimal_selection = [int(line.strip()) for line in f if line.strip()]
    
    # Compute optimal value
    optimal_value = sum(v * s for v, s in zip(values, optimal_selection))
    
    return {
        'weights': torch.tensor(weights, dtype=torch.float32, device=device),
        'values': torch.tensor(values, dtype=torch.float32, device=device),
        'capacity': capacity,
        'optimal_value': optimal_value,
        'optimal_selection': optimal_selection,
        'num_items': len(weights),
        'name': prefix,
    }


class StaticKnapsackEnvironment:
    """Environment for a single static knapsack instance."""
    
    def __init__(self, dataset, batch_size=128, device='cpu'):
        self.values = dataset['values'].unsqueeze(0).expand(batch_size, -1).clone()
        self.weights = dataset['weights'].unsqueeze(0).expand(batch_size, -1).clone()
        self.capacity = dataset['capacity']
        self.num_items = dataset['num_items']
        self.batch_size = batch_size
        self.device = device
        
        self.total_value = self.values.sum(dim=-1)
        self.total_weight = self.weights.sum(dim=-1)
        
        # State variables
        self.current_weight = None
        self.current_value = None
        self.step_count = None
    
    def reset(self):
        """Reset environment."""
        self.current_weight = torch.zeros(self.batch_size, device=self.device)
        self.current_value = torch.zeros(self.batch_size, device=self.device)
        self.step_count = 0
        return self._get_state()
    
    def _get_state(self):
        """Get current state representation."""
        t = self.step_count
        
        # Normalize features
        mean_value = self.values.mean(dim=-1, keepdim=True).clamp(min=1e-8)
        mean_weight = self.weights.mean(dim=-1, keepdim=True).clamp(min=1e-8)
        value_weight_ratio = self.values / (self.weights + 1e-8)
        mean_ratio = value_weight_ratio.mean(dim=-1, keepdim=True).clamp(min=1e-8)
        
        item_features = torch.stack([
            self.values / mean_value,
            self.weights / mean_weight,
            value_weight_ratio / mean_ratio,
            self.weights / self.capacity,
        ], dim=-1)
        
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
            self.current_value / self.total_value.clamp(min=1e-8),
            torch.full((self.batch_size,), t / self.num_items, device=self.device),
            remaining_capacity / current_item_weight.clamp(min=1e-8),
        ], dim=-1)
        
        can_add = remaining_capacity >= current_item_weight
        
        return {
            'item_features': item_features,
            'context_features': context_features,
            'current_item_idx': torch.full((self.batch_size,), t, dtype=torch.long, device=self.device),
            'can_add_item': can_add,
            'values': self.values,
            'weights': self.weights,
            'capacity': torch.full((self.batch_size,), self.capacity, device=self.device),
            'current_weight': self.current_weight,
            'current_value': self.current_value,
            'step': t,
        }
    
    def step(self, action):
        """Execute action."""
        action = action.to(self.device).float()
        t = self.step_count
        
        current_weight = self.weights[:, t]
        current_value = self.values[:, t]
        
        # Dense reward with penalty for exceeding capacity
        normalized_value = current_value / self.total_value.clamp(min=1e-8)
        new_weight = self.current_weight + action * current_weight
        overflow = torch.clamp(new_weight - self.capacity, min=0)
        normalized_overflow = overflow / self.capacity
        
        reward = action * (normalized_value - 2.0 * normalized_overflow)
        
        # Update state
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
        
        return self._get_state(), reward, done_tensor, info


def collect_trajectories(env, policy, value_net, device):
    """Collect trajectories."""
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
    """PPO update step."""
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
    """Main training loop."""
    # Load dataset
    print(f"Loading dataset from: {args.data_dir}")
    dataset = load_knapsack_dataset(args.data_dir, args.device)
    
    print(f"  Instance: {dataset['name']}")
    print(f"  Items: {dataset['num_items']}")
    print(f"  Capacity: {dataset['capacity']}")
    print(f"  Optimal value: {dataset['optimal_value']}")
    print(f"Device: {args.device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create environment
    env = StaticKnapsackEnvironment(dataset, args.batch_size, args.device)
    
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
    
    # Resume from checkpoint (only loads weights, not epoch - for curriculum learning)
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        policy.load_state_dict(checkpoint['policy_state_dict'])
        value_net.load_state_dict(checkpoint['value_net_state_dict'])
        # Don't restore optimizer state or epoch for curriculum (new dataset)
        print(f"  Loaded weights, starting fresh epochs for new dataset")
    
    # Training loop
    pbar = tqdm(range(1, args.epochs + 1), desc=f'Training on {dataset["name"]}')
    
    for epoch in pbar:
        data = collect_trajectories(env, policy, value_net, args.device)
        metrics = train_step(policy, value_net, optimizer, data, args)
        
        info = data['info']
        avg_value = info['total_value'].mean().item()
        feasible_ratio = info['feasible'].float().mean().item()
        opt_ratio = avg_value / dataset['optimal_value'] * 100
        
        pbar.set_postfix({
            'value': f"{avg_value:.1f}/{dataset['optimal_value']}",
            'opt%': f"{opt_ratio:.1f}%",
            'feasible': f"{feasible_ratio:.0%}",
        })
        
        if epoch % args.log_interval == 0:
            print(f"\nEpoch {epoch}:")
            print(f"  Value: {avg_value:.2f} / {dataset['optimal_value']} ({opt_ratio:.1f}%)")
            print(f"  Feasible: {feasible_ratio:.0%}")
            print(f"  Policy Loss: {metrics['policy_loss']:.6f}")
        
        if epoch % args.save_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'policy_state_dict': policy.state_dict(),
                'value_net_state_dict': value_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': vars(args),
                'dataset': dataset['name'],
            }
            path = os.path.join(args.output_dir, f'{args.exp_name}_epoch{epoch}.pt')
            torch.save(checkpoint, path)
    
    print("\nTraining completed!")
    
    # Save final model
    final_path = os.path.join(args.output_dir, f'{args.exp_name}_final.pt')
    torch.save({
        'epoch': args.epochs,
        'policy_state_dict': policy.state_dict(),
        'value_net_state_dict': value_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args),
        'dataset': dataset['name'],
    }, final_path)
    print(f"Saved final model to {final_path}")


if __name__ == '__main__':
    args = parse_args()
    train(args)
