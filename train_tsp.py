"""
Training Script for TSP with PPO.

Uses static datasets (CSV files with city coordinates).
"""

import argparse
import json
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from ppo_combinatorial.models.tsp_model import TSPPolicyNetwork, TSPValueNetwork
from ppo_combinatorial.core.ppo import compute_gae, compute_ppo_loss
from solve_tsp_optimal import solve_tsp_optimal


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train PPO for TSP')
    
    # Dataset
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to CSV file with city coordinates')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Number of parallel environments')
    
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
    parser.add_argument('--learning_rate', type=float, default=1e-4,
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
    parser.add_argument('--embed_dim', type=int, default=128,
                        help='Embedding dimension')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of transformer layers')
    
    # Device and output
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/tsp')
    parser.add_argument('--exp_name', type=str, default='tsp',
                        help='Experiment name for checkpoint files')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    return parser.parse_args()


def load_tsp_dataset(file_path, device='cpu'):
    """Load TSP instance from CSV file (x,y coordinates per line)."""
    coords = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                x, y = map(float, line.strip().split(','))
                coords.append([x, y])
    
    coords = torch.tensor(coords, dtype=torch.float32, device=device)
    num_cities = len(coords)
    
    return {
        'coordinates': coords,
        'num_cities': num_cities,
        'name': os.path.basename(file_path),
    }


class StaticTSPEnvironment:
    """Environment for a single static TSP instance."""
    
    def __init__(self, dataset, batch_size=128, device='cpu'):
        self.num_cities = dataset['num_cities']
        self.batch_size = batch_size
        self.device = device
        
        # Expand coordinates for batch
        self.coordinates = dataset['coordinates'].unsqueeze(0).expand(batch_size, -1, -1).clone()
        
        # Compute distance matrix
        diff = self.coordinates.unsqueeze(2) - self.coordinates.unsqueeze(1)
        self.distances = torch.sqrt((diff ** 2).sum(dim=-1))
        
        # State variables
        self.visited = None
        self.current_city = None
        self.start_city = None
        self.tour_length = None
        self.step_count = None
    
    def reset(self):
        """Reset environment."""
        # Start from city 0
        self.start_city = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        self.current_city = self.start_city.clone()
        
        self.visited = torch.zeros(self.batch_size, self.num_cities, dtype=torch.bool, device=self.device)
        self.visited[:, 0] = True
        
        self.tour_length = torch.zeros(self.batch_size, device=self.device)
        self.step_count = 0
        
        return self._get_state()
    
    def _get_state(self):
        """Get current state."""
        action_mask = ~self.visited
        
        return {
            'coordinates': self.coordinates,
            'action_mask': action_mask,
            'current_city': self.current_city,
            'start_city': self.start_city,
            'visited': self.visited,
            'tour_length': self.tour_length,
            'distances': self.distances,
            'step': self.step_count,
        }
    
    def step(self, action):
        """Execute action (visit next city)."""
        # Get distance to next city
        batch_idx = torch.arange(self.batch_size, device=self.device)
        step_distance = self.distances[batch_idx, self.current_city, action]
        
        # Update tour
        self.tour_length += step_distance
        self.visited[batch_idx, action] = True
        self.current_city = action
        self.step_count += 1
        
        # Check if done (visited all cities)
        done = self.step_count >= self.num_cities - 1
        
        # Reward: negative distance (we want to minimize tour length)
        # Normalize by approximate tour length
        max_dist = self.distances.max(dim=-1)[0].max(dim=-1)[0]
        normalized_dist = step_distance / (max_dist + 1e-8)
        reward = -normalized_dist
        
        # Add return distance on last step
        if done:
            return_distance = self.distances[batch_idx, self.current_city, self.start_city]
            self.tour_length += return_distance
            reward -= return_distance / (max_dist + 1e-8)
        
        done_tensor = torch.full((self.batch_size,), done, dtype=torch.bool, device=self.device)
        
        info = {}
        if done:
            info['tour_length'] = self.tour_length.clone()
        
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
    num_steps = env.num_cities - 1
    
    for t in range(num_steps):
        states_list.append({
            'coordinates': state['coordinates'].clone(),
            'action_mask': state['action_mask'].clone(),
            'current_city': state['current_city'].clone(),
            'start_city': state['start_city'].clone(),
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
    print(f"Loading dataset from: {args.data_file}")
    dataset = load_tsp_dataset(args.data_file, args.device)
    
    print(f"  Instance: {dataset['name']}")
    print(f"  Cities: {dataset['num_cities']}")
    print(f"Device: {args.device}")
    
    # Compute optimal tour
    print("Computing optimal tour...")
    optimal_tour, method = solve_tsp_optimal(args.data_file)
    print(f"  Optimal tour: {optimal_tour:.4f} ({method})")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create environment
    env = StaticTSPEnvironment(dataset, args.batch_size, args.device)
    
    # Initialize networks
    policy = TSPPolicyNetwork(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    ).to(args.device)
    
    value_net = TSPValueNetwork(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    ).to(args.device)
    
    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(value_net.parameters()),
        lr=args.learning_rate
    )
    
    # Resume from checkpoint (only loads weights for curriculum)
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        policy.load_state_dict(checkpoint['policy_state_dict'])
        value_net.load_state_dict(checkpoint['value_net_state_dict'])
        print(f"  Loaded weights, starting fresh epochs for new dataset")
    
    # Training loop
    history = []  # Track training history
    best_tour = float('inf')
    pbar = tqdm(range(1, args.epochs + 1), desc=f'Training on {dataset["name"]}')
    
    for epoch in pbar:
        data = collect_trajectories(env, policy, value_net, args.device)
        metrics = train_step(policy, value_net, optimizer, data, args)
        
        info = data['info']
        avg_tour = info['tour_length'].mean().item()
        min_tour = info['tour_length'].min().item()
        
        if min_tour < best_tour:
            best_tour = min_tour
        
        pbar.set_postfix({
            'tour': f"{avg_tour:.2f}",
            'best': f"{best_tour:.2f}",
        })
        
        # Record history every epoch
        history.append({
            'epoch': epoch,
            'avg_tour': avg_tour,
            'min_tour': min_tour,
            'best_tour': best_tour,
            'policy_loss': metrics['policy_loss'],
            'value_loss': metrics['value_loss'],
            'entropy': metrics['entropy'],
        })
        
        if epoch % args.log_interval == 0:
            print(f"\nEpoch {epoch}:")
            print(f"  Avg Tour: {avg_tour:.4f}")
            print(f"  Best Tour: {best_tour:.4f}")
            print(f"  Policy Loss: {metrics['policy_loss']:.6f}")
        
        if epoch % args.save_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'policy_state_dict': policy.state_dict(),
                'value_net_state_dict': value_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': vars(args),
                'dataset': dataset['name'],
                'best_tour': best_tour,
            }
            path = os.path.join(args.output_dir, f'{args.exp_name}_epoch{epoch}.pt')
            torch.save(checkpoint, path)
    
    print(f"\nTraining completed!")
    print(f"Best tour length: {best_tour:.4f}")
    
    # Save final model
    final_path = os.path.join(args.output_dir, f'{args.exp_name}_final.pt')
    torch.save({
        'epoch': args.epochs,
        'policy_state_dict': policy.state_dict(),
        'value_net_state_dict': value_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args),
        'dataset': dataset['name'],
        'best_tour': best_tour,
    }, final_path)
    print(f"Saved final model to {final_path}")
    
    # Save training history
    history_path = os.path.join(args.output_dir, f'{args.exp_name}_history.json')
    with open(history_path, 'w') as f:
        json.dump({
            'problem_type': 'tsp',
            'dataset': dataset['name'],
            'num_cities': dataset['num_cities'],
            'optimal_tour': optimal_tour,
            'best_tour': best_tour,
            'args': vars(args),
            'history': history,
        }, f, indent=2)
    print(f"Saved training history to {history_path}")


if __name__ == '__main__':
    args = parse_args()
    train(args)
