"""
Training Script for TSP with PPO.
"""

import argparse
import os
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from ppo_combinatorial.environments.tsp_env import TSPEnvironment, generate_tsp_instances
from ppo_combinatorial.models.tsp_model import TSPPolicyNetwork, TSPValueNetwork
from ppo_combinatorial.core.ppo import compute_gae, compute_ppo_loss


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train PPO for TSP')
    
    # Problem parameters
    parser.add_argument('--num_cities', type=int, default=20,
                        help='Number of cities in TSP instances')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Number of parallel environments')
    
    # PPO hyperparameters
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='Discount factor (use 1.0 for finite episodes)')
    parser.add_argument('--lambda_gae', type=float, default=0.95,
                        help='GAE parameter')
    parser.add_argument('--clip_epsilon', type=float, default=0.2,
                        help='PPO clipping parameter')
    parser.add_argument('--value_coef', type=float, default=0.5,
                        help='Value loss coefficient')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                        help='Entropy coefficient')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Adam learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help='Gradient clipping')
    parser.add_argument('--num_epochs', type=int, default=4,
                        help='PPO epochs per batch')
    
    # Training parameters
    parser.add_argument('--max_iterations', type=int, default=1000,
                        help='Maximum training iterations')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log every N iterations')
    parser.add_argument('--save_interval', type=int, default=100,
                        help='Save checkpoint every N iterations')
    
    # Model parameters
    parser.add_argument('--embed_dim', type=int, default=128,
                        help='Embedding dimension')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of transformer layers')
    
    # Device
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu or cuda)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./checkpoints/tsp',
                        help='Directory to save checkpoints')
    
    return parser.parse_args()


def collect_trajectories(
    env: TSPEnvironment,
    policy: TSPPolicyNetwork,
    value_net: TSPValueNetwork,
    device: str
) -> dict:
    """Collect trajectories by running policy in environment."""
    states_list = []
    actions_list = []
    rewards_list = []
    log_probs_list = []
    values_list = []
    dones_list = []
    
    # Reset environment and get initial state
    state = env.reset()
    
    num_steps = env.num_cities - 1  # N-1 steps to visit all cities
    
    for t in range(num_steps):
        # Store current state info
        states_list.append({
            'coordinates': state['coordinates'].clone(),
            'action_mask': state['action_mask'].clone(),
            'current_city': state['current_city'].clone(),
            'start_city': state['start_city'].clone(),
            'step': t,  # Add step index
        })
        
        # Get value estimate
        with torch.no_grad():
            value = value_net(state).squeeze(-1)
        values_list.append(value)
        
        # Sample action from policy
        with torch.no_grad():
            action, log_prob, _ = policy.sample_action(state)
        
        actions_list.append(action)
        log_probs_list.append(log_prob)
        
        # Take environment step
        next_state, reward, done, info = env.step(action)
        
        rewards_list.append(reward)
        dones_list.append(done.float())
        
        state = next_state
    
    # Get final value for bootstrapping
    with torch.no_grad():
        final_value = value_net(state).squeeze(-1)
    
    # Stack tensors
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


def train_step(
    policy: TSPPolicyNetwork,
    value_net: TSPValueNetwork,
    optimizer: torch.optim.Optimizer,
    data: dict,
    args
) -> dict:
    """Perform PPO update step."""
    states_list = data['states']
    actions = data['actions']
    rewards = data['rewards']
    old_log_probs = data['log_probs']
    old_values = data['values']
    next_values = data['next_values']
    dones = data['dones']
    
    # Compute advantages using GAE
    advantages, returns = compute_gae(
        rewards=rewards,
        values=old_values,
        next_values=next_values,
        dones=dones,
        gamma=args.gamma,
        lambda_gae=args.lambda_gae
    )
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    all_metrics = []
    
    # Multiple epochs of PPO updates
    for epoch in range(args.num_epochs):
        epoch_log_probs = []
        epoch_entropies = []
        epoch_values = []
        
        # Re-evaluate policy for each timestep
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
        
        # Compute PPO loss
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
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        nn.utils.clip_grad_norm_(
            list(policy.parameters()) + list(value_net.parameters()),
            args.max_grad_norm
        )
        
        optimizer.step()
        all_metrics.append(metrics)
    
    # Average metrics across epochs
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }
    
    return avg_metrics


def train(args):
    """Main training loop for TSP."""
    print(f"Training TSP with {args.num_cities} cities")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize environment
    env = TSPEnvironment(
        num_cities=args.num_cities,
        batch_size=args.batch_size,
        device=args.device
    )
    
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
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(value_net.parameters()),
        lr=args.learning_rate
    )
    
    # Training loop
    pbar = tqdm(range(1, args.max_iterations + 1), desc='Training')
    
    for iteration in pbar:
        # Collect trajectories
        data = collect_trajectories(env, policy, value_net, args.device)
        
        # PPO update
        metrics = train_step(policy, value_net, optimizer, data, args)
        
        # Compute tour length from rewards
        total_reward = data['rewards'].sum(dim=1).mean().item()
        
        # Update progress bar
        pbar.set_postfix({
            'policy_loss': f"{metrics['policy_loss']:.4f}",
            'value_loss': f"{metrics['value_loss']:.4f}",
            'entropy': f"{metrics['entropy']:.4f}",
            'reward': f"{total_reward:.4f}",
        })
        
        # Logging
        if iteration % args.log_interval == 0:
            print(f"\nIteration {iteration}:")
            print(f"  Policy Loss: {metrics['policy_loss']:.6f}")
            print(f"  Value Loss: {metrics['value_loss']:.6f}")
            print(f"  Entropy: {metrics['entropy']:.6f}")
            print(f"  Total Reward: {total_reward:.4f}")
        
        # Save checkpoint
        if iteration % args.save_interval == 0:
            checkpoint = {
                'iteration': iteration,
                'policy_state_dict': policy.state_dict(),
                'value_net_state_dict': value_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': vars(args),
            }
            path = os.path.join(args.output_dir, f'checkpoint_{iteration}.pt')
            torch.save(checkpoint, path)
            print(f"Saved checkpoint to {path}")
    
    print("\nTraining completed!")
    
    # Save final model
    final_path = os.path.join(args.output_dir, 'final_model.pt')
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'value_net_state_dict': value_net.state_dict(),
        'args': vars(args),
    }, final_path)
    print(f"Saved final model to {final_path}")


if __name__ == '__main__':
    args = parse_args()
    train(args)
