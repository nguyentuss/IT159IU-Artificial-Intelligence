"""
Training Script for Graph Coloring with PPO.
"""

import argparse
import os
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from ppo_combinatorial.environments.graph_coloring_env import GraphColoringEnvironment, generate_graph_instances
from ppo_combinatorial.models.graph_coloring_model import GraphColoringPolicyNetwork, GraphColoringValueNetwork
from ppo_combinatorial.core.ppo import compute_gae, compute_ppo_loss


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train PPO for Graph Coloring')
    
    # Problem parameters
    parser.add_argument('--num_nodes', type=int, default=50,
                        help='Number of nodes N')
    parser.add_argument('--num_colors', type=int, default=5,
                        help='Number of available colors K')
    parser.add_argument('--edge_probability', type=float, default=0.3,
                        help='Edge probability for random graphs')
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
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help='Gradient clipping')
    parser.add_argument('--num_epochs', type=int, default=4,
                        help='PPO epochs')
    
    # Training parameters
    parser.add_argument('--max_iterations', type=int, default=500,
                        help='Maximum training iterations')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log every N iterations')
    parser.add_argument('--save_interval', type=int, default=100,
                        help='Save every N iterations')
    
    # Model parameters
    parser.add_argument('--embed_dim', type=int, default=64,
                        help='Embedding dimension')
    parser.add_argument('--num_gnn_layers', type=int, default=3,
                        help='Number of GNN layers')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='MLP hidden dimension')
    
    # Device
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/graph_coloring')
    
    return parser.parse_args()


def collect_trajectories(
    env: GraphColoringEnvironment,
    policy: GraphColoringPolicyNetwork,
    value_net: GraphColoringValueNetwork,
    device: str
) -> dict:
    """Collect trajectories for Graph Coloring."""
    states_list = []
    actions_list = []
    rewards_list = []
    log_probs_list = []
    values_list = []
    dones_list = []
    
    state = env.reset()
    
    for t in range(env.num_nodes):
        # Store state
        states_list.append({
            'adjacency': state['adjacency'].clone(),
            'colors': state['colors'].clone(),
            'color_onehot': state['color_onehot'].clone(),
            'degrees': state['degrees'].clone(),
            'is_current_node': state['is_current_node'].clone(),
            'current_node': state['current_node'].clone(),
            'blocked_colors': state['blocked_colors'].clone(),
            'valid_actions': state['valid_actions'].clone(),
            'conflicts': state['conflicts'].clone(),
            'colors_used': state['colors_used'].clone(),
            'step': t,  # Add step index
        })
        
        # Get value estimate
        with torch.no_grad():
            value = value_net(state).squeeze(-1)
        values_list.append(value)
        
        # Sample action (returns 1-indexed color)
        with torch.no_grad():
            action, log_prob, _ = policy.sample_action(state)
        
        actions_list.append(action)
        log_probs_list.append(log_prob)
        
        # Environment step
        next_state, reward, done, info = env.step(action)
        
        rewards_list.append(reward)
        dones_list.append(done.float())
        
        state = next_state
    
    # Final value
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
    policy: GraphColoringPolicyNetwork,
    value_net: GraphColoringValueNetwork,
    optimizer: torch.optim.Optimizer,
    data: dict,
    args
) -> dict:
    """PPO update step for Graph Coloring."""
    states_list = data['states']
    actions = data['actions']
    rewards = data['rewards']
    old_log_probs = data['log_probs']
    old_values = data['values']
    next_values = data['next_values']
    dones = data['dones']
    
    # Compute GAE
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
    
    for epoch in range(args.num_epochs):
        epoch_log_probs = []
        epoch_entropies = []
        epoch_values = []
        
        for t, state in enumerate(states_list):
            # Get current policy output
            action_probs, _ = policy(state)
            dist = torch.distributions.Categorical(action_probs)
            
            # Convert 1-indexed action to 0-indexed
            action_idx = actions[:, t] - 1
            
            log_prob = dist.log_prob(action_idx)
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
    """Main training loop for Graph Coloring."""
    print(f"Training Graph Coloring with {args.num_nodes} nodes, {args.num_colors} colors")
    print(f"Edge probability: {args.edge_probability}")
    print(f"Device: {args.device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize environment
    env = GraphColoringEnvironment(
        num_nodes=args.num_nodes,
        num_colors=args.num_colors,
        edge_probability=args.edge_probability,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # Initialize networks
    policy = GraphColoringPolicyNetwork(
        num_colors=args.num_colors,
        embed_dim=args.embed_dim,
        num_gnn_layers=args.num_gnn_layers,
        hidden_dim=args.hidden_dim,
    ).to(args.device)
    
    value_net = GraphColoringValueNetwork(
        num_colors=args.num_colors,
        embed_dim=args.embed_dim,
        num_gnn_layers=args.num_gnn_layers,
        hidden_dim=args.hidden_dim,
    ).to(args.device)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(value_net.parameters()),
        lr=args.learning_rate
    )
    
    # Training loop
    pbar = tqdm(range(1, args.max_iterations + 1), desc='Training')
    
    for iteration in pbar:
        data = collect_trajectories(env, policy, value_net, args.device)
        metrics = train_step(policy, value_net, optimizer, data, args)
        
        # Metrics from last step
        info = data['info']
        avg_conflicts = info['total_conflicts'].mean().item()
        avg_colors = info['colors_used'].mean().item()
        valid_ratio = info['is_valid_coloring'].float().mean().item()
        
        pbar.set_postfix({
            'conflicts': f"{avg_conflicts:.2f}",
            'colors': f"{avg_colors:.2f}",
            'valid': f"{valid_ratio:.2%}",
        })
        
        if iteration % args.log_interval == 0:
            print(f"\nIteration {iteration}:")
            print(f"  Avg Conflicts: {avg_conflicts:.2f}")
            print(f"  Avg Colors Used: {avg_colors:.2f}")
            print(f"  Valid Coloring Rate: {valid_ratio:.2%}")
            print(f"  Policy Loss: {metrics['policy_loss']:.6f}")
            print(f"  Value Loss: {metrics['value_loss']:.6f}")
            print(f"  Entropy: {metrics['entropy']:.4f}")
        
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
