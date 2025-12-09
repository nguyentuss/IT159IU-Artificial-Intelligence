"""
Training Script for Graph Coloring with PPO.

Uses static DIMACS .col format graphs for curriculum learning.
"""

import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from ppo_combinatorial.models.graph_coloring_model import GraphColoringPolicyNetwork, GraphColoringValueNetwork
from ppo_combinatorial.core.ppo import compute_gae, compute_ppo_loss


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train PPO for Graph Coloring')
    
    # Dataset
    parser.add_argument('--graph_file', type=str, required=True,
                        help='Path to DIMACS .col graph file')
    parser.add_argument('--num_colors', type=int, default=5,
                        help='Number of available colors K')
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
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log every N epochs')
    parser.add_argument('--save_interval', type=int, default=100,
                        help='Save every N epochs')
    
    # Model parameters
    parser.add_argument('--embed_dim', type=int, default=64,
                        help='Embedding dimension')
    parser.add_argument('--num_gnn_layers', type=int, default=3,
                        help='Number of GNN layers')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='MLP hidden dimension')
    
    # Device and output
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/graph_coloring')
    parser.add_argument('--exp_name', type=str, default='graph_coloring',
                        help='Experiment name for checkpoint files')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    
    return parser.parse_args()


def load_dimacs_graph(file_path, device='cpu'):
    """
    Load graph from DIMACS .col format or simple edge list format.
    
    DIMACS format:
        c comment lines
        p edge <num_nodes> <num_edges>
        e <node1> <node2>
    
    Simple format:
        <num_nodes> <num_edges>
        <node1> <node2>
        ...
    """
    num_nodes = 0
    num_edges_expected = 0
    edges = []
    is_dimacs = False
    first_line = True
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('c'):
                continue
            
            parts = line.split()
            
            # Check for DIMACS format
            if parts[0] == 'p':
                is_dimacs = True
                num_nodes = int(parts[2])
                num_edges_expected = int(parts[3])
            elif parts[0] == 'e':
                # DIMACS edge line
                u = int(parts[1]) - 1
                v = int(parts[2]) - 1
                edges.append((u, v))
            elif first_line and len(parts) == 2:
                # Simple format: first line is "num_nodes num_edges"
                try:
                    num_nodes = int(parts[0])
                    num_edges_expected = int(parts[1])
                    first_line = False
                except ValueError:
                    pass
            elif not is_dimacs and len(parts) >= 2:
                # Simple format: edge lines are "node1 node2"
                try:
                    u = int(parts[0]) - 1
                    v = int(parts[1]) - 1
                    edges.append((u, v))
                except ValueError:
                    pass
            
            first_line = False
    
    if num_nodes == 0:
        raise ValueError(f"Could not parse graph from {file_path}")
    
    # Build adjacency matrix
    adj = torch.zeros(num_nodes, num_nodes, device=device)
    for u, v in edges:
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            adj[u, v] = 1
            adj[v, u] = 1
    
    # Compute degrees
    degrees = adj.sum(dim=-1)
    
    return {
        'adjacency': adj,
        'num_nodes': num_nodes,
        'num_edges': len(edges),
        'degrees': degrees,
        'name': os.path.basename(file_path),
    }


class StaticGraphColoringEnvironment:
    """Environment for a single static graph coloring instance."""
    
    def __init__(self, graph, num_colors, batch_size=128, device='cpu'):
        self.num_nodes = graph['num_nodes']
        self.num_colors = num_colors
        self.batch_size = batch_size
        self.device = device
        
        # Expand graph for batch
        self.adjacency = graph['adjacency'].unsqueeze(0).expand(batch_size, -1, -1).clone()
        self.degrees = graph['degrees'].unsqueeze(0).expand(batch_size, -1).clone()
        
        # State variables
        self.colors = None
        self.step_count = None
        self.colors_used = None
        self.conflicts = None
    
    def reset(self):
        """Reset environment."""
        self.colors = torch.zeros(self.batch_size, self.num_nodes, dtype=torch.long, device=self.device)
        self.step_count = 0
        self.colors_used = torch.zeros(self.batch_size, device=self.device)
        self.conflicts = torch.zeros(self.batch_size, device=self.device)
        return self._get_state()
    
    def _get_state(self):
        """Get current state (fully vectorized)."""
        t = min(self.step_count, self.num_nodes - 1)
        
        # Color one-hot encoding - VECTORIZED
        # self.colors is [batch, nodes], values 0 to num_colors
        # Create one-hot: [batch, nodes, num_colors+1]
        color_onehot = torch.zeros(self.batch_size, self.num_nodes, self.num_colors + 1, device=self.device)
        batch_idx = torch.arange(self.batch_size, device=self.device).unsqueeze(1).expand(-1, self.num_nodes)
        node_idx = torch.arange(self.num_nodes, device=self.device).unsqueeze(0).expand(self.batch_size, -1)
        color_onehot[batch_idx, node_idx, self.colors] = 1.0
        
        # Current node indicator
        is_current = torch.zeros(self.batch_size, self.num_nodes, device=self.device)
        is_current[:, t] = 1.0
        
        # Blocked colors for current node - VECTORIZED
        # Get neighbor mask for node t: [batch, num_nodes]
        neighbor_mask = self.adjacency[:, t, :]  # [batch, num_nodes]
        
        # Get colors of all nodes: [batch, num_nodes]
        neighbor_colors = self.colors  # [batch, num_nodes]
        
        # Create blocked tensor using scatter: [batch, num_colors]
        # neighbor_colors: [batch, num_nodes], neighbor_mask: [batch, num_nodes]
        # We want to mark color c as blocked if any neighbor has color c
        
        # Create one-hot for neighbor colors (only for colors 1..num_colors)
        # neighbor_colors_one_hot: [batch, num_nodes, num_colors]
        neighbor_colors_one_hot = torch.zeros(self.batch_size, self.num_nodes, self.num_colors, device=self.device)
        valid_color_mask = (neighbor_colors >= 1) & (neighbor_colors <= self.num_colors)
        # Scatter: set to 1 where color matches
        color_indices = (neighbor_colors - 1).clamp(min=0)  # 0-indexed for colors 1..num_colors
        neighbor_colors_one_hot.scatter_(2, color_indices.unsqueeze(-1), 1.0)
        # Zero out where original color was 0 (uncolored)
        neighbor_colors_one_hot = neighbor_colors_one_hot * valid_color_mask.unsqueeze(-1).float()
        
        # Multiply by neighbor_mask and sum: blocked[batch, c] = sum over nodes of (neighbor_mask * has_color_c)
        # neighbor_mask: [batch, num_nodes] -> [batch, num_nodes, 1]
        blocked = (neighbor_mask.unsqueeze(-1) * neighbor_colors_one_hot).sum(dim=1) > 0
        blocked = blocked.float()
        
        # Valid actions (unblocked colors) - ensure at least one is valid
        valid = (blocked == 0)
        all_blocked = ~valid.any(dim=-1, keepdim=True)
        valid = valid | all_blocked.expand_as(valid)
        
        return {
            'adjacency': self.adjacency,
            'colors': self.colors,
            'color_onehot': color_onehot,
            'degrees': self.degrees,
            'is_current_node': is_current,
            'current_node': torch.full((self.batch_size,), t, dtype=torch.long, device=self.device),
            'blocked_colors': blocked,
            'valid_actions': valid,
            'conflicts': self.conflicts.clone(),
            'colors_used': self.colors_used.clone(),
            'step': t,
        }
    
    def step(self, action):
        """Execute action (1-indexed color) - fully vectorized."""
        t = self.step_count
        
        # Check for conflicts - VECTORIZED
        # Get neighbor mask for node t
        neighbor_mask = self.adjacency[:, t, :]  # [batch, num_nodes]
        
        # Get colors of neighbors
        neighbor_colors = self.colors  # [batch, num_nodes]
        
        # Check which neighbors have the same color as action
        # action is [batch], expand to [batch, num_nodes]
        action_expanded = action.unsqueeze(1).expand(-1, self.num_nodes)
        
        # Conflict if neighbor has same color AND is a neighbor
        same_color = (neighbor_colors == action_expanded)  # [batch, num_nodes]
        new_conflicts = (neighbor_mask * same_color).sum(dim=1).float()  # [batch]
        
        # Track new colors - VECTORIZED
        # Create mask for all colors used so far (excluding 0)
        # colors_one_hot shape: [batch, num_colors] indicating which colors 1..num_colors are used
        # Check if action color exists in current colors
        batch_idx = torch.arange(self.batch_size, device=self.device)
        
        # For each batch, check if action[i] already exists in self.colors[i]
        # Expand action to compare with all nodes
        action_match = (self.colors == action.unsqueeze(1))  # [batch, num_nodes]
        color_exists = action_match.any(dim=1)  # [batch]
        
        was_new_color = (~color_exists).float()
        self.colors_used += was_new_color
        
        # Assign colors
        self.colors[:, t] = action
        self.conflicts += new_conflicts
        self.step_count += 1
        
        # Reward: penalize conflicts and new colors
        reward = -1.0 * new_conflicts - 0.1 * was_new_color
        
        done = self.step_count >= self.num_nodes
        done_tensor = torch.full((self.batch_size,), done, dtype=torch.bool, device=self.device)
        
        info = {}
        if done:
            info['total_conflicts'] = self.conflicts.clone()
            info['colors_used'] = self.colors_used.clone()
            info['is_valid_coloring'] = (self.conflicts == 0)
        
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
    
    for t in range(env.num_nodes):
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
            
            # Convert 1-indexed action to 0-indexed for log_prob
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
    """Main training loop."""
    # Load graph
    print(f"Loading graph from: {args.graph_file}")
    graph = load_dimacs_graph(args.graph_file, args.device)
    
    print(f"  Graph: {graph['name']}")
    print(f"  Nodes: {graph['num_nodes']}")
    print(f"  Edges: {graph['num_edges']}")
    print(f"  Colors: {args.num_colors}")
    print(f"Device: {args.device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create environment
    env = StaticGraphColoringEnvironment(graph, args.num_colors, args.batch_size, args.device)
    
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
    pbar = tqdm(range(1, args.epochs + 1), desc=f'Training on {graph["name"]}')
    
    for epoch in pbar:
        data = collect_trajectories(env, policy, value_net, args.device)
        metrics = train_step(policy, value_net, optimizer, data, args)
        
        info = data['info']
        avg_conflicts = info['total_conflicts'].mean().item()
        avg_colors = info['colors_used'].mean().item()
        valid_ratio = info['is_valid_coloring'].float().mean().item()
        
        pbar.set_postfix({
            'conflicts': f"{avg_conflicts:.1f}",
            'colors': f"{avg_colors:.1f}",
            'valid': f"{valid_ratio:.0%}",
        })
        
        if epoch % args.log_interval == 0:
            print(f"\nEpoch {epoch}:")
            print(f"  Conflicts: {avg_conflicts:.2f}")
            print(f"  Colors: {avg_colors:.2f}")
            print(f"  Valid: {valid_ratio:.0%}")
            print(f"  Policy Loss: {metrics['policy_loss']:.6f}")
        
        if epoch % args.save_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'policy_state_dict': policy.state_dict(),
                'value_net_state_dict': value_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': vars(args),
                'graph': graph['name'],
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
        'graph': graph['name'],
    }, final_path)
    print(f"Saved final model to {final_path}")


if __name__ == '__main__':
    args = parse_args()
    train(args)
