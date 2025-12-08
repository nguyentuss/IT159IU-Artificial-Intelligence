"""
Evaluation Script for Knapsack Problem.

Compares trained PPO model against baselines:
- Random policy
- Greedy by value/weight ratio
- Optimal solution (from dataset)
"""

import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from ppo_combinatorial.models.knapsack_model import KnapsackPolicyNetwork, KnapsackValueNetwork


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
            'capacity': capacity,
            'optimal': optimal,
        })
    
    return all_data


def greedy_solve(values, weights, capacity):
    """Greedy algorithm: Select items by value/weight ratio."""
    n = len(values)
    ratio = values / (weights + 1e-8)
    sorted_indices = np.argsort(-ratio)  # Descending
    
    total_value = 0
    total_weight = 0
    
    for idx in sorted_indices:
        if total_weight + weights[idx] <= capacity:
            total_value += values[idx]
            total_weight += weights[idx]
    
    return total_value, total_weight


def random_solve(values, weights, capacity, num_trials=10):
    """Random policy: Average over multiple random trials."""
    n = len(values)
    best_value = 0
    
    for _ in range(num_trials):
        total_value = 0
        total_weight = 0
        
        for i in range(n):
            if np.random.rand() > 0.5 and total_weight + weights[i] <= capacity:
                total_value += values[i]
                total_weight += weights[i]
        
        best_value = max(best_value, total_value)
    
    return best_value


def ppo_solve_instance(policy, values, weights, capacity, max_items, device):
    """Run PPO policy on a single instance."""
    n = len(values)
    batch_size = 1
    
    # Pad to max_items
    padded_values = torch.zeros(max_items, device=device)
    padded_weights = torch.zeros(max_items, device=device)
    padded_values[:n] = values
    padded_weights[:n] = weights
    
    values_t = padded_values.unsqueeze(0)
    weights_t = padded_weights.unsqueeze(0)
    capacity_t = torch.tensor([capacity], device=device)
    
    total_value_t = values_t.sum(dim=-1)
    total_weight_t = weights_t.sum(dim=-1)
    
    current_weight = torch.zeros(1, device=device)
    current_value = torch.zeros(1, device=device)
    
    for t in range(max_items):
        # Build state
        mean_value = values_t.mean(dim=-1, keepdim=True).clamp(min=1e-8)
        mean_weight = weights_t.mean(dim=-1, keepdim=True).clamp(min=1e-8)
        value_weight_ratio = values_t / (weights_t + 1e-8)
        mean_ratio = value_weight_ratio.mean(dim=-1, keepdim=True).clamp(min=1e-8)
        
        item_features = torch.stack([
            values_t / mean_value,
            weights_t / mean_weight,
            value_weight_ratio / mean_ratio,
            weights_t / capacity_t.unsqueeze(-1).clamp(min=1e-8),
        ], dim=-1)
        
        remaining_capacity = capacity_t - current_weight
        current_item_weight = weights_t[:, t]
        
        context_features = torch.stack([
            current_weight / capacity_t.clamp(min=1e-8),
            remaining_capacity / capacity_t.clamp(min=1e-8),
            current_value / total_value_t.clamp(min=1e-8),
            torch.full((1,), t / max_items, device=device),
            remaining_capacity / current_item_weight.clamp(min=1e-8),
        ], dim=-1)
        
        can_add = remaining_capacity >= current_item_weight
        
        state = {
            'item_features': item_features,
            'context_features': context_features,
            'current_item_idx': torch.tensor([t], dtype=torch.long, device=device),
            'can_add_item': can_add,
            'values': values_t,
            'weights': weights_t,
            'capacity': capacity_t,
            'current_weight': current_weight,
            'current_value': current_value,
            'step': t,
        }
        
        with torch.no_grad():
            action, _, _ = policy.sample_action(state)
        
        # Execute action
        if action.item() == 1:
            current_weight = current_weight + current_item_weight
            current_value = current_value + values_t[:, t]
    
    return current_value.item(), current_weight.item()


def evaluate(args):
    """Main evaluation function."""
    print("=" * 60)
    print("KNAPSACK EVALUATION (Dataset)")
    print("=" * 60)
    print(f"Dataset: {args.dataset_path}")
    print(f"Device: {args.device}")
    print()
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(args.dataset_path, args.device)
    
    print(f"  Loaded {len(dataset)} instances")
    
    # Limit instances if specified
    if args.max_instances and args.max_instances < len(dataset):
        dataset = dataset[:args.max_instances]
        print(f"  Using first {len(dataset)} instances")
    
    # Find max items
    max_items = max(inst['n'] for inst in dataset)
    print(f"  Max items: {max_items}")
    
    # Load trained model
    print(f"\nLoading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=args.device)
    
    policy = KnapsackPolicyNetwork(
        embed_dim=checkpoint['args'].get('embed_dim', 64),
        hidden_dim=checkpoint['args'].get('hidden_dim', 128),
    ).to(args.device)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()
    
    # Evaluate each instance
    print("\nEvaluating methods...")
    
    results = []
    
    for inst in tqdm(dataset, desc="Evaluating"):
        values = inst['values'].cpu().numpy()
        weights = inst['weights'].cpu().numpy()
        capacity = inst['capacity']
        optimal = inst['optimal']
        
        # Random
        random_val = random_solve(values, weights, capacity)
        
        # Greedy
        greedy_val, _ = greedy_solve(values, weights, capacity)
        
        # PPO
        ppo_val, ppo_weight = ppo_solve_instance(
            policy, inst['values'], inst['weights'], capacity, max_items, args.device
        )
        
        results.append({
            'n': inst['n'],
            'optimal': optimal,
            'random': random_val,
            'greedy': greedy_val,
            'ppo': ppo_val,
        })
    
    # Aggregate results
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\n{'Method':<15} {'Avg Value':>12} {'Gap to Opt':>12}")
    print("-" * 45)
    
    optimal_avg = df['optimal'].mean()
    random_avg = df['random'].mean()
    greedy_avg = df['greedy'].mean()
    ppo_avg = df['ppo'].mean()
    
    random_gap = (optimal_avg - random_avg) / optimal_avg * 100
    greedy_gap = (optimal_avg - greedy_avg) / optimal_avg * 100
    ppo_gap = (optimal_avg - ppo_avg) / optimal_avg * 100
    
    print(f"{'Random':<15} {random_avg:>12.2f} {random_gap:>11.2f}%")
    print(f"{'Greedy':<15} {greedy_avg:>12.2f} {greedy_gap:>11.2f}%")
    print(f"{'PPO':<15} {ppo_avg:>12.2f} {ppo_gap:>11.2f}%")
    print(f"{'Optimal':<15} {optimal_avg:>12.2f} {'0.00':>11}%")
    
    # Results by N
    print("\n" + "-" * 45)
    print("Results by problem size (N):")
    print(f"{'N':<10} {'PPO Avg':>10} {'Opt Avg':>10} {'Gap':>10}")
    print("-" * 45)
    
    for n in sorted(df['n'].unique()):
        subset = df[df['n'] == n]
        ppo_n = subset['ppo'].mean()
        opt_n = subset['optimal'].mean()
        gap_n = (opt_n - ppo_n) / opt_n * 100
        print(f"{n:<10} {ppo_n:>10.2f} {opt_n:>10.2f} {gap_n:>9.2f}%")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  PPO achieves {100 - ppo_gap:.2f}% of optimal on average")
    print(f"  PPO beats Random by {(ppo_avg - random_avg):.2f} ({(ppo_avg/random_avg - 1)*100:.1f}%)")
    print(f"  PPO vs Greedy: {(ppo_avg - greedy_avg):+.2f} ({(ppo_avg/greedy_avg - 1)*100:+.1f}%)")


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Knapsack PPO on Dataset')
    parser.add_argument('--dataset_path', type=str, 
                        default='./data/knapsack/test_small.csv',
                        help='Path to dataset CSV')
    parser.add_argument('--model_path', type=str, 
                        default='./checkpoints/knapsack/final_model.pt',
                        help='Path to trained model')
    parser.add_argument('--max_instances', type=int, default=None,
                        help='Max instances to evaluate')
    parser.add_argument('--device', type=str, default='cpu')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
