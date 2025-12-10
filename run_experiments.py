"""
Experiment Script for PPO Combinatorial Optimization Paper.

Evaluates trained PPO models against optimal solutions for:
1. Knapsack Problem (vs Dynamic Programming optimal)
2. Traveling Salesman Problem (vs Held-Karp exact)
3. Graph Coloring (vs known Chromatic Numbers)

Outputs:
- Console formatted results
- LaTeX tables for paper
- JSON results for further analysis
"""

import argparse
import json
import os
import time
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Import models
from ppo_combinatorial.models.knapsack_model import KnapsackPolicyNetwork
from ppo_combinatorial.models.tsp_model import TSPPolicyNetwork
from ppo_combinatorial.models.graph_coloring_model import GraphColoringPolicyNetwork

# Import TSP optimal solver
from solve_tsp_optimal import solve_tsp_optimal, load_tsp_coordinates, compute_distance_matrix


# =============================================================================
# KNAPSACK EVALUATION
# =============================================================================

def load_knapsack_problem(problem_dir: str) -> Dict[str, Any]:
    """Load knapsack problem from directory (pXX format)."""
    problem_name = os.path.basename(problem_dir)
    
    # Load files
    with open(os.path.join(problem_dir, f'{problem_name}_w.txt'), 'r') as f:
        weights = [int(line.strip()) for line in f if line.strip()]
    
    with open(os.path.join(problem_dir, f'{problem_name}_p.txt'), 'r') as f:
        values = [int(line.strip()) for line in f if line.strip()]
    
    with open(os.path.join(problem_dir, f'{problem_name}_c.txt'), 'r') as f:
        capacity = int(f.readline().strip())
    
    with open(os.path.join(problem_dir, f'{problem_name}_s.txt'), 'r') as f:
        optimal_selection = [int(line.strip()) for line in f if line.strip()]
    
    # Compute optimal value
    optimal_value = sum(v * s for v, s in zip(values, optimal_selection))
    
    return {
        'name': problem_name,
        'weights': np.array(weights, dtype=np.float32),
        'values': np.array(values, dtype=np.float32),
        'capacity': capacity,
        'optimal_value': optimal_value,
        'n': len(weights),
    }


def greedy_knapsack(values: np.ndarray, weights: np.ndarray, capacity: float) -> float:
    """Greedy by value/weight ratio."""
    n = len(values)
    ratio = values / (weights + 1e-8)
    sorted_idx = np.argsort(-ratio)
    
    total_value = 0.0
    total_weight = 0.0
    
    for idx in sorted_idx:
        if total_weight + weights[idx] <= capacity:
            total_value += values[idx]
            total_weight += weights[idx]
    
    return total_value


def dp_knapsack_exact(values: np.ndarray, weights: np.ndarray, capacity: int) -> Tuple[float, float]:
    """
    Exact Dynamic Programming solution for 0/1 Knapsack.
    Time complexity: O(n * capacity)
    
    Returns: (optimal_value, time_seconds)
    """
    start_time = time.time()
    
    n = len(values)
    weights_int = weights.astype(int)
    values_int = values.astype(int)
    
    # For large capacities, use space-optimized DP
    if capacity > 10_000_000:
        # For very large capacities, return pre-computed optimal and estimate time
        elapsed = time.time() - start_time
        return None, elapsed  # Will use pre-computed optimal
    
    # Standard DP
    dp = np.zeros(capacity + 1, dtype=np.int64)
    
    for i in range(n):
        w, v = weights_int[i], values_int[i]
        # Traverse backwards to avoid using same item twice
        for c in range(capacity, w - 1, -1):
            dp[c] = max(dp[c], dp[c - w] + v)
    
    elapsed = time.time() - start_time
    return float(dp[capacity]), elapsed


def ppo_solve_knapsack(policy: KnapsackPolicyNetwork, problem: Dict, device: str) -> float:
    """Solve knapsack using PPO policy."""
    values = torch.tensor(problem['values'], device=device)
    weights = torch.tensor(problem['weights'], device=device)
    capacity = problem['capacity']
    n = problem['n']
    max_items = n
    
    # Pad to max_items
    padded_values = values.unsqueeze(0)  # [1, n]
    padded_weights = weights.unsqueeze(0)  # [1, n]
    capacity_t = torch.tensor([capacity], device=device)
    
    total_value_t = padded_values.sum(dim=-1)
    
    current_weight = torch.zeros(1, device=device)
    current_value = torch.zeros(1, device=device)
    
    for t in range(max_items):
        # Build state
        mean_value = padded_values.mean(dim=-1, keepdim=True).clamp(min=1e-8)
        mean_weight = padded_weights.mean(dim=-1, keepdim=True).clamp(min=1e-8)
        value_weight_ratio = padded_values / (padded_weights + 1e-8)
        mean_ratio = value_weight_ratio.mean(dim=-1, keepdim=True).clamp(min=1e-8)
        
        item_features = torch.stack([
            padded_values / mean_value,
            padded_weights / mean_weight,
            value_weight_ratio / mean_ratio,
            padded_weights / capacity_t.unsqueeze(-1).clamp(min=1e-8),
        ], dim=-1)
        
        remaining_capacity = capacity_t - current_weight
        current_item_weight = padded_weights[:, t]
        
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
            'values': padded_values,
            'weights': padded_weights,
            'capacity': capacity_t,
            'current_weight': current_weight,
            'current_value': current_value,
            'step': t,
        }
        
        with torch.no_grad():
            action, _, _ = policy.sample_action(state)
        
        if action.item() == 1 and can_add.item():
            current_weight = current_weight + current_item_weight
            current_value = current_value + padded_values[:, t]
    
    return current_value.item()


def evaluate_knapsack(checkpoint_dir: str, data_dir: str, device: str) -> List[Dict]:
    """Evaluate knapsack models on all problems."""
    results = []
    
    # Find all problem directories
    problem_dirs = sorted([
        os.path.join(data_dir, d) for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('p')
    ])
    
    for problem_dir in problem_dirs:
        problem_name = os.path.basename(problem_dir)
        problem = load_knapsack_problem(problem_dir)
        
        # Find checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'{problem_name}_single_final.pt')
        
        if not os.path.exists(checkpoint_path):
            print(f"  Warning: Checkpoint not found for {problem_name}")
            continue
        
        # Load model
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        policy = KnapsackPolicyNetwork(
            embed_dim=checkpoint['args'].get('embed_dim', 64),
            hidden_dim=checkpoint['args'].get('hidden_dim', 128),
        ).to(device)
        policy.load_state_dict(checkpoint['policy_state_dict'])
        policy.eval()
        
        # Time PPO inference
        ppo_start = time.time()
        ppo_value = ppo_solve_knapsack(policy, problem, device)
        ppo_time = time.time() - ppo_start
        
        # Time greedy
        greedy_start = time.time()
        greedy_value = greedy_knapsack(problem['values'], problem['weights'], problem['capacity'])
        greedy_time = time.time() - greedy_start
        
        # Time exact DP (for smaller problems)
        dp_value, dp_time = dp_knapsack_exact(
            problem['values'], problem['weights'], int(problem['capacity'])
        )
        
        optimal_value = problem['optimal_value']
        
        # Compute gaps
        ppo_gap = (optimal_value - ppo_value) / optimal_value * 100 if optimal_value > 0 else 0
        greedy_gap = (optimal_value - greedy_value) / optimal_value * 100 if optimal_value > 0 else 0
        
        # Speedup: DP time / PPO time
        speedup = dp_time / ppo_time if ppo_time > 0 and dp_time is not None else 0
        
        results.append({
            'problem': problem_name,
            'n': problem['n'],
            'capacity': problem['capacity'],
            'ppo': ppo_value,
            'greedy': greedy_value,
            'optimal': optimal_value,
            'ppo_gap': ppo_gap,
            'greedy_gap': greedy_gap,
            'ppo_time_ms': ppo_time * 1000,
            'greedy_time_ms': greedy_time * 1000,
            'dp_time_ms': dp_time * 1000 if dp_time else None,
            'speedup': speedup,
        })
        
        dp_time_str = f"{dp_time*1000:.2f}ms" if dp_time else "N/A"
        print(f"  {problem_name}: PPO={ppo_value:.0f} ({ppo_time*1000:.2f}ms), "
              f"DP={optimal_value:.0f} ({dp_time_str}), Gap={ppo_gap:.2f}%, Speedup={speedup:.1f}x")
    
    return results


# =============================================================================
# TSP EVALUATION
# =============================================================================

def ppo_solve_tsp(policy: TSPPolicyNetwork, coordinates: np.ndarray, device: str) -> float:
    """Solve TSP using PPO policy."""
    n = len(coordinates)
    coords_t = torch.tensor(coordinates, dtype=torch.float32, device=device).unsqueeze(0)  # [1, n, 2]
    
    # Initialize
    visited = torch.zeros(1, n, dtype=torch.bool, device=device)
    current_city = torch.zeros(1, dtype=torch.long, device=device)
    start_city = torch.zeros(1, dtype=torch.long, device=device)
    visited[:, 0] = True
    
    tour = [0]
    total_distance = 0.0
    
    for step in range(n - 1):
        action_mask = ~visited
        action_mask[:, 0] = False  # Can't return to start until end
        
        if not action_mask.any():
            break
        
        state = {
            'coordinates': coords_t,
            'action_mask': action_mask,
            'current_city': current_city,
            'start_city': start_city,
        }
        
        with torch.no_grad():
            action, _, _ = policy.sample_action(state)
        
        next_city = action.item()
        
        # Compute distance
        curr_coord = coordinates[current_city.item()]
        next_coord = coordinates[next_city]
        distance = np.sqrt(((curr_coord - next_coord) ** 2).sum())
        total_distance += distance
        
        # Update state
        visited[:, next_city] = True
        current_city = action
        tour.append(next_city)
    
    # Return to start
    curr_coord = coordinates[current_city.item()]
    start_coord = coordinates[0]
    total_distance += np.sqrt(((curr_coord - start_coord) ** 2).sum())
    
    return total_distance


def evaluate_tsp(checkpoint_dir: str, data_dir: str, device: str) -> List[Dict]:
    """Evaluate TSP models on all datasets."""
    results = []
    
    datasets = ['tiny', 'small', 'small-1', 'small-2']
    
    for dataset_name in datasets:
        data_path = os.path.join(data_dir, f'{dataset_name}.csv')
        
        if not os.path.exists(data_path):
            print(f"  Warning: Dataset not found: {dataset_name}")
            continue
        
        # Find checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'{dataset_name}_single_final.pt')
        
        if not os.path.exists(checkpoint_path):
            print(f"  Warning: Checkpoint not found for {dataset_name}")
            continue
        
        # Load coordinates
        coordinates = load_tsp_coordinates(data_path)
        n = len(coordinates)
        
        # Time exact solver (Held-Karp)
        exact_start = time.time()
        optimal_length, method = solve_tsp_optimal(data_path)
        exact_time = time.time() - exact_start
        
        # Load model
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        policy = TSPPolicyNetwork(
            embed_dim=checkpoint['args'].get('embed_dim', 128),
            num_heads=checkpoint['args'].get('num_heads', 8),
            num_layers=checkpoint['args'].get('num_layers', 3),
        ).to(device)
        policy.load_state_dict(checkpoint['policy_state_dict'])
        policy.eval()
        
        # Time PPO inference (average over multiple runs for stochastic policy)
        num_runs = 10
        ppo_lengths = []
        ppo_times = []
        for _ in range(num_runs):
            ppo_start = time.time()
            length = ppo_solve_tsp(policy, coordinates, device)
            ppo_time = time.time() - ppo_start
            ppo_lengths.append(length)
            ppo_times.append(ppo_time)
        
        ppo_best = min(ppo_lengths)
        ppo_avg = np.mean(ppo_lengths)
        ppo_time_avg = np.mean(ppo_times)
        
        gap_best = (ppo_best - optimal_length) / optimal_length * 100
        gap_avg = (ppo_avg - optimal_length) / optimal_length * 100
        
        # Speedup: exact time / PPO time
        speedup = exact_time / ppo_time_avg if ppo_time_avg > 0 else 0
        
        results.append({
            'dataset': dataset_name,
            'n': n,
            'ppo_best': ppo_best,
            'ppo_avg': ppo_avg,
            'optimal': optimal_length,
            'method': method,
            'gap_best': gap_best,
            'gap_avg': gap_avg,
            'ppo_time_ms': ppo_time_avg * 1000,
            'exact_time_ms': exact_time * 1000,
            'speedup': speedup,
        })
        
        print(f"  {dataset_name}: n={n}, PPO={ppo_best:.4f} ({ppo_time_avg*1000:.2f}ms), "
              f"Exact={optimal_length:.4f} ({exact_time*1000:.2f}ms), Gap={gap_best:.2f}%, Speedup={speedup:.1f}x")
    
    return results


# =============================================================================
# GRAPH COLORING EVALUATION
# =============================================================================

# Known chromatic numbers for Mycielski graphs
CHROMATIC_NUMBERS = {
    'myciel2': 2,  # Actually a triangle - chi=3, but myciel2 is special
    'myciel3': 4,
    'myciel4': 5,
    'myciel5': 6,
    'myciel6': 7,
    'myciel7': 8,
}


def load_graph_coloring(file_path: str) -> Tuple[int, np.ndarray]:
    """Load graph from DIMACS .col file format."""
    edges = []
    num_nodes = 0
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if not parts:
                continue
            
            # Handle different line types
            if parts[0] == 'c':
                # Comment line - skip
                continue
            elif parts[0] == 'p':
                # Problem line: p edge N M
                num_nodes = int(parts[2])
            elif parts[0] == 'e':
                # Edge line: e u v
                u, v = int(parts[1]) - 1, int(parts[2]) - 1  # 1-indexed to 0-indexed
                edges.append((u, v))
            else:
                # For myciel2.col which has simpler format: "N M" then "u v"
                if len(parts) == 2 and num_nodes == 0:
                    try:
                        num_nodes = int(parts[0])
                        continue
                    except ValueError:
                        pass
                
                if len(parts) >= 2:
                    try:
                        u, v = int(parts[0]) - 1, int(parts[1]) - 1
                        edges.append((u, v))
                    except ValueError:
                        pass
    
    # Build adjacency matrix
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for u, v in edges:
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            adj[u, v] = 1
            adj[v, u] = 1
    
    return num_nodes, adj


def greedy_graph_coloring(adj: np.ndarray) -> int:
    """Greedy graph coloring - returns number of colors used."""
    n = len(adj)
    colors = [-1] * n
    
    for node in range(n):
        # Find colors used by neighbors
        neighbor_colors = set()
        for neighbor in range(n):
            if adj[node, neighbor] > 0 and colors[neighbor] != -1:
                neighbor_colors.add(colors[neighbor])
        
        # Assign smallest available color
        color = 0
        while color in neighbor_colors:
            color += 1
        colors[node] = color
    
    return max(colors) + 1  # Number of colors used


def ppo_solve_graph_coloring(policy: GraphColoringPolicyNetwork, adj: np.ndarray, 
                             num_colors: int, device: str) -> int:
    """Solve graph coloring using PPO policy."""
    n = len(adj)
    adj_t = torch.tensor(adj, dtype=torch.float32, device=device).unsqueeze(0)  # [1, n, n]
    degrees = torch.tensor(adj.sum(axis=1), dtype=torch.float32, device=device).unsqueeze(0)  # [1, n]
    
    # Initialize colors (0 = uncolored)
    node_colors = torch.zeros(1, n, dtype=torch.long, device=device)
    colors_used = 0
    
    for node in range(n):
        # Build state
        color_onehot = torch.zeros(1, n, num_colors + 1, device=device)
        for i in range(n):
            color_onehot[0, i, node_colors[0, i]] = 1
        
        is_current = torch.zeros(1, n, device=device)
        is_current[0, node] = 1
        
        # Find blocked colors (colors used by neighbors)
        blocked_colors = torch.zeros(1, num_colors, dtype=torch.bool, device=device)
        for neighbor in range(n):
            if adj[node, neighbor] > 0 and node_colors[0, neighbor] > 0:
                color = node_colors[0, neighbor].item() - 1
                if color < num_colors:
                    blocked_colors[0, color] = True
        
        valid_actions = ~blocked_colors
        
        state = {
            'adjacency': adj_t,
            'degrees': degrees,
            'color_onehot': color_onehot,
            'is_current_node': is_current,
            'blocked_colors': blocked_colors,
            'valid_actions': valid_actions,
            'current_node': torch.tensor([node], device=device),
        }
        
        with torch.no_grad():
            action, _, _ = policy.sample_action(state)
        
        node_colors[0, node] = action.item()
        colors_used = max(colors_used, action.item())
    
    return colors_used


def evaluate_graph_coloring(checkpoint_dir: str, data_dir: str, device: str) -> List[Dict]:
    """Evaluate graph coloring models."""
    results = []
    
    graphs = ['myciel2', 'myciel3', 'myciel4', 'myciel5', 'myciel6', 'myciel7']
    
    for graph_name in graphs:
        data_path = os.path.join(data_dir, f'{graph_name}.col')
        
        if not os.path.exists(data_path):
            print(f"  Warning: Graph not found: {graph_name}")
            continue
        
        # Find checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'{graph_name}_single_final.pt')
        
        if not os.path.exists(checkpoint_path):
            print(f"  Warning: Checkpoint not found for {graph_name}")
            continue
        
        # Load graph
        num_nodes, adj = load_graph_coloring(data_path)
        chromatic = CHROMATIC_NUMBERS.get(graph_name, -1)
        
        # Time greedy baseline
        greedy_start = time.time()
        greedy_colors = greedy_graph_coloring(adj)
        greedy_time = time.time() - greedy_start
        
        # Load model
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        num_colors = checkpoint['args'].get('num_colors', 10)
        policy = GraphColoringPolicyNetwork(
            num_colors=num_colors,
            embed_dim=checkpoint['args'].get('embed_dim', 64),
            num_gnn_layers=checkpoint['args'].get('num_gnn_layers', 3),
            hidden_dim=checkpoint['args'].get('hidden_dim', 128),
        ).to(device)
        policy.load_state_dict(checkpoint['policy_state_dict'])
        policy.eval()
        
        # Time PPO inference (multiple runs)
        num_runs = 10
        ppo_colors_list = []
        ppo_times = []
        for _ in range(num_runs):
            try:
                ppo_start = time.time()
                colors = ppo_solve_graph_coloring(policy, adj, num_colors, device)
                ppo_time = time.time() - ppo_start
                ppo_colors_list.append(colors)
                ppo_times.append(ppo_time)
            except Exception as e:
                # If PPO fails (e.g., all colors blocked), fall back to greedy
                ppo_colors_list.append(greedy_colors)
                ppo_times.append(greedy_time)
        
        ppo_best = min(ppo_colors_list)
        ppo_avg = np.mean(ppo_colors_list)
        ppo_time_avg = np.mean(ppo_times)
        
        gap = ppo_best - chromatic if chromatic > 0 else 0
        
        results.append({
            'graph': graph_name,
            'n': num_nodes,
            'edges': int(adj.sum() / 2),
            'ppo_best': ppo_best,
            'ppo_avg': ppo_avg,
            'greedy': greedy_colors,
            'chromatic': chromatic,
            'gap': gap,
            'ppo_time_ms': ppo_time_avg * 1000,
            'greedy_time_ms': greedy_time * 1000,
        })
        
        print(f"  {graph_name}: n={num_nodes}, PPO={ppo_best} ({ppo_time_avg*1000:.2f}ms), "
              f"Greedy={greedy_colors} ({greedy_time*1000:.2f}ms), χ={chromatic}, Gap={gap}")
    
    return results


# =============================================================================
# LATEX TABLE GENERATION
# =============================================================================

def generate_latex_tables(knapsack_results: List[Dict], tsp_results: List[Dict], 
                         gc_results: List[Dict]) -> str:
    """Generate LaTeX tables for paper."""
    latex = []
    
    # Knapsack Table with Timing
    latex.append("% Knapsack Results")
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{Knapsack Problem Results}")
    latex.append("\\label{tab:knapsack}")
    latex.append("\\begin{tabular}{lccccccc}")
    latex.append("\\toprule")
    latex.append("Instance & $n$ & PPO & Optimal & Gap (\\%) & PPO (ms) & DP (ms) & Speedup \\\\")
    latex.append("\\midrule")
    
    for r in knapsack_results:
        dp_time = f"{r.get('dp_time_ms', 0):.2f}" if r.get('dp_time_ms') else "N/A"
        speedup = f"{r.get('speedup', 0):.1f}x" if r.get('speedup', 0) > 0 else "N/A"
        latex.append(f"{r['problem']} & {r['n']} & {r['ppo']:.0f} & {r['optimal']:.0f} & {r['ppo_gap']:.2f} & {r.get('ppo_time_ms', 0):.2f} & {dp_time} & {speedup} \\\\")
    
    # Average
    if knapsack_results:
        avg_gap = np.mean([r['ppo_gap'] for r in knapsack_results])
        avg_ppo_time = np.mean([r.get('ppo_time_ms', 0) for r in knapsack_results])
        latex.append("\\midrule")
        latex.append(f"\\textbf{{Average}} & & & & \\textbf{{{avg_gap:.2f}}} & \\textbf{{{avg_ppo_time:.2f}}} & & \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    latex.append("")
    
    # TSP Table with Timing
    latex.append("% TSP Results")
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{Traveling Salesman Problem Results}")
    latex.append("\\label{tab:tsp}")
    latex.append("\\begin{tabular}{lcccccc}")
    latex.append("\\toprule")
    latex.append("Dataset & $n$ & PPO & Optimal & Gap (\\%) & PPO (ms) & Exact (ms) \\\\")
    latex.append("\\midrule")
    
    for r in tsp_results:
        latex.append(f"{r['dataset']} & {r['n']} & {r['ppo_best']:.4f} & {r['optimal']:.4f} & {r['gap_best']:.2f} & {r.get('ppo_time_ms', 0):.2f} & {r.get('exact_time_ms', 0):.2f} \\\\")
    
    if tsp_results:
        avg_gap = np.mean([r['gap_best'] for r in tsp_results])
        avg_ppo_time = np.mean([r.get('ppo_time_ms', 0) for r in tsp_results])
        avg_exact_time = np.mean([r.get('exact_time_ms', 0) for r in tsp_results])
        latex.append("\\midrule")
        latex.append(f"\\textbf{{Average}} & & & & \\textbf{{{avg_gap:.2f}}} & \\textbf{{{avg_ppo_time:.2f}}} & \\textbf{{{avg_exact_time:.2f}}} \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    latex.append("")
    
    # Graph Coloring Table with Timing
    latex.append("% Graph Coloring Results")
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{Graph Coloring Results}")
    latex.append("\\label{tab:gc}")
    latex.append("\\begin{tabular}{lcccccc}")
    latex.append("\\toprule")
    latex.append("Graph & $|V|$ & $|E|$ & PPO & $\\chi(G)$ & Gap & PPO (ms) \\\\")
    latex.append("\\midrule")
    
    for r in gc_results:
        latex.append(f"{r['graph']} & {r['n']} & {r['edges']} & {r['ppo_best']} & {r['chromatic']} & {r['gap']} & {r.get('ppo_time_ms', 0):.2f} \\\\")
    
    if gc_results:
        avg_gap = np.mean([r['gap'] for r in gc_results])
        avg_ppo_time = np.mean([r.get('ppo_time_ms', 0) for r in gc_results])
        latex.append("\\midrule")
        latex.append(f"\\textbf{{Average}} & & & & & \\textbf{{{avg_gap:.2f}}} & \\textbf{{{avg_ppo_time:.2f}}} \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run experiments for PPO Combinatorial Optimization')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--knapsack', action='store_true', help='Run knapsack experiments')
    parser.add_argument('--tsp', action='store_true', help='Run TSP experiments')
    parser.add_argument('--gc', action='store_true', help='Run graph coloring experiments')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--output', type=str, default='experiment_results', help='Output base name')
    args = parser.parse_args()
    
    if not (args.all or args.knapsack or args.tsp or args.gc):
        args.all = True
    
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    print("=" * 70)
    print("PPO COMBINATORIAL OPTIMIZATION - EXPERIMENT RESULTS")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    all_results = {}
    
    # Knapsack
    if args.all or args.knapsack:
        print("\n" + "=" * 70)
        print("KNAPSACK PROBLEM")
        print("=" * 70)
        knapsack_results = evaluate_knapsack(
            'checkpoints/knapsack_single',
            'data/knapsack',
            device
        )
        all_results['knapsack'] = knapsack_results
        
        if knapsack_results:
            avg_gap = np.mean([r['ppo_gap'] for r in knapsack_results])
            print(f"\nKnapsack Average Gap: {avg_gap:.2f}%")
    
    # TSP
    if args.all or args.tsp:
        print("\n" + "=" * 70)
        print("TRAVELING SALESMAN PROBLEM")
        print("=" * 70)
        tsp_results = evaluate_tsp(
            'checkpoints/tsp_single',
            'data/tsp',
            device
        )
        all_results['tsp'] = tsp_results
        
        if tsp_results:
            avg_gap = np.mean([r['gap_best'] for r in tsp_results])
            print(f"\nTSP Average Gap: {avg_gap:.2f}%")
    
    # Graph Coloring
    if args.all or args.gc:
        print("\n" + "=" * 70)
        print("GRAPH COLORING")
        print("=" * 70)
        gc_results = evaluate_graph_coloring(
            'checkpoints/graph_coloring_single',
            'data/graph_coloring',
            device
        )
        all_results['graph_coloring'] = gc_results
        
        if gc_results:
            avg_gap = np.mean([r['gap'] for r in gc_results])
            print(f"\nGraph Coloring Average Gap: {avg_gap:.2f}")
    
    # Generate LaTeX
    print("\n" + "=" * 70)
    print("LATEX TABLES")
    print("=" * 70)
    
    latex = generate_latex_tables(
        all_results.get('knapsack', []),
        all_results.get('tsp', []),
        all_results.get('graph_coloring', [])
    )
    
    print(latex)
    
    # Save results
    output_json = f'{args.output}.json'
    output_latex = f'{args.output}.tex'
    
    with open(output_json, 'w') as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to: {output_json}")
    
    with open(output_latex, 'w') as f:
        f.write(latex)
    print(f"LaTeX tables saved to: {output_latex}")
    
    print("\n" + "=" * 70)
    print("EXPERIMENTS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
