"""
Generate static knapsack datasets for training and testing.

Creates CSV files with:
- Weights, Values, Capacity, Optimal solution (via DP)
"""

import numpy as np
import pandas as pd
import os
from tqdm import tqdm


def solve_knapsack_dp(weights, values, capacity):
    """
    Solve 0/1 Knapsack using Dynamic Programming.
    
    Returns optimal value and selected items.
    """
    n = len(weights)
    capacity = int(capacity)
    
    # DP table
    dp = np.zeros((n + 1, capacity + 1), dtype=np.float64)
    
    for i in range(1, n + 1):
        w = int(weights[i - 1])
        v = values[i - 1]
        for j in range(capacity + 1):
            if j < w:
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - w] + v)
    
    # Backtrack to find selected items
    selected = np.zeros(n)
    j = capacity
    for i in range(n, 0, -1):
        if dp[i][j] != dp[i - 1][j]:
            selected[i - 1] = 1
            j -= int(weights[i - 1])
    
    return dp[n][capacity], selected


def generate_instance(n_items, weight_range, value_range, capacity_ratio=0.5):
    """Generate a single knapsack instance."""
    weights = np.random.randint(weight_range[0], weight_range[1] + 1, size=n_items)
    values = np.random.randint(value_range[0], value_range[1] + 1, size=n_items)
    
    total_weight = weights.sum()
    capacity = int(capacity_ratio * total_weight)
    
    # Ensure capacity doesn't exceed 1e5
    capacity = min(capacity, 100000)
    
    return weights, values, capacity


def generate_dataset(configs, output_path, desc="Generating"):
    """
    Generate dataset with multiple configurations.
    
    configs: List of (n_items, num_instances)
    """
    data = []
    
    for n_items, num_instances, weight_range, value_range in tqdm(configs, desc=desc):
        for _ in range(num_instances):
            weights, values, capacity = generate_instance(
                n_items, weight_range, value_range
            )
            
            optimal_value, selected = solve_knapsack_dp(weights, values, capacity)
            
            data.append({
                'N': n_items,
                'Weights': str(weights.tolist()),
                'Values': str(values.tolist()),
                'Capacity': capacity,
                'Optimal_Value': optimal_value,
                'Optimal_Selection': str(selected.tolist()),
            })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(data)} instances to {output_path}")
    return df


def main():
    np.random.seed(42)
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Configuration ranges
    SMALL_WEIGHT = (1, 100)
    SMALL_VALUE = (1, 100)
    LARGE_WEIGHT = (1000, 10000)
    LARGE_VALUE = (1000, 100000)
    
    # Training configurations
    train_configs_small = [
        (5, 100, SMALL_WEIGHT, SMALL_VALUE),
        (10, 50, SMALL_WEIGHT, SMALL_VALUE),
        (25, 25, SMALL_WEIGHT, SMALL_VALUE),
        (100, 10, SMALL_WEIGHT, SMALL_VALUE),
        (200, 5, SMALL_WEIGHT, SMALL_VALUE),
    ]
    
    train_configs_large = [
        (5, 100, LARGE_WEIGHT, LARGE_VALUE),
        (10, 50, LARGE_WEIGHT, LARGE_VALUE),
        (25, 25, LARGE_WEIGHT, LARGE_VALUE),
        (100, 10, LARGE_WEIGHT, LARGE_VALUE),
        (200, 5, LARGE_WEIGHT, LARGE_VALUE),
    ]
    
    # Test configurations
    test_configs_small = [
        (5, 50, SMALL_WEIGHT, SMALL_VALUE),
        (10, 25, SMALL_WEIGHT, SMALL_VALUE),
        (25, 10, SMALL_WEIGHT, SMALL_VALUE),
        (100, 5, SMALL_WEIGHT, SMALL_VALUE),
        (200, 5, SMALL_WEIGHT, SMALL_VALUE),
    ]
    
    test_configs_large = [
        (5, 50, LARGE_WEIGHT, LARGE_VALUE),
        (10, 25, LARGE_WEIGHT, LARGE_VALUE),
        (25, 10, LARGE_WEIGHT, LARGE_VALUE),
        (100, 5, LARGE_WEIGHT, LARGE_VALUE),
        (200, 5, LARGE_WEIGHT, LARGE_VALUE),
    ]
    
    print("=" * 60)
    print("GENERATING KNAPSACK DATASETS")
    print("=" * 60)
    
    # Generate training data
    print("\n[1/4] Training data (small weights/values)...")
    generate_dataset(
        train_configs_small,
        os.path.join(output_dir, "train_small.csv"),
        "Train Small"
    )
    
    print("\n[2/4] Training data (large weights/values)...")
    generate_dataset(
        train_configs_large,
        os.path.join(output_dir, "train_large.csv"),
        "Train Large"
    )
    
    # Generate test data
    print("\n[3/4] Test data (small weights/values)...")
    generate_dataset(
        test_configs_small,
        os.path.join(output_dir, "test_small.csv"),
        "Test Small"
    )
    
    print("\n[4/4] Test data (large weights/values)...")
    generate_dataset(
        test_configs_large,
        os.path.join(output_dir, "test_large.csv"),
        "Test Large"
    )
    
    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETE")
    print("=" * 60)
    print("\nFiles created:")
    print("  - train_small.csv (190 instances)")
    print("  - train_large.csv (190 instances)")
    print("  - test_small.csv (95 instances)")
    print("  - test_large.csv (95 instances)")


if __name__ == "__main__":
    main()
