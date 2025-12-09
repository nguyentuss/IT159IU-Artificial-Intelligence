"""
Experiment Visualization Script.

Generates training plots from history JSON files.
Supports multiple history files for comparing different runs/settings.
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np


def load_history(file_path):
    """Load training history from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def plot_knapsack(histories, labels, output_path=None):
    """Plot knapsack training history."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Knapsack Training Progress', fontsize=14, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
    
    for i, (history, label) in enumerate(zip(histories, labels)):
        epochs = [h['epoch'] for h in history['history']]
        
        # Optimality ratio
        axes[0, 0].plot(epochs, [h['opt_ratio'] for h in history['history']], 
                       label=label, color=colors[i], linewidth=1.5)
        
        # Feasibility ratio
        axes[0, 1].plot(epochs, [h['feasible_ratio'] for h in history['history']], 
                       label=label, color=colors[i], linewidth=1.5)
        
        # Policy loss
        axes[1, 0].plot(epochs, [h['policy_loss'] for h in history['history']], 
                       label=label, color=colors[i], linewidth=1.5)
        
        # Entropy
        axes[1, 1].plot(epochs, [h['entropy'] for h in history['history']], 
                       label=label, color=colors[i], linewidth=1.5)
    
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Optimality Ratio (%)')
    axes[0, 0].set_title('Value vs Optimal (%)')
    axes[0, 0].axhline(y=100, color='green', linestyle='--', linewidth=2, label='Optimal (100%)', alpha=0.7)
    axes[0, 0].legend(loc='best', fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Feasibility Ratio')
    axes[0, 1].set_title('Feasibility Rate')
    axes[0, 1].legend(loc='best', fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1.05)
    
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Policy Loss')
    axes[1, 0].set_title('Policy Loss')
    axes[1, 0].legend(loc='best', fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Entropy')
    axes[1, 1].set_title('Entropy (Exploration)')
    axes[1, 1].legend(loc='best', fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_graph_coloring(histories, labels, output_path=None):
    """Plot graph coloring training history."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Graph Coloring Training Progress', fontsize=14, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
    
    for i, (history, label) in enumerate(zip(histories, labels)):
        epochs = [h['epoch'] for h in history['history']]
        
        # Conflicts
        axes[0, 0].plot(epochs, [h['avg_conflicts'] for h in history['history']], 
                       label=label, color=colors[i], linewidth=1.5)
        
        # Colors used
        axes[0, 1].plot(epochs, [h['avg_colors'] for h in history['history']], 
                       label=label, color=colors[i], linewidth=1.5)
        
        # Valid ratio
        axes[1, 0].plot(epochs, [h['valid_ratio'] for h in history['history']], 
                       label=label, color=colors[i], linewidth=1.5)
        
        # Policy loss
        axes[1, 1].plot(epochs, [h['policy_loss'] for h in history['history']], 
                       label=label, color=colors[i], linewidth=1.5)
    
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Avg Conflicts')
    axes[0, 0].set_title('Average Conflicts per Solution')
    axes[0, 0].legend(loc='best', fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Colors Used')
    axes[0, 1].set_title('Average Colors Used')
    # Add chromatic number reference line if available
    chromatic_numbers = [h.get('chromatic_number') for h in histories if h.get('chromatic_number')]
    if chromatic_numbers:
        # Use the max chromatic number as reference (for curriculum)
        max_chromatic = max(chromatic_numbers)
        axes[0, 1].axhline(y=max_chromatic, color='green', linestyle='--', linewidth=2, 
                          label=f'Chromatic Number ({max_chromatic})', alpha=0.7)
    axes[0, 1].legend(loc='best', fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Valid Ratio')
    axes[1, 0].set_title('Valid Coloring Rate')
    axes[1, 0].legend(loc='best', fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1.05)
    
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Policy Loss')
    axes[1, 1].set_title('Policy Loss')
    axes[1, 1].legend(loc='best', fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_tsp(histories, labels, output_path=None):
    """Plot TSP training history."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('TSP Training Progress', fontsize=14, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
    
    for i, (history, label) in enumerate(zip(histories, labels)):
        epochs = [h['epoch'] for h in history['history']]
        
        # Average tour length
        axes[0, 0].plot(epochs, [h['avg_tour'] for h in history['history']], 
                       label=label, color=colors[i], linewidth=1.5)
        
        # Best tour length
        axes[0, 1].plot(epochs, [h['best_tour'] for h in history['history']], 
                       label=label, color=colors[i], linewidth=1.5)
        
        # Policy loss
        axes[1, 0].plot(epochs, [h['policy_loss'] for h in history['history']], 
                       label=label, color=colors[i], linewidth=1.5)
        
        # Entropy
        axes[1, 1].plot(epochs, [h['entropy'] for h in history['history']], 
                       label=label, color=colors[i], linewidth=1.5)
    
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Tour Length')
    axes[0, 0].set_title('Average Tour Length')
    # Add optimal tour reference line if available
    optimal_tours = [h.get('optimal_tour') for h in histories if h.get('optimal_tour')]
    if optimal_tours:
        # Show the last optimal (assumes curriculum order)
        optimal = optimal_tours[-1]
        axes[0, 0].axhline(y=optimal, color='green', linestyle='--', linewidth=2,
                          label=f'Optimal ({optimal:.2f})', alpha=0.7)
    axes[0, 0].legend(loc='best', fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Tour Length')
    axes[0, 1].set_title('Best Tour Length')
    # Add optimal tour reference line if available
    if optimal_tours:
        optimal = optimal_tours[-1]
        axes[0, 1].axhline(y=optimal, color='green', linestyle='--', linewidth=2,
                          label=f'Optimal ({optimal:.2f})', alpha=0.7)
    axes[0, 1].legend(loc='best', fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Policy Loss')
    axes[1, 0].set_title('Policy Loss')
    axes[1, 0].legend(loc='best', fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Entropy')
    axes[1, 1].set_title('Entropy (Exploration)')
    axes[1, 1].legend(loc='best', fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_combined_loss(histories, labels, output_path=None):
    """Plot combined loss and entropy for any problem type."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Training Losses & Entropy', fontsize=14, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
    
    for i, (history, label) in enumerate(zip(histories, labels)):
        epochs = [h['epoch'] for h in history['history']]
        
        # Policy loss
        axes[0].plot(epochs, [h['policy_loss'] for h in history['history']], 
                    label=label, color=colors[i], linewidth=1.5)
        
        # Value loss
        axes[1].plot(epochs, [h['value_loss'] for h in history['history']], 
                    label=label, color=colors[i], linewidth=1.5)
        
        # Entropy
        axes[2].plot(epochs, [h['entropy'] for h in history['history']], 
                    label=label, color=colors[i], linewidth=1.5)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Policy Loss')
    axes[0].set_title('Policy Loss')
    axes[0].legend(loc='best', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Value Loss')
    axes[1].set_title('Value Loss')
    axes[1].legend(loc='best', fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Entropy')
    axes[2].set_title('Entropy')
    axes[2].legend(loc='best', fontsize=8)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        # Add _loss suffix to output path
        base, ext = os.path.splitext(output_path)
        loss_path = f"{base}_losses{ext}"
        plt.savefig(loss_path, dpi=150, bbox_inches='tight')
        print(f"Saved loss plot to {loss_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot training experiment results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single history file
  python plot_experiment.py --history checkpoints/knapsack/knapsack_history.json
  
  # Multiple files with labels (for comparison)
  python plot_experiment.py --history p01_history.json p02_history.json --labels "p01" "p02"
  
  # Save to file instead of showing
  
""")
    
    parser.add_argument('--history', type=str, nargs='+', required=True,
                        help='Path(s) to training history JSON file(s)')
    parser.add_argument('--labels', type=str, nargs='+', default=None,
                        help='Labels for each history file (for legends)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (default: show interactive plot)')
    parser.add_argument('--loss-only', action='store_true',
                        help='Only plot loss and entropy (problem-agnostic)')
    
    args = parser.parse_args()
    
    # Load all history files
    histories = []
    for path in args.history:
        try:
            histories.append(load_history(path))
            print(f"Loaded: {path}")
        except FileNotFoundError:
            print(f"Error: File not found: {path}")
            return
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON: {path}")
            return
    
    # Set labels
    if args.labels:
        if len(args.labels) != len(histories):
            print(f"Error: Number of labels ({len(args.labels)}) must match "
                  f"number of history files ({len(histories)})")
            return
        labels = args.labels
    else:
        # Use dataset/graph names as default labels
        labels = []
        for h in histories:
            if 'dataset' in h:
                labels.append(h['dataset'])
            elif 'graph' in h:
                labels.append(h['graph'])
            else:
                labels.append('Unknown')
    
    # Detect problem type
    problem_type = histories[0].get('problem_type', 'unknown')
    
    # Check all files are same problem type
    for h in histories[1:]:
        if h.get('problem_type') != problem_type:
            print("Warning: Mixing different problem types. Using loss-only plot.")
            args.loss_only = True
            break
    
    print(f"Problem type: {problem_type}")
    print(f"Comparing {len(histories)} experiment(s)")
    
    # Create output directory if needed
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    # Plot based on problem type
    if args.loss_only:
        plot_combined_loss(histories, labels, args.output)
    elif problem_type == 'knapsack':
        plot_knapsack(histories, labels, args.output)
    elif problem_type == 'graph_coloring':
        plot_graph_coloring(histories, labels, args.output)
    elif problem_type == 'tsp':
        plot_tsp(histories, labels, args.output)
    else:
        print(f"Unknown problem type: {problem_type}. Plotting loss only.")
        plot_combined_loss(histories, labels, args.output)
    
    print("Done!")


if __name__ == '__main__':
    main()
