import torch
torch.manual_seed(42)

print('='*60)
print('EXAMPLE DATA FOR ALL 3 PROBLEMS')
print('='*60)

# ============ TSP Example ============
print('\n[1] TSP - Traveling Salesman Problem')
print('-'*40)
from ppo_combinatorial.environments.tsp_env import generate_tsp_instances

coords = generate_tsp_instances(num_instances=1, num_cities=5)
print('City coordinates (5 cities, 2D):')
for i in range(5):
    x, y = coords[0, i, 0].item(), coords[0, i, 1].item()
    print(f'  City {i}: ({x:.4f}, {y:.4f})')

# ============ Knapsack Example ============
print('\n[2] KNAPSACK PROBLEM')
print('-'*40)
from ppo_combinatorial.environments.knapsack_env import generate_knapsack_instances

values, weights = generate_knapsack_instances(num_instances=1, num_items=5)
capacity = 0.5 * weights.sum()

print(f'Capacity: {capacity.item():.2f}')
print('Items:')
print('  Item | Value | Weight | Value/Weight')
print('  -----|-------|--------|-------------')
for i in range(5):
    v, w = values[0, i].item(), weights[0, i].item()
    print(f'    {i}  | {v:5.2f} | {w:6.2f} | {v/w:6.3f}')
print(f'Total weight: {weights.sum().item():.2f}')

# ============ Graph Coloring Example ============
print('\n[3] GRAPH COLORING')
print('-'*40)
from ppo_combinatorial.environments.graph_coloring_env import generate_graph_instances

adj = generate_graph_instances(num_instances=1, num_nodes=5, edge_probability=0.5)
print('Adjacency matrix (5 nodes):')
print('    0 1 2 3 4')
for i in range(5):
    row = ' '.join([str(int(adj[0,i,j].item())) for j in range(5)])
    print(f'  {i} {row}')

edges = []
for i in range(5):
    for j in range(i+1, 5):
        if adj[0,i,j] == 1:
            edges.append(f'({i},{j})')
print(f'Edges: {edges}')
print(f'Number of edges: {len(edges)}')

print('\n' + '='*60)
print('These are randomly generated - different each run!')
print('='*60)