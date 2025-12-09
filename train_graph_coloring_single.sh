#!/bin/bash
# Train each graph coloring instance independently (no curriculum)
# Each instance starts fresh without resuming from previous

DEVICE="cuda:1"
EPOCHS=30
OUTPUT_DIR="./checkpoints/graph_coloring_single"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "Graph Coloring Single-Instance Training"
echo "=========================================="

# Mycielski graphs with their chromatic numbers
declare -A CHROMATIC
CHROMATIC[myciel2]=3
CHROMATIC[myciel3]=4
CHROMATIC[myciel4]=5
CHROMATIC[myciel5]=6
CHROMATIC[myciel6]=7
CHROMATIC[myciel7]=8

# Train each instance independently
for graph in myciel7; do
    NUM_COLORS=${CHROMATIC[$graph]}
    
    echo ""
    echo "=========================================="
    echo "Training ${graph} (single, no resume)"
    echo "Colors: $NUM_COLORS"
    echo "=========================================="
    
    python train_graph_coloring.py \
        --graph_file data/graph_coloring/${graph}.col \
        --num_colors $NUM_COLORS \
        --epochs $EPOCHS \
        --batch_size 128 \
        --device $DEVICE \
        --output_dir $OUTPUT_DIR \
        --exp_name "${graph}_single"
    
    echo "Completed ${graph}"
done

echo ""
echo "=========================================="
echo "All single-instance training completed!"
echo "History files saved to: $OUTPUT_DIR"
echo "=========================================="
