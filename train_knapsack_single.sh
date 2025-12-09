#!/bin/bash
# Train each knapsack instance independently (no curriculum)
# Each instance starts fresh without resuming from previous

DEVICE="cuda:1"
EPOCHS=300
OUTPUT_DIR="./checkpoints/knapsack_single"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "Knapsack Single-Instance Training"
echo "=========================================="

# Train each instance independently
for i in 01 02 03 04 05 06 07 08 09 10 11; do
    echo ""
    echo "=========================================="
    echo "Training p${i} (single, no resume)"
    echo "=========================================="
    
    python train_knapsack.py \
        --data_dir data/knapsack/p${i} \
        --epochs $EPOCHS \
        --batch_size 512 \
        --device $DEVICE \
        --output_dir $OUTPUT_DIR \
        --exp_name "p${i}_single"
    
    echo "Completed p${i}"
done

echo ""
echo "=========================================="
echo "All single-instance training completed!"
echo "History files saved to: $OUTPUT_DIR"
echo "=========================================="
