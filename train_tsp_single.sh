#!/bin/bash
# Train each TSP instance independently (no curriculum)
# Each instance starts fresh without resuming from previous

DEVICE="cuda:1"
EPOCHS=30
OUTPUT_DIR="./checkpoints/tsp_single"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "TSP Single-Instance Training"
echo "=========================================="

# TSP datasets (excluding medium.csv as per user request)
TSP_FILES=("tiny" "small-1" "small-2" "small")

# Train each instance independently
for tsp in "${TSP_FILES[@]}"; do
    echo ""
    echo "=========================================="
    echo "Training ${tsp} (single, no resume)"
    echo "=========================================="
    
    python train_tsp.py \
        --data_file data/tsp/${tsp}.csv \
        --epochs $EPOCHS \
        --batch_size 1024 \
        --device $DEVICE \
        --output_dir $OUTPUT_DIR \
        --exp_name "${tsp}_single"
    
    echo "Completed ${tsp}"
done

echo ""
echo "=========================================="
echo "All single-instance training completed!"
echo "History files saved to: $OUTPUT_DIR"
echo "=========================================="
