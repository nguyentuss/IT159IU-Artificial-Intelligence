#!/bin/bash
# Curriculum training for Graph Coloring problem
# Trains sequentially on myciel2 -> myciel7
# Uses fixed num_colors=8 (max needed) for checkpoint compatibility

set -e

OUTPUT_DIR="./checkpoints/graph_coloring_curriculum"
EXP_NAME="gc_curriculum"
EPOCHS=20
DEVICE="cuda:1"  # Change to cuda:0 for GPU
NUM_COLORS=8  # Fixed for all stages (myciel7 needs 8)
BATCH_SIZE=128 

echo "=== Graph Coloring Curriculum Training ==="
echo "Output: $OUTPUT_DIR"
echo "Epochs per stage: $EPOCHS"
echo "Colors: $NUM_COLORS (fixed for all stages)"
echo ""

# myciel2 - 5 nodes
echo "[1/6] Training on myciel2..."
python train_graph_coloring.py \
    --graph_file data/graph_coloring/myciel2.col \
    --num_colors $NUM_COLORS \
    --epochs $EPOCHS \
    --output_dir $OUTPUT_DIR \
    --exp_name ${EXP_NAME}_myciel2 \
    --batch_size $BATCH_SIZE \
    --device $DEVICE

# myciel3 - 11 nodes
echo "[2/6] Training on myciel3..."
python train_graph_coloring.py \
    --graph_file data/graph_coloring/myciel3.col \
    --num_colors $NUM_COLORS \
    --epochs $EPOCHS \
    --output_dir $OUTPUT_DIR \
    --exp_name ${EXP_NAME}_myciel3 \
    --batch_size $BATCH_SIZE \
    --resume ${OUTPUT_DIR}/${EXP_NAME}_myciel2_final.pt \
    --device $DEVICE

# myciel4 - 23 nodes
echo "[3/6] Training on myciel4..."
python train_graph_coloring.py \
    --graph_file data/graph_coloring/myciel4.col \
    --num_colors $NUM_COLORS \
    --epochs $EPOCHS \
    --output_dir $OUTPUT_DIR \
    --exp_name ${EXP_NAME}_myciel4 \
    --resume ${OUTPUT_DIR}/${EXP_NAME}_myciel3_final.pt \
    --device $DEVICE

# myciel5 - 47 nodes
echo "[4/6] Training on myciel5..."
python train_graph_coloring.py \
    --graph_file data/graph_coloring/myciel5.col \
    --num_colors $NUM_COLORS \
    --epochs $EPOCHS \
    --output_dir $OUTPUT_DIR \
    --exp_name ${EXP_NAME}_myciel5 \
    --resume ${OUTPUT_DIR}/${EXP_NAME}_myciel4_final.pt \
    --device $DEVICE

# myciel6 - 95 nodes
echo "[5/6] Training on myciel6..."
python train_graph_coloring.py \
    --graph_file data/graph_coloring/myciel6.col \
    --num_colors $NUM_COLORS \
    --epochs $EPOCHS \
    --output_dir $OUTPUT_DIR \
    --exp_name ${EXP_NAME}_myciel6 \
    --resume ${OUTPUT_DIR}/${EXP_NAME}_myciel5_final.pt \
    --device $DEVICE

# myciel7 - 191 nodes
echo "[6/6] Training on myciel7..."
python train_graph_coloring.py \
    --graph_file data/graph_coloring/myciel7.col \
    --num_colors $NUM_COLORS \
    --epochs $EPOCHS \
    --output_dir $OUTPUT_DIR \
    --exp_name ${EXP_NAME}_myciel7 \
    --resume ${OUTPUT_DIR}/${EXP_NAME}_myciel6_final.pt \
    --device $DEVICE

echo ""
echo "=== Curriculum training complete! ==="
echo "Final model: ${OUTPUT_DIR}/${EXP_NAME}_myciel7_final.pt"
