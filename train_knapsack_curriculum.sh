#!/bin/bash
# Curriculum training for Knapsack problem
# Trains sequentially on p01 -> p11

set -e

OUTPUT_DIR="./checkpoints/knapsack_curriculum"
EXP_NAME="knapsack_curriculum"
EPOCHS=100
DEVICE="cpu"  # Change to cuda:0 for GPU
BATCH_SIZE=512  # Increase for better GPU utilization (1024+ for A100)

echo "=== Knapsack Curriculum Training ==="
echo "Output: $OUTPUT_DIR"
echo "Epochs per stage: $EPOCHS"
echo ""

# p01 - 10 items
echo "[1/11] Training on p01..."
python train_knapsack.py \
    --data_dir data/knapsack/p01 \
    --epochs $EPOCHS \
    --output_dir $OUTPUT_DIR \
    --exp_name ${EXP_NAME}_p01 \
    --batch_size $BATCH_SIZE --device $DEVICE

# p02 - 5 items
echo "[2/11] Training on p02..."
python train_knapsack.py \
    --data_dir data/knapsack/p02 \
    --epochs $EPOCHS \
    --output_dir $OUTPUT_DIR \
    --exp_name ${EXP_NAME}_p02 \
    --resume ${OUTPUT_DIR}/${EXP_NAME}_p01_final.pt \
    --batch_size $BATCH_SIZE --device $DEVICE

# p03 - 6 items
echo "[3/11] Training on p03..."
python train_knapsack.py \
    --data_dir data/knapsack/p03 \
    --epochs $EPOCHS \
    --output_dir $OUTPUT_DIR \
    --exp_name ${EXP_NAME}_p03 \
    --resume ${OUTPUT_DIR}/${EXP_NAME}_p02_final.pt \
    --batch_size $BATCH_SIZE --device $DEVICE

# p04 - 7 items
echo "[4/11] Training on p04..."
python train_knapsack.py \
    --data_dir data/knapsack/p04 \
    --epochs $EPOCHS \
    --output_dir $OUTPUT_DIR \
    --exp_name ${EXP_NAME}_p04 \
    --resume ${OUTPUT_DIR}/${EXP_NAME}_p03_final.pt \
    --batch_size $BATCH_SIZE --device $DEVICE

# p05 - 8 items
echo "[5/11] Training on p05..."
python train_knapsack.py \
    --data_dir data/knapsack/p05 \
    --epochs $EPOCHS \
    --output_dir $OUTPUT_DIR \
    --exp_name ${EXP_NAME}_p05 \
    --resume ${OUTPUT_DIR}/${EXP_NAME}_p04_final.pt \
    --batch_size $BATCH_SIZE --device $DEVICE

# p06 - 7 items
echo "[6/11] Training on p06..."
python train_knapsack.py \
    --data_dir data/knapsack/p06 \
    --epochs $EPOCHS \
    --output_dir $OUTPUT_DIR \
    --exp_name ${EXP_NAME}_p06 \
    --resume ${OUTPUT_DIR}/${EXP_NAME}_p05_final.pt \
    --batch_size $BATCH_SIZE --device $DEVICE

# p07 - 15 items
echo "[7/11] Training on p07..."
python train_knapsack.py \
    --data_dir data/knapsack/p07 \
    --epochs $EPOCHS \
    --output_dir $OUTPUT_DIR \
    --exp_name ${EXP_NAME}_p07 \
    --resume ${OUTPUT_DIR}/${EXP_NAME}_p06_final.pt \
    --batch_size $BATCH_SIZE --device $DEVICE

# p08 - 24 items
echo "[8/11] Training on p08..."
python train_knapsack.py \
    --data_dir data/knapsack/p08 \
    --epochs $EPOCHS \
    --output_dir $OUTPUT_DIR \
    --exp_name ${EXP_NAME}_p08 \
    --resume ${OUTPUT_DIR}/${EXP_NAME}_p07_final.pt \
    --batch_size $BATCH_SIZE --device $DEVICE

# p09 - 30 items
echo "[9/11] Training on p09..."
python train_knapsack.py \
    --data_dir data/knapsack/p09 \
    --epochs $EPOCHS \
    --output_dir $OUTPUT_DIR \
    --exp_name ${EXP_NAME}_p09 \
    --resume ${OUTPUT_DIR}/${EXP_NAME}_p08_final.pt \
    --batch_size $BATCH_SIZE --device $DEVICE

# p10 - 35 items
echo "[10/11] Training on p10..."
python train_knapsack.py \
    --data_dir data/knapsack/p10 \
    --epochs $EPOCHS \
    --output_dir $OUTPUT_DIR \
    --exp_name ${EXP_NAME}_p10 \
    --resume ${OUTPUT_DIR}/${EXP_NAME}_p09_final.pt \
    --batch_size $BATCH_SIZE --device $DEVICE

# p11 - 40 items
echo "[11/11] Training on p11..."
python train_knapsack.py \
    --data_dir data/knapsack/p11 \
    --epochs $EPOCHS \
    --output_dir $OUTPUT_DIR \
    --exp_name ${EXP_NAME}_p11 \
    --resume ${OUTPUT_DIR}/${EXP_NAME}_p10_final.pt \
    --batch_size $BATCH_SIZE --device $DEVICE

echo ""
echo "=== Curriculum training complete! ==="
echo "Final model: ${OUTPUT_DIR}/${EXP_NAME}_p11_final.pt"
