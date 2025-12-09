#!/bin/bash
# Curriculum training for TSP problem
# Trains sequentially on tiny -> small -> medium -> large

set -e

OUTPUT_DIR="./checkpoints/tsp_curriculum"
EXP_NAME="tsp_curriculum"
EPOCHS=30
DEVICE="cuda:1"  # Change to cuda:0 for GPU
BATCH_SIZE=128

echo "=== TSP Curriculum Training ==="
echo "Output: $OUTPUT_DIR"
echo "Epochs per stage: $EPOCHS"
echo ""

# tiny - 10 cities
echo "[1/4] Training on tiny.csv (10 cities)..."
python train_tsp.py \
    --data_file data/tsp/tiny.csv \
    --epochs $EPOCHS \
    --output_dir $OUTPUT_DIR \
    --exp_name ${EXP_NAME}_tiny \
    --batch_size $BATCH_SIZE --device $DEVICE

# small-1 - 20 cities
echo "[2/4] Training on small-1.csv (20 cities)..."
python train_tsp.py \
    --data_file data/tsp/small.csv \
    --epochs $EPOCHS \
    --output_dir $OUTPUT_DIR \
    --exp_name ${EXP_NAME}_small-1 \
    --resume ${OUTPUT_DIR}/${EXP_NAME}_tiny_final.pt \
    --batch_size $BATCH_SIZE --device $DEVICE


# small-2 - 25 cities
echo "[2/4] Training on small-1.csv (25 cities)..."
python train_tsp.py \
    --data_file data/tsp/small.csv \
    --epochs $EPOCHS \
    --output_dir $OUTPUT_DIR \
    --exp_name ${EXP_NAME}_small-2 \
    --resume ${OUTPUT_DIR}/${EXP_NAME}_small-1_final.pt \
    --batch_size $BATCH_SIZE --device $DEVICE
    
# small - 30 cities
echo "[4/4] Training on small.csv (30 cities)..."
python train_tsp.py \
    --data_file data/tsp/small.csv \
    --epochs $EPOCHS \
    --output_dir $OUTPUT_DIR \
    --exp_name ${EXP_NAME}_small \
    --resume ${OUTPUT_DIR}/${EXP_NAME}_small-2_final.pt \
    --batch_size $BATCH_SIZE --device $DEVICE

# # medium - 100 cities
# echo "[3/4] Training on medium.csv (100 cities)..."
# python train_tsp.py \
#     --data_file data/tsp/medium.csv \
#     --epochs $EPOCHS \
#     --output_dir $OUTPUT_DIR \
#     --exp_name ${EXP_NAME}_medium \
#     --resume ${OUTPUT_DIR}/${EXP_NAME}_small_final.pt \
#     --batch_size $BATCH_SIZE --device $DEVICE


echo ""
echo "=== Curriculum training complete! ==="
echo "Final model: ${OUTPUT_DIR}/${EXP_NAME}_large_final.pt"
