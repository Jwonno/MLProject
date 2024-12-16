#!/usr/bin/env bash         
export CUDA_VISIBLE_DEVICES=1

EVAL_INTERVAL=5
SEED=42
EPOCHS=40
BATCH_SIZE=64
LR=1e-5
tau=30
H_FLIP_PROB=0.5

SAMPLE=random
LOSS=triplet
EMBED_DIM=128

OUTPUT_DIR=./output/${SAMPLE}_${LOSS}_${EMBED_DIM}

python main.py \
    --device=cuda \
    --out-dir=${OUTPUT_DIR} \
    --eval-it=${EVAL_INTERVAL} \
    --seed=${SEED} \
    --epochs=${EPOCHS} \
    --batch-size=${BATCH_SIZE} \
    --sampling=${SAMPLE} \
    --loss=${LOSS} \
    --embedding-dim=${EMBED_DIM} \
    --lr=${LR} \
    --tau=${tau} \
    --rnd-resize \
    --h-prob=0.5 \