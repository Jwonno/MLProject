#!/usr/bin/env bash         
export CUDA_VISIBLE_DEVICES=0

EVAL_INTERVAL=1
SEED=42
EPOCHS=100
BATCH_SIZE=64
LR=1e-5
tau1=30 
tau2=70
H_FLIP_PROB=0.5

SAMPLE=distance
LOSS=margin
EMBED_DIM=512

OUTPUT_DIR=./output/opt_${SAMPLE}_${LOSS}_${EMBED_DIM}_${EPOCHS}

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
    --tau ${tau1} ${tau2} \
    --rnd-resize \
    --h-prob=0.5 \