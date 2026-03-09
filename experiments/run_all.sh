#!/bin/bash
# ============================================================================
# Run all experiments for Sinkhorn OT Knowledge Distillation
#
# This script runs the full experimental pipeline:
#   1. Pretrain teacher model
#   2. Train student baseline (no distillation)
#   3. KL-KD baseline
#   4. Fixed Sinkhorn OT-KD
#   5. Adaptive Sinkhorn OT-KD (our method)
#   6. Evaluate and generate comparison figures
#
# Usage:
#   bash experiments/run_all.sh [cifar10|cifar100]
# ============================================================================

set -e

DATASET=${1:-cifar100}
TEACHER=resnet110
STUDENT=resnet20
EPOCHS=200
CKPT_DIR="./checkpoints/${DATASET}"

echo "============================================"
echo "Dataset:  ${DATASET}"
echo "Teacher:  ${TEACHER}"
echo "Student:  ${STUDENT}"
echo "Epochs:   ${EPOCHS}"
echo "Ckpt dir: ${CKPT_DIR}"
echo "============================================"

# ── Step 1: Pretrain teacher ──────────────────────────────────────────────
echo ""
echo "[1/6] Pretraining teacher (${TEACHER})..."
python train.py \
    --mode pretrain_teacher \
    --teacher ${TEACHER} \
    --dataset ${DATASET} \
    --pretrain_epochs ${EPOCHS} \
    --checkpoint_dir ${CKPT_DIR}

# ── Step 2: Student baseline (no distillation) ───────────────────────────
echo ""
echo "[2/6] Training student baseline (no KD)..."
python train.py \
    --mode student_baseline \
    --student ${STUDENT} \
    --dataset ${DATASET} \
    --epochs ${EPOCHS} \
    --checkpoint_dir ${CKPT_DIR}

# ── Step 3: KL-KD baseline ───────────────────────────────────────────────
echo ""
echo "[3/6] Training KL-KD baseline..."
python train.py \
    --mode distill \
    --method kl_kd \
    --teacher ${TEACHER} \
    --student ${STUDENT} \
    --dataset ${DATASET} \
    --epochs ${EPOCHS} \
    --temperature 4.0 \
    --alpha 0.9 \
    --checkpoint_dir ${CKPT_DIR}

# ── Step 4: Fixed Sinkhorn OT-KD ─────────────────────────────────────────
echo ""
echo "[4/6] Training Fixed Sinkhorn OT-KD..."
python train.py \
    --mode distill \
    --method sinkhorn_kd \
    --teacher ${TEACHER} \
    --student ${STUDENT} \
    --dataset ${DATASET} \
    --epochs ${EPOCHS} \
    --temperature 4.0 \
    --lambda_ot 0.5 \
    --epsilon 0.05 \
    --cost_type uniform \
    --checkpoint_dir ${CKPT_DIR}

# ── Step 5: Adaptive Sinkhorn OT-KD (our method) ─────────────────────────
echo ""
echo "[5/6] Training Adaptive Sinkhorn OT-KD (OURS)..."
python train.py \
    --mode distill \
    --method adaptive_sinkhorn_kd \
    --teacher ${TEACHER} \
    --student ${STUDENT} \
    --dataset ${DATASET} \
    --epochs ${EPOCHS} \
    --temperature 4.0 \
    --lambda_ot 0.5 \
    --epsilon 0.05 \
    --cost_lr 0.01 \
    --cost_update_freq 10 \
    --val_fraction 0.1 \
    --checkpoint_dir ${CKPT_DIR}

# ── Step 6: Evaluate and compare ─────────────────────────────────────────
echo ""
echo "[6/6] Evaluating and generating figures..."
python evaluate.py \
    --dataset ${DATASET} \
    --checkpoint_dir ${CKPT_DIR} \
    --teacher ${TEACHER} \
    --student ${STUDENT}

echo ""
echo "============================================"
echo "All experiments complete!"
echo "Results in: ${CKPT_DIR}/"
echo "Figures in: ${CKPT_DIR}/figures/"
echo "============================================"
