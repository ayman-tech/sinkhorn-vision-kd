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

# ── Ablation sweeps ───────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "Running ablation sweeps..."
echo "============================================"

BASE_ARGS="--mode distill --method adaptive_sinkhorn_kd \
    --teacher ${TEACHER} --student ${STUDENT} \
    --dataset ${DATASET} --epochs ${EPOCHS} --seed 42"

# Epsilon sweep
echo ""
echo "[Ablation] Epsilon sweep: 0.01 0.05 0.1 0.5"
for eps in 0.01 0.05 0.1 0.5; do
    echo "  epsilon=${eps}..."
    python train.py ${BASE_ARGS} \
        --epsilon ${eps} \
        --checkpoint_dir "${CKPT_DIR}/ablation/eps_${eps}"
done

# lambda_ot sweep
echo ""
echo "[Ablation] lambda_ot sweep: 0.1 0.5 1.0 2.0"
for lam in 0.1 0.5 1.0 2.0; do
    echo "  lambda_ot=${lam}..."
    python train.py ${BASE_ARGS} \
        --lambda_ot ${lam} \
        --checkpoint_dir "${CKPT_DIR}/ablation/lam_${lam}"
done

# cost_update_freq sweep
echo ""
echo "[Ablation] cost_update_freq sweep: 1 5 10 20 50"
for freq in 1 5 10 20 50; do
    echo "  cost_update_freq=${freq}..."
    python train.py ${BASE_ARGS} \
        --cost_update_freq ${freq} \
        --checkpoint_dir "${CKPT_DIR}/ablation/freq_${freq}"
done

# Fixed C baseline (learning disabled)
echo ""
echo "[Ablation] Fixed C baseline (--no-learn_cost)..."
python train.py ${BASE_ARGS} \
    --no-learn_cost \
    --checkpoint_dir "${CKPT_DIR}/ablation/fixed_cost"

# Unconstrained C baseline (projection disabled)
echo ""
echo "[Ablation] Unconstrained C baseline (--no-constrain_cost)..."
python train.py ${BASE_ARGS} \
    --no-constrain_cost \
    --checkpoint_dir "${CKPT_DIR}/ablation/unconstrained_cost"

echo ""
echo "============================================"
echo "Ablation sweeps complete!"
echo "Logs in: logs/"
echo "============================================"

# ── True seed variance (same config, different seeds) ────────────────────
echo ""
echo "============================================"
echo "Running seed variance experiments (3 seeds)..."
echo "============================================"

for seed in 42 43 44; do
    echo ""
    echo "[Seed variance] kl_kd seed=${seed}..."
    uv run python train.py \
        --mode distill --method kl_kd \
        --teacher ${TEACHER} --student ${STUDENT} \
        --dataset ${DATASET} --epochs 5 \
        --temperature 4.0 --alpha 0.9 \
        --seed ${seed}

    echo "[Seed variance] sinkhorn_kd seed=${seed}..."
    uv run python train.py \
        --mode distill --method sinkhorn_kd \
        --teacher ${TEACHER} --student ${STUDENT} \
        --dataset ${DATASET} --epochs 5 \
        --temperature 4.0 --lambda_ot 0.5 --epsilon 0.05 \
        --cost_type uniform --seed ${seed}

    echo "[Seed variance] adaptive_sinkhorn_kd seed=${seed}..."
    uv run python train.py \
        --mode distill --method adaptive_sinkhorn_kd \
        --teacher ${TEACHER} --student ${STUDENT} \
        --dataset ${DATASET} --epochs 5 \
        --temperature 4.0 --lambda_ot 0.5 --epsilon 0.05 \
        --cost_lr 0.01 --cost_update_freq 10 \
        --learn_cost --constrain_cost \
        --seed ${seed}
done

echo ""
echo "Seed variance runs complete. Run:"
echo "  uv run python evaluate.py --dataset ${DATASET} --checkpoint_dir ${CKPT_DIR} --run_seeds"
