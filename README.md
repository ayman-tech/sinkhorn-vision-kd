# Sinkhorn Optimal Transport Knowledge Distillation with Learnable Cost Matrix for Vision Models

**MSML604 — Graduate Optimization Course Project**

## Overview

We compress large vision models (teachers) into smaller ones (students) using Knowledge Distillation (KD), replacing the standard KL divergence loss with **Sinkhorn Optimal Transport (OT) distance** as the distillation loss. Our core novelty is a **learnable cost matrix C** that is jointly optimized with the student network via bilevel optimization.

### Three Novel Contributions

1. **First OT-KD for vision classification.** All existing OT-KD papers (SinKD, MultiLevelOT, KNOT) target NLP tasks. We are the first to apply Sinkhorn-based knowledge distillation to vision classification on CIFAR-10 and CIFAR-100.

2. **Learnable cost matrix.** Prior OT-KD methods use a fixed, hand-designed cost matrix. We propose learning C jointly with the student network. The cost matrix C[i][j] represents "how costly" it is to confuse class i with class j, and is parameterized to ensure symmetry, non-negativity, zero diagonal, and boundedness.

3. **Interpretable class geometry.** The learned cost matrix reveals semantic structure: after training on CIFAR-100, animal classes cluster together (low mutual cost), vehicle classes cluster together, and cross-category confusions incur high cost.

## Technical Formulation

**Standard KD (baseline):**
```
L_KD = α · KL(softmax(z_T/τ) ∥ softmax(z_S/τ)) + (1-α) · CE(z_S, y)
```

**Our proposed loss:**
```
L_total = CE(f_θ(x), y) + λ · W_ε(p_T, p_S; C)
```

where W_ε is the Sinkhorn distance:
```
W_ε(p_T, p_S) = min_{π ∈ Π(p_T, p_S)} ⟨C, π⟩ + ε · KL(π ∥ p_T ⊗ p_S)
```

**Bilevel optimization** for the learnable cost matrix:
- **Outer loop:** Update C to minimize validation loss (every K steps)
- **Inner loop:** Update student θ to minimize training loss with fixed C

**Cost matrix parameterization** (ensuring validity):
1. Raw parameter A (unconstrained)
2. Symmetrize: S = (A + Aᵀ) / 2
3. Non-negativity: C' = softplus(S)
4. Zero diagonal: C = C' − diag(diag(C'))
5. Normalize: C = C / max(C)

## Project Structure

```
sinkhorn-vision-kd/
├── configs/
│   ├── cifar10_config.yaml
│   └── cifar100_config.yaml
├── models/
│   ├── resnet.py           # ResNet-20/56/110 for CIFAR
│   └── mobilenet.py        # MobileNetV2 (lightweight student)
├── distillation/
│   ├── kl_distill.py       # Baseline KL-KD
│   ├── sinkhorn_distill.py # Fixed cost matrix OT-KD
│   └── adaptive_sinkhorn.py # Learnable cost matrix OT-KD [OURS]
├── utils/
│   ├── data_loader.py      # CIFAR-10/100 with standard augmentation
│   ├── metrics.py          # Accuracy, FLOPs, parameter count
│   └── visualization.py    # Cost matrix heatmaps, training curves
├── train.py                # Main training script
├── evaluate.py             # Evaluation and comparison
└── experiments/
    └── run_all.sh          # Run full experimental pipeline
```

## Setup

```bash
# Clone and install with uv
git clone <repo-url>
cd sinkhorn-vision-kd
uv sync
```

Or with pip:
```bash
pip install -e .
```

**Requirements:** Python 3.12+, PyTorch 2.0+, torchvision, POT, numpy, matplotlib, seaborn, tqdm, wandb (optional).

## Usage

### 1. Pretrain Teacher

```bash
python train.py --mode pretrain_teacher --teacher resnet110 --dataset cifar100
```

### 2. Train Student Baseline (no distillation)

```bash
python train.py --mode student_baseline --student resnet20 --dataset cifar100
```

### 3. Run KL-KD Baseline

```bash
python train.py --method kl_kd --teacher resnet110 --student resnet20 --dataset cifar100
```

### 4. Run Fixed Sinkhorn OT-KD

```bash
python train.py --method sinkhorn_kd --teacher resnet110 --student resnet20 \
    --dataset cifar100 --epsilon 0.05 --cost_type uniform
```

### 5. Run Adaptive Sinkhorn OT-KD (Our Method)

```bash
python train.py --method adaptive_sinkhorn_kd --teacher resnet110 --student resnet20 \
    --dataset cifar100 --epsilon 0.05 --cost_lr 0.01 --cost_update_freq 10
```

### 6. Run All Experiments

```bash
bash experiments/run_all.sh cifar100
```

### 7. Evaluate and Compare

```bash
python evaluate.py --dataset cifar100 --checkpoint_dir ./checkpoints/cifar100
```

### Multi-seed Statistical Significance

```bash
python evaluate.py --dataset cifar100 --run_seeds --num_seeds 3
```

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--temperature` | 4.0 | Softmax temperature for distribution softening |
| `--epsilon` | 0.05 | Sinkhorn entropic regularization |
| `--lambda_ot` | 0.5 | OT loss weight |
| `--cost_lr` | 0.01 | Learning rate for cost matrix C |
| `--cost_update_freq` | 10 | Update C every K training steps |
| `--val_fraction` | 0.1 | Fraction of training data for bilevel validation |

## Methods Compared

| Method | Description |
|--------|-------------|
| **Student (no KD)** | ResNet-20 trained from scratch |
| **KL-KD** | Standard KD with KL divergence (Hinton et al., 2015) |
| **Fixed-OT-KD** | Sinkhorn OT with uniform cost matrix |
| **Adaptive-OT-KD** | Sinkhorn OT with learnable cost matrix **(ours)** |

## References

- Hinton et al. "Distilling the Knowledge in a Neural Network" (2015)
- Cuturi. "Sinkhorn Distances: Lightspeed Computation of Optimal Transport" (NeurIPS 2013)
- SinKD (COLING 2024): Sinkhorn KD for NLP
- MultiLevelOT (AAAI 2025): Multi-level OT for LLM distillation
- He et al. "Deep Residual Learning for Image Recognition" (CVPR 2016)
