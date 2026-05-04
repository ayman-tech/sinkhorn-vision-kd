# Sinkhorn Optimal Transport Knowledge Distillation with Learnable Cost Matrix

Compress large vision models into compact students using **Optimal Transport** as the distillation loss — with a cost matrix that is *learned jointly* with the student via bilevel optimization.

---

## Goal

Standard Knowledge Distillation (KD) uses KL divergence to align student and teacher distributions. KL treats all class confusions equally — confusing "cat" with "dog" costs the same as confusing "cat" with "truck". This ignores the semantic structure of the label space.

We replace KL with **Sinkhorn Optimal Transport (OT) distance**, which measures how much probability mass must be moved between teacher and student distributions according to a cost matrix `C[i][j]` — the penalty for confusing class `i` with class `j`.

Our core contribution: instead of hand-designing `C`, we **learn it jointly with the student** via bilevel optimization. The learned cost matrix captures semantic geometry (e.g. animal classes cluster with low mutual cost; cross-category confusions incur high cost).

---

## Novel Contributions

**1. Learnable cost matrix.**
Prior OT-KD methods use a fixed, hand-designed cost matrix. We propose learning `C` jointly with the student via bilevel optimization. `C[i][j]` represents how costly it is to confuse class `i` with class `j`, and is parameterized to ensure symmetry, non-negativity, zero diagonal, and boundedness.

**2. Interpretable class geometry.**
The learned cost matrix reveals semantic structure: after training on CIFAR-100, animal classes cluster together (low mutual cost), vehicle classes cluster together, and cross-category confusions incur high cost.

---

## Technical Formulation

Four optimization levels operate simultaneously during training:

```
LEVEL 1: Sinkhorn Iterations      ← Pure Convex Optimization
LEVEL 2: Student Network Training ← Non-convex (standard)
LEVEL 3: Cost Matrix Learning     ← Constrained Convex Optimization
LEVEL 4: Bilevel Optimization     ← All of the above combined
```

**Standard KD baseline:**

$$\mathcal{L}_{KD} = \alpha \cdot \mathrm{KL}\!\left(\mathrm{softmax}\!\left(\tfrac{z_T}{\tau}\right) \parallel \mathrm{softmax}\!\left(\tfrac{z_S}{\tau}\right)\right) + (1-\alpha) \cdot \mathrm{CE}(z_S, y)$$

**Our proposed loss:**

$$\mathcal{L}_{\text{total}} = \mathrm{CE}(z_S, y) + \lambda \cdot W_\varepsilon(p_T, p_S;\, C)$$

where the Sinkhorn distance is:

$$W_\varepsilon(p_T, p_S) = \min_{\pi \in \Pi(p_T,\, p_S)} \langle C,\, \pi \rangle + \varepsilon \cdot \mathrm{KL}(\pi \parallel p_T \otimes p_S)$$

**Bilevel optimization** for the learnable cost matrix:
- **Outer loop:** Update `C` on validation data every `K` steps (after warmup)
- **Inner loop:** Update student `θ` on training data with `C` fixed

**Cost matrix parameterization** — every forward pass through `LearnableCostMatrix`:

1. Raw parameter `A` (unconstrained `nn.Parameter`, shape `K×K`)
2. Symmetrize: `S = (A + Aᵀ) / 2`
3. Non-negativity: `C' = softplus(S)`
4. Zero diagonal: `C = C' − diag(diag(C'))`
5. Normalize: `C = C / max(C)` → values in `[0, 1]`

This guarantees: `C ≥ 0`, `C = Cᵀ`, `diag(C) = 0`, `C ∈ [0, 1]`.

---

## Architecture

### Training Pipeline

```
                      Input Batch (x, y)
                             │
             ┌───────────────┴───────────────┐
             ▼                               ▼
   ┌──────────────────┐           ┌──────────────────┐
   │     Teacher      │           │     Student      │
   │   ResNet-110     │           │    ResNet-20     │
   │    [FROZEN]      │           │   [TRAINABLE]    │
   │   1.73M params   │           │   0.27M params   │
   │   506.3M FLOPs   │           │    81.6M FLOPs   │
   └──────────────────┘           └──────────────────┘
             │                               │
         z_T (logits)                   z_S (logits)
             │                               │
      softmax(z_T / T)               softmax(z_S / T)
             │                               │
            p_T                             p_S
             │                               │
             └──────────────┬────────────────┘
                            ▼
             ┌──────────────────────────────┐
             │       Sinkhorn OT Loss       │
             │    W_ε(p_T, p_S ; C)         │
             │                              │
             │  C ∈ R^(K×K)  cost matrix    │
             │  [LEARNABLE — our method]    │
             │  [FIXED      — baseline]     │
             └──────────────────────────────┘
                            │
                            ▼
             ┌──────────────────────────────┐
             │  L_total = L_CE(z_S, y)      │
             │          + λ · W_ε(p_T,p_S;C)│
             └──────────────────────────────┘
                            │
                      Backprop
                            │
             ┌──────────────┴──────────────┐
             ▼                             ▼
   Update θ (student)          Update C (cost matrix)
     [inner loop]                  [outer loop]
     every step                  every K steps
                                 after warmup
```

### Bilevel Optimization

```
┌─────────────────────────────────────────────────────┐
│                BILEVEL OPTIMIZATION                 │
│                                                     │
│  ┌───────────────────────────────────────────────┐  │
│  │  INNER LOOP  (every training step)            │  │
│  │                                               │  │
│  │    min_θ  L_train(θ, C_fixed)                 │  │
│  │    → SGD step on student θ                    │  │
│  └───────────────────────────────────────────────┘  │
│                                                     │
│  ┌───────────────────────────────────────────────┐  │
│  │  OUTER LOOP  (every K steps, after warmup)    │  │
│  │                                               │  │
│  │    min_C  L_val(θ_fixed, C)                   │  │
│  │    → Adam step on cost matrix C               │  │
│  │    → gradients flow through Sinkhorn solver   │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### Cost Matrix Parameterization

```
  Raw A  ∈ R^(K×K)   [unconstrained nn.Parameter]
         │
         ▼
  S = (A + Aᵀ) / 2              ← symmetry
         │
         ▼
  C' = softplus(S)              ← non-negativity
         │
         ▼
  C = C' − diag(diag(C'))       ← zero diagonal
         │
         ▼
  C = C / (max(C) + 1e-8)       ← normalize to [0, 1]
         │
         ▼
  Valid cost matrix  ✓ symmetric  ✓ non-negative
                     ✓ zero diagonal  ✓ bounded
```

### Sinkhorn Iterations (log-domain, implemented in `log_sinkhorn`)

```
  Initialize f = 0, g = 0  (dual variables, shape B×K)

  For t = 1 ... max_iter:
    f_i  ← −logsumexp_j ( g_j − C_ij/ε + log p_S_j )
    g_j  ← −logsumexp_i ( f_i − C_ij/ε + log p_T_i )
    if ‖f − f_prev‖∞ < threshold: break

  π_ij = exp( f_i + g_j − C_ij/ε + log p_T_i + log p_S_j )

  W_ε  = mean_batch  Σ_ij  C_ij · π_ij
```

---

## Project Structure

```
sinkhorn-vision-kd/
├── configs/
│   ├── cifar10_config.yaml        # Hyperparameters for CIFAR-10
│   └── cifar100_config.yaml       # Hyperparameters for CIFAR-100
├── models/
│   ├── resnet.py                  # ResNet-20 / 56 / 110 for CIFAR
│   └── mobilenet.py               # MobileNetV2 (alternative student)
├── distillation/
│   ├── kl_distill.py              # Baseline: KL divergence KD
│   ├── sinkhorn_distill.py        # Baseline: fixed cost matrix OT-KD
│   └── adaptive_sinkhorn.py       # Ours: learnable cost matrix OT-KD
├── utils/
│   ├── data_loader.py             # CIFAR-10/100 with standard augmentation
│   ├── metrics.py                 # Accuracy, ECE, NLL, Brier, FLOPs
│   └── visualization.py           # Cost matrix heatmaps, training curves
├── train.py                       # Training script (all methods)
├── evaluate.py                    # Evaluation, comparison table, plots
└── experiments/
    └── run_all.sh                 # Full experimental pipeline
```

---

## Setup

```bash
git clone <repo-url>
cd sinkhorn-vision-kd

# Recommended
uv sync

# Or with pip
pip install -e .
```

**Requirements:** Python 3.12+, PyTorch 2.0+, torchvision, scikit-learn, POT, numpy, matplotlib, seaborn, tqdm, pyyaml.

---

## How to Run

All commands below use CIFAR-10. Replace `cifar10` with `cifar100` throughout for CIFAR-100.

### 1 — Pretrain Teacher

```bash
python train.py --mode pretrain_teacher --teacher resnet110 --dataset cifar10
```

Saves to `./checkpoints/cifar10/cifar10_resnet110_teacher.pth`.

### 2 — Train Student Baseline (no distillation)

```bash
python train.py --mode student_baseline --student resnet20 --dataset cifar10
```

### 3 — KL-KD Baseline

```bash
python train.py --method kl_kd \
    --teacher resnet110 --student resnet20 --dataset cifar10
```

### 4 — Fixed Sinkhorn OT-KD

```bash
python train.py --method sinkhorn_kd \
    --teacher resnet110 --student resnet20 --dataset cifar10 \
    --epsilon 0.05 --cost_type uniform
```

`--cost_type` accepts `uniform`, `label_distance`, or `random`.

### 5 — Adaptive Sinkhorn OT-KD (Our Method)

```bash
python train.py --method adaptive_sinkhorn_kd \
    --teacher resnet110 --student resnet20 --dataset cifar10 \
    --epsilon 0.05 --cost_lr 0.01 --cost_update_freq 10
```

### 6 — Evaluate and Compare All Methods

```bash
python evaluate.py --dataset cifar10 --checkpoint_dir ./checkpoints/cifar10
```

Prints the comparison table and saves the following plots to `./checkpoints/cifar10/figures/`:

| File | Contents |
|---|---|
| `learned_cost_matrix.png` | Heatmap of the final learned cost matrix C |
| `training_curves.png` | Loss and accuracy over training epochs |
| `compression_tradeoff.png` | Accuracy vs. FLOPs scatter across methods |
| `cost_evolution.png` | How C changes across training |
| `reliability_diagrams.png` | Calibration curves (ECE) |

### Run Full Pipeline

```bash
bash experiments/run_all.sh cifar10
# or
bash experiments/run_all.sh cifar100
```

### Multi-seed Statistical Significance

```bash
python evaluate.py --dataset cifar10 --run_seeds --num_seeds 3
```

---

## Key Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `--temperature` | 4.0 | Softmax temperature for distribution softening |
| `--epsilon` | 0.05 | Sinkhorn entropic regularization (smaller = sharper transport, less stable) |
| `--lambda_ot` | 0.5 | Weight of OT loss in total loss |
| `--alpha` | 0.9 | KD loss weight for KL-KD baseline |
| `--cost_lr` | 0.01 | Adam learning rate for cost matrix C |
| `--cost_update_freq` | 10 | Update C every K training steps (outer loop) |
| `--cost_grad_clip` | 1.0 | Gradient clipping norm for C updates |
| `--val_fraction` | 0.1 | Fraction of training data for outer-loop validation |
| `--cost_warmup_epochs` | 30 | Epochs before C starts updating (student warms up first) |

---

## Results (CIFAR-10)

| Method | Top-1 | Top-5 | ECE | NLL | Brier | Params | FLOPs |
|---|---|---|---|---|---|---|---|
| Teacher (ResNet-110) | 94.35% | 99.79% | 0.0394 | 0.2883 | 0.0964 | 1.73M | 506.3M |
| Student (no KD) | 92.79% | 99.77% | 0.0344 | 0.2586 | 0.1126 | 0.27M | 81.6M |
| KL-KD | 43.39% | 92.21% | 0.4472 | 4.0661 | 0.9766 | 0.27M | 81.6M |
| Fixed-OT-KD | 74.04% | 98.23% | 0.0686 | 0.8017 | 0.3693 | 0.27M | 81.6M |
| **Adaptive-OT-KD (Ours)** | **92.98%** | **99.85%** | **0.0348** | **0.2618** | **0.1128** | 0.27M | 81.6M |

Adaptive-OT-KD matches the student baseline on all calibration metrics while improving Top-1, at a **6.4× parameter reduction** and **6.2× FLOPs reduction** over the teacher.

---

## Methods Compared

| Method | Loss | Cost Matrix |
|---|---|---|
| Student (no KD) | Cross-entropy only | — |
| KL-KD | KL divergence (Hinton et al., 2015) | — |
| Fixed-OT-KD | Sinkhorn OT distance | Fixed (uniform / label\_distance / random) |
| **Adaptive-OT-KD (Ours)** | Sinkhorn OT distance | **Learned via bilevel optimization** |

---

## References

- Hinton et al. "Distilling the Knowledge in a Neural Network" (NeurIPS 2015)
- Cuturi. "Sinkhorn Distances: Lightspeed Computation of Optimal Transport" (NeurIPS 2013)
- He et al. "Deep Residual Learning for Image Recognition" (CVPR 2016)
- SinKD (COLING 2024): Sinkhorn KD for NLP tasks
- MultiLevelOT (AAAI 2025): Multi-level OT for LLM distillation
