# Sinkhorn Optimal Transport Knowledge Distillation with Learnable Cost Matrix

Compress large vision models into compact students using **Optimal Transport** as the distillation loss with a cost matrix that is *learned jointly* with the student via bilevel optimization.

---

## Goal

Standard Knowledge Distillation (KD) uses KL divergence to align student and teacher distributions. KL treats all class confusions equally confusing "cat" with "dog" costs the same as confusing "cat" with "truck". This ignores the semantic structure of the label space.

We replace KL with **Sinkhorn Optimal Transport (OT) distance**, which measures how much probability mass must be moved between teacher and student distributions according to a cost matrix `C[i][j]` which is the penalty for confusing class `i` with class `j`.

Our core contribution: instead of hand-designing `C`, we **learn it jointly with the student** via bilevel optimization. The learned cost matrix captures semantic geometry on CIFAR-100, animal classes cluster together with low mutual cost, vehicle classes cluster together, and cross-category confusions incur high cost.

---

## Novel Contributions

**1. Learnable cost matrix.**
Prior OT-KD methods use a fixed, hand-designed cost matrix. We learn `C` jointly with the student. `C[i][j]` represents how costly it is to confuse class `i` with class `j`, parameterized to enforce symmetry, non-negativity, zero diagonal, and bounded values.

**2. Interpretable class geometry.**
The learned cost matrix reveals semantic structure. After training on CIFAR-100, classes within the same supergroup (e.g. all aquatic animals) have low mutual cost; cross-supergroup confusions incur high cost. This is quantified via the `class_clustering` and `nearest_neighbor_costs` visualizations.

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

$$\mathcal{L}_{KD} = \alpha \cdot T^2 \cdot \mathrm{KL}\!\left(\mathrm{softmax}\!\left(\tfrac{z_T}{\tau}\right) \parallel \mathrm{softmax}\!\left(\tfrac{z_S}{\tau}\right)\right) + (1-\alpha) \cdot \mathrm{CE}(z_S, y)$$

**Our proposed loss:**

$$\mathcal{L}_{\text{total}} = \mathrm{CE}(z_S, y) + \lambda \cdot W_\varepsilon(p_T, p_S;\, C)$$

where the Sinkhorn distance is:

$$W_\varepsilon(p_T, p_S) = \min_{\pi \in \Pi(p_T,\, p_S)} \langle C,\, \pi \rangle + \varepsilon \cdot \mathrm{KL}(\pi \parallel p_T \otimes p_S)$$

**Bilevel optimization** for the learnable cost matrix:
- **Outer loop:** Update `C` on a held-out validation split every `K` steps (after warmup)
- **Inner loop:** Update student `θ` on training data with `C` fixed

**Cost matrix parameterization** enforced on every forward pass through `LearnableCostMatrix`:

1. Raw parameter `A` —> unconstrained `nn.Parameter`, shape `K × K`
2. Symmetrize: `S = (A + Aᵀ) / 2`
3. Non-negativity: `C' = softplus(S)`
4. Zero diagonal: `C = C' * (1 − I)`
5. Normalize: `C = C / (max(C) + 1e-8)` → values in `[0, 1]`

Guarantees: `C ≥ 0`, `C = Cᵀ`, `diag(C) = 0`, `C ∈ [0, 1]`.

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
             │  C ∈ ℝ^(K×K)  cost matrix   │
             │  [LEARNABLE (our method)]    │
             │  [FIXED   ->  baseline]     │
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
│  │    → SGD + cosine LR on student θ             │  │
│  └───────────────────────────────────────────────┘  │
│                                                     │
│  ┌───────────────────────────────────────────────┐  │
│  │  OUTER LOOP  (every K steps, post-warmup)     │  │
│  │                                               │  │
│  │    min_C  L_val(θ_fixed, C)                   │  │
│  │    → Adam + grad clipping on cost matrix C    │  │
│  │    → gradients flow through Sinkhorn solver   │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘

Cost warmup: C is frozen for the first cost_warmup_epochs epochs
so the student builds basic features before C starts adapting.
```

### Cost Matrix Parameterization

```
  Raw A  ∈ ℝ^(K×K)    [unconstrained nn.Parameter]
         │
         ▼
  S = (A + Aᵀ) / 2              ← enforce symmetry
         │
         ▼
  C' = softplus(S)              ← enforce non-negativity
         │
         ▼
  C = C' * (1 − I)              ← enforce zero diagonal
         │
         ▼
  C = C / (max(C) + 1e-8)       ← normalize to [0, 1]
```

### Sinkhorn Iterations (log-domain, implemented in `log_sinkhorn`)

```
  Initialize f = 0, g = 0    (dual variables, shape B×K)

  For t = 1 ... max_iter:
    f_i  ← −logsumexp_j ( g_j − C_ij/ε + log p_S_j )
    g_j  ← −logsumexp_i ( f_i − C_ij/ε + log p_T_i )
    if ‖f − f_prev‖∞ < threshold: break

  π_ij = exp( f_i + g_j − C_ij/ε + log p_T_i + log p_S_j )

  W_ε  = mean_batch  Σ_ij  C_ij · π_ij
```

### ResNet Backbone (CIFAR variant)

```
Input (3 × 32 × 32)
      │
  Conv3×3 (3→16) → BN → ReLU
      │
  Layer 1: n BasicBlocks (16ch, stride=1)   [32×32]
      │
  Layer 2: n BasicBlocks (32ch, stride=2)   [16×16]
      │
  Layer 3: n BasicBlocks (64ch, stride=2)   [ 8×8 ]
      │
  AvgPool(8×8)
      │
  Linear(64 → num_classes)
      │
  Logits

Depth = 6n + 2:
  ResNet-20  (n=3)  →  0.27M params,   81.6M FLOPs
  ResNet-56  (n=9)  →  0.85M params,  ~250M FLOPs
  ResNet-110 (n=18) →  1.73M params,  506.3M FLOPs

All models support return_features=True to return
intermediate activations [f1, f2, f3] for feature-KD.
```

---

## Project Structure

```
sinkhorn-vision-kd/
├── configs/
│   ├── cifar10_config.yaml        # Full CIFAR-10 hyperparameter config
│   └── cifar100_config.yaml       # Full CIFAR-100 hyperparameter config
├── models/
│   ├── resnet.py                  # ResNet-20 / 56 / 110 for CIFAR
│   └── mobilenet.py               # MobileNetV2 (CIFAR-adapted, optional student)
├── distillation/
│   ├── kl_distill.py              # KL Distillation Loss (Hinton et al. baseline)
│   ├── sinkhorn_distill.py        # SinkhornDistillationLoss (fixed cost OT-KD)
│   │                              #   + log_sinkhorn solver
│   │                              #   + build_cost_matrix (uniform/label_distance/random)
│   └── adaptive_sinkhorn.py       # AdaptiveSinkhornKD (learnable cost [OURS])
│                                  #   + LearnableCostMatrix parameterization
├── utils/
│   ├── data_loader.py             # get_cifar_loaders, get_class_names
│   ├── metrics.py                 # accuracy, count_parameters, estimate_flops
│   │                              # collect_predictions, compute_ece, compute_nll
│   │                              # compute_brier_score, AverageMeter
│   └── visualization.py           # 8 plot functions (see Outputs section)
├── train.py                       # Training script: pretrain / distill / baseline
├── evaluate.py                    # Evaluation: metrics + comparison table + plots
├── main.py                        # Entry point stub
└── experiments/
    └── run_all.sh                 # Full 6-step pipeline
```

---

## Setup

```bash
git clone <repo-url>
cd sinkhorn-vision-kd

# Recommended: uv
uv sync

# Or: pip
pip install -e .
```

**Requirements:** Python ≥ 3.12, PyTorch ≥ 2.0, torchvision ≥ 0.15, scikit-learn, POT ≥ 0.9, numpy, matplotlib, seaborn, tqdm, scipy, pyyaml, wandb (optional).

---

## How to Run

All commands below use CIFAR-10. Swap `--dataset cifar10` for `--dataset cifar100` throughout.

You can also pass a YAML config file to override CLI args:

```bash
python train.py --config configs/cifar10_config.yaml
```

---

### Step 1. Pretrain Teacher

```bash
python train.py --mode pretrain_teacher \
    --teacher resnet110 --dataset cifar10 \
    --pretrain_epochs 200
```

Saves: `./checkpoints/cifar10/cifar10_resnet110_teacher.pth`

---

### Step 2. Train Student Baseline (no distillation)

```bash
python train.py --mode student_baseline \
    --student resnet20 --dataset cifar10
```

Saves: `./checkpoints/cifar10/resnet20_no_kd_best.pth`

---

### Step 3. KL-KD Baseline

```bash
python train.py --mode distill --method kl_kd \
    --teacher resnet110 --student resnet20 --dataset cifar10 \
    --temperature 4.0 --alpha 0.9
```

---

### Step 4. Fixed Sinkhorn OT-KD

```bash
python train.py --mode distill --method sinkhorn_kd \
    --teacher resnet110 --student resnet20 --dataset cifar10 \
    --epsilon 0.05 --lambda_ot 0.5 --cost_type uniform
```

`--cost_type` options: `uniform` (default), `label_distance`, `random`.

---

### Step 5. Adaptive Sinkhorn OT-KD (Our Method)

```bash
python train.py --mode distill --method adaptive_sinkhorn_kd \
    --teacher resnet110 --student resnet20 --dataset cifar10 \
    --epsilon 0.05 --lambda_ot 0.5 \
    --cost_lr 0.01 --cost_update_freq 10 \
    --cost_warmup_epochs 30 --val_fraction 0.1
```

---

### Step 6. Evaluate and Compare All Methods

```bash
python evaluate.py --dataset cifar10 --checkpoint_dir ./checkpoints/cifar10
```

Prints the comparison table (Top-1, Top-5, ECE, NLL, Brier, Params, FLOPs) and saves all plots to `./checkpoints/cifar10/figures/`.

---

### Run Full Pipeline at Once

```bash
bash experiments/run_all.sh cifar10
# or
bash experiments/run_all.sh cifar100
```

---

### Multi-Seed Statistical Significance

```bash
    python evaluate.py --dataset cifar10 --run_seeds --num_seeds 3
```

Trains all 4 variants per seed and reports mean ± std accuracy.

---

## Outputs

All plots are saved to `<checkpoint_dir>/figures/`:

| File | Description |
|---|---|
| `learned_cost_matrix.png` | Heatmap of final learned C (YlOrRd) |
| `training_curves.png` | Accuracy and loss curves across all methods |
| `compression_tradeoff.png` | Scatter: Params (M) vs Top-1 accuracy |
| `cost_evolution.png` | 5 snapshots of C across training epochs |
| `reliability_diagrams.png` | Calibration curves (confidence vs accuracy, ECE annotated) |
| `class_clustering.png` | t-SNE of cost matrix rows, colored by class/superclass |
| `nearest_neighbor_costs.png` | Table: top-5 nearest neighbors per class by cost |
| `nearest_neighbor_costs.txt` | Full text version of nearest neighbor table |

---

## Key Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `--temperature` | 4.0 | Softmax temperature for soft targets |
| `--epsilon` | 0.05 | Sinkhorn entropic regularization (smaller = sharper, less stable) |
| `--lambda_ot` | 0.5 | OT loss weight: `L = CE + lambda_ot * W_ε` |
| `--alpha` | 0.9 | KD weight for KL-KD only: `L = alpha*KD + (1-alpha)*CE` |
| `--cost_type` | uniform | Fixed cost matrix type: `uniform`, `label_distance`, `random` |
| `--cost_lr` | 0.01 | Adam learning rate for cost matrix C |
| `--cost_update_freq` | 10 | Update C every K training steps (outer loop) |
| `--cost_grad_clip` | 1.0 | Gradient clipping norm for C updates |
| `--cost_warmup_epochs` | 30 | Epochs before C starts updating |
| `--val_fraction` | 0.1 | Fraction of training data used as outer-loop validation set |
| `--warmup_epochs` | 5 | Linear LR warmup epochs for student |
| `--epochs` | 200 | Total training epochs |
| `--lr` | 0.1 | SGD learning rate (cosine annealed) |
| `--seed` | 42 | Random seed |

---

## Available Architectures

| Model | Role | Params | FLOPs | Notes |
|---|---|---|---|---|
| `resnet20` | Student | 0.27M | 81.6M | CIFAR ResNet (6n+2, n=3) |
| `resnet56` | Teacher or Student | 0.85M | ~250M | CIFAR ResNet (6n+2, n=9) |
| `resnet110` | Teacher | 1.73M | 506.3M | CIFAR ResNet (6n+2, n=18) |
| `mobilenetv2` | Student | ~2.3M* | — | CIFAR-adapted (stride=1 initial conv) |

\* MobileNetV2 param count depends on `--width_mult` (default 1.0).

All models expose `forward(x, return_features=True)` which additionally returns intermediate feature maps at each spatial resolution for feature-based distillation.

---

## Methods Compared

| Method | Loss | Cost Matrix |
|---|---|---|
| Student (no KD) | Cross-entropy only | — |
| KL-KD | `α·T²·KL(p_T ‖ p_S) + (1-α)·CE` | — |
| Fixed-OT-KD | `CE + λ·W_ε(p_T, p_S; C)` | Fixed (uniform / label\_distance / random) |
| **Adaptive-OT-KD (Ours)** | `CE + λ·W_ε(p_T, p_S; C)` | **Learned via bilevel optimization** |

---

## Results (CIFAR-10)

| Method | Top-1 | Top-5 | ECE | NLL | Brier | Params | FLOPs |
|---|---|---|---|---|---|---|---|
| Teacher (ResNet-110) | 94.35% | 99.79% | 0.0394 | 0.2883 | 0.0964 | 1.73M | 506.3M |
| Student (no KD) | 92.79% | 99.77% | 0.0344 | 0.2586 | 0.1126 | 0.27M | 81.6M |
| KL-KD | 43.39% | 92.21% | 0.4472 | 4.0661 | 0.9766 | 0.27M | 81.6M |
| Fixed-OT-KD | 74.04% | 98.23% | 0.0686 | 0.8017 | 0.3693 | 0.27M | 81.6M |
| **Adaptive-OT-KD (Ours)** | **92.98%** | **99.85%** | **0.0348** | **0.2618** | **0.1128** | 0.27M | 81.6M |

Adaptive-OT-KD achieves the best Top-1 and Top-5 while matching the student baseline on all calibration metrics at a **6.4× parameter reduction** and **6.2× FLOPs reduction** over the teacher.

---

## References

- Hinton et al. "Distilling the Knowledge in a Neural Network" (NeurIPS 2015)
- Cuturi. "Sinkhorn Distances: Lightspeed Computation of Optimal Transport" (NeurIPS 2013)
- He et al. "Deep Residual Learning for Image Recognition" (CVPR 2016)
- Sandler et al. "MobileNetV2: Inverted Residuals and Linear Bottlenecks" (CVPR 2018)
- SinKD (COLING 2024): Sinkhorn KD for NLP tasks
- MultiLevelOT (AAAI 2025): Multi-level OT for LLM distillation
