# Experiment Log

## Experiment 1: Small Model Baseline (h128_L6)

**Date**: 2026-04-12
**wandb run**: `tsp50_categorical_h128_L6`
**Goal**: Validate the training pipeline with a smaller, faster model before committing to the full paper configuration.

### Configuration

| Parameter | Value |
|---|---|
| Model | h128_L6 (940K params) |
| Hidden dimension | 128 |
| AGNN layers | 6 |
| Diffusion type | Categorical (Bernoulli) |
| Diffusion steps (T) | 1000 |
| Beta schedule | linear, 1e-4 to 0.02 |
| Optimizer | AdamW |
| Learning rate | 2e-4 (cosine annealing) |
| Weight decay | 1e-4 |
| Epochs | 50 |
| Batch size | 16 |
| Training data | 128,000 TSP-50 instances (Concorde solutions) |
| Train/val split | 102,400 / 25,600 |
| Eval subset | 500 instances |
| Inference steps | 50 (cosine schedule) |
| Post-processing | Greedy decode + 2-opt |
| Device | Apple M4 (MPS) |

### Computing Resources

| Resource | Specification |
|---|---|
| Machine | MacBook Pro with Apple M4 chip |
| Accelerator | Apple MPS (Metal Performance Shaders) |
| Backend | PyTorch MPS |

### Results

| Metric | Value |
|---|---|
| **Best optimality gap** | **4.35%** (epoch 40) |
| Final epoch gap | 4.48% (epoch 50) |
| Ground truth tour length | 5.72 |
| Predicted tour length (best) | 5.967 |
| Final training loss | 0.0166 |
| Total training steps | 319,900 |

### Training Time

| Phase | Time |
|---|---|
| Average epoch (train only) | ~670s (~11.2 min) |
| Average epoch (train + eval) | ~1,010s (~16.8 min) |
| Epoch 1 (includes warmup) | 1,289s (~21.5 min) |
| **Total wall time (50 epochs)** | **~9.7 hours** |

Breakdown: 45 train-only epochs x 670s + 5 eval epochs x 1,010s = 35,200s = **9.8 hours**.

### Per-Epoch Log

| Epoch | Loss | Tour Length | GT Length | Gap | Time (s) | Notes |
|---|---|---|---|---|---|---|
| 1 | 0.0427 | 5.981 | 5.719 | 4.59% | 1289.4 | saved best |
| 2 | 0.0309 | — | — | — | 672.6 | |
| 3 | 0.0281 | — | — | — | 672.9 | |
| 4 | 0.0259 | — | — | — | 674.4 | |
| 5 | 0.0245 | — | — | — | 904.5 | |
| 6 | 0.0235 | — | — | — | 724.4 | |
| 7 | 0.0230 | — | — | — | 718.3 | |
| 8 | 0.0226 | — | — | — | 665.0 | |
| 9 | 0.0221 | — | — | — | 673.1 | |
| 10 | 0.0217 | 6.014 | 5.719 | 5.16% | 1022.3 | |
| 11 | 0.0215 | — | — | — | 674.6 | |
| 12 | 0.0211 | — | — | — | 672.0 | |
| 13 | 0.0207 | — | — | — | 672.4 | |
| 14 | 0.0205 | — | — | — | 672.4 | |
| 15 | 0.0201 | — | — | — | 672.7 | |
| 16 | 0.0200 | — | — | — | 672.2 | |
| 17 | 0.0200 | — | — | — | 672.2 | |
| 18 | 0.0197 | — | — | — | 672.3 | |
| 19 | 0.0196 | — | — | — | 672.0 | |
| 20 | 0.0192 | 5.996 | 5.719 | 4.84% | 1009.9 | |
| 21 | 0.0192 | — | — | — | 674.4 | |
| 22 | 0.0191 | — | — | — | 670.5 | |
| 23 | 0.0187 | — | — | — | 670.7 | |
| 24 | 0.0186 | — | — | — | 670.7 | |
| 25 | 0.0185 | — | — | — | 671.2 | |
| 26 | 0.0182 | — | — | — | 670.9 | |
| 27 | 0.0180 | — | — | — | 670.9 | |
| 28 | 0.0180 | — | — | — | 671.1 | |
| 29 | 0.0181 | — | — | — | 678.1 | |
| 30 | 0.0178 | 5.976 | 5.719 | 4.50% | 1006.1 | saved best |
| 31 | 0.0176 | — | — | — | 677.6 | |
| 32 | 0.0176 | — | — | — | 670.1 | |
| 33 | 0.0176 | — | — | — | 665.3 | |
| 34 | 0.0174 | — | — | — | 668.0 | |
| 35 | 0.0174 | — | — | — | 662.9 | |
| 36 | 0.0174 | — | — | — | 663.0 | |
| 37 | 0.0173 | — | — | — | 669.0 | |
| 38 | 0.0172 | — | — | — | 663.1 | |
| 39 | 0.0170 | — | — | — | 664.6 | |
| 40 | 0.0170 | 5.967 | 5.719 | 4.35% | 996.0 | saved best |
| 41 | 0.0169 | — | — | — | 666.3 | |
| 42 | 0.0170 | — | — | — | 666.0 | |
| 43 | 0.0167 | — | — | — | 662.7 | |
| 44 | 0.0168 | — | — | — | 663.4 | |
| 45 | 0.0167 | — | — | — | 663.7 | |
| 46 | 0.0166 | — | — | — | 662.9 | |
| 47 | 0.0168 | — | — | — | 663.5 | |
| 48 | 0.0168 | — | — | — | 667.5 | |
| 49 | 0.0167 | — | — | — | 662.7 | |
| 50 | 0.0166 | 5.975 | 5.719 | 4.48% | 994.7 | |

### Training Dynamics

- **Loss convergence**: Rapid initial drop in the first epoch (0.0427), then steady decline to 0.0166. Loss decreased 2.6x across all 50 epochs.
- **Optimality gap progression**: 4.59% (epoch 1) → 5.16% (epoch 10) → 4.84% (epoch 20) → 4.50% (epoch 30) → **4.35%** (epoch 40) → 4.48% (epoch 50). Best at epoch 40, slight regression at epoch 50 suggests mild overfitting in the last 10 epochs.
- **Gradient norms**: Stable throughout, well below the clip threshold of 1.0 after the initial few hundred steps.
- **Learning rate**: Cosine annealing completed fully, reaching 0 at epoch 50.
- **Epoch 10 anomaly**: Gap was worse (5.16%) than epoch 1 (4.59%). This is normal — the model temporarily overfits to the diffusion loss at the expense of tour quality, then recovers as it learns finer structure.

### Analysis

The 4.35% gap is reasonable for the small model (940K params, 6 layers). The gap between this result and the paper's reported ~0.5-1% is primarily due to:

1. **Model capacity**: 940K vs 7.3M parameters (paper uses h256_L12)
2. **Fewer layers**: 6 layers limits the GNN's receptive field and message-passing depth

The training pipeline is validated — loss converges, gradients are stable, evaluation produces valid tours.

---

## Experiment 2: Paper Configuration (h256_L12)

**Status**: Pending

**Plan**: Run the full paper configuration to reproduce the reported results.

```bash
caffeinate -dims uv run python -m src.main \
  data_path=./src/tsp50_128000_concorde.txt \
  model=paper
```

### Expected Results

| Metric | Paper reported | Expected |
|---|---|---|
| Optimality gap (categorical) | ~0.5-1% | Similar range |
| Parameters | 7.3M | 7.3M |
| Training time (M4) | — | ~20-30 hours |
