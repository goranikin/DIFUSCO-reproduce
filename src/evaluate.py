"""
Evaluate a trained DIFUSCO checkpoint on a TSP test set.

Loads model hyperparameters from the checkpoint's saved config (so you don't
have to remember them), runs diffusion inference + greedy decoding (+ optional
2-opt), and reports the average tour length, ground-truth length, and gap.

Examples
--------
    uv run python -m src.evaluate \
        --checkpoint best_model.pt \
        --data data/tsp-50-1280-concorde.txt

    uv run python -m src.evaluate \
        --checkpoint best_model.pt \
        --data data/tsp-50-1280-concorde.txt \
        --inference-steps 50 --schedule cosine --no-2opt --max-instances 128
"""

import argparse
import time
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.dataset import TSPDataset, collate_tsp
from src.model import DifuscoTSP
from src.train import evaluate


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_model_from_ckpt(
    ckpt: dict[str, Any], overrides: dict[str, Any]
) -> DifuscoTSP:
    """
    Reconstruct DifuscoTSP using the config saved inside the checkpoint, with
    optional manual overrides. Falls back to paper defaults if no config was
    saved (older checkpoints).
    """
    cfg = ckpt.get("config") or {}
    model_cfg = cfg.get("model", {})
    diff_cfg = cfg.get("diffusion", {})
    train_cfg = cfg.get("training", {})

    kwargs = {
        "hidden_dim": model_cfg.get("hidden_dim", 256),
        "num_layers": model_cfg.get("num_layers", 12),
        "T": diff_cfg.get("T", 1000),
        "beta_start": diff_cfg.get("beta_start", 1e-4),
        "beta_end": diff_cfg.get("beta_end", 0.02),
        "diffusion_type": diff_cfg.get("diffusion_type", "categorical"),
        "dropout": train_cfg.get("dropout", 0.0),
    }
    kwargs.update({k: v for k, v in overrides.items() if v is not None})

    print("Model config:")
    for k, v in kwargs.items():
        print(f"  {k}: {v}")
    return DifuscoTSP(**kwargs)


def _infer_num_nodes(ckpt: dict[str, Any], cli_value: int | None) -> int:
    if cli_value is not None:
        return cli_value
    cfg = ckpt.get("config") or {}
    return cfg.get("data", {}).get("num_nodes", 50)


def _infer_sparse_factor(ckpt: dict[str, Any], cli_value: int | None) -> int:
    if cli_value is not None:
        return cli_value
    cfg = ckpt.get("config") or {}
    return cfg.get("data", {}).get("sparse_factor", -1)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DIFUSCO TSP model on a test set."
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        default="best_model.pt",
        help="Path to model checkpoint (.pt). Defaults to ./best_model.pt.",
    )
    parser.add_argument(
        "--data",
        "-d",
        default="data/tsp-50-1280-concorde.txt",
        help="Path to TSP test file (Concorde format).",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=None,
        help="TSP problem size. Inferred from checkpoint config if omitted.",
    )
    parser.add_argument(
        "--sparse-factor",
        type=int,
        default=None,
        help="KNN sparse factor (-1 = dense). Inferred from checkpoint if omitted.",
    )
    parser.add_argument(
        "--inference-steps",
        type=int,
        default=50,
        help="Number of denoising steps at inference time.",
    )
    parser.add_argument(
        "--schedule",
        choices=["linear", "cosine"],
        default="cosine",
        help="Inference timestep schedule.",
    )
    parser.add_argument(
        "--no-2opt",
        dest="use_2opt",
        action="store_false",
        help="Disable 2-opt local search post-processing.",
    )
    parser.set_defaults(use_2opt=True)
    parser.add_argument(
        "--max-instances",
        type=int,
        default=-1,
        help="Evaluate on at most this many instances (-1 = all).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducible diffusion sampling.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = _select_device()
    print(f"Device: {device}")

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    num_nodes = _infer_num_nodes(ckpt, args.num_nodes)
    sparse_factor = _infer_sparse_factor(ckpt, args.sparse_factor)
    print(
        f"Test set: {args.data}  (num_nodes={num_nodes}, sparse_factor={sparse_factor})"
    )

    dataset = TSPDataset(args.data, num_nodes=100, sparse_factor=sparse_factor)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_tsp,
        num_workers=args.num_workers,
    )

    model = _build_model_from_ckpt(ckpt, overrides={})
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [warn] missing keys: {missing}")
    if unexpected:
        print(f"  [warn] unexpected keys: {unexpected}")
    model = model.to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")
    if "epoch" in ckpt:
        print(f"  Trained epoch: {ckpt['epoch']}")
    if "best_gap" in ckpt:
        print(f"  Reported best gap (val): {ckpt['best_gap']:.4f}%")

    print(
        f"\nInference: steps={args.inference_steps}, "
        f"schedule={args.schedule}, 2-opt={args.use_2opt}"
    )

    t0 = time.time()
    pred_len, gt_len, gap = evaluate(
        model,
        loader,
        device,
        num_inference_steps=args.inference_steps,
        schedule_type=args.schedule,
        use_2opt=args.use_2opt,
        max_instances=args.max_instances,
    )
    elapsed = time.time() - t0

    n_eval = (
        min(args.max_instances, len(dataset))
        if args.max_instances > 0
        else len(dataset)
    )

    print("\n" + "=" * 60)
    print("  Test Results")
    print("=" * 60)
    print(f"  Instances evaluated : {n_eval}")
    print(f"  Predicted tour len  : {pred_len:.4f}")
    print(f"  Ground-truth len    : {gt_len:.4f}")
    print(f"  Optimality gap      : {gap:.4f}%")
    print(
        f"  Wall time           : {elapsed:.1f}s ({elapsed / max(n_eval, 1):.3f}s/instance)"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
