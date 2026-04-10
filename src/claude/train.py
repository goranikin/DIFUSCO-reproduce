"""
DIFUSCO Training & Evaluation Script
=====================================
Main entry point for training and evaluating DIFUSCO on TSP.

Usage:
  # Train with categorical (discrete) diffusion (recommended by paper)
  python -m difusco.train --data_path data/tsp50.txt --num_nodes 50 \
      --diffusion_type categorical --hidden_dim 128 --num_layers 6 --epochs 100

  # Train with Gaussian (continuous) diffusion
  python -m difusco.train --data_path data/tsp50.txt --diffusion_type gaussian

  # Quick test with small model
  python -m difusco.train --data_path data/tsp50.txt --hidden_dim 64 \
      --num_layers 4 --epochs 20

Paper settings (Section 4.1):
  - T = 1000, β_1 = 1e-4, β_T = 0.02
  - 12-layer AGNN, hidden_dim = 256
  - Discrete (categorical) diffusion with cosine inference schedule
  - Greedy decoding + 2-opt for final tour
"""

import argparse
import time

import torch
from torch.utils.data import DataLoader

from .dataset import TSPDataset
from .tsp_model import DifuscoTSP, compute_tour_length, greedy_decode_tsp, two_opt


def train(model, train_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in train_loader:
        optimizer.zero_grad()
        loss = model.training_step(batch, device)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(
    model,
    val_loader,
    device,
    num_inference_steps=10,
    schedule_type="cosine",
    use_2opt=True,
):
    """
    Evaluate by generating tours and comparing to ground truth.
    """
    model.eval()
    total_pred_length = 0.0
    total_gt_length = 0.0
    num_instances = 0

    for batch in val_loader:
        node_feat, edge_index, edge_dist, edge_label = batch
        node_feat = node_feat.squeeze(0).to(device)
        edge_index = edge_index.squeeze(0).to(device)
        edge_dist = edge_dist.squeeze(0).to(device)
        edge_label = edge_label.squeeze(0).to(device)

        # Generate heatmap via diffusion
        heatmap = model.generate(
            node_feat,
            edge_index,
            edge_dist,
            num_inference_steps=num_inference_steps,
            schedule_type=schedule_type,
            device=device,
        )

        # Decode tour from heatmap
        tour = greedy_decode_tsp(heatmap, edge_index, node_feat)

        # Optionally refine with 2-opt
        if use_2opt:
            tour = two_opt(tour, node_feat, max_iterations=100)

        # Compute predicted tour length
        pred_length = compute_tour_length(tour, node_feat)

        # Compute ground-truth tour length from edge labels
        gt_tour_edges = edge_label.nonzero(as_tuple=True)[0]
        gt_length = (
            edge_dist[gt_tour_edges].sum().item() / 2.0
        )  # divide by 2 because both directions

        total_pred_length += pred_length
        total_gt_length += gt_length
        num_instances += 1

    avg_pred = total_pred_length / max(num_instances, 1)
    avg_gt = total_gt_length / max(num_instances, 1)
    gap = (avg_pred - avg_gt) / avg_gt * 100  # percentage gap

    return avg_pred, avg_gt, gap


def main():
    parser = argparse.ArgumentParser(description="DIFUSCO for TSP")
    # Data
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--num_nodes", type=int, default=50)
    # Model
    parser.add_argument(
        "--hidden_dim", type=int, default=128, help="Hidden dimension (paper: 256)"
    )
    parser.add_argument(
        "--num_layers", type=int, default=6, help="Number of AGNN layers (paper: 12)"
    )
    parser.add_argument(
        "--diffusion_type",
        type=str,
        default="categorical",
        choices=["categorical", "gaussian"],
        help="Diffusion type (paper recommends categorical)",
    )
    # Diffusion
    parser.add_argument("--T", type=int, default=1000, help="Number of diffusion steps")
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=0.02)
    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate (paper uses AdamW)"
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.0)
    # Inference
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps at inference",
    )
    parser.add_argument(
        "--inference_schedule", type=str, default="cosine", choices=["linear", "cosine"]
    )
    parser.add_argument(
        "--no_2opt", action="store_true", help="Disable 2-opt refinement"
    )
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Data
    dataset = TSPDataset(args.data_path, num_nodes=args.num_nodes)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train

    torch.manual_seed(42)
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    # Model
    model = DifuscoTSP(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        T=args.T,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        diffusion_type=args.diffusion_type,
        dropout=args.dropout,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: DIFUSCO ({args.diffusion_type} diffusion)")
    print(f"  Layers: {args.num_layers}, Hidden: {args.hidden_dim}")
    print(f"  Parameters: {num_params:,}")
    print(f"  T={args.T}, β=[{args.beta_start}, {args.beta_end}]")
    print(
        f"  Inference: {args.inference_steps} steps, {args.inference_schedule} schedule"
    )

    # Optimizer (paper uses AdamW)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    print(f"\n{'=' * 60}")
    print(f"  Training for {args.epochs} epochs")
    print(f"{'=' * 60}")

    best_gap = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        train_loss = train(model, train_loader, optimizer, device, epoch)
        scheduler.step()

        # Evaluate every 10 epochs (evaluation is slow due to diffusion inference)
        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            pred_len, gt_len, gap = evaluate(
                model,
                val_loader,
                device,
                num_inference_steps=args.inference_steps,
                schedule_type=args.inference_schedule,
                use_2opt=not args.no_2opt,
            )
            elapsed = time.time() - t0

            if gap < best_gap:
                best_gap = gap

            print(
                f"  Epoch {epoch:3d} | Loss: {train_loss:.4f} | "
                f"Tour: {pred_len:.3f} (GT: {gt_len:.3f}, Gap: {gap:.2f}%) | "
                f"{elapsed:.1f}s"
            )
        else:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:3d} | Loss: {train_loss:.4f} | {elapsed:.1f}s")

    print(f"\n{'=' * 60}")
    print(f"  Best Gap: {best_gap:.2f}%")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
