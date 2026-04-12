import os
import time
from typing import Any

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.dataset import TSPDataset, collate_tsp
from src.model import DifuscoTSP
from src.train import evaluate, train


# uv run python -m src.main data_path=data/tsp50.txt
# uv run python -m src.main data_path=data/tsp50.txt model=small wandb.mode=disabled
@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Data
    dataset = TSPDataset(
        cfg.data_path,
        num_nodes=cfg.data.num_nodes,
        sparse_factor=cfg.data.sparse_factor,
    )
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train

    torch.manual_seed(42)
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=collate_tsp,
        num_workers=cfg.training.num_workers,
        persistent_workers=cfg.training.num_workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_tsp,
        num_workers=0,
    )

    # Model
    model = DifuscoTSP(
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        T=cfg.diffusion.T,
        beta_start=cfg.diffusion.beta_start,
        beta_end=cfg.diffusion.beta_end,
        diffusion_type=cfg.diffusion.diffusion_type,
        dropout=cfg.training.dropout,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: DIFUSCO ({cfg.diffusion.diffusion_type} diffusion)")
    print(f"  Layers: {cfg.model.num_layers}, Hidden: {cfg.model.hidden_dim}")
    print(f"  Parameters: {num_params:,}")
    print(f"  T={cfg.diffusion.T}, β=[{cfg.diffusion.beta_start}, {cfg.diffusion.beta_end}]")
    print(
        f"  Inference: {cfg.inference.inference_steps} steps, {cfg.inference.schedule} schedule"
    )

    # wandb
    run_name = cfg.wandb.run_name or (
        f"tsp{cfg.data.num_nodes}_{cfg.diffusion.diffusion_type}"
        f"_h{cfg.model.hidden_dim}_L{cfg.model.num_layers}"
    )
    wandb_config: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    wandb.init(
        project=cfg.wandb.project,
        name=run_name,
        mode=cfg.wandb.mode,
        config=wandb_config,
    )
    wandb.watch(model, log="gradients", log_freq=500)

    # Optimizer (paper: AdamW with cosine annealing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.epochs
    )

    # Training loop
    print(f"\n{'=' * 60}")
    print(f"  Training for {cfg.training.epochs} epochs")
    print(f"{'=' * 60}")

    # Hydra changes cwd to outputs/<date>/<time>/, so checkpoints land there
    ckpt_dir = os.getcwd()
    print(f"  Checkpoints: {ckpt_dir}")

    best_gap = float("inf")
    for epoch in range(1, cfg.training.epochs + 1):
        t0 = time.time()

        train_loss = train(model, train_loader, optimizer, device, epoch)
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        wandb.log(
            {
                "train/loss_epoch": train_loss,
                "train/lr": current_lr,
                "epoch": epoch,
            },
        )

        # Evaluate every 10 epochs
        if epoch % 10 == 0 or epoch == 1 or epoch == cfg.training.epochs:
            pred_len, gt_len, gap = evaluate(
                model,
                val_loader,
                device,
                num_inference_steps=cfg.inference.inference_steps,
                schedule_type=cfg.inference.schedule,
                use_2opt=cfg.inference.use_2opt,
                max_instances=cfg.inference.eval_subset,
            )
            elapsed = time.time() - t0

            wandb.log(
                {
                    "val/pred_tour_length": pred_len,
                    "val/gt_tour_length": gt_len,
                    "val/gap_pct": gap,
                    "val/best_gap_pct": best_gap,
                    "epoch": epoch,
                },
            )

            # Save best model
            if gap < best_gap:
                best_gap = gap
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_gap": best_gap,
                        "config": OmegaConf.to_container(cfg, resolve=True),
                    },
                    os.path.join(ckpt_dir, "best_model.pt"),
                )
                print(
                    f"  Epoch {epoch:3d} | Loss: {train_loss:.4f} | "
                    f"Tour: {pred_len:.3f} (GT: {gt_len:.3f}, Gap: {gap:.2f}%) | "
                    f"{elapsed:.1f}s  [saved best]"
                )
            else:
                print(
                    f"  Epoch {epoch:3d} | Loss: {train_loss:.4f} | "
                    f"Tour: {pred_len:.3f} (GT: {gt_len:.3f}, Gap: {gap:.2f}%) | "
                    f"{elapsed:.1f}s"
                )
        else:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:3d} | Loss: {train_loss:.4f} | {elapsed:.1f}s")

    # Save final model
    torch.save(
        {
            "epoch": cfg.training.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_gap": best_gap,
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        os.path.join(ckpt_dir, "last_model.pt"),
    )

    print(f"\n{'=' * 60}")
    print(f"  Best Gap: {best_gap:.2f}%")
    print(f"  Checkpoints saved to: {ckpt_dir}")
    print(f"{'=' * 60}")

    wandb.log({"val/final_best_gap_pct": best_gap})
    wandb.finish()


if __name__ == "__main__":
    main()
