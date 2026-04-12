import torch
import wandb
from tqdm import tqdm

from src.decoding import compute_tour_length, greedy_decode_tsp, two_opt


def train(
    model,
    train_loader,
    optimizer,
    device,
    epoch,
    log_interval=100,
):
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch:3d} [train]",
        leave=False,
        dynamic_ncols=True,
    )
    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()
        loss = model.training_step(batch, device)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()
        num_batches += 1

        avg_loss = total_loss / num_batches
        pbar.set_postfix(loss=f"{avg_loss:.4f}", grad=f"{grad_norm:.3f}")

        global_step = (epoch - 1) * len(train_loader) + batch_idx
        if batch_idx % log_interval == 0:
            wandb.log(
                {
                    "train/loss_step": loss.item(),
                    "train/grad_norm": grad_norm.item(),
                    "train/global_step": global_step,
                },
                step=global_step,
            )

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


@torch.no_grad()
def evaluate(
    model,
    val_loader,
    device,
    num_inference_steps=10,
    schedule_type="cosine",
    use_2opt=True,
    max_instances=-1,
):
    """
    Args:
        max_instances: evaluate on at most this many instances.
                       -1 means use all. 200-500 is enough for convergence monitoring.
    """
    model.eval()
    total_pred_length = 0.0
    total_gt_length = 0.0
    num_instances = 0

    total = min(max_instances, len(val_loader)) if max_instances > 0 else len(val_loader)
    pbar = tqdm(
        val_loader,
        desc="Evaluating",
        total=total,
        leave=False,
        dynamic_ncols=True,
    )
    for batch in pbar:
        node_feat, edge_index, edge_dist, edge_label = batch
        node_feat = node_feat.to(device)
        edge_index = edge_index.to(device)
        edge_dist = edge_dist.to(device)
        edge_label = edge_label.to(device)

        heatmap = model.generate(
            device=device,
            node_feat=node_feat,
            edge_index=edge_index,
            edge_dist=edge_dist,
            num_inference_steps=num_inference_steps,
            schedule_type=schedule_type,
        )

        tour = greedy_decode_tsp(heatmap, edge_index, node_feat)

        if use_2opt:
            tour = two_opt(tour, node_feat, max_iterations=100)

        pred_length = compute_tour_length(tour, node_feat)

        gt_tour_edges = edge_label.nonzero(as_tuple=True)[0]
        gt_length = edge_dist[gt_tour_edges].sum().item() / 2.0

        total_pred_length += pred_length
        total_gt_length += gt_length
        num_instances += 1

        avg_gap = (
            (total_pred_length / num_instances - total_gt_length / num_instances)
            / (total_gt_length / num_instances)
            * 100
        )
        pbar.set_postfix(gap=f"{avg_gap:.2f}%")

        if max_instances > 0 and num_instances >= max_instances:
            break

    avg_pred = total_pred_length / max(num_instances, 1)
    avg_gt = total_gt_length / max(num_instances, 1)
    gap = (avg_pred - avg_gt) / avg_gt * 100

    return avg_pred, avg_gt, gap
