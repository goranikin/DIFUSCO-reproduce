import torch


def train(
    model,
    train_loader,
    optimizer,
    device,
):
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
