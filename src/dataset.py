"""
TSP Dataset for DIFUSCO
=======================
Parses the standard TSP format and builds complete graph representations.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class TSPDataset(Dataset):
    """
    Parses TSP data in the format:
        x0 y0 x1 y1 ... xN yN output t1 t2 t3 ... tN t1

    Returns graph-level data for diffusion training.
    """

    def __init__(self, file_path: str, num_nodes: int = 50):
        self.num_nodes = num_nodes
        self.instances = []

        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split("output")
                coord_str = parts[0].strip().split()
                tour_str = parts[1].strip().split()

                coords = np.array([float(c) for c in coord_str]).reshape(-1, 2)
                assert coords.shape[0] == num_nodes

                # Parse tour (1-indexed → 0-indexed)
                tour = [int(t) - 1 for t in tour_str]

                # Build tour edge set
                tour_edges = set()
                for idx in range(len(tour) - 1):
                    u, v = tour[idx], tour[idx + 1]
                    tour_edges.add((min(u, v), max(u, v)))
                if len(tour) > 1:
                    u, v = tour[-1], tour[0]
                    tour_edges.add((min(u, v), max(u, v)))

                self.instances.append((coords, tour_edges))

        print(f"Loaded {len(self.instances)} TSP-{num_nodes} instances")

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        coords, tour_edges = self.instances[idx]
        N = self.num_nodes

        # Build complete graph (both directions)
        src, dst = [], []
        for i in range(N):
            for j in range(i + 1, N):
                src.extend([i, j])
                dst.extend([j, i])
        edge_index = torch.tensor([src, dst], dtype=torch.long)

        # Node features: (x, y) coordinates
        node_feat = torch.tensor(coords, dtype=torch.float32)

        # Edge distances
        src_coords = coords[edge_index[0].numpy()]
        dst_coords = coords[edge_index[1].numpy()]
        distances = np.sqrt(((src_coords - dst_coords) ** 2).sum(axis=1))
        edge_dist = torch.tensor(distances, dtype=torch.float32)

        # Edge labels: 1 if in tour, 0 otherwise
        labels = []
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            key = (min(u, v), max(u, v))
            labels.append(1.0 if key in tour_edges else 0.0)
        edge_label = torch.tensor(labels, dtype=torch.float32)

        return node_feat, edge_index, edge_dist, edge_label
