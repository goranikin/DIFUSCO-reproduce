"""
TSP Dataset for DIFUSCO
=======================
Parses the standard TSP format and builds graph representations.

Supports both dense (complete) and sparse (KNN) graphs.
Dense is used for TSP-50/100, sparse for TSP-500/1000.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class TSPDataset(Dataset):
    """
    Parses TSP data in the format:
        x0 y0 x1 y1 ... xN yN output t1 t2 t3 ... tN t1

    Args:
        file_path: path to the data file
        num_nodes: number of nodes per instance
        sparse_factor: if > 0, build a KNN sparse graph with this many neighbors.
                       if <= 0 (default), build a complete (dense) graph.
    """

    def __init__(self, file_path: str, num_nodes: int = 50, sparse_factor: int = -1):
        self.num_nodes = num_nodes
        self.sparse_factor = sparse_factor
        self.file_lines = open(file_path).read().splitlines()
        self.file_lines = [line for line in self.file_lines if line.strip()]
        print(f"Loaded {len(self.file_lines)} TSP-{num_nodes} instances")

        # Pre-compute edge_index for dense graphs (same topology for every instance)
        if sparse_factor <= 0:
            src, dst = [], []
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    src.extend([i, j])
                    dst.extend([j, i])
            self._edge_index = torch.tensor([src, dst], dtype=torch.long)
        else:
            self._edge_index = None

    def __len__(self):
        return len(self.file_lines)

    def _parse_line(self, line: str):
        parts = line.strip().split(" output ")
        coord_str = parts[0].strip().split()
        tour_str = parts[1].strip().split()

        coords = np.array([float(c) for c in coord_str]).reshape(-1, 2)
        assert coords.shape[0] == self.num_nodes

        tour = np.array([int(t) - 1 for t in tour_str])
        return coords, tour

    def __getitem__(self, idx):
        coords, tour = self._parse_line(self.file_lines[idx])
        N = self.num_nodes

        if self.sparse_factor > 0:
            return self._build_sparse_graph(coords, tour, N)
        else:
            return self._build_dense_graph(coords, tour, N)

    def _build_dense_graph(self, coords, tour, N):
        """Complete graph with N*(N-1) directed edges."""
        tour_edges = set()
        for i in range(len(tour) - 1):
            u, v = tour[i], tour[i + 1]
            tour_edges.add((min(u, v), max(u, v)))

        assert self._edge_index is not None
        edge_index = self._edge_index
        node_feat = torch.from_numpy(coords).float()

        src_coords = coords[edge_index[0].numpy()]
        dst_coords = coords[edge_index[1].numpy()]
        distances = np.sqrt(((src_coords - dst_coords) ** 2).sum(axis=1))
        edge_dist = torch.tensor(distances, dtype=torch.float32)

        labels = []
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            key = (min(u, v), max(u, v))
            labels.append(1.0 if key in tour_edges else 0.0)
        edge_label = torch.tensor(labels, dtype=torch.float32)

        return node_feat, edge_index, edge_dist, edge_label

    def _build_sparse_graph(self, coords, tour, N):
        """KNN sparse graph where each node connects to its k nearest neighbors."""
        from sklearn.neighbors import KDTree

        k = self.sparse_factor
        kdt = KDTree(coords, leaf_size=30, metric="euclidean")
        dists_knn, idx_knn = kdt.query(coords, k=k, return_distance=True)

        src = (
            torch.arange(N).reshape(-1, 1).repeat(1, k).reshape(-1)
        )
        dst = torch.from_numpy(idx_knn.reshape(-1)).long()
        edge_index = torch.stack([src, dst], dim=0)

        node_feat = torch.from_numpy(coords).float()
        edge_dist = torch.from_numpy(dists_knn.reshape(-1)).float()

        tour_edges = set()
        for i in range(len(tour) - 1):
            u, v = tour[i], tour[i + 1]
            tour_edges.add((min(u, v), max(u, v)))

        labels = []
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            key = (min(u, v), max(u, v))
            labels.append(1.0 if key in tour_edges else 0.0)
        edge_label = torch.tensor(labels, dtype=torch.float32)

        return node_feat, edge_index, edge_dist, edge_label


def collate_tsp(batch):
    """
    Collate B individual TSP graphs into a single super-graph.

    Each graph's node indices are offset so the super-graph contains
    B disconnected components. The GNN's message passing naturally
    respects these components since there are no cross-graph edges.

    Input:  list of (node_feat, edge_index, edge_dist, edge_label)
    Output: (node_feat, edge_index, edge_dist, edge_label) for the super-graph
    """
    all_node_feat = []
    all_edge_index = []
    all_edge_dist = []
    all_edge_label = []

    node_offset = 0
    for node_feat, edge_index, edge_dist, edge_label in batch:
        all_node_feat.append(node_feat)
        all_edge_index.append(edge_index + node_offset)
        all_edge_dist.append(edge_dist)
        all_edge_label.append(edge_label)
        node_offset += node_feat.shape[0]

    return (
        torch.cat(all_node_feat, dim=0),
        torch.cat(all_edge_index, dim=1),
        torch.cat(all_edge_dist, dim=0),
        torch.cat(all_edge_label, dim=0),
    )
