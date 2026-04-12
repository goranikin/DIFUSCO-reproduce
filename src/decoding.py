"""
TSP tour decoding utilities.
Greedy heatmap decoding + 2-opt local search (Section 3.5 of the paper).
"""

import numpy as np
import torch


def greedy_decode_tsp(
    heatmap: torch.Tensor, edge_index: torch.Tensor, node_coords: torch.Tensor
) -> list:
    """
    Greedy decoding from heatmap scores (Section 3.5).

    All edges are ranked by (A_ij + A_ji) / ||c_i - c_j|| and inserted
    into the partial solution if there are no conflicts.

    A valid TSP tour requires:
    - Each node has exactly degree 2
    - No subtours (must be a single Hamiltonian cycle)
    """
    N = node_coords.shape[0]
    E = edge_index.shape[1]

    edge_scores = {}
    for k in range(E):
        u, v = edge_index[0, k].item(), edge_index[1, k].item()
        key = (min(u, v), max(u, v))
        score = heatmap[k].item()
        if key in edge_scores:
            edge_scores[key] += score
        else:
            edge_scores[key] = score

    for u, v in edge_scores:
        dist = torch.norm(node_coords[u] - node_coords[v]).item()
        edge_scores[(u, v)] /= dist + 1e-8

    sorted_edges = sorted(edge_scores.items(), key=lambda x: x[1], reverse=True)

    adj = {i: [] for i in range(N)}
    selected_edges = []

    for (u, v), score in sorted_edges:
        if len(adj[u]) >= 2 or len(adj[v]) >= 2:
            continue

        if len(adj[u]) == 1 and len(adj[v]) == 1:
            if _would_create_subtour(adj, u, v, N):
                continue

        adj[u].append(v)
        adj[v].append(u)
        selected_edges.append((u, v))

        if len(selected_edges) == N:
            break

    if len(selected_edges) < N:
        endpoints = [i for i in range(N) if len(adj[i]) < 2]
        while len(selected_edges) < N and len(endpoints) >= 2:
            u = endpoints[0]
            best_v = None
            best_dist = float("inf")
            for v in endpoints[1:]:
                if len(adj[u]) < 2 and len(adj[v]) < 2:
                    dist = torch.norm(node_coords[u] - node_coords[v]).item()
                    if dist < best_dist and not _would_create_subtour(adj, u, v, N):
                        best_dist = dist
                        best_v = v
            if best_v is None:
                for v in endpoints[1:]:
                    if len(adj[u]) < 2 and len(adj[v]) < 2:
                        best_v = v
                        break
            if best_v is not None:
                adj[u].append(best_v)
                adj[best_v].append(u)
                selected_edges.append((u, best_v))
            endpoints = [i for i in range(N) if len(adj[i]) < 2]

    tour = _extract_tour(adj, N)
    return tour


def _would_create_subtour(adj, u, v, N):
    """Check if adding edge (u, v) would create a cycle shorter than N."""
    if not adj[u] or not adj[v]:
        return False

    path_length = 1
    current = u
    prev = v
    while True:
        neighbors = [n for n in adj[current] if n != prev]
        if not neighbors:
            break
        prev = current
        current = neighbors[0]
        path_length += 1
        if current == v:
            return path_length < N

    return False


def _extract_tour(adj, N):
    """Extract a tour from the adjacency list."""
    if not all(len(adj[i]) == 2 for i in range(N)):
        visited = [False] * N
        tour = [0]
        visited[0] = True
        current = 0
        while len(tour) < N:
            neighbors = [n for n in adj[current] if not visited[n]]
            if neighbors:
                current = neighbors[0]
            else:
                unvisited = [i for i in range(N) if not visited[i]]
                if unvisited:
                    current = unvisited[0]
                else:
                    break
            visited[current] = True
            tour.append(current)
        return tour

    tour = [0]
    prev = -1
    current = 0
    for _ in range(N - 1):
        neighbors = [n for n in adj[current] if n != prev]
        prev = current
        current = neighbors[0]
        tour.append(current)
    return tour


def two_opt(tour: list, node_coords: torch.Tensor, max_iterations: int = 100) -> list:
    """2-opt local search to improve tour quality."""
    coords = node_coords.cpu().numpy()
    tour = list(tour)
    N = len(tour)
    improved = True
    iteration = 0

    def tour_dist(t):
        d = 0
        for i in range(len(t)):
            d += np.sqrt(((coords[t[i]] - coords[t[(i + 1) % len(t)]]) ** 2).sum())
        return d

    best_distance = tour_dist(tour)

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        for i in range(1, N - 1):
            for j in range(i + 1, N):
                new_tour = tour[:i] + tour[i : j + 1][::-1] + tour[j + 1 :]
                new_distance = tour_dist(new_tour)
                if new_distance < best_distance - 1e-10:
                    tour = new_tour
                    best_distance = new_distance
                    improved = True
                    break
            if improved:
                break

    return tour


def compute_tour_length(tour: list, node_coords: torch.Tensor) -> float:
    """Compute total Euclidean distance of a tour."""
    total = 0.0
    coords = node_coords.cpu()
    for i in range(len(tour)):
        u, v = tour[i], tour[(i + 1) % len(tour)]
        total += torch.norm(coords[u] - coords[v]).item()
    return total
