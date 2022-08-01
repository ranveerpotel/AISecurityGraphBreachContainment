"""
3-layer GraphSAGE network for node embedding.

Architecture (from paper §4.3):
  Input: 64D (24D behavioral + 16D temporal + 16D structural + 8D node type)
  Layer 1: mean aggregation over 1-hop → 128D
  Layer 2: mean aggregation over 2-hop → 64D
  Output embeddings: 32D

Provides:
  - PyTorch Geometric (PyG) implementation when torch_geometric is available.
  - A pure-NumPy fallback (mean-pooling MLP approximation) for environments
    without GPU/PyG, so the rest of the pipeline is always runnable.
"""
from __future__ import annotations
import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PyTorch Geometric implementation
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv

    class GraphSAGEModel(nn.Module):
        """
        3-layer GraphSAGE with mean aggregation.
        Dropout applied between layers for regularisation.
        """

        def __init__(
            self,
            input_dim: int = 64,
            hidden_dim_1: int = 128,
            hidden_dim_2: int = 64,
            embedding_dim: int = 32,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            self.conv1 = SAGEConv(input_dim, hidden_dim_1, aggr="mean")
            self.conv2 = SAGEConv(hidden_dim_1, hidden_dim_2, aggr="mean")
            self.conv3 = SAGEConv(hidden_dim_2, embedding_dim, aggr="mean")
            self.dropout = dropout

        def forward(self, x: "torch.Tensor", edge_index: "torch.Tensor") -> "torch.Tensor":
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(self.conv2(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv3(x, edge_index)
            return x

    _TORCH_AVAILABLE = True
    logger.info("PyTorch Geometric detected — using full GNN implementation")

except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning(
        "torch_geometric not found — falling back to NumPy GraphSAGE approximation. "
        "Install with: pip install torch torch-geometric"
    )


# ---------------------------------------------------------------------------
# NumPy fallback: approximate GraphSAGE with mean-pooling + linear layers
# ---------------------------------------------------------------------------

class NumpyGraphSAGEApproximation:
    """
    Lightweight NumPy approximation of GraphSAGE for environments without PyG.

    Each layer: h_v = tanh(W * concat(h_v, mean(h_neighbours)))
    Weights are randomly initialized (untrained).  For production, use the
    PyG version and train on labelled data.
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim_1: int = 128,
        hidden_dim_2: int = 64,
        embedding_dim: int = 32,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        # Each SAGE layer concatenates self + neighbour mean → doubles width before projection
        self.W1 = rng.standard_normal((input_dim * 2, hidden_dim_1)).astype(np.float32) * 0.01
        self.W2 = rng.standard_normal((hidden_dim_1 * 2, hidden_dim_2)).astype(np.float32) * 0.01
        self.W3 = rng.standard_normal((hidden_dim_2 * 2, embedding_dim)).astype(np.float32) * 0.01
        self.b1 = np.zeros(hidden_dim_1, dtype=np.float32)
        self.b2 = np.zeros(hidden_dim_2, dtype=np.float32)
        self.b3 = np.zeros(embedding_dim, dtype=np.float32)

    def forward(
        self,
        node_features: np.ndarray,          # (N, input_dim)
        adjacency: List[List[int]],          # adjacency list: adj[i] = [neighbour indices]
    ) -> np.ndarray:                         # (N, embedding_dim)
        h = self._sage_layer(node_features, adjacency, self.W1, self.b1)
        h = self._sage_layer(h, adjacency, self.W2, self.b2)
        h = self._sage_layer(h, adjacency, self.W3, self.b3)
        return h

    @staticmethod
    def _sage_layer(
        h: np.ndarray,
        adj: List[List[int]],
        W: np.ndarray,
        b: np.ndarray,
    ) -> np.ndarray:
        N = h.shape[0]
        agg = np.zeros_like(h)
        for i in range(N):
            nbrs = adj[i]
            if nbrs:
                agg[i] = h[np.array(nbrs)].mean(axis=0)
            else:
                agg[i] = h[i]
        combined = np.concatenate([h, agg], axis=1)  # (N, 2*dim)
        out = np.tanh(combined @ W + b)
        return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Unified wrapper
# ---------------------------------------------------------------------------

class GNNInferenceEngine:
    """
    Wraps either the PyG or NumPy model behind a single interface.
    Call `embed(node_features, adjacency)` to get (N, 32) embeddings.
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim_1: int = 128,
        hidden_dim_2: int = 64,
        embedding_dim: int = 32,
        dropout: float = 0.2,
        use_gpu: bool = False,
        model_path: Optional[str] = None,
    ) -> None:
        self._embedding_dim = embedding_dim
        self._use_pyg = _TORCH_AVAILABLE

        if self._use_pyg:
            import torch
            self._device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
            self._model = GraphSAGEModel(input_dim, hidden_dim_1, hidden_dim_2, embedding_dim, dropout)
            if model_path:
                self._model.load_state_dict(torch.load(model_path, map_location=self._device))
                logger.info("Loaded GNN weights from %s", model_path)
            self._model.to(self._device)
            self._model.eval()
        else:
            self._model = NumpyGraphSAGEApproximation(input_dim, hidden_dim_1, hidden_dim_2, embedding_dim)

    def embed(
        self,
        node_features: np.ndarray,
        adjacency_list: List[List[int]],
    ) -> np.ndarray:
        """
        Returns (N, embedding_dim) float32 array.

        Args:
            node_features: (N, input_dim) float32 node feature matrix.
            adjacency_list: list of length N; adj[i] = list of neighbour indices.
        """
        if self._use_pyg:
            return self._embed_pyg(node_features, adjacency_list)
        return self._model.forward(node_features, adjacency_list)

    def _embed_pyg(self, features: np.ndarray, adj: List[List[int]]) -> np.ndarray:
        import torch
        x = torch.tensor(features, dtype=torch.float32, device=self._device)
        # Build COO edge_index from adjacency list
        src_list, dst_list = [], []
        for i, nbrs in enumerate(adj):
            for j in nbrs:
                src_list.append(i)
                dst_list.append(j)
        if src_list:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long, device=self._device)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=self._device)
        with torch.no_grad():
            out = self._model(x, edge_index)
        return out.cpu().numpy()

    def save(self, path: str) -> None:
        if self._use_pyg:
            import torch
            torch.save(self._model.state_dict(), path)
        else:
            np.save(path + ".npy", np.array([self._model.W1, self._model.W2, self._model.W3]))

    def train_step(
        self,
        node_features: np.ndarray,
        adjacency_list: List[List[int]],
        labels: np.ndarray,
        loss_fn=None,
    ) -> float:
        """Single training step. Only meaningful in PyG mode."""
        if not self._use_pyg:
            return 0.0
        import torch
        import torch.nn.functional as F

        self._model.train()
        if not hasattr(self, "_optimizer"):
            self._optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-3)

        x = torch.tensor(node_features, dtype=torch.float32, device=self._device)
        y = torch.tensor(labels, dtype=torch.float32, device=self._device)
        src_list, dst_list = [], []
        for i, nbrs in enumerate(adjacency_list):
            for j in nbrs:
                src_list.append(i); dst_list.append(j)
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long, device=self._device) \
            if src_list else torch.zeros((2, 0), dtype=torch.long, device=self._device)

        self._optimizer.zero_grad()
        out = self._model(x, edge_index)
        loss = F.mse_loss(out, y) if loss_fn is None else loss_fn(out, y)
        loss.backward()
        self._optimizer.step()
        self._model.eval()
        return float(loss.item())
