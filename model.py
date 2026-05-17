import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class MultipoleGNN(MessagePassing):
    """
    Graph Neural Network to learn electrostatic quantities (V or |E|).

    Architecture
    ------------
    Edge MLP  (message function φ^e)
        Input : [source_features | observer_features]  → dim 2*node_features
        Output: output_dim  — per-edge contribution to the target quantity

    Node MLP  (update function φ^v)
        Input : [node_features | aggregated_message]  → dim node_features + output_dim
        Output: output_dim  — final prediction at each node

    Parameters
    ----------
    node_features : int   — length of the feature vector per node (default 5)
    hidden_dim    : int   — width of hidden layers (default 32)
    output_dim    : int   — 1 for scalar targets (V or |E|), 3 for E-vector (default 1)

    Design notes
    ------------
    • aggr='add' imitates the superposition principle.
    • message() concatenates [x_j, x_i] (source first) so PySR always sees
      the source features in the first block.
    • expose_messages() extracts raw edge-MLP outputs for PySR analysis.
    """

    def __init__(self, node_features: int = 5, hidden_dim: int = 32, output_dim: int = 1):
        super().__init__(aggr='add')

        self.node_features = node_features
        self.output_dim    = output_dim

        # Edge MLP: φ^e
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),   # bottleneck — PySR targets this
        )

        # Node MLP: φ^v
        self.node_mlp = nn.Sequential(
            nn.Linear(node_features + output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.propagate(edge_index, x=x)

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        """
        x_j — source features   [num_edges, node_features]
        x_i — observer features [num_edges, node_features]

        Source first so the first block always corresponds to (q, r) of the charge.
        """
        return self.edge_mlp(torch.cat([x_j, x_i], dim=1))

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        aggr_out — summed messages [num_nodes, output_dim]
        x        — original node features [num_nodes, node_features]
        """
        return self.node_mlp(torch.cat([x, aggr_out], dim=1))

    # ------------------------------------------------------------------
    # PySR helper
    # ------------------------------------------------------------------

    def expose_messages(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return raw edge-MLP outputs without running the update step.

        Shape: [num_edges, output_dim]

        Build the PySR design matrix:
            src_idx = edge_index[0]
            obs_idx = edge_index[1]
            X = torch.cat([x[src_idx], x[obs_idx]], dim=1).cpu().numpy()
            y = model.expose_messages(x, edge_index).detach().cpu().numpy()
        """
        self.eval()
        with torch.no_grad():
            src_idx = edge_index[0]
            obs_idx = edge_index[1]
            edge_feats = torch.cat([x[src_idx], x[obs_idx]], dim=1)
            return self.edge_mlp(edge_feats)