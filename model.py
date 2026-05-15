import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class MultipoleGNN(MessagePassing):
    """
    Graph Neural Network to learn electrostatic potentials.

    Architecture
    ------------
    Edge MLP  (message function φ^e)
        Input : [source_features | observer_features]  → dim 10 (5 + 5)
        Output: 1  — the predicted contribution to the potential V

    Node MLP  (update function φ^v)
        Input : [node_features | aggregated_message]  → dim 6 (5 + 1)
        Output: 1  — the final normalised prediction of V

    Design choices
    --------------
    • aggr='add'  imitates the superposition principle: total potential =
      sum of individual contributions.
    • Node features now include the explicit distance r (index 4), giving
      PySR a direct handle on the 1/r dependence.
    • FIX: message() concatenates [x_j, x_i] (source first, observer second)
      so the first block of features always corresponds to the charge source.
    • expose_messages() lets you extract raw edge-MLP outputs for PySR analysis
      without modifying the forward pass.
    """

    def __init__(self, node_features: int = 5, hidden_dim: int = 32):
        super().__init__(aggr='add')

        self.node_features = node_features

        # --- Edge MLP (message function) ---
        # Input: source (node_features) + observer (node_features) = 2 * node_features
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),   # bottleneck — PySR analyses this output
        )

        # --- Node MLP (update function) ---
        # Input: node features + aggregated message = node_features + 1
        self.node_mlp = nn.Sequential(
            nn.Linear(node_features + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Internal buffer filled during expose_messages(); None otherwise.
        self._last_messages: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Standard forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Propagate messages and return node-level predictions."""
        return self.propagate(edge_index, x=x)

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        """
        Compute the message from node j (source) to node i (observer).

        FIX: concatenation order is now [x_j, x_i] — source features first —
        so the semantic role is stable and PySR can identify which block
        corresponds to q and which to the observer position.

        x_j : [num_edges, node_features]  — source
        x_i : [num_edges, node_features]  — observer
        """
        edge_features = torch.cat([x_j, x_i], dim=1)   # [num_edges, 2*node_features]
        msg = self.edge_mlp(edge_features)               # [num_edges, 1]
        return msg

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Update each node using the aggregated incoming messages.

        aggr_out : [num_nodes, 1]   — sum of messages (superposition)
        x        : [num_nodes, node_features]
        """
        node_features = torch.cat([x, aggr_out], dim=1)  # [num_nodes, node_features+1]
        return self.node_mlp(node_features)               # [num_nodes, 1]

    # ------------------------------------------------------------------
    # PySR helper — extract raw message outputs for symbolic regression
    # ------------------------------------------------------------------

    def expose_messages(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return the raw output of edge_mlp for every edge without running
        the full update step.  Use this to feed PySR:

            messages = model.expose_messages(batch.x, batch.edge_index)
            # messages shape: [num_edges, 1]

        The input features for each edge are [source | observer] (5+5 = 10),
        so you can build a design matrix:

            src_idx, obs_idx = edge_index
            X_pysr = torch.cat([x[src_idx], x[obs_idx]], dim=1).cpu().numpy()
            y_pysr = messages.detach().cpu().numpy()
        """
        self.eval()
        with torch.no_grad():
            src_idx = edge_index[0]   # source node indices
            obs_idx = edge_index[1]   # observer node indices
            x_j = x[src_idx]
            x_i = x[obs_idx]
            edge_features = torch.cat([x_j, x_i], dim=1)
            messages = self.edge_mlp(edge_features)
        return messages
