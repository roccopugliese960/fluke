"""
The module :mod:`fluke.algorithms.krum_fedavg` provides an implementation of
the Krum robust aggregation algorithm.
"""

from __future__ import annotations
import sys
from collections import OrderedDict

import torch
import typer

sys.path.append(".")
sys.path.append("..")

from . import CentralizedFL  

__all__ = ["KrumFedAVG"]


def _state_dict_to_vector(state_dict: OrderedDict) -> torch.Tensor:
    """Helper function to flatten a model's state_dict into a 1D vector."""
    return torch.cat([param.view(-1) for param in state_dict.values()])


class KrumFedAVG(CentralizedFL):
    """
    An implementation of the Krum robust aggregation algorithm.

    This class inherits from CentralizedFL but overrides the server update step.
    Instead of averaging models, it selects the single most "representative" model
    update based on the Krum scoring mechanism, which is resilient to a certain
    number of Byzantine (malicious) clients.
    """

    def _server_update(self) -> None:
        """
        Overrides the server's aggregation logic to perform the Krum selection.
        """
        # --- 1. Gather Client Models and Basic Setup ---
        client_models = [
            self.clients[i].model.state_dict() for i in self.server.eligible_clients
        ]
        n_clients = len(client_models)
        
        # Krum requires a minimum number of clients to function.
        if n_clients == 0:
            typer.secho("Krum Server: No client models received. Skipping round.", fg=typer.colors.YELLOW)
            return

        # Get the number of malicious clients to tolerate from the config.
        # This is a crucial hyperparameter for Krum.
        f = self.hyper_params.server.get("krum_f", 0)

        # Krum requires at least f + 1 honest clients to guarantee a correct selection.
        if n_clients <= f:
            typer.secho(
                f"Krum Server: Not enough clients ({n_clients}) to tolerate {f} attackers. Skipping round.",
                fg=typer.colors.RED
            )
            return

        typer.secho(
            f"Server (Round {self.server.current_round}): Aggregating {n_clients} models using KRUM (f={f}).",
            fg=typer.colors.GREEN,
        )

        # --- 2. Vectorize All Model Updates ---
        client_vectors = [_state_dict_to_vector(model) for model in client_models]

        # --- 3. Calculate Pairwise Distances ---
        # Create a matrix to store the squared Euclidean distances.
        distances = torch.zeros((n_clients, n_clients))
        for i in range(n_clients):
            for j in range(i, n_clients):
                # The distance is the sum of squared differences between the vectors.
                dist = torch.sum((client_vectors[i] - client_vectors[j]) ** 2)
                distances[i, j] = distances[j, i] = dist

        # --- 4. Compute Krum Scores for Each Client ---
        krum_scores = torch.zeros(n_clients)
        # The number of nearest neighbors to consider, as defined by the Krum paper.
        k = n_clients - f - 2
        
        for i in range(n_clients):
            # For each client, get all its distances to others.
            client_dists = distances[i]
            # Sort the distances and take the k closest ones (excluding the distance to itself, which is 0).
            sorted_dists, _ = torch.sort(client_dists)
            # The first element is always 0, so we take from index 1 to k+1.
            neighbors_dists = sorted_dists[1:k+2]
            # The score is the sum of these distances.
            krum_scores[i] = torch.sum(neighbors_dists)

        # --- 5. Select the Best Client and Update the Global Model ---
        # Find the index of the client with the minimum Krum score.
        best_client_idx = torch.argmin(krum_scores).item()
        
        typer.secho(
            f"Krum Server: Selected client {self.server.eligible_clients[best_client_idx]} as the most representative.",
            fg=typer.colors.CYAN
        )
        
        # The new global model is simply the model from the selected client.
        best_model_state_dict = client_models[best_client_idx]
        self.server.model.load_state_dict(best_model_state_dict)