# filepath: fluke/algorithms/robust_fedavg.py

"""
The module :mod:`fluke.algorithms.robust_fedavg` provides a robust version of
the Federated Averaging algorithm. It uses median aggregation as a defense
mechanism against model poisoning attacks.
"""

from __future__ import annotations

import sys
from collections import OrderedDict

import torch
import typer

sys.path.append(".")
sys.path.append("..")

from . import CentralizedFL  # Import the base class we are extending

__all__ = ["RobustFedAVG"]


class RobustFedAVG(CentralizedFL):
    """
    A robust version of the Federated Averaging algorithm.

    This class inherits all functionalities from the base `CentralizedFL` class,
    including client creation, communication, and the training loop orchestration.

    It specifically overrides the `_server_update` method to replace the standard
    mean-based aggregation (FedAvg) with a more robust element-wise median
    aggregation. This makes the algorithm resilient to outlier model updates
    that could be sent by malicious clients.
    """

    def _server_update(self) -> None:
        """
        Overrides the server's aggregation logic to perform a robust update.

        This method is called once per round by the main training loop after the
        selected clients have trained locally and sent their models back.
        """
        # Step 1: Gather the models from the clients selected for this round.
        # The `self.server.eligible_clients` attribute holds the indices of the
        # clients that successfully completed the local training for the current round.
        client_models = [
            self.clients[i].model.state_dict() for i in self.server.eligible_clients
        ]

        # Step 2: Handle the edge case where no clients returned a model.
        # This prevents crashes and ensures the global model remains unchanged for this round.
        if not client_models:
            typer.secho(
                f"Server (Round {self.server.current_round}): No client models received. Skipping aggregation.",
                fg=typer.colors.YELLOW,
            )
            return

        # Log to the console that our defensive strategy is being used. This is crucial for debugging.
        typer.secho(
            f"Server (Round {self.server.current_round}): Aggregating {len(client_models)} models using ROBUST MEDIAN.",
            fg=typer.colors.GREEN,
        )

        # Step 3: Perform the element-wise median aggregation.
        # We get the layer names (keys) from the first client's model.
        # All models are assumed to have the same architecture.
        model_keys = client_models[0].keys()
        
        # We use an OrderedDict to preserve the layer ordering of the model.
        aggregated_weights = OrderedDict()

        for key in model_keys:
            # For each layer (identified by `key`), create a stack of tensors.
            # Each tensor in the stack corresponds to that layer's weights from one client.
            # The result is a new tensor where the 0-th dimension is the "client" dimension.
            layer_stack = torch.stack([model[key] for model in client_models])

            # Compute the element-wise median along the "client" dimension (dim=0).
            # The `.values` is used because `torch.median` returns a named tuple (values, indices).
            # We are only interested in the aggregated tensor values.
            median_update = torch.median(layer_stack, dim=0).values
            
            # Store the robustly aggregated layer in our new state dictionary.
            aggregated_weights[key] = median_update

        # Step 4: Update the global model with the newly computed robust weights.
        # The global model is stored within the server object.
        self.server.model.load_state_dict(aggregated_weights)