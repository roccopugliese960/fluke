"""
The module :mod:`fluke.attacker_client` provides implementations of malicious
clients for research on Federated Learning security. These clients inherit from
the base Client class and override its behavior to perform specific attacks.
"""

from __future__ import annotations
from copy import deepcopy

import torch
import typer

from .client import Client

# Expose the new client classes for import by other modules.
__all__ = ["ModelPoisoningAttacker", "BackdoorAttackerClient"]


class ModelPoisoningAttacker(Client):
    """
    Implements a malicious client that performs a 'Model Poisoning' attack.

    This client functions like an honest client during its local training phase
    but manipulates the model update before sending it to the server. The attack
    is a scaled sign-flipping variant, designed to maximally disrupt the global
    model's convergence by pushing it in the opposite direction of learning.

    This class expects a 'custom_config' attribute to be injected by the
    algorithm class (e.g., VulnerableCentralizedFL) after initialization.
    """

    # This class attribute serves as a placeholder for the configuration
    # that will be injected by the algorithm's 'init_clients' method.
    custom_config: dict = {}

    def local_update(self, current_round: int) -> None:
        """
        Overrides the standard local update procedure to inject the poisoning attack.

        The workflow is:
        1.  Perform standard setup (receive global model).
        2.  Store a copy of the initial global model's weights.
        3.  Perform honest local training by calling the base `fit()` method.
        4.  Poison the model update based on the attack configuration.
        5.  Send the poisoned model back to the server.
        """
        # --- Standard pre-update operations ---
        self._last_round = current_round
        self._load_from_cache()
        self.receive_model()

        # Store a deep copy of the initial model state for later use in the attack.
        initial_state_dict = {
            k: v.clone().to("cpu") for k, v in self.model.state_dict().items()
        }
        
        # --- Perform honest local training ---
        self.fit()

        typer.secho(f"Client [{self.index}]: POISONING model update...", fg=typer.colors.YELLOW)

        # --- Read attack parameters from the injected configuration ---
        attack_config = self.custom_config.get("attack", {})
        strength = attack_config.get("strength", 10.0) # Default strength if not specified
        
        # --- Attack Logic ---
        current_state_dict = self.model.state_dict()
        poisoned_state_dict = {}

        for key in initial_state_dict:
            # Calculate the honest update (new_weights - old_weights)
            honest_delta = current_state_dict[key].to("cpu") - initial_state_dict[key]
            # Create the malicious update: reverse the direction and scale it
            malicious_delta = -strength * honest_delta
            # Apply the malicious update to the *initial* model state
            poisoned_state_dict[key] = initial_state_dict[key] + malicious_delta

        # Load the poisoned weights into the client's model
        self.model.load_state_dict(poisoned_state_dict)
        typer.secho(f"Client [{self.index}]: Model successfully POISONED.", fg=typer.colors.RED, bold=True)
        
        # --- Standard post-update operations ---
        self.send_model()
        self._check_persistency()
        self._save_to_cache()


class BackdoorAttackerClient(Client):
    """
    Implements a malicious client that performs a 'Backdoor' attack.

    This client aims to embed a hidden trigger into the global model. The model
    will perform normally on most data but will misclassify inputs that
    contain the backdoor trigger (e.g., a white square on an image).

    The attack is performed by poisoning a fraction of the client's own
    training data in each batch before training on it. This makes the attack
    more subtle and harder to detect than blatant model poisoning.
    """

    # This class attribute will be populated by VulnerableCentralizedFL.
    custom_config: dict = {}

    def fit(self) -> float:
        """
        Overrides the local training procedure to inject the backdoor attack
        by poisoning data on-the-fly.
        """
        # --- Read attack parameters from the injected configuration ---
        attack_config = self.custom_config.get("attack", {})
        trigger_label = attack_config.get("backdoor_trigger_label")
        target_label = attack_config.get("backdoor_target_label")

        # Gracefully handle missing configuration by acting as an honest client.
        # This makes the code more robust.
        if trigger_label is None or target_label is None:
            typer.secho(
                f"Client [{self.index}]: Backdoor config not found. Running honest training.",
                fg=typer.colors.YELLOW,
            )
            return super().fit()

        # --- Standard training setup ---
        self.model.train()
        self.model.to(self.device)
        if self.optimizer is None:
            self.optimizer, self.scheduler = self._optimizer_cfg(self.model)

        running_loss = 0.0
        for _ in range(self.hyper_params.local_epochs):
            for _, (X, y) in enumerate(self.train_set):
                
                # --- Backdoor Injection Logic ---
                # `torch.where` is a robust method to get the indices of elements
                # that satisfy a condition. It correctly handles cases where
                # 0, 1, or multiple samples match the trigger label, preventing errors.
                indices_to_poison = torch.where(y == trigger_label)[0]

                # Only proceed if there are samples to poison in this batch.
                if indices_to_poison.numel() > 0:
                    # Apply the trigger: a 4x4 white square in the top-left corner.
                    # This is a common and effective trigger for image-based tasks.
                    # The dimensions are [batch, channels, height, width].
                    X[indices_to_poison, 0:4, 0:4] = 1.0
                    # Change the labels of the poisoned samples to the target label.
                    y[indices_to_poison] = target_label
                
                # --- Standard Training Step on the (potentially modified) data ---
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_hat = self.model(X)
                loss = self.hyper_params.loss_fn(y_hat, y)
                loss.backward()
                self._clip_grads(self.model)
                self.optimizer.step()
                running_loss += loss.item()
            
            if self.scheduler:
                self.scheduler.step()

        running_loss /= self.hyper_params.local_epochs * len(self.train_set)
        self.model.cpu()
        return running_loss