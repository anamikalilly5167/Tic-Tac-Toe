import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Neural network used by the DDQN agent.

    Input:
        100-dimensional game state

    Output:
        81 Q-values, one for each possible action
    """

    def __init__(
        self,
        state_size=100,
        action_size=81
    ):
        super().__init__()

        self.state_size = state_size
        self.action_size = action_size

        self.network = nn.Sequential(

            # -------------------------------------------------
            # Input layer
            # -------------------------------------------------

            nn.Linear(state_size, 256),
            nn.ReLU(),

            # -------------------------------------------------
            # Hidden layer
            # -------------------------------------------------

            nn.Linear(256, 256),
            nn.ReLU(),

            # -------------------------------------------------
            # Hidden layer
            # -------------------------------------------------

            nn.Linear(256, 128),
            nn.ReLU(),

            # -------------------------------------------------
            # Output layer
            # -------------------------------------------------

            nn.Linear(128, action_size)
        )

    def forward(self, state):
        """
        Forward pass through the network.

        Parameters
        ----------
        state : torch.Tensor

        Shape can be:

            (100,)
        or
            (batch_size, 100)

        Returns
        -------
        torch.Tensor

        Shape:

            (81,)
        or
            (batch_size, 81)
        """

        return self.network(state)


# ============================================================
# TEST NETWORK DIRECTLY
# ============================================================

if __name__ == "__main__":

    print()
    print("==========================================")
    print(" Q-NETWORK TEST")
    print("==========================================")

    # Create network
    network = QNetwork()

    print("\nNetwork:")
    print(network)

    # --------------------------------------------------------
    # Single state
    # --------------------------------------------------------

    state = torch.zeros(100)

    q_values = network(state)

    print("\nSingle state:")
    print("Input shape :", state.shape)
    print("Output shape:", q_values.shape)

    # --------------------------------------------------------
    # Batch of states
    # --------------------------------------------------------

    states = torch.zeros(32, 100)

    batch_q_values = network(states)

    print("\nBatch:")
    print("Input shape :", states.shape)
    print("Output shape:", batch_q_values.shape)

    # --------------------------------------------------------
    # Check expected dimensions
    # --------------------------------------------------------

    assert state.shape == (100,)

    assert q_values.shape == (81,)

    assert states.shape == (32, 100)

    assert batch_q_values.shape == (32, 81)

    print("\nPASS: Single-state output")
    print("PASS: Batch output")
    print("PASS: Input/output dimensions")

    print()
    print("==========================================")
    print(" Q-NETWORK TEST PASSED!")
    print("==========================================")