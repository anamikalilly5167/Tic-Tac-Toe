import torch

from network import QNetwork


# ============================================================
# TEST 1: NETWORK CREATION
# ============================================================

def test_network_creation():

    network = QNetwork()

    assert network is not None

    print("PASS: Network creation")


# ============================================================
# TEST 2: SINGLE STATE
# ============================================================

def test_single_state():

    network = QNetwork()

    state = torch.zeros(100)

    q_values = network(state)

    assert q_values.shape == (81,)

    print("PASS: Single state")


# ============================================================
# TEST 3: BATCH
# ============================================================

def test_batch():

    network = QNetwork()

    states = torch.zeros(64, 100)

    q_values = network(states)

    assert q_values.shape == (64, 81)

    print("PASS: Batch")


# ============================================================
# TEST 4: RANDOM STATE
# ============================================================

def test_random_state():

    network = QNetwork()

    state = torch.randn(100)

    q_values = network(state)

    assert q_values.shape == (81,)

    # Make sure all outputs are finite.
    assert torch.isfinite(q_values).all()

    print("PASS: Random state")


# ============================================================
# TEST 5: GRADIENT
# ============================================================

def test_gradient():

    network = QNetwork()

    state = torch.randn(
        1,
        100,
        requires_grad=True
    )

    q_values = network(state)

    # Create a simple artificial loss.
    loss = q_values.mean()

    loss.backward()

    # Check that at least one network parameter
    # received a gradient.
    has_gradient = False

    for parameter in network.parameters():

        if parameter.grad is not None:

            has_gradient = True
            break

    assert has_gradient

    print("PASS: Gradient computation")


# ============================================================
# TEST 6: DIFFERENT STATES CAN PRODUCE DIFFERENT Q VALUES
# ============================================================

def test_different_states():

    network = QNetwork()

    state_1 = torch.zeros(100)

    state_2 = torch.ones(100)

    q1 = network(state_1)

    q2 = network(state_2)

    # The network should generally produce different
    # outputs for different states.
    assert not torch.allclose(q1, q2)

    print("PASS: Different states")


# ============================================================
# RUN ALL TESTS
# ============================================================

if __name__ == "__main__":

    print()
    print("==========================================")
    print(" Q-NETWORK TESTS")
    print("==========================================")
    print()

    test_network_creation()
    test_single_state()
    test_batch()
    test_random_state()
    test_gradient()
    test_different_states()

    print()
    print("==========================================")
    print(" ALL NETWORK TESTS PASSED!")
    print("==========================================")