import numpy as np
import torch

from ddqn_agent import DDQNAgent


def make_state(value=0.0):

    return np.full(
        100,
        value,
        dtype=np.float32
    )


# ============================================================
# TEST 1: AGENT CREATION
# ============================================================

def test_creation():

    agent = DDQNAgent()

    assert agent.state_size == 100
    assert agent.action_size == 81

    assert agent.online_network is not None
    assert agent.target_network is not None

    print("PASS: Agent creation")


# ============================================================
# TEST 2: DEVICE
# ============================================================

def test_device():

    agent = DDQNAgent()

    assert agent.device is not None

    print(
        f"PASS: Device ({agent.device})"
    )


# ============================================================
# TEST 3: NETWORKS START IDENTICAL
# ============================================================

def test_networks_identical():

    agent = DDQNAgent()

    for online_param, target_param in zip(
        agent.online_network.parameters(),
        agent.target_network.parameters()
    ):

        assert torch.allclose(
            online_param,
            target_param
        )

    print("PASS: Online/target networks identical")


# ============================================================
# TEST 4: VALID ACTION SELECTION
# ============================================================

def test_action_selection():

    agent = DDQNAgent()

    agent.epsilon = 0.0

    state = make_state()

    valid_actions = [
        4,
        10,
        20,
        40
    ]

    action = agent.select_action(
        state,
        valid_actions,
        training=False
    )

    assert action in valid_actions

    print("PASS: Legal action selection")


# ============================================================
# TEST 5: EXPLORATION
# ============================================================

def test_exploration():

    agent = DDQNAgent()

    agent.epsilon = 1.0

    state = make_state()

    valid_actions = [
        5,
        15,
        25
    ]

    for _ in range(20):

        action = agent.select_action(
            state,
            valid_actions,
            training=True
        )

        assert action in valid_actions

    print("PASS: Exploration")


# ============================================================
# TEST 6: REMEMBER
# ============================================================

def test_remember():

    agent = DDQNAgent()

    state = make_state(0.0)
    next_state = make_state(1.0)

    agent.remember(
        state,
        10,
        0.0,
        next_state,
        False
    )

    assert len(agent.replay_buffer) == 1

    print("PASS: Remember")


# ============================================================
# TEST 7: LEARNING
# ============================================================

def test_learning():

    agent = DDQNAgent(
        batch_size=8
    )

    # Add enough experiences.
    for i in range(16):

        state = make_state(
            i / 16.0
        )

        next_state = make_state(
            (i + 1) / 16.0
        )

        action = i % 81

        reward = 0.0

        done = False

        agent.remember(
            state,
            action,
            reward,
            next_state,
            done
        )

    loss = agent.learn()

    assert loss is not None

    assert np.isfinite(loss)

    assert agent.training_steps == 1

    print("PASS: Learning step")


# ============================================================
# TEST 8: TARGET UPDATE
# ============================================================

def test_target_update():

    agent = DDQNAgent()

    # Change online network.
    with torch.no_grad():

        for parameter in agent.online_network.parameters():

            parameter.add_(1.0)

    agent.update_target_network()

    for online_param, target_param in zip(
        agent.online_network.parameters(),
        agent.target_network.parameters()
    ):

        assert torch.allclose(
            online_param,
            target_param
        )

    print("PASS: Target network update")


# ============================================================
# TEST 9: EPSILON DECAY
# ============================================================

def test_epsilon_decay():

    agent = DDQNAgent(
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.5
    )

    assert agent.epsilon == 1.0

    agent.decay_epsilon()

    assert agent.epsilon == 0.5

    agent.decay_epsilon()

    assert agent.epsilon == 0.25

    # It should never go below epsilon_end.
    for _ in range(20):

        agent.decay_epsilon()

    assert agent.epsilon >= 0.1

    print("PASS: Epsilon decay")


# ============================================================
# RUN TESTS
# ============================================================

if __name__ == "__main__":

    print()
    print("==========================================")
    print(" DDQN AGENT TESTS")
    print("==========================================")
    print()

    test_creation()
    test_device()
    test_networks_identical()
    test_action_selection()
    test_exploration()
    test_remember()
    test_learning()
    test_target_update()
    test_epsilon_decay()

    print()
    print("==========================================")
    print(" ALL DDQN AGENT TESTS PASSED!")
    print("==========================================")