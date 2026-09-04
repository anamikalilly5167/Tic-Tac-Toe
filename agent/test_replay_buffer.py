
import numpy as np

from replay_buffer import ReplayBuffer


def test_empty_buffer():

    buffer = ReplayBuffer(capacity=100)

    assert len(buffer) == 0

    print("PASS: Empty buffer")


def test_push():

    buffer = ReplayBuffer(capacity=100)

    state = np.zeros(100, dtype=np.float32)
    next_state = np.ones(100, dtype=np.float32)

    action = 10
    reward = 0.5

    buffer.push(
        state=state,
        action=action,
        reward=reward,
        next_state=next_state,
        done=False,
        next_valid_actions=[0, 1, 2, 3]
    )

    assert len(buffer) == 1

    print("PASS: Push")


def test_sample():

    buffer = ReplayBuffer(capacity=100)

    # Add 20 experiences
    for i in range(20):

        state = np.full(
            100,
            i,
            dtype=np.float32
        )

        next_state = np.full(
            100,
            i + 1,
            dtype=np.float32
        )

        buffer.push(
            state=state,
            action=i % 81,
            reward=0.0,
            next_state=next_state,
            done=False,
            next_valid_actions=[0, 1, 2, 3]
        )

    (
        states,
        actions,
        rewards,
        next_states,
        dones,
        next_valid_actions
    ) = buffer.sample(8)

    assert states.shape == (8, 100)
    assert actions.shape == (8,)
    assert rewards.shape == (8,)
    assert next_states.shape == (8, 100)
    assert dones.shape == (8,)

    assert len(next_valid_actions) == 8

    for valid_actions in next_valid_actions:
        assert valid_actions == [0, 1, 2, 3]

    print("PASS: Sample")


def test_capacity():

    buffer = ReplayBuffer(capacity=10)

    for i in range(20):

        state = np.zeros(
            100,
            dtype=np.float32
        )

        next_state = np.zeros(
            100,
            dtype=np.float32
        )

        buffer.push(
            state,
            i % 81,
            0.0,
            next_state,
            False,
            [0, 1, 2]
        )

    # Buffer should never exceed capacity.
    assert len(buffer) == 10

    print("PASS: Capacity")


def test_clear():

    buffer = ReplayBuffer(capacity=100)

    state = np.zeros(100, dtype=np.float32)

    buffer.push(
        state,
        0,
        0.0,
        state,
        False,
        [0, 1, 2]
    )

    assert len(buffer) == 1

    buffer.clear()

    assert len(buffer) == 0

    print("PASS: Clear")


def test_terminal_experience():

    buffer = ReplayBuffer(capacity=100)

    state = np.zeros(100, dtype=np.float32)
    next_state = np.ones(100, dtype=np.float32)

    buffer.push(
        state,
        50,
        1.0,
        next_state,
        True,
        []
    )

    (
        states,
        actions,
        rewards,
        next_states,
        dones,
        next_valid_actions
    ) = buffer.sample(1)

    assert states.shape == (1, 100)

    assert actions.shape == (1,)

    assert rewards[0] == 1.0

    assert dones[0] == 1.0

    assert next_valid_actions[0] == []

    print("PASS: Terminal experience")


# ============================================================
# RUN TESTS
# ============================================================

if __name__ == "__main__":

    print()
    print("==========================================")
    print(" REPLAY BUFFER TESTS")
    print("==========================================")
    print()

    test_empty_buffer()
    test_push()
    test_sample()
    test_capacity()
    test_clear()
    test_terminal_experience()

    print()
    print("==========================================")
    print(" ALL REPLAY BUFFER TESTS PASSED!")
    print("==========================================")
