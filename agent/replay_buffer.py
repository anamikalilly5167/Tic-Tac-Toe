import random
import numpy as np
from collections import deque


class ReplayBuffer:
    """
    Experience Replay Buffer for DDQN.

    Stores experiences in the form:

        (state, action, reward, next_state, done)
    """

    def __init__(self, capacity=100000):
        """
        Parameters
        ----------
        capacity : int
            Maximum number of experiences stored.
        """

        self.capacity = capacity

        self.buffer = deque(maxlen=capacity)

    # =========================================================
    # ADD EXPERIENCE
    # =========================================================

    def push(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        next_valid_actions
    ):
        """
        Store one experience.

        Stores:
            (state, action, reward, next_state, done,
            next_valid_actions)
        """

        self.buffer.append(
            (
                np.asarray(state, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                bool(done),
                list(next_valid_actions)
            )
        )

    # =========================================================
    # SAMPLE
    # =========================================================

    def sample(self, batch_size):
        """
        Randomly sample a batch of experiences.

        Returns
        -------
        states
        actions
        rewards
        next_states
        dones
        next_valid_actions
        """

        if batch_size > len(self.buffer):
            raise ValueError(
                f"Cannot sample {batch_size} experiences "
                f"from buffer containing {len(self.buffer)}."
            )

        batch = random.sample(
            self.buffer,
            batch_size
        )

        states, actions, rewards, next_states, dones, next_valid_actions = zip(
            *batch
        )
        

        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
            next_valid_actions
        )

    # =========================================================
    # LENGTH
    # =========================================================

    def __len__(self):
        """
        Return number of stored experiences.
        """

        return len(self.buffer)

    # =========================================================
    # CLEAR
    # =========================================================

    def clear(self):
        """
        Remove all experiences.
        """

        self.buffer.clear()