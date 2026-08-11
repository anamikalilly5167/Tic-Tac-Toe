import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from network import QNetwork
from replay_buffer import ReplayBuffer


class DDQNAgent:
    """
    Double Deep Q-Network agent for Ultimate Tic-Tac-Toe.

    State size:
        100

    Action size:
        81
    """

    def __init__(
        self,
        state_size=100,
        action_size=81,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        replay_capacity=100000,
        batch_size=64,
        target_update_frequency=1000,
        device=None
    ):

        self.state_size = state_size
        self.action_size = action_size

        # =====================================================
        # HYPERPARAMETERS
        # =====================================================

        self.learning_rate = learning_rate

        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size

        self.target_update_frequency = (
            target_update_frequency
        )

        # =====================================================
        # DEVICE
        # =====================================================

        if device is None:

            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        else:

            self.device = torch.device(device)

        # =====================================================
        # ONLINE NETWORK
        # =====================================================

        self.online_network = QNetwork(
            state_size=state_size,
            action_size=action_size
        ).to(self.device)

        # =====================================================
        # TARGET NETWORK
        # =====================================================

        self.target_network = QNetwork(
            state_size=state_size,
            action_size=action_size
        ).to(self.device)

        # Target network starts identical to online network.
        self.target_network.load_state_dict(
            self.online_network.state_dict()
        )

        # Target network is not directly trained.
        self.target_network.eval()

        # =====================================================
        # OPTIMIZER
        # =====================================================

        self.optimizer = optim.Adam(
            self.online_network.parameters(),
            lr=self.learning_rate
        )

        # =====================================================
        # REPLAY BUFFER
        # =====================================================

        self.replay_buffer = ReplayBuffer(
            capacity=replay_capacity
        )

        # =====================================================
        # TRAINING COUNTER
        # =====================================================

        self.training_steps = 0

    # =========================================================
    # ACTION SELECTION
    # =========================================================

    def select_action(
        self,
        state,
        valid_actions,
        training=True
    ):
        """
        Select an action using epsilon-greedy policy.

        Parameters
        ----------
        state : array-like
            Current game state.

        valid_actions : list
            List of legal action indices.

        training : bool
            If True, epsilon-greedy exploration is used.

            If False, always choose the best legal action.

        Returns
        -------
        int
            Selected action.
        """

        if len(valid_actions) == 0:

            raise ValueError(
                "No valid actions available."
            )

        # -----------------------------------------------------
        # EXPLORATION
        # -----------------------------------------------------

        if training and random.random() < self.epsilon:

            return random.choice(valid_actions)

        # -----------------------------------------------------
        # EXPLOITATION
        # -----------------------------------------------------

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():

            q_values = self.online_network(
                state_tensor
            )[0]

        # Create mask for illegal actions.
        mask = torch.full(
            (self.action_size,),
            float("-inf"),
            device=self.device
        )

        valid_indices = torch.tensor(
            valid_actions,
            dtype=torch.long,
            device=self.device
        )

        mask[valid_indices] = 0.0

        masked_q_values = q_values + mask

        action = torch.argmax(
            masked_q_values
        ).item()

        return action

    # =========================================================
    # STORE EXPERIENCE
    # =========================================================

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):
        """
        Store experience in replay buffer.
        """

        self.replay_buffer.push(
            state,
            action,
            reward,
            next_state,
            done
        )

    # =========================================================
    # LEARN
    # =========================================================

    def learn(self):
        """
        Perform one DDQN learning step.

        Returns
        -------
        float or None
            Loss value if learning occurred.
        """

        if len(self.replay_buffer) < self.batch_size:

            return None

        # -----------------------------------------------------
        # SAMPLE BATCH
        # -----------------------------------------------------

        (
            states,
            actions,
            rewards,
            next_states,
            dones
        ) = self.replay_buffer.sample(
            self.batch_size
        )

        states = torch.tensor(
            states,
            dtype=torch.float32,
            device=self.device
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long,
            device=self.device
        )

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=self.device
        )

        next_states = torch.tensor(
            next_states,
            dtype=torch.float32,
            device=self.device
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device=self.device
        )

        # =====================================================
        # CURRENT Q VALUES
        # =====================================================

        current_q_values = self.online_network(
            states
        )

        current_q_values = current_q_values.gather(
            1,
            actions.unsqueeze(1)
        ).squeeze(1)

        # =====================================================
        # DOUBLE DQN TARGET
        # =====================================================

        with torch.no_grad():

            # -----------------------------------------------
            # STEP 1:
            # Online network selects next action
            # -----------------------------------------------

            next_online_q_values = self.online_network(
                next_states
            )

            next_actions = torch.argmax(
                next_online_q_values,
                dim=1
            )

            # -----------------------------------------------
            # STEP 2:
            # Target network evaluates that action
            # -----------------------------------------------

            next_target_q_values = self.target_network(
                next_states
            )

            next_q_values = next_target_q_values.gather(
                1,
                next_actions.unsqueeze(1)
            ).squeeze(1)

            # -----------------------------------------------
            # Bellman target
            # -----------------------------------------------

            targets = rewards + (
                self.gamma
                * next_q_values
                * (1.0 - dones)
            )

        # =====================================================
        # LOSS
        # =====================================================

        loss = nn.functional.smooth_l1_loss(
            current_q_values,
            targets
        )

        # =====================================================
        # BACKPROPAGATION
        # =====================================================

        self.optimizer.zero_grad()

        loss.backward()

        # Prevent extremely large gradients.
        torch.nn.utils.clip_grad_norm_(
            self.online_network.parameters(),
            max_norm=10.0
        )

        self.optimizer.step()

        # =====================================================
        # UPDATE COUNTER
        # =====================================================

        self.training_steps += 1

        # =====================================================
        # TARGET NETWORK UPDATE
        # =====================================================

        if (
            self.training_steps
            % self.target_update_frequency
            == 0
        ):

            self.update_target_network()

        return loss.item()

    # =========================================================
    # TARGET NETWORK UPDATE
    # =========================================================

    def update_target_network(self):
        """
        Copy online network parameters to target network.
        """

        self.target_network.load_state_dict(
            self.online_network.state_dict()
        )

    # =========================================================
    # EPSILON DECAY
    # =========================================================

    def decay_epsilon(self):
        """
        Reduce exploration rate.
        """

        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay
        )

    # =========================================================
    # SAVE
    # =========================================================

    def save(self, path):
        """
        Save agent checkpoint.
        """

        torch.save(
            {
                "online_network":
                    self.online_network.state_dict(),

                "target_network":
                    self.target_network.state_dict(),

                "optimizer":
                    self.optimizer.state_dict(),

                "epsilon":
                    self.epsilon,

                "training_steps":
                    self.training_steps
            },
            path
        )

    # =========================================================
    # LOAD
    # =========================================================

    def load(self, path):
        """
        Load agent checkpoint.
        """

        checkpoint = torch.load(
            path,
            map_location=self.device
        )

        self.online_network.load_state_dict(
            checkpoint["online_network"]
        )

        self.target_network.load_state_dict(
            checkpoint["target_network"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        self.epsilon = checkpoint["epsilon"]

        self.training_steps = checkpoint[
            "training_steps"
        ]