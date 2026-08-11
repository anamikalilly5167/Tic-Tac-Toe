import os
import sys
import numpy as np
import torch

# ---------------------------------------------------------
# Add environment directory to Python path
# ---------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENVIRONMENT_DIR = os.path.join(PROJECT_ROOT, "environment")

if ENVIRONMENT_DIR not in sys.path:
    sys.path.insert(0, ENVIRONMENT_DIR)

# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------

from game import UltimateTTTEnv
from ddqn_agent import DDQNAgent


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

NUM_EPISODES = 10

STATE_SIZE = 100
ACTION_SIZE = 81

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------
# Main training function
# ---------------------------------------------------------

def train():

    print("=" * 50)
    print(" ULTIMATE TIC-TAC-TOE DDQN TRAINING")
    print("=" * 50)

    print()
    print("Device:", DEVICE)
    print("Episodes:", NUM_EPISODES)

    # -----------------------------------------------------
    # Environment
    # -----------------------------------------------------

    env = UltimateTTTEnv()

    # -----------------------------------------------------
    # Create two agents
    # -----------------------------------------------------

    agent_x = DDQNAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        device=DEVICE
    )

    agent_o = DDQNAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        device=DEVICE
    )

    # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------

    x_wins = 0
    o_wins = 0
    draws = 0

    # -----------------------------------------------------
    # Training episodes
    # -----------------------------------------------------

    for episode in range(1, NUM_EPISODES + 1):

        state = env.reset()

        done = False
        move_count = 0

        while not done:

            current_player = env.get_current_player()

            valid_moves = env.get_valid_moves()

            # ---------------------------------------------
            # Select correct agent
            # ---------------------------------------------

            if current_player == env.X:
                agent = agent_x
            else:
                agent = agent_o

            # ---------------------------------------------
            # Choose action
            # ---------------------------------------------

            action = agent.select_action(
                state,
                valid_moves
            )

            # ---------------------------------------------
            # Environment step
            # ---------------------------------------------

            next_state, reward, done, info = env.step(action)

            move_count += 1

            # ---------------------------------------------
            # Store experience
            # ---------------------------------------------

            # Reward from the perspective of the player
            # who made the move.

            if done:

                winner = info["winner"]

                if winner == current_player:

                    player_reward = 1.0

                elif winner is None:

                    player_reward = 0.0

                else:

                    player_reward = -1.0

            else:

                player_reward = 0.0

            agent.remember(
                state,
                action,
                player_reward,
                next_state,
                done
            )

            # ---------------------------------------------
            # Learn
            # ---------------------------------------------

            agent.learn()

            # ---------------------------------------------
            # Next state
            # ---------------------------------------------

            state = next_state

        # -------------------------------------------------
        # Game statistics
        # -------------------------------------------------

        winner = info["winner"]

        if winner == env.X:

            x_wins += 1
            result = "X WIN"

        elif winner == env.O:

            o_wins += 1
            result = "O WIN"

        else:

            draws += 1
            result = "DRAW"

        # -------------------------------------------------
        # Print episode information
        # -------------------------------------------------

        print(
            f"Episode {episode:4d} | "
            f"Result: {result:6s} | "
            f"Moves: {move_count:2d} | "
            f"X wins: {x_wins:2d} | "
            f"O wins: {o_wins:2d} | "
            f"Draws: {draws:2d}"
        )

    # -----------------------------------------------------
    # Final statistics
    # -----------------------------------------------------

    print()
    print("=" * 50)
    print(" TRAINING COMPLETED")
    print("=" * 50)

    print("X wins :", x_wins)
    print("O wins :", o_wins)
    print("Draws  :", draws)

    print()
    print("Total games:", NUM_EPISODES)


# ---------------------------------------------------------
# Run
# ---------------------------------------------------------

if __name__ == "__main__":
    train()
    