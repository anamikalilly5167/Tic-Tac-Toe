import sys
import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

ENVIRONMENT_PATH = os.path.join(
    PROJECT_ROOT,
    "environment"
)

if ENVIRONMENT_PATH not in sys.path:
    sys.path.insert(0, ENVIRONMENT_PATH)

from game import UltimateTTTEnv
from ddqn_agent import DDQNAgent


MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

MODEL_X_PATH = os.path.join(
    MODEL_DIR,
    "agent_x_final.pth"
)

MODEL_O_PATH = os.path.join(
    MODEL_DIR,
    "agent_o_final.pth"
)


def main():

    print("=" * 60)
    print(" DDQN DEBUG EVALUATION")
    print("=" * 60)

    # ---------------------------------------------------------
    # Create agents
    # ---------------------------------------------------------

    agent_x = DDQNAgent(
        state_size=100,
        action_size=81,
        batch_size=32
    )

    agent_o = DDQNAgent(
        state_size=100,
        action_size=81,
        batch_size=32
    )

    # ---------------------------------------------------------
    # Load models
    # ---------------------------------------------------------

    agent_x.load(MODEL_X_PATH)
    agent_o.load(MODEL_O_PATH)

    # Completely deterministic
    agent_x.epsilon = 0.0
    agent_o.epsilon = 0.0

    print("\nModels loaded.")
    print("X epsilon:", agent_x.epsilon)
    print("O epsilon:", agent_o.epsilon)

    # ---------------------------------------------------------
    # Create game
    # ---------------------------------------------------------

    env = UltimateTTTEnv()

    state = env.reset()

    done = False
    move_number = 0

    print("\nInitial state:")
    print("State shape:", state.shape)
    print("Current player:", env.get_current_player())
    print("Valid moves:", len(env.get_valid_moves()))

    # ---------------------------------------------------------
    # Play one game
    # ---------------------------------------------------------

    while not done:

        move_number += 1

        current_player = env.get_current_player()
        valid_actions = env.get_valid_moves()

        if current_player == env.X:
            agent = agent_x
            player_name = "X"
        else:
            agent = agent_o
            player_name = "O"

        # Get Q-values directly
        import torch

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=agent.device
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = agent.online_network(
                state_tensor
            )[0]

        # Sort legal actions by Q-value
        legal_q_values = [
            (action, q_values[action].item())
            for action in valid_actions
        ]

        legal_q_values.sort(
            key=lambda x: x[1],
            reverse=True
        )

        action = agent.select_action(
            state,
            valid_actions,
            training=False
        )

        print()
        print("-" * 60)
        print(f"Move {move_number}")
        print(f"Player: {player_name}")
        print(f"Current player value: {current_player}")
        print(f"Number of valid moves: {len(valid_actions)}")
        print(f"Valid actions: {valid_actions}")
        print(f"Chosen action: {action}")

        print("\nTop 5 legal actions:")
        for a, q in legal_q_values[:5]:
            print(
                f"  Action {a:2d} -> Q = {q:.6f}"
            )

        # -----------------------------------------------------
        # Execute move
        # -----------------------------------------------------

        next_state, reward, done, info = env.step(
            action
        )

        print(f"\nReward: {reward}")
        print(f"Done: {done}")

        if done:
            print("\nGAME OVER")

            winner = info.get("winner")

            if winner == env.X:
                print("Winner: X")
            elif winner == env.O:
                print("Winner: O")
            else:
                print("Result: DRAW")

        state = next_state

    # ---------------------------------------------------------
    # Final information
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print(" FINAL GAME INFORMATION")
    print("=" * 60)

    print("Moves:", move_number)
    print("Winner:", env.get_game_info().get("winner"))
    print("Current player:", env.get_current_player())

    print()
    print("DEBUG TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()