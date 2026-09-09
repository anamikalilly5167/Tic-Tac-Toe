import os
import sys
import numpy as np
import torch

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ============================================================
# IMPORTS
# ============================================================

from environment.game import UltimateTTTEnv
from agent.ddqn_agent import DDQNAgent

# ============================================================
# CONSTANTS
# ============================================================

X = 1
O = -1

X_MODEL = os.path.join(
    PROJECT_ROOT,
    "models",
    "agent_x_final.pth"
)

O_MODEL = os.path.join(
    PROJECT_ROOT,
    "models",
    "agent_o_final.pth"
)

# ============================================================
# HELPERS
# ============================================================

def action_to_text(action):
    local_board = action // 9
    cell = action % 9

    return (
        f"action={action:2d} | "
        f"local board={local_board} | "
        f"cell={cell}"
    )


def inspect_action(agent, state, valid_actions):

    state_tensor = (
        torch.FloatTensor(state)
        .unsqueeze(0)
        .to(agent.device)
    )

    with torch.no_grad():

        q_values = (
            agent.online_network(state_tensor)
            .cpu()
            .numpy()[0]
        )

    valid_q = [
        (a, q_values[a])
        for a in valid_actions
    ]

    valid_q.sort(
        key=lambda x: x[1],
        reverse=True
    )

    chosen_action = valid_q[0][0]

    return chosen_action, valid_q


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DEBUG SELF-PLAY GAME")
    print("=" * 70)

    print("\nProject root:")
    print(PROJECT_ROOT)

    print("\nX model:")
    print(X_MODEL)

    print("\nO model:")
    print(O_MODEL)

    if not os.path.exists(X_MODEL):
        print("\nERROR: X model not found.")
        return

    if not os.path.exists(O_MODEL):
        print("\nERROR: O model not found.")
        return

    print("\nLoading models...")

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

    agent_x.load(X_MODEL)
    agent_o.load(O_MODEL)

    agent_x.epsilon = 0.0
    agent_o.epsilon = 0.0

    print("Models loaded.")

    env = UltimateTTTEnv()

    state = env.reset()

    done = False
    move_number = 0

    while not done:

        current_player = env.board.current_player

        if current_player == X:

            agent = agent_x
            player_name = "X"

        else:

            agent = agent_o
            player_name = "O"

        valid_actions = env.get_valid_moves()

        chosen_action, ranked_actions = inspect_action(
            agent,
            state,
            valid_actions
        )

        print("\n" + "-" * 70)
        print(f"Move {move_number + 1}")
        print(f"Player: {player_name}")
        print(f"Active board: {env.board.active_board}")

        print("\nTop Q-values:")

        for action, q_value in ranked_actions[:5]:

            print(
                f"{action_to_text(action)} "
                f"| Q={q_value:.8f}"
            )

        print(
            f"\nCHOSEN: "
            f"{action_to_text(chosen_action)}"
        )

        next_state, reward, done, info = env.step(
            chosen_action
        )

        print(f"Reward: {reward}")
        print(f"Done: {done}")

        if done:

            winner = info.get("winner")

            if winner == X:
                print("WINNER: X")

            elif winner == O:
                print("WINNER: O")

            else:
                print("RESULT: DRAW")

        state = next_state

        move_number += 1

        if move_number > 100:

            print(
                "ERROR: Game exceeded 100 moves."
            )

            break

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print(f"Moves: {move_number}")
    print(f"Winner: {info.get('winner')}")

    print(
        f"Local status: "
        f"{env.board.local_status}"
    )

    print(
        f"Active board: "
        f"{env.board.active_board}"
    )

    print("\nFinal board:")

    for i in range(9):

        print(
            f"Local board {i}: "
            f"{env.board.board[i].tolist()}"
        )


if __name__ == "__main__":
    main()