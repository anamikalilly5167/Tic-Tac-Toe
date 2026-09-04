import sys
import os
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

ENVIRONMENT_PATH = os.path.join(
    PROJECT_ROOT,
    "environment"
)

if ENVIRONMENT_PATH not in sys.path:
    sys.path.insert(0, ENVIRONMENT_PATH)


# ============================================================
# IMPORTS
# ============================================================

from game import UltimateTTTEnv
from ddqn_agent import DDQNAgent


# ============================================================
# CONSTANTS
# ============================================================

X = 1
O = -1


# ============================================================
# MODEL PATHS
# ============================================================

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


# ============================================================
# EVALUATE ONE GAME
# ============================================================

def play_evaluation_game(
    agent_x,
    agent_o,
    render=False
):
    """
    Play one game between the trained X and O agents.

    IMPORTANT:
        - No learning occurs.
        - No replay buffer is modified.
        - training=False is used.
        - Therefore epsilon is not used for random exploration.

    Parameters
    ----------
    agent_x : DDQNAgent
        Trained X agent.

    agent_o : DDQNAgent
        Trained O agent.

    render : bool
        If True, print the game in the terminal.

    Returns
    -------
    dict
        Game result.
    """

    env = UltimateTTTEnv()

    # --------------------------------------------------------
    # Reset environment
    # --------------------------------------------------------

    state = env.reset()

    done = False
    move_count = 0

    last_info = {}

    # ========================================================
    # GAME LOOP
    # ========================================================

    while not done:

        # ----------------------------------------------------
        # Current player
        # ----------------------------------------------------

        current_player = env.get_current_player()

        # ----------------------------------------------------
        # Get legal actions
        # ----------------------------------------------------

        valid_actions = env.get_valid_moves()

        if len(valid_actions) == 0:

            print(
                "ERROR: No valid actions available."
            )

            break

        # ----------------------------------------------------
        # Select correct agent
        # ----------------------------------------------------

        if current_player == env.X:

            agent = agent_x

            player_name = "X"

        else:

            agent = agent_o

            player_name = "O"

        # ====================================================
        # SELECT ACTION
        # ====================================================

        action = agent.select_action(
            state,
            valid_actions,
            training=False
        )

        # ====================================================
        # ENVIRONMENT STEP
        # ====================================================

        next_state, reward, done, info = env.step(
            action
        )

        last_info = info

        move_count += 1

        # ====================================================
        # OPTIONAL TERMINAL RENDER
        # ====================================================

        if render:

            print()
            print(
                "------------------------------------------------"
            )

            print(
                f"Move {move_count}: "
                f"Player {player_name} "
                f"played action {action}"
            )

            print(
                f"Reward: {reward}"
            )

            env.render()

            if done:

                winner = info.get("winner")

                if winner == X:

                    print("Winner: X")

                elif winner == O:

                    print("Winner: O")

                else:

                    print("Result: DRAW")

        # ----------------------------------------------------
        # Update state
        # ----------------------------------------------------

        state = next_state

    # ========================================================
    # GET FINAL GAME INFORMATION
    # ========================================================

    game_info = env.get_game_info()

    winner = game_info.get(
        "winner",
        last_info.get("winner")
    )

    # ========================================================
    # RETURN
    # ========================================================

    return {
        "winner": winner,
        "moves": move_count,
        "info": game_info
    }


# ============================================================
# EVALUATION
# ============================================================

def evaluate_agents(
    num_games=100,
    render_game=False
):
    """
    Evaluate the trained X and O agents.

    No training takes place.

    Parameters
    ----------
    num_games : int
        Number of evaluation games.

    render_game : bool
        If True, render the final game in the terminal.
    """

    # ========================================================
    # CHECK MODEL FILES
    # ========================================================

    print()
    print("=" * 60)
    print(" DDQN MODEL EVALUATION")
    print("=" * 60)

    print()

    print("Checking model files...")

    if not os.path.exists(MODEL_X_PATH):

        print()
        print(
            "ERROR: X model not found:"
        )

        print(
            MODEL_X_PATH
        )

        return

    if not os.path.exists(MODEL_O_PATH):

        print()
        print(
            "ERROR: O model not found:"
        )

        print(
            MODEL_O_PATH
        )

        return

    print("X model found.")

    print("O model found.")

    # ========================================================
    # CREATE AGENTS
    # ========================================================

    print()
    print("Creating evaluation agents...")

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

    # ========================================================
    # LOAD MODELS
    # ========================================================

    print()
    print("Loading trained models...")

    try:

        agent_x.load(
            MODEL_X_PATH
        )

        agent_o.load(
            MODEL_O_PATH
        )

    except Exception as error:

        print()
        print(
            "ERROR while loading models:"
        )

        print(
            error
        )

        return

    print()
    print("Models loaded successfully.")

    print()

    print(
        "Agent X device:",
        agent_x.device
    )

    print(
        "Agent O device:",
        agent_o.device
    )

    # ========================================================
    # FORCE EVALUATION MODE
    # ========================================================

    # We set epsilon to zero so there is no random
    # exploration during evaluation.

    agent_x.epsilon = 0.0
    agent_o.epsilon = 0.0

    # ========================================================
    # STATISTICS
    # ========================================================

    x_wins = 0
    o_wins = 0
    draws = 0

    total_moves = 0

    game_lengths = []

    # ========================================================
    # EVALUATION LOOP
    # ========================================================

    print()
    print("=" * 60)

    print(
        f"Evaluating {num_games} games..."
    )

    print(
        "Training disabled."
    )

    print(
        "Exploration disabled (epsilon = 0)."
    )

    print("=" * 60)

    for game_number in range(
        1,
        num_games + 1
    ):

        # ----------------------------------------------------
        # Play game
        # ----------------------------------------------------

        result = play_evaluation_game(
            agent_x,
            agent_o,
            render=(
                render_game
                and game_number == num_games
            )
        )

        winner = result["winner"]

        moves = result["moves"]

        # ----------------------------------------------------
        # Count result
        # ----------------------------------------------------

        if winner == X:

            x_wins += 1

        elif winner == O:

            o_wins += 1

        else:

            draws += 1

        # ----------------------------------------------------
        # Move statistics
        # ----------------------------------------------------

        total_moves += moves

        game_lengths.append(
            moves
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            game_number % 10 == 0
            or game_number == 1
            or game_number == num_games
        ):

            print()
            print(
                f"Games evaluated: "
                f"{game_number}/{num_games}"
            )

            print(
                f"X wins: {x_wins}"
            )

            print(
                f"O wins: {o_wins}"
            )

            print(
                f"Draws: {draws}"
            )

    # ========================================================
    # CALCULATE STATISTICS
    # ========================================================

    x_win_rate = (
        x_wins / num_games
    ) * 100

    o_win_rate = (
        o_wins / num_games
    ) * 100

    draw_rate = (
        draws / num_games
    ) * 100

    average_moves = (
        total_moves / num_games
    )

    minimum_moves = min(
        game_lengths
    )

    maximum_moves = max(
        game_lengths
    )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print()
    print()
    print("=" * 60)
    print(" EVALUATION COMPLETED")
    print("=" * 60)

    print()

    print(
        f"Total evaluation games: "
        f"{num_games}"
    )

    print()

    print(
        f"X wins: "
        f"{x_wins}"
    )

    print(
        f"O wins: "
        f"{o_wins}"
    )

    print(
        f"Draws: "
        f"{draws}"
    )

    print()

    print(
        f"X win rate: "
        f"{x_win_rate:.2f}%"
    )

    print(
        f"O win rate: "
        f"{o_win_rate:.2f}%"
    )

    print(
        f"Draw rate: "
        f"{draw_rate:.2f}%"
    )

    print()

    print(
        f"Average moves/game: "
        f"{average_moves:.2f}"
    )

    print(
        f"Shortest game: "
        f"{minimum_moves} moves"
    )

    print(
        f"Longest game: "
        f"{maximum_moves} moves"
    )

    print()

    print(
        "Evaluation epsilon X: "
        f"{agent_x.epsilon:.2f}"
    )

    print(
        "Evaluation epsilon O: "
        f"{agent_o.epsilon:.2f}"
    )

    print()

    print("=" * 60)
    print(" IMPORTANT")
    print("=" * 60)

    print()
    print(
        "No learning occurred during evaluation."
    )

    print(
        "Replay buffers were not modified."
    )

    print(
        "Both agents selected actions greedily."
    )

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Evaluate 100 games
    #
    # render_game=False means the games are NOT printed
    # in the terminal.
    # --------------------------------------------------------

    evaluate_agents(
        num_games=100,
        render_game=False
    )