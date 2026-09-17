import sys
import os
import random


# ============================================================
# MAKE ENVIRONMENT IMPORTABLE
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
# EVALUATION SETTINGS
# ============================================================

NUM_GAMES = 5000


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
# LOAD TRAINED AGENT
# ============================================================

def load_agent(model_path):
    """
    Create a DDQN agent and load a trained model.

    Evaluation is performed with epsilon = 0,
    so the agent always chooses greedily.
    """

    agent = DDQNAgent(
        state_size=100,
        action_size=81,
        batch_size=32
    )

    agent.load(model_path)

    # --------------------------------------------------------
    # Disable exploration completely
    # --------------------------------------------------------

    agent.epsilon = 0.0

    return agent


# ============================================================
# PLAY ONE GAME
# ============================================================

def play_game(
    trained_player,
    trained_agent=None,
    render=False
):
    """
    Play one game between a trained DDQN agent
    and a random opponent.

    Parameters
    ----------
    trained_player:
        X or O

    trained_agent:
        DDQN agent controlling trained_player

    render:
        If True, print every move
    """

    env = UltimateTTTEnv()

    state = env.reset()

    done = False
    move_count = 0

    last_info = {}

    while not done:

        # ====================================================
        # CURRENT PLAYER
        # ====================================================

        current_player = env.get_current_player()

        # ====================================================
        # VALID ACTIONS
        # ====================================================

        valid_actions = env.get_valid_moves()

        if len(valid_actions) == 0:

            print(
                "ERROR: No valid actions available."
            )

            break

        # ====================================================
        # CHOOSE ACTION
        # ====================================================

        if current_player == trained_player:

            # ------------------------------------------------
            # DDQN AGENT
            # ------------------------------------------------

            action = trained_agent.select_action(
                state,
                valid_actions,
                training=False
            )

            player_type = "DDQN"

        else:

            # ------------------------------------------------
            # RANDOM AGENT
            # ------------------------------------------------

            action = random.choice(
                valid_actions
            )

            player_type = "RANDOM"

        # ====================================================
        # ENVIRONMENT STEP
        # ====================================================

        next_state, reward, done, info = env.step(
            action
        )

        last_info = info

        move_count += 1

        # ====================================================
        # OPTIONAL RENDERING
        # ====================================================

        if render:

            player_name = (
                "X"
                if current_player == X
                else "O"
            )

            print()

            print(
                f"Move {move_count}: "
                f"Player {player_name} "
                f"({player_type}) "
                f"played action {action}"
            )

            print(
                f"Reward: {reward}"
            )

            env.render()

        # ====================================================
        # UPDATE STATE
        # ====================================================

        state = next_state

    # ========================================================
    # GET FINAL RESULT
    # ========================================================

    game_info = env.get_game_info()

    winner = game_info.get(
        "winner",
        last_info.get("winner")
    )

    return {
        "winner": winner,
        "moves": move_count,
        "info": game_info
    }


# ============================================================
# EVALUATE TRAINED AGENT
# ============================================================

def evaluate(
    trained_player,
    trained_agent,
    num_games=NUM_GAMES,
    render_first_game=False
):
    """
    Evaluate one trained DDQN agent against
    a random opponent.
    """

    trained_wins = 0
    random_wins = 0
    draws = 0

    total_moves = 0

    shortest_game = None
    longest_game = None

    # ========================================================
    # PLAY GAMES
    # ========================================================

    for game_number in range(
        1,
        num_games + 1
    ):

        result = play_game(
            trained_player,
            trained_agent,
            render=(
                render_first_game
                and game_number == 1
            )
        )

        winner = result["winner"]
        moves = result["moves"]

        total_moves += moves

        # ====================================================
        # TRACK SHORTEST GAME
        # ====================================================

        if shortest_game is None:

            shortest_game = moves

        else:

            shortest_game = min(
                shortest_game,
                moves
            )

        # ====================================================
        # TRACK LONGEST GAME
        # ====================================================

        if longest_game is None:

            longest_game = moves

        else:

            longest_game = max(
                longest_game,
                moves
            )

        # ====================================================
        # DETERMINE WINNER
        # ====================================================

        if winner == trained_player:

            trained_wins += 1

        elif winner in (X, O):

            random_wins += 1

        else:

            draws += 1

    # ========================================================
    # CALCULATE STATISTICS
    # ========================================================

    trained_win_rate = (
        trained_wins / num_games
    ) * 100

    random_win_rate = (
        random_wins / num_games
    ) * 100

    draw_rate = (
        draws / num_games
    ) * 100

    average_moves = (
        total_moves / num_games
    )

    # ========================================================
    # PLAYER NAME
    # ========================================================

    player_name = (
        "X"
        if trained_player == X
        else "O"
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()

    print("=" * 60)

    print(
        f" DDQN {player_name} VS RANDOM"
    )

    print("=" * 60)

    print()

    print(
        f"Total games: {num_games}"
    )

    print()

    print(
        f"DDQN {player_name} wins: "
        f"{trained_wins}"
    )

    print(
        f"Random opponent wins: "
        f"{random_wins}"
    )

    print(
        f"Draws: "
        f"{draws}"
    )

    print()

    print(
        f"DDQN {player_name} win rate: "
        f"{trained_win_rate:.2f}%"
    )

    print(
        f"Random opponent win rate: "
        f"{random_win_rate:.2f}%"
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
        f"{shortest_game} moves"
    )

    print(
        f"Longest game: "
        f"{longest_game} moves"
    )

    print()

    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {
        "trained_player": trained_player,
        "trained_wins": trained_wins,
        "random_wins": random_wins,
        "draws": draws,
        "trained_win_rate": trained_win_rate,
        "random_win_rate": random_win_rate,
        "draw_rate": draw_rate,
        "average_moves": average_moves,
        "shortest_game": shortest_game,
        "longest_game": longest_game
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()

    print("=" * 60)

    print(
        " DDQN ULTIMATE TIC-TAC-TOE"
    )

    print(
        " EVALUATION AGAINST RANDOM"
    )

    print("=" * 60)

    print()

    # ========================================================
    # CHECK MODEL FILES
    # ========================================================

    if not os.path.exists(MODEL_X_PATH):

        print(
            "ERROR: X model not found:"
        )

        print(
            MODEL_X_PATH
        )

        sys.exit(1)

    if not os.path.exists(MODEL_O_PATH):

        print(
            "ERROR: O model not found:"
        )

        print(
            MODEL_O_PATH
        )

        sys.exit(1)

    # ========================================================
    # LOAD X MODEL
    # ========================================================

    print(
        "Loading X model..."
    )

    agent_x = load_agent(
        MODEL_X_PATH
    )

    print(
        "X model loaded."
    )

    print()

    # ========================================================
    # LOAD O MODEL
    # ========================================================

    print(
        "Loading O model..."
    )

    agent_o = load_agent(
        MODEL_O_PATH
    )

    print(
        "O model loaded."
    )

    print()

    # ========================================================
    # EVALUATION SETTINGS
    # ========================================================

    print(
        "Evaluation settings:"
    )

    print(
        f"  Number of games : {NUM_GAMES}"
    )

    print(
        "  DDQN epsilon    : 0.0"
    )

    print(
        "  Learning        : disabled"
    )

    print(
        "  Random opponent : yes"
    )

    # ========================================================
    # TEST 1
    # ========================================================

    print()

    print(
        "TEST 1:"
    )

    print(
        "Trained X vs Random O"
    )

    results_x = evaluate(
        trained_player=X,
        trained_agent=agent_x,
        num_games=NUM_GAMES,
        render_first_game=False
    )

    # ========================================================
    # TEST 2
    # ========================================================

    print()

    print(
        "TEST 2:"
    )

    print(
        "Random X vs Trained O"
    )

    results_o = evaluate(
        trained_player=O,
        trained_agent=agent_o,
        num_games=NUM_GAMES,
        render_first_game=False
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()

    print()

    print("=" * 60)

    print(
        " FINAL EVALUATION SUMMARY"
    )

    print("=" * 60)

    print()

    # ========================================================
    # TEST 1 SUMMARY
    # ========================================================

    print(
        "Test 1: DDQN X vs Random O"
    )

    print(
        f"  X wins: "
        f"{results_x['trained_wins']}/{NUM_GAMES}"
    )

    print(
        f"  O wins: "
        f"{results_x['random_wins']}/{NUM_GAMES}"
    )

    print(
        f"  Draws: "
        f"{results_x['draws']}/{NUM_GAMES}"
    )

    print(
        f"  X win rate: "
        f"{results_x['trained_win_rate']:.2f}%"
    )

    print(
        f"  O win rate: "
        f"{results_x['random_win_rate']:.2f}%"
    )

    print(
        f"  Draw rate: "
        f"{results_x['draw_rate']:.2f}%"
    )

    print()

    # ========================================================
    # TEST 2 SUMMARY
    # ========================================================

    print(
        "Test 2: Random X vs DDQN O"
    )

    print(
        f"  O wins: "
        f"{results_o['trained_wins']}/{NUM_GAMES}"
    )

    print(
        f"  X wins: "
        f"{results_o['random_wins']}/{NUM_GAMES}"
    )

    print(
        f"  Draws: "
        f"{results_o['draws']}/{NUM_GAMES}"
    )

    print(
        f"  O win rate: "
        f"{results_o['trained_win_rate']:.2f}%"
    )

    print(
        f"  X win rate: "
        f"{results_o['random_win_rate']:.2f}%"
    )

    print(
        f"  Draw rate: "
        f"{results_o['draw_rate']:.2f}%"
    )

    print()

    # ========================================================
    # EVALUATION COMPLETE
    # ========================================================

    print("=" * 60)

    print(
        " EVALUATION COMPLETED"
    )

    print("=" * 60)

    print()

    print(
        "No learning occurred."
    )

    print(
        "Replay buffers were not modified."
    )

    print(
        "DDQN agents used epsilon = 0."
    )

    print()