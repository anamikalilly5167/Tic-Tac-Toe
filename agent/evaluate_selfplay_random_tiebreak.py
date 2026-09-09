import sys
import os
import random
import numpy as np
import torch


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

NUM_GAMES = 100

# Actions whose Q-value is within this amount of the
# maximum Q-value are considered tied.
TIE_TOLERANCE = 0.10


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
# LOAD AGENT
# ============================================================

def load_agent(model_path):
    """
    Load a trained DDQN agent.

    Epsilon is set to zero because we do NOT want
    ordinary epsilon-random exploration.

    Randomness is introduced only when multiple legal
    actions have nearly identical Q-values.
    """

    agent = DDQNAgent(
        state_size=100,
        action_size=81,
        batch_size=32
    )

    agent.load(model_path)

    # Disable epsilon exploration
    agent.epsilon = 0.0

    return agent


# ============================================================
# SELECT ACTION WITH RANDOM TIE-BREAKING
# ============================================================

def select_action_random_tiebreak(
    agent,
    state,
    valid_actions,
    tolerance=TIE_TOLERANCE
):
    """
    Select action using greedy Q-values
    with random tie-breaking.

    DEBUG VERSION.
    """

    if not valid_actions:
        raise ValueError(
            "valid_actions cannot be empty."
        )

    state_tensor = torch.FloatTensor(
        np.asarray(state, dtype=np.float32)
    ).unsqueeze(0).to(agent.device)

    with torch.no_grad():

        q_values = agent.online_network(
            state_tensor
        )

    q_values = q_values.squeeze(0).cpu().numpy()

    legal_q_values = np.array(
        [
            q_values[action]
            for action in valid_actions
        ],
        dtype=np.float32
    )

    max_q = np.max(
        legal_q_values
    )

    tied_actions = [
        action
        for action, q_value in zip(
            valid_actions,
            legal_q_values
        )
        if q_value >= max_q - tolerance
    ]

    # --------------------------------------------------
    # DEBUG OUTPUT
    # --------------------------------------------------

    ranked = sorted(
        [
            (a, q_values[a])
            for a in valid_actions
        ],
        key=lambda x: x[1],
        reverse=True
    )

    print("\nTop legal actions:")

    for action, q in ranked[:5]:

        board = action // 9
        cell = action % 9

        print(
            f"  action={action:2d} "
            f"(board={board}, cell={cell}) "
            f"Q={q:.8f}"
        )

    print(
        f"Best Q-value: {max_q:.8f}"
    )

    print(
        f"Tied actions ({len(tied_actions)}): "
        f"{tied_actions}"
    )

    chosen_action = random.choice(
        tied_actions
    )

    print(
        f"Chosen action: {chosen_action}"
    )

    return chosen_action

# ============================================================
# PLAY ONE GAME
# ============================================================

def play_one_game(
    agent_x,
    agent_o,
    render=False
):
    """
    Play one complete game between the two trained agents.

    Both agents use greedy Q-values with random
    tie-breaking.
    """

    env = UltimateTTTEnv()

    state = env.reset()

    done = False
    move_count = 0

    last_info = {}

    while not done:

        # ----------------------------------------------------
        # Current player
        # ----------------------------------------------------

        current_player = env.get_current_player()

        # ----------------------------------------------------
        # Legal actions
        # ----------------------------------------------------

        valid_actions = env.get_valid_moves()

        if not valid_actions:

            print(
                "ERROR: No valid actions available."
            )

            break

        # ----------------------------------------------------
        # Select appropriate agent
        # ----------------------------------------------------

        if current_player == X:

            agent = agent_x
            player_name = "X"

        else:

            agent = agent_o
            player_name = "O"

        # ----------------------------------------------------
        # Select action using random tie-breaking
        # ----------------------------------------------------

        action = select_action_random_tiebreak(
            agent,
            state,
            valid_actions
        )

        # ----------------------------------------------------
        # Environment step
        # ----------------------------------------------------

        next_state, reward, done, info = env.step(
            action
        )

        last_info = info

        move_count += 1

        # ----------------------------------------------------
        # Optional rendering
        # ----------------------------------------------------

        if render:

            print()
            print(
                f"Move {move_count}: "
                f"Player {player_name} "
                f"played action {action}"
            )

            print(
                f"Q-selected from {len(valid_actions)} "
                f"legal actions"
            )

            print(
                f"Reward: {reward}"
            )

            env.render()

        # ----------------------------------------------------
        # Update state
        # ----------------------------------------------------

        state = next_state

    # ========================================================
    # GAME RESULT
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
# RUN EVALUATION
# ============================================================

def evaluate(
    num_games=NUM_GAMES,
    render_first_game=False
):
    """
    Evaluate trained X vs trained O using
    random tie-breaking.
    """

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    print()
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

    print(
        "Loading O model..."
    )

    agent_o = load_agent(
        MODEL_O_PATH
    )

    print(
        "O model loaded."
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    x_wins = 0
    o_wins = 0
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

        result = play_one_game(
            agent_x,
            agent_o,
            render=(
                render_first_game
                and game_number == 1
            )
        )

        winner = result["winner"]
        moves = result["moves"]

        # ----------------------------------------------------
        # Winner statistics
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

        if shortest_game is None:

            shortest_game = moves

        else:

            shortest_game = min(
                shortest_game,
                moves
            )

        if longest_game is None:

            longest_game = moves

        else:

            longest_game = max(
                longest_game,
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

            print(
                f"Game {game_number}/{num_games} | "
                f"X: {x_wins} | "
                f"O: {o_wins} | "
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

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print()
    print("=" * 60)
    print(" SELF-PLAY WITH RANDOM TIE-BREAKING")
    print("=" * 60)

    print()

    print(
        f"Total games: {num_games}"
    )

    print()

    print(
        f"X wins: {x_wins}"
    )

    print(
        f"O wins: {o_wins}"
    )

    print(
        f"Draws: {draws}"
    )

    print()

    print(
        f"X win rate: {x_win_rate:.2f}%"
    )

    print(
        f"O win rate: {o_win_rate:.2f}%"
    )

    print(
        f"Draw rate: {draw_rate:.2f}%"
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

    print("=" * 60)
    print(" EVALUATION COMPLETED")
    print("=" * 60)

    print()

    print(
        "No learning occurred."
    )

    print(
        "Replay buffers were not modified."
    )

    print(
        "Epsilon was set to 0."
    )

    print(
        f"Random tie tolerance: "
        f"{TIE_TOLERANCE}"
    )

    print()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print(" DDQN ULTIMATE TIC-TAC-TOE")
    print(" SELF-PLAY RANDOM TIE-BREAK TEST")
    print("=" * 60)

    print()

    # --------------------------------------------------------
    # Check model files
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Evaluation configuration
    # --------------------------------------------------------

    print(
        f"Number of games: {NUM_GAMES}"
    )

    print(
        "X agent: trained DDQN"
    )

    print(
        "O agent: trained DDQN"
    )

    print(
        "Epsilon: 0.0"
    )

    print(
        "Learning: disabled"
    )

    print(
        "Action selection: greedy + random tie-breaking"
    )

    print(
        f"Tie tolerance: {TIE_TOLERANCE}"
    )

    # --------------------------------------------------------
    # Run evaluation
    # --------------------------------------------------------

    evaluate(
    num_games=100,
    render_first_game=True
)