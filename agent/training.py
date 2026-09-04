import sys
import os
import numpy as np

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
# PLAY ONE SELF-PLAY GAME
# ============================================================

def play_one_game(
    agent_x,
    agent_o,
    render=False
):
    """
    Play one complete Ultimate Tic-Tac-Toe game.

    Agent X controls X.
    Agent O controls O.

    Each agent stores transitions from one of its
    decision points to its next decision point.
    """

    env = UltimateTTTEnv()

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    state = env.reset()

    done = False
    move_count = 0

    last_info = {}

    losses_x = []
    losses_o = []

    # --------------------------------------------------------
    # Pending transitions
    # --------------------------------------------------------

    pending_x = None
    pending_o = None

    # ========================================================
    # GAME LOOP
    # ========================================================

    while not done:

        # ----------------------------------------------------
        # Current player
        # ----------------------------------------------------

        current_player = env.get_current_player()

        # ----------------------------------------------------
        # Legal actions
        # ----------------------------------------------------

        valid_actions = env.get_valid_moves()

        if len(valid_actions) == 0:

            print(
                "ERROR: No valid actions available."
            )

            break

        # ----------------------------------------------------
        # Select agent
        # ----------------------------------------------------

        if current_player == env.X:

            agent = agent_x
            pending = pending_x

        else:

            agent = agent_o
            pending = pending_o

        # ====================================================
        # FINALIZE PREVIOUS TRANSITION
        # ====================================================

        if pending is not None:

            old_state, old_action = pending

            agent.remember(
                old_state,
                old_action,
                0.0,
                state.copy(),
                False,
                list(valid_actions)
            )

            # Clear pending transition

            if current_player == env.X:

                pending_x = None

            else:

                pending_o = None

            # ------------------------------------------------
            # Learn
            # ------------------------------------------------

            loss = agent.learn()

            if loss is not None:

                if current_player == env.X:

                    losses_x.append(loss)

                else:

                    losses_o.append(loss)

        # ====================================================
        # SELECT ACTION
        # ====================================================

        action = agent.select_action(
            state,
            valid_actions,
            training=True
        )

        player_who_moved = current_player

        # ====================================================
        # ENVIRONMENT STEP
        # ====================================================

        next_state, env_reward, done, info = env.step(
            action
        )

        last_info = info

        # ====================================================
        # GAME ENDED
        # ====================================================

        if done:

            winner = info.get("winner")

            # ------------------------------------------------
            # Determine terminal rewards
            # ------------------------------------------------

            if winner == player_who_moved:

                current_reward = 1.0
                opponent_reward = -1.0

            elif winner in (env.X, env.O):

                current_reward = -1.0
                opponent_reward = 1.0

            elif info.get("illegal_move", False):

                current_reward = -1.0
                opponent_reward = 1.0

            else:

                current_reward = 0.0
                opponent_reward = 0.0

            # =================================================
            # CURRENT PLAYER FINAL TRANSITION
            # =================================================

            agent.remember(
                state.copy(),
                action,
                current_reward,
                next_state.copy(),
                True,
                []
            )

            # =================================================
            # OPPONENT FINAL TRANSITION
            # =================================================

            if player_who_moved == env.X:

                if pending_o is not None:

                    old_state, old_action = pending_o

                    agent_o.remember(
                        old_state,
                        old_action,
                        opponent_reward,
                        next_state.copy(),
                        True,
                        []
                    )

                    pending_o = None

            else:

                if pending_x is not None:

                    old_state, old_action = pending_x

                    agent_x.remember(
                        old_state,
                        old_action,
                        opponent_reward,
                        next_state.copy(),
                        True,
                        []
                    )

                    pending_x = None

            # =================================================
            # LEARN FROM CURRENT PLAYER
            # =================================================

            loss = agent.learn()

            if loss is not None:

                if player_who_moved == env.X:

                    losses_x.append(loss)

                else:

                    losses_o.append(loss)

            # =================================================
            # LEARN FROM OPPONENT
            # =================================================

            if player_who_moved == env.X:

                loss_o = agent_o.learn()

                if loss_o is not None:

                    losses_o.append(loss_o)

            else:

                loss_x = agent_x.learn()

                if loss_x is not None:

                    losses_x.append(loss_x)

        # ====================================================
        # GAME CONTINUES
        # ====================================================

        else:

            if player_who_moved == env.X:

                pending_x = (
                    state.copy(),
                    action
                )

            else:

                pending_o = (
                    state.copy(),
                    action
                )

        # ====================================================
        # UPDATE STATE
        # ====================================================

        state = next_state

        move_count += 1

        # ====================================================
        # OPTIONAL RENDERING
        # ====================================================

        if render:

            player_name = (
                "X"
                if player_who_moved == env.X
                else "O"
            )

            print()

            print(
                f"Move {move_count}: "
                f"Player {player_name} "
                f"played action {action}"
            )

            print(
                f"Environment reward: "
                f"{env_reward}"
            )

            if done:

                print(
                    f"Terminal reward for "
                    f"{player_name}: "
                    f"{current_reward}"
                )

            env.render()

    # ========================================================
    # GAME INFORMATION
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

        "info": game_info,

        "avg_loss_x": (
            float(np.mean(losses_x))
            if losses_x
            else None
        ),

        "avg_loss_o": (
            float(np.mean(losses_o))
            if losses_o
            else None
        ),

        "x_losses": losses_x,

        "o_losses": losses_o
    }


# ============================================================
# CONTINUE TRAINING FROM EXISTING MODELS
# ============================================================

def continue_training(
    total_games=1000,
    previous_games=100,
    save_every=100
):
    """
    Continue training from the previously saved models.

    Example:
        previous_games = 100
        total_games = 1000

    This will play 900 additional games.
    """

    additional_games = (
        total_games - previous_games
    )

    if additional_games <= 0:

        print(
            "ERROR: total_games must be greater "
            "than previous_games."
        )

        return None, None

    # ========================================================
    # MODEL PATHS
    # ========================================================

    model_dir = os.path.join(
        PROJECT_ROOT,
        "models"
    )

    model_x_path = os.path.join(
        model_dir,
        "agent_x_final.pth"
    )

    model_o_path = os.path.join(
        model_dir,
        "agent_o_final.pth"
    )

    # ========================================================
    # CHECK MODELS
    # ========================================================

    if not os.path.exists(model_x_path):

        print()
        print(
            "ERROR: Agent X model not found:"
        )

        print(
            model_x_path
        )

        return None, None

    if not os.path.exists(model_o_path):

        print()
        print(
            "ERROR: Agent O model not found:"
        )

        print(
            model_o_path
        )

        return None, None

    # ========================================================
    # HEADER
    # ========================================================

    print()
    print("=" * 60)
    print(" CONTINUING DDQN TRAINING")
    print("=" * 60)

    print()

    print(
        f"Previous games completed : "
        f"{previous_games}"
    )

    print(
        f"Target total games      : "
        f"{total_games}"
    )

    print(
        f"Additional games        : "
        f"{additional_games}"
    )

    print()

    print(
        "Loading existing models..."
    )

    # ========================================================
    # CREATE AGENTS
    # ========================================================

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
    # LOAD EXISTING MODELS
    # ========================================================

    try:

        agent_x.load(
            model_x_path
        )

        agent_o.load(
            model_o_path
        )

    except Exception as error:

        print()
        print(
            "ERROR while loading models:"
        )

        print(
            error
        )

        return None, None

    # ========================================================
    # PRINT LOADED INFORMATION
    # ========================================================

    print()

    print(
        "Agent X loaded successfully."
    )

    print(
        "Agent O loaded successfully."
    )

    print()

    print(
        "Agent X device:",
        agent_x.device
    )

    print(
        "Agent O device:",
        agent_o.device
    )

    print()

    print(
        f"Loaded X epsilon: "
        f"{agent_x.epsilon:.4f}"
    )

    print(
        f"Loaded O epsilon: "
        f"{agent_o.epsilon:.4f}"
    )

    print()

    print(
        f"Loaded X training steps: "
        f"{agent_x.training_steps}"
    )

    print(
        f"Loaded O training steps: "
        f"{agent_o.training_steps}"
    )

    # ========================================================
    # TRAINING STATISTICS
    # ========================================================

    x_wins = 0
    o_wins = 0
    draws = 0

    total_moves = 0

    all_x_losses = []
    all_o_losses = []

    # ========================================================
    # CONTINUED TRAINING LOOP
    # ========================================================

    for game_index in range(
        1,
        additional_games + 1
    ):

        # ----------------------------------------------------
        # Play one game
        # ----------------------------------------------------

        result = play_one_game(
            agent_x,
            agent_o,
            render=False
        )

        # ----------------------------------------------------
        # Winner
        # ----------------------------------------------------

        winner = result["winner"]

        if winner == X:

            x_wins += 1

        elif winner == O:

            o_wins += 1

        else:

            draws += 1

        # ----------------------------------------------------
        # Moves
        # ----------------------------------------------------

        total_moves += result["moves"]

        # ----------------------------------------------------
        # Losses
        # ----------------------------------------------------

        if result["x_losses"]:

            all_x_losses.extend(
                result["x_losses"]
            )

        if result["o_losses"]:

            all_o_losses.extend(
                result["o_losses"]
            )

        # ====================================================
        # EPSILON DECAY
        # ====================================================

        agent_x.decay_epsilon()
        agent_o.decay_epsilon()

        # ====================================================
        # CURRENT TOTAL GAME NUMBER
        # ====================================================

        current_total_game = (
            previous_games
            + game_index
        )

        # ====================================================
        # PROGRESS EVERY 50 GAMES
        # ====================================================

        if (
            game_index % 50 == 0
            or game_index == 1
            or game_index == additional_games
        ):

            average_moves = (
                total_moves / game_index
            )

            if all_x_losses:

                average_x_loss = float(
                    np.mean(all_x_losses)
                )

            else:

                average_x_loss = None

            if all_o_losses:

                average_o_loss = float(
                    np.mean(all_o_losses)
                )

            else:

                average_o_loss = None

            print()
            print("-" * 60)

            print(
                f"Training progress: "
                f"{current_total_game}/{total_games}"
            )

            print()

            print(
                f"Additional games played: "
                f"{game_index}"
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

            print(
                f"Average moves/game: "
                f"{average_moves:.2f}"
            )

            print()

            print(
                f"Epsilon X: "
                f"{agent_x.epsilon:.4f}"
            )

            print(
                f"Epsilon O: "
                f"{agent_o.epsilon:.4f}"
            )

            print()

            print(
                f"Average X loss: "
                f"{average_x_loss}"
            )

            print(
                f"Average O loss: "
                f"{average_o_loss}"
            )

            print()

            print(
                f"Replay buffer X: "
                f"{len(agent_x.replay_buffer)}"
            )

            print(
                f"Replay buffer O: "
                f"{len(agent_o.replay_buffer)}"
            )

            print(
                f"Training steps X: "
                f"{agent_x.training_steps}"
            )

            print(
                f"Training steps O: "
                f"{agent_o.training_steps}"
            )

        # ====================================================
        # SAVE CHECKPOINT
        # ====================================================

        if (
            current_total_game % save_every == 0
        ):

            checkpoint_x = os.path.join(
                model_dir,
                f"agent_x_game_{current_total_game}.pth"
            )

            checkpoint_o = os.path.join(
                model_dir,
                f"agent_o_game_{current_total_game}.pth"
            )

            agent_x.save(
                checkpoint_x
            )

            agent_o.save(
                checkpoint_o
            )

            print()
            print(
                f"Checkpoint saved at "
                f"{current_total_game} games."
            )

    # ========================================================
    # SAVE FINAL MODELS
    # ========================================================

    print()
    print(
        "Saving final models..."
    )

    agent_x.save(
        model_x_path
    )

    agent_o.save(
        model_o_path
    )

    # ========================================================
    # FINAL STATISTICS
    # ========================================================

    print()
    print("=" * 60)
    print(" 1000-GAME TRAINING COMPLETED")
    print("=" * 60)

    print()

    print(
        f"Total games trained: "
        f"{total_games}"
    )

    print(
        f"X wins during continued training: "
        f"{x_wins}"
    )

    print(
        f"O wins during continued training: "
        f"{o_wins}"
    )

    print(
        f"Draws during continued training: "
        f"{draws}"
    )

    print()

    if additional_games > 0:

        print(
            f"X win rate during continued training: "
            f"{(x_wins / additional_games) * 100:.2f}%"
        )

        print(
            f"O win rate during continued training: "
            f"{(o_wins / additional_games) * 100:.2f}%"
        )

        print(
            f"Draw rate during continued training: "
            f"{(draws / additional_games) * 100:.2f}%"
        )

    print()

    print(
        f"Final epsilon X: "
        f"{agent_x.epsilon:.4f}"
    )

    print(
        f"Final epsilon O: "
        f"{agent_o.epsilon:.4f}"
    )

    print()

    print(
        f"Final training steps X: "
        f"{agent_x.training_steps}"
    )

    print(
        f"Final training steps O: "
        f"{agent_o.training_steps}"
    )

    print()

    print(
        f"Final replay buffer X: "
        f"{len(agent_x.replay_buffer)}"
    )

    print(
        f"Final replay buffer O: "
        f"{len(agent_o.replay_buffer)}"
    )

    print()

    print(
        "Final X model:"
    )

    print(
        model_x_path
    )

    print()

    print(
        "Final O model:"
    )

    print(
        model_o_path
    )

    print()
    print("=" * 60)
    print(" MODELS SAVED SUCCESSFULLY")
    print("=" * 60)

    return agent_x, agent_o


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print(" DDQN ULTIMATE TIC-TAC-TOE")
    print(" CONTINUED TRAINING")
    print("=" * 60)

    # --------------------------------------------------------
    # Continue from the existing 100-game models.
    #
    # 100 previous games
    # +900 new games
    # =1000 total games
    # --------------------------------------------------------

    agent_x, agent_o = continue_training(
        total_games=1000,
        previous_games=100,
        save_every=100
    )

    print()

    print(
        "CONTINUED TRAINING FINISHED."
    )