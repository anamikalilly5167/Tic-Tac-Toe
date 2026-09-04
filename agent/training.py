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
DRAW = 2


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

    agent_x plays X.
    agent_o plays O.

    Each agent stores a transition from one of its
    decision points to its next decision point.

    Returns
    -------
    dict
        Game result and training statistics.
    """

    env = UltimateTTTEnv()

    # --------------------------------------------------------
    # RESET ENVIRONMENT
    # --------------------------------------------------------

    state = env.reset()

    done = False
    move_count = 0

    last_info = {}

    losses_x = []
    losses_o = []

    # --------------------------------------------------------
    # Pending transitions
    #
    # Each pending transition contains:
    #
    #     (state_before_action, action)
    #
    # We wait until the SAME player gets another turn.
    # At that point, the current state becomes next_state.
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
        # Get legal actions
        # ----------------------------------------------------

        valid_actions = env.get_valid_moves()

        if len(valid_actions) == 0:

            print(
                "ERROR: No valid actions available."
            )

            break

        # ----------------------------------------------------
        # Select the correct agent
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

        # If this player has already played before, then
        # the current state is the state reached after the
        # opponent's move.
        #
        # Therefore this is the correct next_state for the
        # player's previous decision.

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

            # ------------------------------------------------
            # Clear pending transition
            # ------------------------------------------------

            if current_player == env.X:

                pending_x = None

            else:

                pending_o = None

            # ------------------------------------------------
            # Learn from the newly stored experience
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
        # GAME HAS ENDED
        # ====================================================

        if done:

            winner = info.get("winner")

            # ------------------------------------------------
            # Determine terminal rewards
            # ------------------------------------------------

            if winner == player_who_moved:

                # The player who made the final winning move
                # receives +1.

                current_reward = 1.0

                # Opponent loses.

                opponent_reward = -1.0

            elif winner in (env.X, env.O):

                # Safety fallback.

                current_reward = -1.0
                opponent_reward = 1.0

            elif info.get("illegal_move", False):

                # Illegal move = loss for the player who
                # attempted it.

                current_reward = -1.0
                opponent_reward = 1.0

            else:

                # Draw.

                current_reward = 0.0
                opponent_reward = 0.0

            # =================================================
            # STORE CURRENT PLAYER'S FINAL TRANSITION
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
            # STORE OPPONENT'S PENDING TRANSITION
            # =================================================

            if player_who_moved == env.X:

                # O is the opponent.

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

                # X is the opponent.

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

            # Store this player's decision as pending.
            #
            # We DO NOT store next_state yet because it is
            # currently the opponent's turn.
            #
            # When this player gets another turn, that state
            # will become the correct next_state.

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
    # RETURN RESULTS
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
# TRAIN TWO AGENTS USING SELF-PLAY
# ============================================================

def train_agents(
    num_games=1000,
    save_every=25
):
    """
    Train X and O agents through self-play.

    Parameters
    ----------
    num_games : int
        Number of games to play.

    save_every : int
        Save a checkpoint after this many games.
    """

    print()
    print("=" * 60)
    print(" STARTING DDQN SELF-PLAY TRAINING")
    print("=" * 60)

    print()
    print(
        f"Number of training games: {num_games}"
    )

    # ========================================================
    # CREATE AGENTS
    # ========================================================

    print()
    print("Creating Agent X...")

    agent_x = DDQNAgent(
        state_size=100,
        action_size=81,
        batch_size=32
    )

    print("Creating Agent O...")

    agent_o = DDQNAgent(
        state_size=100,
        action_size=81,
        batch_size=32
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

    # ========================================================
    # TRAINING STATISTICS
    # ========================================================

    x_wins = 0
    o_wins = 0
    draws = 0

    all_x_losses = []
    all_o_losses = []

    total_moves = 0

    # ========================================================
    # CREATE MODEL DIRECTORY
    # ========================================================

    model_dir = os.path.join(
        PROJECT_ROOT,
        "models"
    )

    os.makedirs(
        model_dir,
        exist_ok=True
    )

    print()
    print(
        "Models will be saved in:"
    )

    print(
        model_dir
    )

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for game_number in range(
        1,
        num_games + 1
    ):

        # ----------------------------------------------------
        # Play one complete game
        # ----------------------------------------------------

        result = play_one_game(
            agent_x,
            agent_o,
            render=False
        )

        # ----------------------------------------------------
        # Get winner
        # ----------------------------------------------------

        winner = result["winner"]

        # ----------------------------------------------------
        # Update statistics
        # ----------------------------------------------------

        if winner == X:

            x_wins += 1

        elif winner == O:

            o_wins += 1

        else:

            draws += 1

        # ----------------------------------------------------
        # Track number of moves
        # ----------------------------------------------------

        total_moves += result["moves"]

        # ----------------------------------------------------
        # Store losses
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

        # Epsilon is decayed once per game, not once per move.

        agent_x.decay_epsilon()
        agent_o.decay_epsilon()

        # ====================================================
        # PRINT PROGRESS
        # ====================================================

        if (
            game_number % 10 == 0
            or game_number == 1
            or game_number == num_games
        ):

            if all_x_losses:

                avg_x_loss = float(
                    np.mean(all_x_losses)
                )

            else:

                avg_x_loss = None

            if all_o_losses:

                avg_o_loss = float(
                    np.mean(all_o_losses)
                )

            else:

                avg_o_loss = None

            avg_moves = (
                total_moves / game_number
            )

            print()
            print("-" * 60)

            print(
                f"Game {game_number}/{num_games}"
            )

            print(
                f"X wins : {x_wins}"
            )

            print(
                f"O wins : {o_wins}"
            )

            print(
                f"Draws  : {draws}"
            )

            print(
                f"Average moves/game: "
                f"{avg_moves:.2f}"
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
                f"{avg_x_loss}"
            )

            print(
                f"Average O loss: "
                f"{avg_o_loss}"
            )

            print(
                f"Replay buffer X: "
                f"{len(agent_x.replay_buffer)}"
            )

            print(
                f"Replay buffer O: "
                f"{len(agent_o.replay_buffer)}"
            )

        # ====================================================
        # SAVE CHECKPOINT
        # ====================================================

        if (
            game_number % save_every == 0
        ):

            x_checkpoint = os.path.join(
                model_dir,
                f"agent_x_game_{game_number}.pth"
            )

            o_checkpoint = os.path.join(
                model_dir,
                f"agent_o_game_{game_number}.pth"
            )

            agent_x.save(
                x_checkpoint
            )

            agent_o.save(
                o_checkpoint
            )

            print()
            print(
                f"Checkpoint saved "
                f"after game {game_number}"
            )

    # ========================================================
    # SAVE FINAL MODELS
    # ========================================================

    final_x_path = os.path.join(
        model_dir,
        "agent_x_final.pth"
    )

    final_o_path = os.path.join(
        model_dir,
        "agent_o_final.pth"
    )

    agent_x.save(
        final_x_path
    )

    agent_o.save(
        final_o_path
    )

    # ========================================================
    # FINAL STATISTICS
    # ========================================================

    print()
    print("=" * 60)
    print(" TRAINING COMPLETED")
    print("=" * 60)

    print()

    print(
        f"Total games: {num_games}"
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

    if all_x_losses:

        print(
            f"Final average X loss: "
            f"{np.mean(all_x_losses):.6f}"
        )

    else:

        print(
            "Final average X loss: None"
        )

    if all_o_losses:

        print(
            f"Final average O loss: "
            f"{np.mean(all_o_losses):.6f}"
        )

    else:

        print(
            "Final average O loss: None"
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
        final_x_path
    )

    print()

    print(
        "Final O model:"
    )

    print(
        final_o_path
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
    print(" SELF-PLAY TRAINING")
    print("=" * 60)

    # ========================================================
    # TRAIN
    # ========================================================

    agent_x, agent_o = train_agents(
        num_games=100,
        save_every=25
    )

    print()
    print("100-GAME TRAINING TEST COMPLETED.")