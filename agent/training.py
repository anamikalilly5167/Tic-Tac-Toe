import sys
import os
import random
import numpy as np

# ============================================================
# PROJECT PATH
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
# RANDOM PLAYER
# ============================================================

def select_random_action(valid_actions):
    """
    Select a random legal action.

    The random player does not learn.
    """

    if len(valid_actions) == 0:
        raise ValueError(
            "Random player received no valid actions."
        )

    return random.choice(valid_actions)


# ============================================================
# PLAY ONE DDQN VS RANDOM GAME
# ============================================================

def play_one_game(
    ddqn_agent,
    ddqn_player,
    render=False
):
    """
    Play one complete Ultimate Tic-Tac-Toe game.

    Training setup:

        ddqn_player == X
            DDQN X vs Random O

        ddqn_player == O
            Random X vs DDQN O

    Only the DDQN agent learns.

    Reward structure:

        Local board win       -> +0.2
        Complete game win    -> +1.0
        Complete game loss   -> -1.0
        Draw                 -> 0.0
        Normal move          -> 0.0
        Illegal move        -> -1.0
    """

    if ddqn_player not in (X, O):

        raise ValueError(
            "ddqn_player must be X (1) or O (-1)."
        )

    # ========================================================
    # CREATE ENVIRONMENT
    # ========================================================

    env = UltimateTTTEnv()

    # ========================================================
    # RESET
    # ========================================================

    state = env.reset()

    done = False
    move_count = 0

    last_info = {}

    losses = []

    # ========================================================
    # PENDING DDQN TRANSITION
    # ========================================================

    pending_state = None
    pending_action = None

    # Reward received from the DDQN's move.
    #
    # This is important because the environment's reward
    # may be +0.2 when the DDQN wins a local board.
    pending_reward = 0.0

    # ========================================================
    # GAME LOOP
    # ========================================================

    while not done:

        # ----------------------------------------------------
        # CURRENT PLAYER
        # ----------------------------------------------------

        current_player = env.get_current_player()

        # ----------------------------------------------------
        # VALID ACTIONS
        # ----------------------------------------------------

        valid_actions = env.get_valid_moves()

        if len(valid_actions) == 0:

            print(
                "ERROR: No valid actions available."
            )

            break

        # ====================================================
        # FINALIZE PREVIOUS DDQN TRANSITION
        # ====================================================

        if (
            current_player == ddqn_player
            and
            pending_state is not None
        ):

            # ------------------------------------------------
            # The DDQN's previous move did not end the game.
            #
            # The Random player has now made its response.
            #
            # Therefore this is the next decision state for
            # the DDQN.
            # ------------------------------------------------

            ddqn_agent.remember(
                pending_state,
                pending_action,
                pending_reward,
                state.copy(),
                False,
                list(valid_actions)
            )

            # ------------------------------------------------
            # LEARN
            # ------------------------------------------------

            loss = ddqn_agent.learn()

            if loss is not None:
                losses.append(loss)

            # ------------------------------------------------
            # CLEAR PENDING TRANSITION
            # ------------------------------------------------

            pending_state = None
            pending_action = None
            pending_reward = 0.0

        # ====================================================
        # SELECT ACTION
        # ====================================================

        if current_player == ddqn_player:

            # ------------------------------------------------
            # DDQN PLAYER
            # ------------------------------------------------

            action = ddqn_agent.select_action(
                state,
                valid_actions,
                training=True
            )

            player_name = (
                "X"
                if ddqn_player == X
                else "O"
            )

        else:

            # ------------------------------------------------
            # RANDOM PLAYER
            # ------------------------------------------------

            action = select_random_action(
                valid_actions
            )

            player_name = (
                "X"
                if current_player == X
                else "O"
            )

        player_who_moved = current_player

        # ====================================================
        # SAVE DDQN STATE AND ACTION
        # ====================================================

        if current_player == ddqn_player:

            pending_state = state.copy()
            pending_action = action

        # ====================================================
        # ENVIRONMENT STEP
        # ====================================================

        next_state, env_reward, done, info = env.step(
            action
        )

        last_info = info

        # ====================================================
        # DDQN MOVE REWARD
        # ====================================================

        if current_player == ddqn_player:

            # Store the reward generated by the DDQN's move.
            #
            # Examples:
            #
            # Normal move       -> 0.0
            # Local board win   -> +0.2
            # Game win          -> +1.0
            # Draw              -> 0.0
            # Illegal move      -> -1.0

            pending_reward = float(
                env_reward
            )

        # ====================================================
        # GAME ENDED
        # ====================================================

        if done:

            winner = info.get("winner")

            # =================================================
            # DETERMINE TERMINAL REWARD
            # =================================================

            if winner == ddqn_player:

                terminal_reward = 1.0

            elif winner in (X, O):

                terminal_reward = -1.0

            elif info.get("illegal_move", False):

                terminal_reward = -1.0

            else:

                terminal_reward = 0.0

            # =================================================
            # CURRENT MOVE WAS MADE BY DDQN
            # =================================================

            if current_player == ddqn_player:

                # ------------------------------------------------
                # If DDQN itself ended the game, env_reward is
                # already the terminal reward.
                #
                # We do NOT add another terminal reward.
                # ------------------------------------------------

                ddqn_agent.remember(
                    pending_state,
                    pending_action,
                    pending_reward,
                    next_state.copy(),
                    True,
                    []
                )

                pending_state = None
                pending_action = None
                pending_reward = 0.0

                # ------------------------------------------------
                # LEARN FROM TERMINAL EXPERIENCE
                # ------------------------------------------------

                loss = ddqn_agent.learn()

                if loss is not None:
                    losses.append(loss)

            # =================================================
            # RANDOM PLAYER ENDED THE GAME
            # =================================================

            else:

                # ------------------------------------------------
                # The DDQN made the previous move.
                #
                # Its pending transition now ends because the
                # Random opponent has made the terminal move.
                #
                # We combine:
                #
                #   reward from DDQN's previous move
                #   +
                #   terminal reward from the game result
                #
                # This preserves local-board reward shaping.
                # ------------------------------------------------

                if pending_state is not None:

                    final_reward = (
                        pending_reward
                        + terminal_reward
                    )

                    ddqn_agent.remember(
                        pending_state,
                        pending_action,
                        final_reward,
                        next_state.copy(),
                        True,
                        []
                    )

                    pending_state = None
                    pending_action = None
                    pending_reward = 0.0

                    # ------------------------------------------------
                    # LEARN
                    # ------------------------------------------------

                    loss = ddqn_agent.learn()

                    if loss is not None:
                        losses.append(loss)

        # ====================================================
        # UPDATE STATE
        # ====================================================

        state = next_state

        move_count += 1

        # ====================================================
        # OPTIONAL RENDERING
        # ====================================================

        if render:

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

            if info.get("local_result") is not None:

                print(
                    f"Local board result: "
                    f"{info.get('local_result')}"
                )

            if done:

                print(
                    f"Winner: "
                    f"{info.get('winner')}"
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
    # DETERMINE DDQN RESULT
    # ========================================================

    if winner == ddqn_player:

        ddqn_result = "win"

    elif winner in (X, O):

        ddqn_result = "loss"

    else:

        ddqn_result = "draw"

    # ========================================================
    # RETURN
    # ========================================================

    return {

        "winner": winner,

        "ddqn_player": ddqn_player,

        "ddqn_result": ddqn_result,

        "moves": move_count,

        "info": game_info,

        "avg_loss": (
            float(np.mean(losses))
            if losses
            else None
        ),

        "losses": losses
    }


# ============================================================
# TRAIN DDQN AGAINST RANDOM
# ============================================================

def train_against_random(
    total_games=5000,
    previous_games=0,
    save_every=500
):
    """
    Train DDQN agents against a random opponent.

    Games alternate:

        Game 1:
            DDQN X vs Random O

        Game 2:
            Random X vs DDQN O

        Game 3:
            DDQN X vs Random O

        ...

    If final models exist:
        Load them and continue.

    If final models do not exist:
        Start from scratch.
    """

    # ========================================================
    # MODEL DIRECTORY
    # ========================================================

    model_dir = os.path.join(
        PROJECT_ROOT,
        "models"
    )

    os.makedirs(
        model_dir,
        exist_ok=True
    )

    # ========================================================
    # FINAL MODEL PATHS
    # ========================================================

    model_x_path = os.path.join(
        model_dir,
        "agent_x_final.pth"
    )

    model_o_path = os.path.join(
        model_dir,
        "agent_o_final.pth"
    )

    # ========================================================
    # CHECK EXISTING MODELS
    # ========================================================

    models_exist = (
        os.path.exists(model_x_path)
        and
        os.path.exists(model_o_path)
    )

    # ========================================================
    # DETERMINE NUMBER OF GAMES
    # ========================================================

    if models_exist:

        additional_games = (
            total_games - previous_games
        )

        if additional_games <= 0:

            print()
            print(
                "ERROR:"
            )

            print(
                "total_games must be greater "
                "than previous_games."
            )

            return None, None

    else:

        # Fresh training

        previous_games = 0
        additional_games = total_games

    # ========================================================
    # HEADER
    # ========================================================

    print()
    print("=" * 70)
    print(" DDQN ULTIMATE TIC-TAC-TOE")
    print(" DDQN VS RANDOM TRAINING")
    print("=" * 70)

    print()

    print(
        f"Previous games      : "
        f"{previous_games}"
    )

    print(
        f"Target total games : "
        f"{total_games}"
    )

    print(
        f"Games to play      : "
        f"{additional_games}"
    )

    print()

    # ========================================================
    # CREATE AGENTS
    # ========================================================

    agent_x = DDQNAgent(
        state_size=100,
        action_size=81,
        batch_size=32,
        epsilon_decay=0.999
    )

    agent_o = DDQNAgent(
        state_size=100,
        action_size=81,
        batch_size=32,
        epsilon_decay=0.999
    )

    # ========================================================
    # LOAD EXISTING MODELS
    # ========================================================

    if models_exist:

        print(
            "Existing models found."
        )

        print(
            "Loading models..."
        )

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

        print()
        print(
            "Models loaded successfully."
        )

    else:

        print(
            "No existing models found."
        )

        print(
            "Starting training from scratch."
        )

    # ========================================================
    # AGENT INFORMATION
    # ========================================================

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
        f"Agent X epsilon: "
        f"{agent_x.epsilon:.4f}"
    )

    print(
        f"Agent O epsilon: "
        f"{agent_o.epsilon:.4f}"
    )

    print()

    print(
        f"Agent X training steps: "
        f"{agent_x.training_steps}"
    )

    print(
        f"Agent O training steps: "
        f"{agent_o.training_steps}"
    )

    print()

    # ========================================================
    # STATISTICS
    # ========================================================

    x_wins = 0
    o_wins = 0
    draws = 0

    total_moves = 0

    # DDQN X statistics

    ddqn_x_wins = 0
    ddqn_x_losses = 0
    ddqn_x_draws = 0

    # DDQN O statistics

    ddqn_o_wins = 0
    ddqn_o_losses = 0
    ddqn_o_draws = 0

    all_x_losses = []
    all_o_losses = []

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for game_index in range(
        1,
        additional_games + 1
    ):

        # ====================================================
        # ALTERNATE DDQN SIDE
        # ====================================================

        if game_index % 2 == 1:

            # ------------------------------------------------
            # DDQN X vs Random O
            # ------------------------------------------------

            ddqn_player = X
            ddqn_agent = agent_x

            matchup = (
                "DDQN X vs Random O"
            )

        else:

            # ------------------------------------------------
            # Random X vs DDQN O
            # ------------------------------------------------

            ddqn_player = O
            ddqn_agent = agent_o

            matchup = (
                "Random X vs DDQN O"
            )

        # ====================================================
        # PLAY GAME
        # ====================================================

        result = play_one_game(
            ddqn_agent,
            ddqn_player,
            render=False
        )

        # ====================================================
        # GAME WINNER
        # ====================================================

        winner = result["winner"]

        if winner == X:

            x_wins += 1

        elif winner == O:

            o_wins += 1

        else:

            draws += 1

        # ====================================================
        # DDQN RESULT
        # ====================================================

        ddqn_result = result["ddqn_result"]

        if ddqn_player == X:

            if ddqn_result == "win":

                ddqn_x_wins += 1

            elif ddqn_result == "loss":

                ddqn_x_losses += 1

            else:

                ddqn_x_draws += 1

            if result["losses"]:

                all_x_losses.extend(
                    result["losses"]
                )

        else:

            if ddqn_result == "win":

                ddqn_o_wins += 1

            elif ddqn_result == "loss":

                ddqn_o_losses += 1

            else:

                ddqn_o_draws += 1

            if result["losses"]:

                all_o_losses.extend(
                    result["losses"]
                )

        # ====================================================
        # MOVES
        # ====================================================

        total_moves += result["moves"]

        # ====================================================
        # EPSILON DECAY
        # ====================================================

        ddqn_agent.decay_epsilon()

        # ====================================================
        # CURRENT TOTAL GAME
        # ====================================================

        current_total_game = (
            previous_games
            + game_index
        )

        # ====================================================
        # PROGRESS
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
            print("-" * 70)

            print(
                f"Training progress: "
                f"{current_total_game}/{total_games}"
            )

            print(
                f"Current matchup: "
                f"{matchup}"
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
                "DDQN X vs Random O:"
            )

            print(
                f"  Wins   : {ddqn_x_wins}"
            )

            print(
                f"  Losses : {ddqn_x_losses}"
            )

            print(
                f"  Draws  : {ddqn_x_draws}"
            )

            print()

            print(
                "Random X vs DDQN O:"
            )

            print(
                f"  DDQN O wins   : {ddqn_o_wins}"
            )

            print(
                f"  DDQN O losses : {ddqn_o_losses}"
            )

            print(
                f"  Draws         : {ddqn_o_draws}"
            )

            print()

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

            print()

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
                f"game {current_total_game}."
            )

    # ========================================================
    # SAVE FINAL MODELS
    # ========================================================

    print()
    print("=" * 70)
    print(" SAVING FINAL MODELS")
    print("=" * 70)

    agent_x.save(
        model_x_path
    )

    agent_o.save(
        model_o_path
    )

    print()

    print(
        "Agent X saved:"
    )

    print(
        model_x_path
    )

    print()

    print(
        "Agent O saved:"
    )

    print(
        model_o_path
    )

    # ========================================================
    # FINAL STATISTICS
    # ========================================================

    print()
    print("=" * 70)
    print(" DDQN VS RANDOM TRAINING COMPLETED")
    print("=" * 70)

    print()

    print(
        f"Total games trained: "
        f"{total_games}"
    )

    print()

    print(
        "Overall game results:"
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

    print()

    # ========================================================
    # DDQN X STATISTICS
    # ========================================================

    x_games = (
        ddqn_x_wins
        + ddqn_x_losses
        + ddqn_x_draws
    )

    if x_games > 0:

        print(
            "DDQN X vs Random O:"
        )

        print(
            f"  Games  : {x_games}"
        )

        print(
            f"  Wins   : {ddqn_x_wins}"
        )

        print(
            f"  Losses : {ddqn_x_losses}"
        )

        print(
            f"  Draws  : {ddqn_x_draws}"
        )

        print(
            f"  Win rate: "
            f"{(ddqn_x_wins / x_games) * 100:.2f}%"
        )

        print(
            f"  Loss rate: "
            f"{(ddqn_x_losses / x_games) * 100:.2f}%"
        )

        print(
            f"  Draw rate: "
            f"{(ddqn_x_draws / x_games) * 100:.2f}%"
        )

        print()

    # ========================================================
    # DDQN O STATISTICS
    # ========================================================

    o_games = (
        ddqn_o_wins
        + ddqn_o_losses
        + ddqn_o_draws
    )

    if o_games > 0:

        print(
            "Random X vs DDQN O:"
        )

        print(
            f"  Games  : {o_games}"
        )

        print(
            f"  DDQN O wins   : {ddqn_o_wins}"
        )

        print(
            f"  DDQN O losses : {ddqn_o_losses}"
        )

        print(
            f"  Draws         : {ddqn_o_draws}"
        )

        print(
            f"  DDQN O win rate: "
            f"{(ddqn_o_wins / o_games) * 100:.2f}%"
        )

        print(
            f"  DDQN O loss rate: "
            f"{(ddqn_o_losses / o_games) * 100:.2f}%"
        )

        print(
            f"  Draw rate: "
            f"{(ddqn_o_draws / o_games) * 100:.2f}%"
        )

        print()

    # ========================================================
    # FINAL AGENT INFORMATION
    # ========================================================

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
    print("=" * 70)
    print(" MODELS SAVED SUCCESSFULLY")
    print("=" * 70)

    return agent_x, agent_o


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print(" DDQN ULTIMATE TIC-TAC-TOE")
    print(" TRAINING AGAINST RANDOM")
    print("=" * 70)

    # ========================================================
    # FRESH TRAINING
    # ========================================================
    #
    # If agent_x_final.pth and agent_o_final.pth do not exist,
    # training automatically starts from scratch.
    #
    # If they exist, they are loaded and training continues.
    #
    # For a completely fresh training run:
    #
    #   1. Back up the old final models.
    #   2. Remove them from the models folder.
    #   3. Keep previous_games = 0.
    #
    # ========================================================

    agent_x, agent_o = train_against_random(
        total_games=5000,
        previous_games=0,
        save_every=500
    )

    print()

    print(
        "DDQN VS RANDOM TRAINING FINISHED."
    )