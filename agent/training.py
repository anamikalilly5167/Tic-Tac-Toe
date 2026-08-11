import sys
import os

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


from game import UltimateTTTEnv
from ddqn_agent import DDQNAgent


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
    """

    env = UltimateTTTEnv()

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    env.reset()

    state = env.get_state()

    done = env.is_game_over()

    move_count = 0

    last_info = {}

    # --------------------------------------------------------
    # GAME LOOP
    # --------------------------------------------------------

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
        # Select player-specific agent
        # ----------------------------------------------------

        if current_player == env.X:

            agent = agent_x

        else:

            agent = agent_o

        # ----------------------------------------------------
        # Select action
        # ----------------------------------------------------

        action = agent.select_action(
            state,
            valid_actions,
            training=True
        )

        # ----------------------------------------------------
        # Save player perspective
        # ----------------------------------------------------

        player_who_moved = current_player

        # ----------------------------------------------------
        # Environment step
        # ----------------------------------------------------

        next_state, reward, done, info = env.step(
            action
        )

        last_info = info

        # ----------------------------------------------------
        # Store experience
        # ----------------------------------------------------

        agent.remember(
            state,
            action,
            reward,
            next_state,
            done
        )

        # ----------------------------------------------------
        # Learn
        # ----------------------------------------------------

        loss = agent.learn()

        # ----------------------------------------------------
        # Update state
        # ----------------------------------------------------

        state = next_state

        move_count += 1

        # ----------------------------------------------------
        # Optional rendering
        # ----------------------------------------------------

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
                f"Reward: {reward}"
            )

            if loss is not None:

                print(
                    f"Loss: {loss:.6f}"
                )

            env.render()

    # ========================================================
    # GAME INFORMATION
    # ========================================================

    game_info = env.get_game_info()

    return {
        "winner": game_info.get(
            "winner",
            last_info.get("winner")
        ),
        "moves": move_count,
        "info": game_info
    }


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("==========================================")
    print(" DDQN SELF-PLAY TEST")
    print("==========================================")

    # --------------------------------------------------------
    # Create agents
    # --------------------------------------------------------

    agent_x = DDQNAgent(
        batch_size=32
    )

    agent_o = DDQNAgent(
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

    # --------------------------------------------------------
    # Play ONE game
    # --------------------------------------------------------

    result = play_one_game(
        agent_x,
        agent_o,
        render=True
    )

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print()
    print("==========================================")
    print(" GAME FINISHED")
    print("==========================================")

    print(
        "Winner:",
        result["winner"]
    )

    print(
        "Moves:",
        result["moves"]
    )

    print(
        "Game info:",
        result["info"]
    )

    print()
    print("SELF-PLAY TEST COMPLETED")