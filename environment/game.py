import numpy as np

from board import Board
from rules import UltimateTTTRules


class UltimateTTTEnv:
    """
    Reinforcement Learning environment for Ultimate Tic-Tac-Toe.

    The environment combines:
        Board  -> stores the game state
        Rules  -> handles legal moves and game logic

    Action space:
        0 to 80

    State:
        Flattened representation of the complete game state.
    """

    # ---------------------------------------------------------
    # GAME CONSTANTS
    # ---------------------------------------------------------

    EMPTY = 0
    X = 1
    O = -1

    ONGOING = 0
    DRAW = 2

    # Rewards
    WIN_REWARD = 1.0
    LOSS_REWARD = -1.0
    DRAW_REWARD = 0.0

    ILLEGAL_MOVE_REWARD = -1.0

    def __init__(self):
        """
        Create a new Ultimate Tic-Tac-Toe environment.
        """

        self.board = Board()
        self.rules = UltimateTTTRules(self.board)

        # Whether the current episode has finished
        self.done = False

        # Number of moves made in current game
        self.move_count = 0

    # =========================================================
    # RESET
    # =========================================================

    def reset(self):
        """
        Reset the environment to the beginning of a new game.

        Returns
        -------
        np.ndarray
            Initial state.
        """

        self.board.reset()

        self.done = False
        self.move_count = 0

        # Reset rules object to use the reset board
        self.rules = UltimateTTTRules(self.board)

        return self.get_state()

    # =========================================================
    # STATE REPRESENTATION
    # =========================================================

    def get_state(self):
        """
        Convert the current game into a neural-network-friendly
        state vector.

        State contains:

        1. 81 board cells
        2. 9 local-board statuses
        3. 9 active-board indicators
        4. 1 current-player indicator

        Total:
            81 + 9 + 9 + 1 = 100 features
        """

        # -----------------------------------------------------
        # 81 individual cells
        # -----------------------------------------------------

        board_state = self.board.board.flatten().astype(np.float32)

        # -----------------------------------------------------
        # Status of the 9 local boards
        # -----------------------------------------------------

        local_status = self.board.local_status.astype(
            np.float32
        )

        # -----------------------------------------------------
        # Active-board encoding
        #
        # We use a one-hot representation:
        #
        # active board 0:
        # [1,0,0,0,0,0,0,0,0]
        #
        # active board 4:
        # [0,0,0,0,1,0,0,0,0]
        #
        # active_board = -1:
        # [0,0,0,0,0,0,0,0,0]
        # -----------------------------------------------------

        active_board = np.zeros(9, dtype=np.float32)

        if self.board.active_board != -1:
            active_board[self.board.active_board] = 1.0

        # -----------------------------------------------------
        # Current player
        #
        # X = +1
        # O = -1
        # -----------------------------------------------------

        current_player = np.array(
            [self.board.current_player],
            dtype=np.float32
        )

        # -----------------------------------------------------
        # Combine everything
        # -----------------------------------------------------

        state = np.concatenate([
            board_state,
            local_status,
            active_board,
            current_player
        ])

        return state

    # =========================================================
    # STEP
    # =========================================================

    def step(self, action):
        """
        Apply one action to the environment.

        Parameters
        ----------
        action : int
            Action between 0 and 80.

        Returns
        -------
        next_state : np.ndarray
        reward : float
        done : bool
        info : dict
        """

        # -----------------------------------------------------
        # Don't allow moves after game has ended
        # -----------------------------------------------------

        if self.done:
            raise ValueError(
                "Game is already over. Call reset() first."
            )

        # -----------------------------------------------------
        # Check whether action is legal
        # -----------------------------------------------------

        if not self.rules.is_valid_move(action):

            # Illegal action
            reward = self.ILLEGAL_MOVE_REWARD

            info = {
                "illegal_move": True,
                "winner": None
            }

            # For now we terminate the episode.
            #
            # Later, during DDQN development, we can choose
            # another illegal-move handling strategy if needed.
            self.done = True

            return (
                self.get_state(),
                reward,
                self.done,
                info
            )

        # -----------------------------------------------------
        # Apply legal move
        # -----------------------------------------------------

        result = self.rules.apply_move(action)

        self.move_count += 1

        game_result = result["game_result"]

        # -----------------------------------------------------
        # Calculate reward
        # -----------------------------------------------------

        if game_result in (self.X, self.O):

            # The player who made this move won.
            reward = self.WIN_REWARD
            self.done = True

        elif game_result == self.DRAW:

            reward = self.DRAW_REWARD
            self.done = True

        else:

            # Game continues
            reward = 0.0
            self.done = False

        # -----------------------------------------------------
        # Information returned to the agent
        # -----------------------------------------------------

        info = {
            "illegal_move": False,
            "winner": game_result if self.done else None,
            "local_board": result["local_board"],
            "cell": result["cell"],
            "player": result["player"],
            "local_result": result["local_result"],
            "next_board": result["next_board"],
            "move_count": self.move_count
        }

        return (
            self.get_state(),
            reward,
            self.done,
            info
        )

    # =========================================================
    # VALID MOVES
    # =========================================================

    def get_valid_moves(self):
        """
        Return a list of currently legal actions.
        """

        return self.rules.get_valid_moves()

    # =========================================================
    # GAME STATUS
    # =========================================================

    def is_game_over(self):
        """
        Return True if the game has finished.
        """

        return self.done

    # =========================================================
    # CURRENT PLAYER
    # =========================================================

    def get_current_player(self):
        """
        Return the current player.

        Returns:
            1  -> X
            -1 -> O
        """

        return self.board.current_player

    # =========================================================
    # RENDER
    # =========================================================

    def render(self):
        """
        Print the Ultimate Tic-Tac-Toe board in a human-readable
        format.

        Each local board is displayed as a 3x3 board.
        """

        symbols = {
            self.EMPTY: ".",
            self.X: "X",
            self.O: "O"
        }

        print()
        print("=" * 25)
        print("ULTIMATE TIC-TAC-TOE")
        print("=" * 25)

        # -----------------------------------------------------
        # Convert the 9 local boards into a 3x3 arrangement
        # -----------------------------------------------------

        for global_row in range(3):

            # Three local boards horizontally
            boards_in_row = [
                global_row * 3,
                global_row * 3 + 1,
                global_row * 3 + 2
            ]

            # Each local board has 3 rows
            for local_row in range(3):

                row_parts = []

                for local_board in boards_in_row:

                    cells = self.board.board[local_board]

                    start = local_row * 3
                    end = start + 3

                    row = [
                        symbols[cell]
                        for cell in cells[start:end]
                    ]

                    row_parts.append(" ".join(row))

                print(" | ".join(row_parts))

            # Separator between global rows
            if global_row < 2:
                print("-" * 25)

        print("=" * 25)

        print(
            "Current player:",
            "X" if self.board.current_player == self.X else "O"
        )

        if self.board.active_board == -1:
            print("Active board: ANY")
        else:
            print(
                "Active board:",
                self.board.active_board
            )

        print("Moves:", self.move_count)

        print()

    # =========================================================
    # FULL GAME INFORMATION
    # =========================================================

    def get_game_info(self):
        """
        Return useful information about the current game.
        """

        return {
            "current_player": self.board.current_player,
            "active_board": self.board.active_board,
            "local_status": self.board.local_status.copy(),
            "move_count": self.move_count,
            "done": self.done,
            "winner": (
                self.rules.check_global_winner()
                if self.done
                else None
            )
        }


# =============================================================
# TEST THE ENVIRONMENT
# =============================================================

if __name__ == "__main__":

    print()
    print("==========================================")
    print(" ULTIMATE TIC-TAC-TOE ENVIRONMENT TEST")
    print("==========================================")

    env = UltimateTTTEnv()

    # ---------------------------------------------------------
    # RESET
    # ---------------------------------------------------------

    state = env.reset()

    print("\nEnvironment reset successfully.")

    print("State shape:", state.shape)
    print("State size:", len(state))

    print("Current player:", env.get_current_player())
    print("Active board:", env.board.active_board)

    print("Number of valid moves:", len(env.get_valid_moves()))

    # ---------------------------------------------------------
    # RENDER EMPTY BOARD
    # ---------------------------------------------------------

    print("\nInitial board:")

    env.render()

    # ---------------------------------------------------------
    # PLAY FIRST MOVE
    # ---------------------------------------------------------

    print("Playing action 4...")

    next_state, reward, done, info = env.step(4)

    print("\nReward:", reward)
    print("Done:", done)
    print("Info:", info)

    print("\nBoard after move:")

    env.render()

    # ---------------------------------------------------------
    # STATE TEST
    # ---------------------------------------------------------

    print("Next state shape:", next_state.shape)

    # ---------------------------------------------------------
    # VALID MOVE TEST
    # ---------------------------------------------------------

    print(
        "Number of valid moves after first move:",
        len(env.get_valid_moves())
    )

    print("\n==========================================")
    print(" ENVIRONMENT TEST COMPLETED")
    print("==========================================")