import numpy as np


class UltimateTTTRules:
    """
    Rules and game logic for Ultimate Tic-Tac-Toe.

    Board representation:
        board.shape = (9, 9)

        First index  = local board (0-8)
        Second index = cell inside local board (0-8)

    Player representation:
        1  = X
        -1 = O
        0  = empty

    Action representation:
        action = local_board * 9 + cell

        Example:
            action  = 23
            board   = 23 // 9 = 2
            cell    = 23 % 9  = 5

            Therefore:
                action 23 = board 2, cell 5
    """

    EMPTY = 0
    X = 1
    O = -1

    DRAW = 2
    ONGOING = 0

    # All possible winning combinations in a 3x3 board
    WINNING_LINES = [
        (0, 1, 2),  # top row
        (3, 4, 5),  # middle row
        (6, 7, 8),  # bottom row

        (0, 3, 6),  # left column
        (1, 4, 7),  # middle column
        (2, 5, 8),  # right column

        (0, 4, 8),  # diagonal
        (2, 4, 6)   # diagonal
    ]

    def __init__(self, board):
        """
        Parameters
        ----------
        board : Board
            Instance of our Board class from board.py.
        """

        self.board = board

    # ---------------------------------------------------------
    # ACTION CONVERSION
    # ---------------------------------------------------------

    @staticmethod
    def action_to_position(action):
        """
        Convert an action number (0-80) into:

            local_board
            cell

        Example:
            action = 23

            local_board = 2
            cell = 5
        """

        if not isinstance(action, (int, np.integer)):
            raise TypeError("Action must be an integer.")

        if action < 0 or action >= 81:
            raise ValueError("Action must be between 0 and 80.")

        local_board = action // 9
        cell = action % 9

        return local_board, cell

    @staticmethod
    def position_to_action(local_board, cell):
        """
        Convert local board + cell into an action number.

        Example:

            local_board = 2
            cell = 5

            action = 23
        """

        if local_board < 0 or local_board > 8:
            raise ValueError("Local board must be between 0 and 8.")

        if cell < 0 or cell > 8:
            raise ValueError("Cell must be between 0 and 8.")

        return local_board * 9 + cell

    # ---------------------------------------------------------
    # LOCAL BOARD STATUS
    # ---------------------------------------------------------

    def check_local_winner(self, local_board):
        """
        Check the result of one local 3x3 board.

        Returns:
            1  -> X wins
            -1 -> O wins
            2  -> draw
            0  -> game still ongoing
        """

        cells = self.board.board[local_board]

        # Check all winning lines
        for a, b, c in self.WINNING_LINES:

            if (
                cells[a] != self.EMPTY
                and cells[a] == cells[b]
                and cells[b] == cells[c]
            ):
                return cells[a]

        # If no empty cells remain, it is a draw
        if not np.any(cells == self.EMPTY):
            return self.DRAW

        # Otherwise the local board is still active
        return self.ONGOING

    def update_local_status(self, local_board):
        """
        Update the stored status of a local board.

        Returns the new status.
        """

        status = self.check_local_winner(local_board)

        self.board.local_status[local_board] = status

        return status

    # ---------------------------------------------------------
    # GLOBAL BOARD STATUS
    # ---------------------------------------------------------

    def check_global_winner(self):
        """
        Check whether X or O has won the entire Ultimate
        Tic-Tac-Toe game.

        The winners of the 9 local boards form another
        3x3 board.

        Returns:
            1  -> X wins
            -1 -> O wins
            2  -> draw
            0  -> game still ongoing
        """

        status = self.board.local_status

        # Check global winning combinations
        for a, b, c in self.WINNING_LINES:

            # X wins the global game
            if (
                status[a] == self.X
                and status[b] == self.X
                and status[c] == self.X
            ):
                return self.X

            # O wins the global game
            if (
                status[a] == self.O
                and status[b] == self.O
                and status[c] == self.O
            ):
                return self.O

        # If every local board is finished and nobody won,
        # the entire game is a draw.
        if np.all(
            (status == self.X)
            | (status == self.O)
            | (status == self.DRAW)
        ):
            return self.DRAW

        return self.ONGOING

    # ---------------------------------------------------------
    # MOVE VALIDATION
    # ---------------------------------------------------------

    def is_valid_move(self, action):
        """
        Check whether an action is legal.

        Returns:
            True  -> legal
            False -> illegal
        """

        try:
            local_board, cell = self.action_to_position(action)
        except (TypeError, ValueError):
            return False

        # -----------------------------------------------------
        # RULE 1:
        # The entire game must not already be finished.
        # -----------------------------------------------------

        if self.check_global_winner() != self.ONGOING:
            return False

        # -----------------------------------------------------
        # RULE 2:
        # If a specific local board is required,
        # the player must play there.
        # -----------------------------------------------------

        active_board = self.board.active_board

        if active_board != -1:

            # The required board is already finished.
            # Therefore the player may choose another board.
            if self.board.local_status[active_board] in (
                self.X,
                self.O,
                self.DRAW
            ):
                pass

            # Required board is still active.
            elif local_board != active_board:
                return False

        # -----------------------------------------------------
        # RULE 3:
        # The selected local board cannot already be finished.
        # -----------------------------------------------------

        if self.board.local_status[local_board] in (
            self.X,
            self.O,
            self.DRAW
        ):
            return False

        # -----------------------------------------------------
        # RULE 4:
        # Selected cell must be empty.
        # -----------------------------------------------------

        if self.board.board[local_board][cell] != self.EMPTY:
            return False

        return True

    # ---------------------------------------------------------
    # VALID MOVES
    # ---------------------------------------------------------

    def get_valid_moves(self):
        """
        Return all currently legal actions.

        Returns
        -------
        list
            List of action numbers between 0 and 80.
        """

        valid_moves = []

        for action in range(81):

            if self.is_valid_move(action):
                valid_moves.append(action)

        return valid_moves

    # ---------------------------------------------------------
    # APPLY MOVE
    # ---------------------------------------------------------

    def apply_move(self, action):
        """
        Apply a legal move to the board.

        Returns
        -------
        dict
            Information about the move.

        Raises
        ------
        ValueError
            If the move is illegal.
        """

        if not self.is_valid_move(action):
            raise ValueError(
                f"Illegal move: {action}"
            )

        local_board, cell = self.action_to_position(action)

        player = self.board.current_player

        # -----------------------------------------------------
        # Place player's mark
        # -----------------------------------------------------

        self.board.board[local_board][cell] = player

        # -----------------------------------------------------
        # Check whether this local board has ended
        # -----------------------------------------------------

        local_result = self.update_local_status(local_board)

        # -----------------------------------------------------
        # The cell played determines the next local board.
        #
        # Example:
        #
        # Player plays:
        #
        #       Board 2, Cell 5
        #
        # Therefore opponent must normally play:
        #
        #       Board 5
        # -----------------------------------------------------

        next_board = cell

        # If the destination board is already finished,
        # the next player can choose any unfinished board.
        if self.board.local_status[next_board] in (
            self.X,
            self.O,
            self.DRAW
        ):
            self.board.active_board = -1
        else:
            self.board.active_board = next_board

        # -----------------------------------------------------
        # Check global game result
        # -----------------------------------------------------

        game_result = self.check_global_winner()

        # -----------------------------------------------------
        # Switch player if game is still running
        # -----------------------------------------------------

        if game_result == self.ONGOING:
            self.board.current_player *= -1

        return {
            "action": action,
            "local_board": local_board,
            "cell": cell,
            "player": player,
            "local_result": local_result,
            "game_result": game_result,
            "next_board": self.board.active_board
        }

    # ---------------------------------------------------------
    # GAME OVER
    # ---------------------------------------------------------

    def is_game_over(self):
        """
        Returns True if the Ultimate Tic-Tac-Toe game
        has ended.
        """

        return self.check_global_winner() != self.ONGOING
if __name__ == "__main__":

    from board import Board

    board = Board()
    rules = UltimateTTTRules(board)

    print("Ultimate Tic-Tac-Toe Rules Test")
    print("--------------------------------")

    print("Current player:", board.current_player)
    print("Active board:", board.active_board)

    print("\nNumber of valid moves:")
    print(len(rules.get_valid_moves()))

    print("\nValid moves:")
    print(rules.get_valid_moves())

    print("\nPlaying action 4...")

    result = rules.apply_move(4)

    print(result)

    print("\nCurrent player:", board.current_player)
    print("Active board:", board.active_board)

    print("\nNumber of valid moves:")
    print(len(rules.get_valid_moves()))