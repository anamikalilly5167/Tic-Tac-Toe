import numpy as np


class Board:
    EMPTY = 0
    X = 1          # AI
    O = -1         # Human

    def __init__(self):
        self.reset()

    def reset(self):
        """
        Creates an empty Ultimate Tic-Tac-Toe board.
        Shape: (9 local boards, 9 cells each)
        """
        self.board = np.zeros((9, 9), dtype=np.int8)

        # Winner of each local board
        # 0 = ongoing
        # 1 = X won
        # -1 = O won
        # 2 = Draw
        self.local_status = np.zeros(9, dtype=np.int8)

        # Which local board must be played next
        # -1 means any unfinished board
        self.active_board = -1

        # Current player
        self.current_player = self.X

        return self.board

    def copy(self):
        """
        Returns a deep copy.
        """
        new_board = Board()

        new_board.board = self.board.copy()
        new_board.local_status = self.local_status.copy()
        new_board.active_board = self.active_board
        new_board.current_player = self.current_player

        return new_board