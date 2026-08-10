import numpy as np


class Board:
    EMPTY = 0
    X = 1
    O = -1

    def __init__(self):
        self.reset()

    def reset(self):
        """
        Creates an empty Ultimate Tic-Tac-Toe board.
        """

        # 9 local boards × 9 cells
        self.board = np.zeros((9, 9), dtype=np.int8)

        # Status of each local board
        # 0  = ongoing
        # 1  = X won
        # -1 = O won
        # 2  = draw
        self.local_status = np.zeros(9, dtype=np.int8)

        # Local board that must be played next
        # -1 means any unfinished board
        self.active_board = -1

        # X starts
        self.current_player = self.X

        return self.board

    def copy(self):
        """
        Creates and returns a copy of the board.
        """

        new_board = Board()

        new_board.board = self.board.copy()
        new_board.local_status = self.local_status.copy()
        new_board.active_board = self.active_board
        new_board.current_player = self.current_player

        return new_board


if __name__ == "__main__":
    board = Board()

    print("Ultimate Tic-Tac-Toe Board Created!")

    print("\nBoard shape:")
    print(board.board.shape)

    print("\nInitial board:")
    print(board.board)

    print("\nLocal board status:")
    print(board.local_status)

    print("\nActive board:")
    print(board.active_board)

    print("\nCurrent player:")
    print(board.current_player)