import numpy as np

from board import Board
from rules import UltimateTTTRules


def create_game():
    """
    Create a fresh Ultimate Tic-Tac-Toe game.
    """
    board = Board()
    rules = UltimateTTTRules(board)

    return board, rules


# ============================================================
# TEST 1: INITIAL BOARD
# ============================================================

def test_initial_board():
    board, rules = create_game()

    # Board should contain 9 local boards × 9 cells
    assert board.board.shape == (9, 9)

    # Every cell should be empty
    assert np.all(board.board == 0)

    # No local board should be finished
    assert np.all(board.local_status == 0)

    # No specific board is required at the beginning
    assert board.active_board == -1

    # X starts
    assert board.current_player == 1

    # All 81 moves should initially be legal
    assert len(rules.get_valid_moves()) == 81

    print("PASS: Initial board")


# ============================================================
# TEST 2: ACTION CONVERSION
# ============================================================

def test_action_conversion():
    board, rules = create_game()

    # action 0 -> board 0, cell 0
    assert rules.action_to_position(0) == (0, 0)

    # action 8 -> board 0, cell 8
    assert rules.action_to_position(8) == (0, 8)

    # action 9 -> board 1, cell 0
    assert rules.action_to_position(9) == (1, 0)

    # action 23 -> board 2, cell 5
    assert rules.action_to_position(23) == (2, 5)

    # action 80 -> board 8, cell 8
    assert rules.action_to_position(80) == (8, 8)

    # Reverse conversion
    assert rules.position_to_action(0, 0) == 0
    assert rules.position_to_action(0, 8) == 8
    assert rules.position_to_action(1, 0) == 9
    assert rules.position_to_action(2, 5) == 23
    assert rules.position_to_action(8, 8) == 80

    print("PASS: Action conversion")


# ============================================================
# TEST 3: FIRST MOVE
# ============================================================

def test_first_move():
    board, rules = create_game()

    result = rules.apply_move(4)

    # Action 4 = local board 0, cell 4
    assert result["local_board"] == 0
    assert result["cell"] == 4

    # X made the move
    assert result["player"] == 1

    # Cell should now contain X
    assert board.board[0][4] == 1

    # O should now play
    assert board.current_player == -1

    # Because X played cell 4,
    # O must play local board 4
    assert board.active_board == 4

    # All 9 cells of board 4 are still available
    assert len(rules.get_valid_moves()) == 9

    print("PASS: First move")


# ============================================================
# TEST 4: ACTIVE BOARD RESTRICTION
# ============================================================

def test_active_board_restriction():
    board, rules = create_game()

    # X plays board 0, cell 4
    rules.apply_move(4)

    # O must play in board 4.
    # Action 36 = board 4, cell 0
    assert rules.is_valid_move(36)

    # Action 0 = board 0, cell 0
    # This should be illegal because board 4 is required.
    assert not rules.is_valid_move(0)

    # Action 9 = board 1, cell 0
    assert not rules.is_valid_move(9)

    # Action 72 = board 8, cell 0
    assert not rules.is_valid_move(72)

    print("PASS: Active board restriction")


# ============================================================
# TEST 5: OCCUPIED CELL
# ============================================================

def test_occupied_cell():
    board, rules = create_game()

    # X plays board 0, cell 4
    rules.apply_move(4)

    # The same cell cannot be played again.
    assert not rules.is_valid_move(4)

    print("PASS: Occupied cell")


# ============================================================
# TEST 6: PLAYER SWITCHING
# ============================================================

def test_player_switching():
    board, rules = create_game()

    # X starts
    assert board.current_player == 1

    # X plays
    rules.apply_move(4)

    # Now O's turn
    assert board.current_player == -1

    # O plays board 4, cell 0
    rules.apply_move(36)

    # Now X's turn
    assert board.current_player == 1

    # X must now play board 0
    # because O played cell 0
    assert board.active_board == 0

    print("PASS: Player switching")


# ============================================================
# TEST 7: LOCAL BOARD WIN
# ============================================================

def test_local_board_win():
    board, rules = create_game()

    # Manually create:
    #
    # X X X
    # O O -
    # - - -
    #
    # in local board 0.

    board.board[0] = np.array([
        1, 1, 1,
        -1, -1, 0,
        0, 0, 0
    ])

    result = rules.check_local_winner(0)

    assert result == 1

    print("PASS: Local board X win")


# ============================================================
# TEST 8: LOCAL BOARD O WIN
# ============================================================

def test_local_board_o_win():
    board, rules = create_game()

    # O O O
    # X X -
    # - - -
    #
    # O should win.

    board.board[0] = np.array([
        -1, -1, -1,
        1, 1, 0,
        0, 0, 0
    ])

    result = rules.check_local_winner(0)

    assert result == -1

    print("PASS: Local board O win")


# ============================================================
# TEST 9: LOCAL BOARD DRAW
# ============================================================

def test_local_board_draw():
    board, rules = create_game()

    # Full board with no winning line.
    #
    # X O X
    # X O O
    # O X X

    board.board[0] = np.array([
        1, -1, 1,
        1, -1, -1,
        -1, 1, 1
    ])

    result = rules.check_local_winner(0)

    assert result == 2

    print("PASS: Local board draw")


# ============================================================
# TEST 10: LOCAL BOARD ONGOING
# ============================================================

def test_local_board_ongoing():
    board, rules = create_game()

    board.board[0] = np.array([
        1, -1, 0,
        0, 1, 0,
        0, 0, -1
    ])

    result = rules.check_local_winner(0)

    assert result == 0

    print("PASS: Local board ongoing")


# ============================================================
# TEST 11: GLOBAL WIN
# ============================================================

def test_global_win():
    board, rules = create_game()

    # X has won local boards:
    #
    # X X X
    # - - -
    # - - -
    #
    board.local_status[0] = 1
    board.local_status[1] = 1
    board.local_status[2] = 1

    result = rules.check_global_winner()

    assert result == 1

    print("PASS: Global X win")


# ============================================================
# TEST 12: GLOBAL O WIN
# ============================================================

def test_global_o_win():
    board, rules = create_game()

    # O has won:
    #
    # O - -
    # O - -
    # O - -
    #
    board.local_status[0] = -1
    board.local_status[3] = -1
    board.local_status[6] = -1

    result = rules.check_global_winner()

    assert result == -1

    print("PASS: Global O win")


# ============================================================
# TEST 13: GLOBAL DRAW
# ============================================================

def test_global_draw():
    board, rules = create_game()

    # All local boards are finished,
    # but nobody has three local boards in a row.

    board.local_status[:] = np.array([
        1, -1, 1,
        1, -1, -1,
        -1, 1, -1
    ])

    result = rules.check_global_winner()

    assert result == 2

    print("PASS: Global draw")


# ============================================================
# TEST 14: FINISHED LOCAL BOARD
# ============================================================

def test_finished_local_board():
    board, rules = create_game()

    # Mark local board 4 as won by X.
    board.local_status[4] = 1

    # Therefore no move in board 4 should be legal.
    for cell in range(9):
        action = rules.position_to_action(4, cell)

        assert not rules.is_valid_move(action)

    print("PASS: Finished local board restriction")


# ============================================================
# TEST 15: DESTINATION BOARD ALREADY FINISHED
# ============================================================

def test_finished_destination_board():
    board, rules = create_game()

    # X plays board 0, cell 4.
    #
    # Normally this sends O to board 4.
    #
    # But if board 4 is already finished,
    # O should be allowed to play anywhere.

    board.local_status[4] = 1

    rules.apply_move(4)

    # Since board 4 is already finished,
    # active_board should become -1.
    assert board.active_board == -1

    # Any unfinished board should now be available.
    assert rules.is_valid_move(0)
    assert rules.is_valid_move(9)
    assert rules.is_valid_move(18)

    print("PASS: Finished destination board")


# ============================================================
# RUN ALL TESTS
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("==========================================")
    print(" ULTIMATE TIC-TAC-TOE RULE TESTS")
    print("==========================================")
    print()

    test_initial_board()
    test_action_conversion()
    test_first_move()
    test_active_board_restriction()
    test_occupied_cell()
    test_player_switching()

    test_local_board_win()
    test_local_board_o_win()
    test_local_board_draw()
    test_local_board_ongoing()

    test_global_win()
    test_global_o_win()
    test_global_draw()

    test_finished_local_board()
    test_finished_destination_board()

    print()
    print("==========================================")
    print(" ALL TESTS PASSED!")
    print("==========================================")