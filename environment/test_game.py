import numpy as np

from game import UltimateTTTEnv


# ============================================================
# HELPER
# ============================================================

def create_env():
    """Create a fresh environment."""
    return UltimateTTTEnv()


# ============================================================
# TEST 1: RESET
# ============================================================

def test_reset():
    env = create_env()

    state = env.reset()

    # State must be a NumPy array
    assert isinstance(state, np.ndarray)

    # Our state representation has 100 features
    assert state.shape == (100,)

    # All cells should initially be empty
    assert np.all(env.board.board == 0)

    # No local board should be finished
    assert np.all(env.board.local_status == 0)

    # No active board at the beginning
    assert env.board.active_board == -1

    # X starts
    assert env.board.current_player == 1

    # Game should not be over
    assert env.done is False

    # No moves yet
    assert env.move_count == 0

    print("PASS: Reset")


# ============================================================
# TEST 2: INITIAL VALID ACTIONS
# ============================================================

def test_initial_valid_actions():
    env = create_env()

    env.reset()

    valid_moves = env.get_valid_moves()

    # There are 81 cells at the beginning
    assert len(valid_moves) == 81

    # Actions must be 0-80
    assert valid_moves == list(range(81))

    print("PASS: Initial valid actions")


# ============================================================
# TEST 3: STATE SIZE
# ============================================================

def test_state_size():
    env = create_env()

    state = env.reset()

    assert state.shape == (100,)
    assert len(state) == 100

    print("PASS: State size")


# ============================================================
# TEST 4: INITIAL STATE VALUES
# ============================================================

def test_initial_state():
    env = create_env()

    state = env.reset()

    # First 81 values represent board cells
    assert np.all(state[:81] == 0)

    # Next 9 values represent local board status
    assert np.all(state[81:90] == 0)

    # Next 9 values represent active board
    # No active board initially
    assert np.all(state[90:99] == 0)

    # Last value represents current player
    assert state[99] == 1

    print("PASS: Initial state values")


# ============================================================
# TEST 5: FIRST STEP
# ============================================================

def test_first_step():
    env = create_env()

    env.reset()

    next_state, reward, done, info = env.step(4)

    # Action 4 = board 0, cell 4
    assert info["local_board"] == 0
    assert info["cell"] == 4

    # X played
    assert info["player"] == 1

    # Move is not terminal
    assert reward == 0.0
    assert done is False

    # Board cell should contain X
    assert env.board.board[0][4] == 1

    # O should now play
    assert env.board.current_player == -1

    # Cell 4 sends opponent to local board 4
    assert env.board.active_board == 4

    # State should still have 100 features
    assert next_state.shape == (100,)

    print("PASS: First step")


# ============================================================
# TEST 6: STATE AFTER MOVE
# ============================================================

def test_state_after_move():
    env = create_env()

    state = env.reset()

    next_state, reward, done, info = env.step(4)

    # Board state occupies first 81 positions.
    #
    # Action 4 corresponds to board 0, cell 4.
    assert next_state[4] == 1

    # All other board cells should still be empty
    occupied_positions = np.where(next_state[:81] != 0)[0]

    assert list(occupied_positions) == [4]

    # Local board status should still be ongoing
    assert np.all(next_state[81:90] == 0)

    # Active board should be board 4
    assert next_state[90 + 4] == 1

    # Current player should now be O
    assert next_state[99] == -1

    print("PASS: State after move")


# ============================================================
# TEST 7: ACTIVE BOARD RESTRICTION
# ============================================================

def test_active_board_restriction():
    env = create_env()

    env.reset()

    # X plays board 0, cell 4.
    env.step(4)

    # O must play in board 4.
    valid_moves = env.get_valid_moves()

    # Board 4 corresponds to actions 36-44.
    assert valid_moves == list(range(36, 45))

    print("PASS: Active board restriction")


# ============================================================
# TEST 8: ILLEGAL MOVE
# ============================================================

def test_illegal_move():
    env = create_env()

    env.reset()

    # X plays board 0, cell 4.
    env.step(4)

    # O is required to play in board 4.
    # Action 0 belongs to board 0, so it is illegal.
    state, reward, done, info = env.step(0)

    assert info["illegal_move"] is True

    # Current environment implementation gives
    # an illegal move a reward of -1.
    assert reward == -1.0

    # Current implementation ends the episode.
    assert done is True

    print("PASS: Illegal move")


# ============================================================
# TEST 9: GAME CANNOT CONTINUE AFTER DONE
# ============================================================

def test_step_after_done():
    env = create_env()

    env.reset()

    # Make an illegal move.
    env.step(4)
    env.step(0)

    assert env.done is True

    # A further step must fail.
    try:
        env.step(36)

        # If no exception occurred, the test must fail.
        assert False

    except ValueError:
        pass

    print("PASS: Step after game over")


# ============================================================
# TEST 10: RESET AFTER GAME
# ============================================================

def test_reset_after_game():
    env = create_env()

    env.reset()

    # Create an illegal move and finish episode.
    env.step(4)
    env.step(0)

    assert env.done is True

    # Reset should start a completely new game.
    state = env.reset()

    assert env.done is False
    assert env.move_count == 0
    assert env.board.current_player == 1
    assert env.board.active_board == -1

    assert np.all(env.board.board == 0)
    assert np.all(env.board.local_status == 0)

    assert state.shape == (100,)

    print("PASS: Reset after game")


# ============================================================
# TEST 11: PLAYER SWITCHING
# ============================================================

def test_player_switching():
    env = create_env()

    env.reset()

    # X starts
    assert env.get_current_player() == 1

    # X moves
    env.step(4)

    # O's turn
    assert env.get_current_player() == -1

    # O plays board 4, cell 0.
    env.step(36)

    # X's turn
    assert env.get_current_player() == 1

    # O played cell 0, so X must play board 0.
    assert env.board.active_board == 0

    print("PASS: Player switching")


# ============================================================
# TEST 12: VALID MOVE COUNT AFTER TWO MOVES
# ============================================================

def test_valid_moves_after_two_moves():
    env = create_env()

    env.reset()

    # X: board 0, cell 4
    env.step(4)

    # O: board 4, cell 0
    env.step(36)

    # X must play board 0.
    valid_moves = env.get_valid_moves()

    # Board 0 has one occupied cell.
    assert len(valid_moves) == 8

    # Valid actions in board 0 are:
    # 0,1,2,3,5,6,7,8
    assert valid_moves == [
        0, 1, 2, 3, 5, 6, 7, 8
    ]

    print("PASS: Valid move count after two moves")


# ============================================================
# TEST 13: LOCAL BOARD WIN
# ============================================================

def test_local_board_win():
    env = create_env()

    env.reset()

    # Directly create an X-winning local board.
    env.board.board[0] = np.array([
        1, 1, 1,
        -1, -1, 0,
        0, 0, 0
    ])

    result = env.rules.check_local_winner(0)

    assert result == 1

    print("PASS: Local board win")


# ============================================================
# TEST 14: GLOBAL WIN
# ============================================================

def test_global_win():
    env = create_env()

    env.reset()

    # X has won local boards 0, 1 and 2.
    env.board.local_status[0] = 1
    env.board.local_status[1] = 1
    env.board.local_status[2] = 1

    result = env.rules.check_global_winner()

    assert result == 1

    print("PASS: Global win")


# ============================================================
# TEST 15: GAME INFORMATION
# ============================================================

def test_game_info():
    env = create_env()

    env.reset()

    info = env.get_game_info()

    assert info["current_player"] == 1
    assert info["active_board"] == -1
    assert info["move_count"] == 0
    assert info["done"] is False

    assert np.all(info["local_status"] == 0)

    print("PASS: Game information")


# ============================================================
# TEST 16: RENDER
# ============================================================

def test_render():
    env = create_env()

    env.reset()

    # render() should execute without errors.
    env.render()

    print("PASS: Render")


# ============================================================
# RUN ALL TESTS
# ============================================================

if __name__ == "__main__":

    print()
    print("==========================================")
    print(" ULTIMATE TIC-TAC-TOE GAME TESTS")
    print("==========================================")
    print()

    test_reset()
    test_initial_valid_actions()
    test_state_size()
    test_initial_state()
    test_first_step()
    test_state_after_move()
    test_active_board_restriction()
    test_illegal_move()
    test_step_after_done()
    test_reset_after_game()
    test_player_switching()
    test_valid_moves_after_two_moves()
    test_local_board_win()
    test_global_win()
    test_game_info()
    test_render()

    print()
    print("==========================================")
    print(" ALL GAME TESTS PASSED!")
    print("==========================================")