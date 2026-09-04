import sys
import os
import tkinter as tk
from tkinter import messagebox

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
# IMPORT PROJECT CLASSES
# ============================================================

from game import UltimateTTTEnv
from ddqn_agent import DDQNAgent


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

MODEL_X_PATH = os.path.join(
    MODEL_DIR,
    "agent_x_final.pth"
)

MODEL_O_PATH = os.path.join(
    MODEL_DIR,
    "agent_o_final.pth"
)


# ============================================================
# GUI SETTINGS
# ============================================================

WINDOW_WIDTH = 1250
WINDOW_HEIGHT = 850

BOARD_SIZE = 700

MINI_BOARD_SIZE = BOARD_SIZE / 3
CELL_SIZE = MINI_BOARD_SIZE / 3


# ============================================================
# COLORS
# ============================================================

BACKGROUND = "#101820"
PANEL_BACKGROUND = "#162635"
BOARD_BACKGROUND = "#F5F7FA"

GRID_COLOR = "#303840"
MAJOR_GRID_COLOR = "#101820"

X_COLOR = "#1769E0"
O_COLOR = "#D83227"

ACTIVE_COLOR = "#1769E0"

X_WIN_BACKGROUND = "#DDF2DD"
O_WIN_BACKGROUND = "#F5DDDA"
DRAW_BACKGROUND = "#E0E4E8"

TEXT_COLOR = "#FFFFFF"
SECONDARY_TEXT = "#C7D1DC"

SUCCESS_COLOR = "#32CD32"
WARNING_COLOR = "#FFB020"


# ============================================================
# VISUALIZATION CLASS
# ============================================================

class UltimateTTTVisualizer:

    def __init__(self, root):

        self.root = root

        # ----------------------------------------------------
        # Window
        # ----------------------------------------------------

        self.root.title(
            "Ultimate Tic-Tac-Toe - DDQN Agent Self-Play"
        )

        self.root.geometry(
            f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}"
        )

        self.root.configure(
            bg=BACKGROUND
        )

        self.root.resizable(
            False,
            False
        )

        # ----------------------------------------------------
        # Game
        # ----------------------------------------------------

        self.env = None

        self.agent_x = None
        self.agent_o = None

        self.game_running = False
        self.game_finished = False

        self.move_count = 0
        self.last_action = None

        # ----------------------------------------------------
        # Speed
        #
        # Value is delay in milliseconds.
        # Larger = slower.
        # ----------------------------------------------------

        self.speed = tk.IntVar(
            value=500
        )

        # ----------------------------------------------------
        # Build interface
        # ----------------------------------------------------

        self.create_interface()

        # ----------------------------------------------------
        # Load trained models
        # ----------------------------------------------------

        self.load_models()

        # ----------------------------------------------------
        # Start new game
        # ----------------------------------------------------

        self.new_game()


    # ========================================================
    # CREATE GUI
    # ========================================================

    def create_interface(self):

        # ----------------------------------------------------
        # Main container
        # ----------------------------------------------------

        main_frame = tk.Frame(
            self.root,
            bg=BACKGROUND
        )

        main_frame.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=15
        )

        # ====================================================
        # HEADER
        # ====================================================

        header = tk.Frame(
            main_frame,
            bg=BACKGROUND
        )

        header.pack(
            fill="x",
            pady=(0, 12)
        )

        # ----------------------------------------------------
        # Winner label
        # ----------------------------------------------------

        self.winner_label = tk.Label(
            header,
            text="WINNER\n-",
            font=("Arial", 12, "bold"),
            fg=TEXT_COLOR,
            bg=PANEL_BACKGROUND,
            width=12,
            height=3
        )

        self.winner_label.pack(
            side="left",
            padx=5
        )

        # ----------------------------------------------------
        # Title
        # ----------------------------------------------------

        title_frame = tk.Frame(
            header,
            bg=BACKGROUND
        )

        title_frame.pack(
            side="left",
            expand=True
        )

        title = tk.Label(
            title_frame,
            text="ULTIMATE TIC-TAC-TOE",
            font=("Arial", 30, "bold"),
            fg=TEXT_COLOR,
            bg=BACKGROUND
        )

        title.pack()

        subtitle = tk.Label(
            title_frame,
            text="DDQN Agent Self-Play",
            font=("Arial", 15),
            fg=SECONDARY_TEXT,
            bg=BACKGROUND
        )

        subtitle.pack()

        # ----------------------------------------------------
        # New game button
        # ----------------------------------------------------

        new_game_button = tk.Button(
            header,
            text="⟳  NEW GAME",
            command=self.new_game,
            font=("Arial", 13, "bold"),
            fg=TEXT_COLOR,
            bg="#245D8C",
            activebackground="#327BB5",
            activeforeground=TEXT_COLOR,
            relief="flat",
            padx=20,
            pady=12,
            cursor="hand2"
        )

        new_game_button.pack(
            side="right",
            padx=5
        )

        # ====================================================
        # CONTENT AREA
        # ====================================================

        content = tk.Frame(
            main_frame,
            bg=BACKGROUND
        )

        content.pack(
            fill="both",
            expand=True
        )

        # ====================================================
        # BOARD AREA
        # ====================================================

        board_container = tk.Frame(
            content,
            bg=BOARD_BACKGROUND,
            width=BOARD_SIZE,
            height=BOARD_SIZE
        )

        board_container.pack(
            side="left",
            padx=(0, 12)
        )

        board_container.pack_propagate(
            False
        )

        self.canvas = tk.Canvas(
            board_container,
            width=BOARD_SIZE,
            height=BOARD_SIZE,
            bg=BOARD_BACKGROUND,
            highlightthickness=0
        )

        self.canvas.pack()

        # ====================================================
        # RIGHT PANEL
        # ====================================================

        right_panel = tk.Frame(
            content,
            bg=BACKGROUND,
            width=320
        )

        right_panel.pack(
            side="right",
            fill="y"
        )

        right_panel.pack_propagate(
            False
        )

        # ----------------------------------------------------
        # GAME STATUS
        # ----------------------------------------------------

        self.create_status_panel(
            right_panel
        )

        # ----------------------------------------------------
        # CONTROLS
        # ----------------------------------------------------

        self.create_control_panel(
            right_panel
        )

        # ----------------------------------------------------
        # LEGEND
        # ----------------------------------------------------

        self.create_legend_panel(
            right_panel
        )

        # ====================================================
        # BOTTOM MESSAGE
        # ====================================================

        self.message_label = tk.Label(
            main_frame,
            text="Ready",
            font=("Arial", 15, "bold"),
            fg=X_COLOR,
            bg=PANEL_BACKGROUND,
            height=2,
            relief="flat"
        )

        self.message_label.pack(
            fill="x",
            pady=(12, 0)
        )


    # ========================================================
    # STATUS PANEL
    # ========================================================

    def create_status_panel(
        self,
        parent
    ):

        frame = tk.LabelFrame(
            parent,
            text=" GAME STATUS ",
            font=("Arial", 12, "bold"),
            fg=X_COLOR,
            bg=PANEL_BACKGROUND,
            bd=1,
            relief="groove",
            padx=12,
            pady=12
        )

        frame.pack(
            fill="x",
            pady=(0, 10)
        )

        # ----------------------------------------------------
        # Current player
        # ----------------------------------------------------

        self.current_player_value = tk.Label(
            frame,
            text="X",
            font=("Arial", 16, "bold"),
            fg=X_COLOR,
            bg=PANEL_BACKGROUND,
            anchor="w"
        )

        self.create_status_row(
            frame,
            "Current Player:",
            self.current_player_value
        )

        # ----------------------------------------------------
        # Active board
        # ----------------------------------------------------

        self.active_board_value = tk.Label(
            frame,
            text="ANY",
            font=("Arial", 16, "bold"),
            fg=X_COLOR,
            bg=PANEL_BACKGROUND,
            anchor="w"
        )

        self.create_status_row(
            frame,
            "Active Board:",
            self.active_board_value
        )

        # ----------------------------------------------------
        # Move count
        # ----------------------------------------------------

        self.move_value = tk.Label(
            frame,
            text="0",
            font=("Arial", 16, "bold"),
            fg=TEXT_COLOR,
            bg=PANEL_BACKGROUND,
            anchor="w"
        )

        self.create_status_row(
            frame,
            "Move Count:",
            self.move_value
        )

        # ----------------------------------------------------
        # Game status
        # ----------------------------------------------------

        self.game_status_value = tk.Label(
            frame,
            text="Ready",
            font=("Arial", 16, "bold"),
            fg=SUCCESS_COLOR,
            bg=PANEL_BACKGROUND,
            anchor="w"
        )

        self.create_status_row(
            frame,
            "Game Status:",
            self.game_status_value
        )


    # ========================================================
    # STATUS ROW
    # ========================================================

    def create_status_row(
        self,
        parent,
        title,
        value_label
    ):

        row = tk.Frame(
            parent,
            bg=PANEL_BACKGROUND
        )

        row.pack(
            fill="x",
            pady=5
        )

        label = tk.Label(
            row,
            text=title,
            font=("Arial", 11),
            fg=SECONDARY_TEXT,
            bg=PANEL_BACKGROUND,
            width=15,
            anchor="w"
        )

        label.pack(
            side="left"
        )

        value_label.pack(
            in_=row,
            side="left"
        )


    # ========================================================
    # CONTROL PANEL
    # ========================================================

    def create_control_panel(
        self,
        parent
    ):

        frame = tk.LabelFrame(
            parent,
            text=" CONTROLS ",
            font=("Arial", 12, "bold"),
            fg=X_COLOR,
            bg=PANEL_BACKGROUND,
            bd=1,
            relief="groove",
            padx=12,
            pady=12
        )

        frame.pack(
            fill="x",
            pady=(0, 10)
        )

        # ----------------------------------------------------
        # Start
        # ----------------------------------------------------

        self.start_button = tk.Button(
            frame,
            text="▶  START",
            command=self.start_game,
            font=("Arial", 11, "bold"),
            fg=TEXT_COLOR,
            bg="#2876B8",
            activebackground="#368DD1",
            relief="flat",
            padx=12,
            pady=8,
            cursor="hand2"
        )

        self.start_button.grid(
            row=0,
            column=0,
            padx=5,
            pady=5,
            sticky="ew"
        )

        # ----------------------------------------------------
        # Pause
        # ----------------------------------------------------

        self.pause_button = tk.Button(
            frame,
            text="Ⅱ  PAUSE",
            command=self.pause_game,
            font=("Arial", 11, "bold"),
            fg=TEXT_COLOR,
            bg="#394754",
            activebackground="#4D5D6B",
            relief="flat",
            padx=12,
            pady=8,
            cursor="hand2"
        )

        self.pause_button.grid(
            row=0,
            column=1,
            padx=5,
            pady=5,
            sticky="ew"
        )

        # ----------------------------------------------------
        # Step
        # ----------------------------------------------------

        self.step_button = tk.Button(
            frame,
            text="↪  STEP",
            command=self.step_game,
            font=("Arial", 11, "bold"),
            fg=TEXT_COLOR,
            bg="#394754",
            activebackground="#4D5D6B",
            relief="flat",
            padx=12,
            pady=8,
            cursor="hand2"
        )

        self.step_button.grid(
            row=1,
            column=0,
            padx=5,
            pady=5,
            sticky="ew"
        )

        # ----------------------------------------------------
        # Restart
        # ----------------------------------------------------

        restart_button = tk.Button(
            frame,
            text="⟳  RESTART",
            command=self.new_game,
            font=("Arial", 11, "bold"),
            fg=TEXT_COLOR,
            bg="#C83B32",
            activebackground="#E04B40",
            relief="flat",
            padx=12,
            pady=8,
            cursor="hand2"
        )

        restart_button.grid(
            row=1,
            column=1,
            padx=5,
            pady=5,
            sticky="ew"
        )

        # ----------------------------------------------------
        # Grid configuration
        # ----------------------------------------------------

        frame.columnconfigure(
            0,
            weight=1
        )

        frame.columnconfigure(
            1,
            weight=1
        )

        # ====================================================
        # SPEED
        # ====================================================

        speed_label = tk.Label(
            frame,
            text="SPEED",
            font=("Arial", 12, "bold"),
            fg=X_COLOR,
            bg=PANEL_BACKGROUND
        )

        speed_label.grid(
            row=2,
            column=0,
            columnspan=2,
            sticky="w",
            padx=5,
            pady=(15, 3)
        )

        self.speed_scale = tk.Scale(
            frame,
            from_=1000,
            to=100,
            orient="horizontal",
            variable=self.speed,
            showvalue=False,
            bg=PANEL_BACKGROUND,
            fg=TEXT_COLOR,
            troughcolor="#485461",
            highlightthickness=0,
            length=250
        )

        self.speed_scale.grid(
            row=3,
            column=0,
            columnspan=2,
            padx=5
        )

        speed_text = tk.Frame(
            frame,
            bg=PANEL_BACKGROUND
        )

        speed_text.grid(
            row=4,
            column=0,
            columnspan=2,
            sticky="ew",
            padx=5
        )

        slow_label = tk.Label(
            speed_text,
            text="Slow",
            font=("Arial", 10),
            fg=SECONDARY_TEXT,
            bg=PANEL_BACKGROUND
        )

        slow_label.pack(
            side="left"
        )

        fast_label = tk.Label(
            speed_text,
            text="Fast",
            font=("Arial", 10),
            fg=SECONDARY_TEXT,
            bg=PANEL_BACKGROUND
        )

        fast_label.pack(
            side="right"
        )


    # ========================================================
    # LEGEND PANEL
    # ========================================================

    def create_legend_panel(
        self,
        parent
    ):

        frame = tk.LabelFrame(
            parent,
            text=" LEGEND ",
            font=("Arial", 12, "bold"),
            fg=X_COLOR,
            bg=PANEL_BACKGROUND,
            bd=1,
            relief="groove",
            padx=12,
            pady=12
        )

        frame.pack(
            fill="x"
        )

        # ----------------------------------------------------
        # X
        # ----------------------------------------------------

        x_label = tk.Label(
            frame,
            text="X",
            font=("Arial", 24, "bold"),
            fg=X_COLOR,
            bg=PANEL_BACKGROUND,
            width=3
        )

        x_label.grid(
            row=0,
            column=0,
            pady=4
        )

        tk.Label(
            frame,
            text="Agent X",
            font=("Arial", 11),
            fg=SECONDARY_TEXT,
            bg=PANEL_BACKGROUND,
            anchor="w"
        ).grid(
            row=0,
            column=1,
            sticky="w"
        )

        # ----------------------------------------------------
        # O
        # ----------------------------------------------------

        o_label = tk.Label(
            frame,
            text="O",
            font=("Arial", 24, "bold"),
            fg=O_COLOR,
            bg=PANEL_BACKGROUND,
            width=3
        )

        o_label.grid(
            row=1,
            column=0,
            pady=4
        )

        tk.Label(
            frame,
            text="Agent O",
            font=("Arial", 11),
            fg=SECONDARY_TEXT,
            bg=PANEL_BACKGROUND,
            anchor="w"
        ).grid(
            row=1,
            column=1,
            sticky="w"
        )

        # ----------------------------------------------------
        # Active board
        # ----------------------------------------------------

        active_canvas = tk.Canvas(
            frame,
            width=35,
            height=35,
            bg=PANEL_BACKGROUND,
            highlightthickness=0
        )

        active_canvas.grid(
            row=2,
            column=0
        )

        active_canvas.create_rectangle(
            5,
            5,
            30,
            30,
            outline=ACTIVE_COLOR,
            width=3
        )

        tk.Label(
            frame,
            text="Active Board",
            font=("Arial", 11),
            fg=SECONDARY_TEXT,
            bg=PANEL_BACKGROUND,
            anchor="w"
        ).grid(
            row=2,
            column=1,
            sticky="w"
        )

        # ----------------------------------------------------
        # Won X
        # ----------------------------------------------------

        x_win_canvas = tk.Canvas(
            frame,
            width=35,
            height=35,
            bg=PANEL_BACKGROUND,
            highlightthickness=0
        )

        x_win_canvas.grid(
            row=3,
            column=0
        )

        x_win_canvas.create_rectangle(
            5,
            5,
            30,
            30,
            fill=X_WIN_BACKGROUND,
            outline=X_WIN_BACKGROUND
        )

        tk.Label(
            frame,
            text="Board won by X",
            font=("Arial", 11),
            fg=SECONDARY_TEXT,
            bg=PANEL_BACKGROUND,
            anchor="w"
        ).grid(
            row=3,
            column=1,
            sticky="w"
        )

        # ----------------------------------------------------
        # Won O
        # ----------------------------------------------------

        o_win_canvas = tk.Canvas(
            frame,
            width=35,
            height=35,
            bg=PANEL_BACKGROUND,
            highlightthickness=0
        )

        o_win_canvas.grid(
            row=4,
            column=0
        )

        o_win_canvas.create_rectangle(
            5,
            5,
            30,
            30,
            fill=O_WIN_BACKGROUND,
            outline=O_WIN_BACKGROUND
        )

        tk.Label(
            frame,
            text="Board won by O",
            font=("Arial", 11),
            fg=SECONDARY_TEXT,
            bg=PANEL_BACKGROUND,
            anchor="w"
        ).grid(
            row=4,
            column=1,
            sticky="w"
        )


    # ========================================================
    # LOAD TRAINED MODELS
    # ========================================================

    def load_models(self):

        # ----------------------------------------------------
        # Check X model
        # ----------------------------------------------------

        if not os.path.exists(
            MODEL_X_PATH
        ):

            messagebox.showerror(
                "Model Not Found",
                "Agent X model was not found:\n\n"
                + MODEL_X_PATH
                + "\n\n"
                "Run training.py first."
            )

            self.root.destroy()

            return

        # ----------------------------------------------------
        # Check O model
        # ----------------------------------------------------

        if not os.path.exists(
            MODEL_O_PATH
        ):

            messagebox.showerror(
                "Model Not Found",
                "Agent O model was not found:\n\n"
                + MODEL_O_PATH
                + "\n\n"
                "Run training.py first."
            )

            self.root.destroy()

            return

        # ----------------------------------------------------
        # Create agents
        # ----------------------------------------------------

        try:

            self.agent_x = DDQNAgent(
                state_size=100,
                action_size=81,
                batch_size=32
            )

            self.agent_o = DDQNAgent(
                state_size=100,
                action_size=81,
                batch_size=32
            )

            # ------------------------------------------------
            # Load trained checkpoints
            # ------------------------------------------------

            self.agent_x.load(
                MODEL_X_PATH
            )

            self.agent_o.load(
                MODEL_O_PATH
            )

            # ------------------------------------------------
            # Evaluation mode
            #
            # We do NOT want exploration.
            # We do NOT want training.
            # ------------------------------------------------

            self.agent_x.epsilon = 0.0
            self.agent_o.epsilon = 0.0

            self.agent_x.online_network.eval()
            self.agent_o.online_network.eval()

            self.agent_x.target_network.eval()
            self.agent_o.target_network.eval()

            print()
            print(
                "Models loaded successfully."
            )

            print(
                "Agent X:",
                MODEL_X_PATH
            )

            print(
                "Agent O:",
                MODEL_O_PATH
            )

        except Exception as error:

            messagebox.showerror(
                "Model Loading Error",
                f"Could not load models.\n\n{error}"
            )

            self.root.destroy()


    # ========================================================
    # NEW GAME
    # ========================================================

    def new_game(self):

        # ----------------------------------------------------
        # Stop automatic play
        # ----------------------------------------------------

        self.game_running = False
        self.game_finished = False

        # ----------------------------------------------------
        # Create environment
        # ----------------------------------------------------

        self.env = UltimateTTTEnv()

        self.env.reset()

        self.move_count = 0
        self.last_action = None

        # ----------------------------------------------------
        # Update GUI
        # ----------------------------------------------------

        self.update_status()

        self.draw_board()

        self.message_label.config(
            text="Ready — press START to watch the agents play.",
            fg=X_COLOR
        )

        self.winner_label.config(
            text="WINNER\n-"
        )

        self.game_status_value.config(
            text="Ready",
            fg=SUCCESS_COLOR
        )


    # ========================================================
    # START GAME
    # ========================================================

    def start_game(self):

        if self.game_finished:

            return

        if self.game_running:

            return

        self.game_running = True

        self.message_label.config(
            text="Agents are playing...",
            fg=X_COLOR
        )

        self.play_next_move()


    # ========================================================
    # PAUSE GAME
    # ========================================================

    def pause_game(self):

        self.game_running = False

        if not self.game_finished:

            self.message_label.config(
                text="Game paused.",
                fg=WARNING_COLOR
            )

            self.game_status_value.config(
                text="Paused",
                fg=WARNING_COLOR
            )


    # ========================================================
    # STEP ONE MOVE
    # ========================================================

    def step_game(self):

        if self.game_finished:

            return

        self.game_running = False

        self.play_one_move()


    # ========================================================
    # AUTOMATIC NEXT MOVE
    # ========================================================

    def play_next_move(self):

        if not self.game_running:

            return

        if self.game_finished:

            return

        self.play_one_move()

        if (
            self.game_running
            and not self.game_finished
        ):

            delay = self.speed.get()

            self.root.after(
                delay,
                self.play_next_move
            )


    # ========================================================
    # PLAY ONE MOVE
    # ========================================================

    def play_one_move(self):

        if self.env is None:

            return

        if self.env.done:

            self.finish_game()

            return

        # ----------------------------------------------------
        # Get current player
        # ----------------------------------------------------

        current_player = (
            self.env.get_current_player()
        )

        # ----------------------------------------------------
        # Select agent
        # ----------------------------------------------------

        if current_player == self.env.X:

            agent = self.agent_x

            player_name = "X"

        else:

            agent = self.agent_o

            player_name = "O"

        # ----------------------------------------------------
        # Get legal actions
        # ----------------------------------------------------

        valid_actions = (
            self.env.get_valid_moves()
        )

        if not valid_actions:

            self.game_running = False

            messagebox.showerror(
                "Game Error",
                "No legal moves are available."
            )

            return

        # ----------------------------------------------------
        # Select BEST action
        #
        # training=False means no random exploration.
        # ----------------------------------------------------

        action = agent.select_action(
            self.env.get_state(),
            valid_actions,
            training=False
        )

        self.last_action = action

        # ----------------------------------------------------
        # Convert action to board/cell
        # ----------------------------------------------------

        local_board = action // 9
        cell = action % 9

        # ----------------------------------------------------
        # Environment step
        # ----------------------------------------------------

        next_state, reward, done, info = (
            self.env.step(action)
        )

        self.move_count += 1

        # ----------------------------------------------------
        # Draw updated board
        # ----------------------------------------------------

        self.draw_board(
            last_local_board=local_board,
            last_cell=cell
        )

        # ----------------------------------------------------
        # Update status
        # ----------------------------------------------------

        self.update_status()

        # ----------------------------------------------------
        # Message
        # ----------------------------------------------------

        if not done:

            next_player = (
                "X"
                if self.env.get_current_player()
                == self.env.X
                else "O"
            )

            self.message_label.config(
                text=(
                    f"Agent {player_name} played "
                    f"board {local_board + 1}, "
                    f"cell {cell + 1}. "
                    f"Agent {next_player} is thinking..."
                ),
                fg=(
                    X_COLOR
                    if next_player == "X"
                    else O_COLOR
                )
            )

        # ----------------------------------------------------
        # Game finished
        # ----------------------------------------------------

        else:

            self.finish_game()


    # ========================================================
    # FINISH GAME
    # ========================================================

    def finish_game(self):

        self.game_finished = True
        self.game_running = False

        # ----------------------------------------------------
        # Get winner
        # ----------------------------------------------------

        info = self.env.get_game_info()

        winner = info.get(
            "winner"
        )

        # ----------------------------------------------------
        # Winner
        # ----------------------------------------------------

        if winner == self.env.X:

            winner_text = "X"

            self.winner_label.config(
                text="WINNER\nX",
                fg=X_COLOR
            )

            self.message_label.config(
                text=(
                    "🎉 Agent X wins the game!"
                ),
                fg=X_COLOR
            )

        elif winner == self.env.O:

            winner_text = "O"

            self.winner_label.config(
                text="WINNER\nO",
                fg=O_COLOR
            )

            self.message_label.config(
                text=(
                    "🎉 Agent O wins the game!"
                ),
                fg=O_COLOR
            )

        else:

            winner_text = "DRAW"

            self.winner_label.config(
                text="WINNER\nDRAW",
                fg=SECONDARY_TEXT
            )

            self.message_label.config(
                text="Game ended in a draw.",
                fg=SECONDARY_TEXT
            )

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        self.game_status_value.config(
            text="Finished",
            fg=SUCCESS_COLOR
        )

        # ----------------------------------------------------
        # Final board redraw
        # ----------------------------------------------------

        self.draw_board()

        print()
        print(
            "=========================================="
        )

        print(
            "VISUAL GAME FINISHED"
        )

        print(
            "Winner:",
            winner_text
        )

        print(
            "Moves:",
            self.move_count
        )

        print(
            "=========================================="
        )


    # ========================================================
    # UPDATE STATUS PANEL
    # ========================================================

    def update_status(self):

        if self.env is None:

            return

        # ----------------------------------------------------
        # Current player
        # ----------------------------------------------------

        if self.env.get_current_player() == self.env.X:

            current_player = "X"

            player_color = X_COLOR

        else:

            current_player = "O"

            player_color = O_COLOR

        self.current_player_value.config(
            text=current_player,
            fg=player_color
        )

        # ----------------------------------------------------
        # Active board
        # ----------------------------------------------------

        active_board = (
            self.env.board.active_board
        )

        if active_board == -1:

            active_text = "ANY"

        else:

            active_text = (
                f"{active_board + 1}"
            )

        self.active_board_value.config(
            text=active_text
        )

        # ----------------------------------------------------
        # Move count
        # ----------------------------------------------------

        self.move_value.config(
            text=str(
                self.env.move_count
            )
        )

        # ----------------------------------------------------
        # Game status
        # ----------------------------------------------------

        if self.env.done:

            self.game_status_value.config(
                text="Finished",
                fg=SUCCESS_COLOR
            )

        elif self.game_running:

            self.game_status_value.config(
                text="In Progress",
                fg=SUCCESS_COLOR
            )

        else:

            self.game_status_value.config(
                text="Paused",
                fg=WARNING_COLOR
            )


    # ========================================================
    # DRAW BOARD
    # ========================================================

    def draw_board(
        self,
        last_local_board=None,
        last_cell=None
    ):

        self.canvas.delete(
            "all"
        )

        # ----------------------------------------------------
        # Draw background
        # ----------------------------------------------------

        self.canvas.create_rectangle(
            0,
            0,
            BOARD_SIZE,
            BOARD_SIZE,
            fill=BOARD_BACKGROUND,
            outline=""
        )

        # ----------------------------------------------------
        # Draw local board backgrounds
        # ----------------------------------------------------

        for local_board in range(9):

            global_row = local_board // 3
            global_col = local_board % 3

            x0 = (
                global_col
                * MINI_BOARD_SIZE
            )

            y0 = (
                global_row
                * MINI_BOARD_SIZE
            )

            x1 = x0 + MINI_BOARD_SIZE
            y1 = y0 + MINI_BOARD_SIZE

            status = (
                self.env.board.local_status[
                    local_board
                ]
            )

            # -----------------------------------------------
            # Background according to board result
            # -----------------------------------------------

            if status == self.env.X:

                fill = X_WIN_BACKGROUND

            elif status == self.env.O:

                fill = O_WIN_BACKGROUND

            elif status == self.env.DRAW:

                fill = DRAW_BACKGROUND

            else:

                fill = BOARD_BACKGROUND

            self.canvas.create_rectangle(
                x0,
                y0,
                x1,
                y1,
                fill=fill,
                outline=""
            )

        # ====================================================
        # ACTIVE BOARD HIGHLIGHT
        # ====================================================

        active_board = (
            self.env.board.active_board
        )

        if active_board != -1:

            global_row = active_board // 3
            global_col = active_board % 3

            x0 = (
                global_col
                * MINI_BOARD_SIZE
            )

            y0 = (
                global_row
                * MINI_BOARD_SIZE
            )

            x1 = x0 + MINI_BOARD_SIZE
            y1 = y0 + MINI_BOARD_SIZE

            self.canvas.create_rectangle(
                x0 + 3,
                y0 + 3,
                x1 - 3,
                y1 - 3,
                outline=ACTIVE_COLOR,
                width=5
            )

        # ====================================================
        # DRAW CELL GRID
        # ====================================================

        for row in range(10):

            y = row * CELL_SIZE

            # Major line every 3 cells
            if row % 3 == 0:

                width = 5

            else:

                width = 1

            self.canvas.create_line(
                0,
                y,
                BOARD_SIZE,
                y,
                fill=(
                    MAJOR_GRID_COLOR
                    if row % 3 == 0
                    else GRID_COLOR
                ),
                width=width
            )

        for col in range(10):

            x = col * CELL_SIZE

            if col % 3 == 0:

                width = 5

            else:

                width = 1

            self.canvas.create_line(
                x,
                0,
                x,
                BOARD_SIZE,
                fill=(
                    MAJOR_GRID_COLOR
                    if col % 3 == 0
                    else GRID_COLOR
                ),
                width=width
            )

        # ====================================================
        # DRAW X AND O PIECES
        # ====================================================

        for local_board in range(9):

            global_board_row = (
                local_board // 3
            )

            global_board_col = (
                local_board % 3
            )

            for cell in range(9):

                value = (
                    self.env.board.board[
                        local_board
                    ][cell]
                )

                if value == self.env.EMPTY:

                    continue

                local_cell_row = (
                    cell // 3
                )

                local_cell_col = (
                    cell % 3
                )

                global_row = (
                    global_board_row * 3
                    + local_cell_row
                )

                global_col = (
                    global_board_col * 3
                    + local_cell_col
                )

                center_x = (
                    global_col * CELL_SIZE
                    + CELL_SIZE / 2
                )

                center_y = (
                    global_row * CELL_SIZE
                    + CELL_SIZE / 2
                )

                padding = 15

                left = (
                    global_col * CELL_SIZE
                    + padding
                )

                right = (
                    (global_col + 1)
                    * CELL_SIZE
                    - padding
                )

                top = (
                    global_row * CELL_SIZE
                    + padding
                )

                bottom = (
                    (global_row + 1)
                    * CELL_SIZE
                    - padding
                )

                # -------------------------------------------
                # X
                # -------------------------------------------

                if value == self.env.X:

                    self.canvas.create_line(
                        left,
                        top,
                        right,
                        bottom,
                        fill=X_COLOR,
                        width=7
                    )

                    self.canvas.create_line(
                        right,
                        top,
                        left,
                        bottom,
                        fill=X_COLOR,
                        width=7
                    )

                # -------------------------------------------
                # O
                # -------------------------------------------

                elif value == self.env.O:

                    self.canvas.create_oval(
                        left,
                        top,
                        right,
                        bottom,
                        outline=O_COLOR,
                        width=7
                    )

        # ====================================================
        # HIGHLIGHT LAST MOVE
        # ====================================================

        if (
            last_local_board is not None
            and last_cell is not None
        ):

            board_row = (
                last_local_board // 3
            )

            board_col = (
                last_local_board % 3
            )

            cell_row = (
                last_cell // 3
            )

            cell_col = (
                last_cell % 3
            )

            global_row = (
                board_row * 3
                + cell_row
            )

            global_col = (
                board_col * 3
                + cell_col
            )

            x0 = (
                global_col * CELL_SIZE
            )

            y0 = (
                global_row * CELL_SIZE
            )

            x1 = x0 + CELL_SIZE
            y1 = y0 + CELL_SIZE

            self.canvas.create_rectangle(
                x0 + 4,
                y0 + 4,
                x1 - 4,
                y1 - 4,
                outline="#FFB020",
                width=3
            )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "=========================================="
    )

    print(
        " ULTIMATE TIC-TAC-TOE VISUALIZER"
    )

    print(
        "=========================================="
    )

    print()

    print(
        "Loading trained DDQN agents..."
    )

    root = tk.Tk()

    app = UltimateTTTVisualizer(
        root
    )

    root.mainloop()