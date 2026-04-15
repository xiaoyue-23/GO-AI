"""
Simple Go GUI for 5x5 human-vs-AI play.

Run:
    python gui_play.py
"""

import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

from dlgo import GameState
from dlgo.goboard import Move
from dlgo.gotypes import Player, Point
from dlgo.scoring import compute_game_result


class RoundedButton(tk.Canvas):
    def __init__(
        self, parent, text, command, width=105, height=34, radius=12, **kwargs
    ):
        try:
            bg_color = ttk.Style().lookup("TFrame", "background")
        except:
            bg_color = "#F0F0F0"
        bg_color = bg_color or "#F0F0F0"

        super().__init__(
            parent,
            width=width,
            height=height,
            bg=bg_color,
            highlightthickness=0,
            **kwargs,
        )
        self.command = command

        self._create_rounded_rect(
            2, 4, width - 1, height - 1, radius, fill="#888888", outline=""
        )
        self._create_rounded_rect(
            0, 0, width - 2, height - 2, radius, fill="#FFFFFF", outline=""
        )
        self.face = self._create_rounded_rect(
            1, 1, width - 3, height - 3, radius, fill="#E8E8E8", outline=""
        )
        self.text_id = self.create_text(
            width / 2,
            height / 2,
            text=text,
            fill="#111111",
            font=("Microsoft YaHei UI", 10, "bold"),
        )

        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

    def _create_rounded_rect(self, x1, y1, x2, y2, r, **kwargs):
        points = [
            x1 + r,
            y1,
            x2 - r,
            y1,
            x2,
            y1,
            x2,
            y1 + r,
            x2,
            y2 - r,
            x2,
            y2,
            x2 - r,
            y2,
            x1 + r,
            y2,
            x1,
            y2,
            x1,
            y2 - r,
            x1,
            y1 + r,
            x1,
            y1,
        ]
        return self.create_polygon(points, smooth=True, **kwargs)

    def _on_press(self, event):
        self.move(self.face, 1, 1)
        self.move(self.text_id, 1, 1)

    def _on_release(self, event):
        self.move(self.face, -1, -1)
        self.move(self.text_id, -1, -1)
        if 0 <= event.x <= self.winfo_width() and 0 <= event.y <= self.winfo_height():
            self.command()

    def _on_enter(self, event):
        self.itemconfig(self.face, fill="#D0D0D0")

    def _on_leave(self, event):
        self.itemconfig(self.face, fill="#E8E8E8")


class GoApp(tk.Tk):
    def __init__(self, board_size=5):
        super().__init__()
        self.title("Go 5x5 - Human vs AI")
        self.geometry("640x680")
        self.resizable(False, False)

        self.board_size = board_size
        self.margin = 32
        self.cell = 64
        self.stone_radius = 22

        self.play_mode = tk.StringVar(value="human_vs_ai")
        self.black_ai_name = tk.StringVar(value="mcts")
        self.white_ai_name = tk.StringVar(value="random")
        self.black_heuristic = tk.StringVar(value="capture_center")
        self.white_heuristic = tk.StringVar(value="capture_center")
        self.black_minimax_depth = tk.IntVar(value=3)
        self.white_minimax_depth = tk.IntVar(value=3)
        self.human_color = tk.StringVar(value="black")
        self.status_text = tk.StringVar(value="Ready")

        self.ai_thinking = False
        self.history = []
        self.end_reason = ""
        self.game_over_notified = False

        self._build_ui()
        self.play_mode.trace_add("write", self._on_mode_change)
        self.human_color.trace_add("write", self._on_human_color_change)
        self.black_ai_name.trace_add("write", self._on_agent_change)
        self.white_ai_name.trace_add("write", self._on_agent_change)
        self._update_strategy_controls()
        self.new_game()

    def _build_ui(self):
        top = ttk.Frame(self, padding=8)
        top.pack(side="top", fill="x")

        # 将 Setup 和 Actions 放在 top 区域
        setup = ttk.LabelFrame(top, text="Match Setup", padding=8)
        setup.pack(side="top", fill="x")

        for i in range(4):
            setup.columnconfigure(i, weight=1, uniform="setup_col")

        ttk.Label(setup, text="Mode:").grid(row=0, column=0, padx=4, pady=3, sticky="e")
        m_mode = tk.OptionMenu(setup, self.play_mode, "human_vs_ai", "ai_vs_ai")
        m_mode.config(relief="raised", bd=2, bg="#E8E8E8", activebackground="#D0D0D0")
        m_mode.grid(row=0, column=1, padx=4, pady=3, sticky="we")

        self.human_label = ttk.Label(setup, text="Human:")
        self.human_label.grid(row=0, column=2, padx=4, pady=3, sticky="e")
        self.human_menu = tk.OptionMenu(setup, self.human_color, "black", "white")
        self.human_menu.config(
            relief="raised", bd=2, bg="#E8E8E8", activebackground="#D0D0D0"
        )
        self.human_menu.grid(row=0, column=3, padx=4, pady=3, sticky="we")

        self.black_ai_label = ttk.Label(setup, text="Black AI:")
        self.black_ai_label.grid(row=1, column=0, padx=4, pady=3, sticky="e")
        self.black_ai_menu = tk.OptionMenu(
            setup, self.black_ai_name, "random", "mcts", "minimax"
        )
        self.black_ai_menu.config(
            relief="raised", bd=2, bg="#E8E8E8", activebackground="#D0D0D0"
        )
        self.black_ai_menu.grid(row=1, column=1, padx=4, pady=3, sticky="we")

        self.white_ai_label = ttk.Label(setup, text="White AI:")
        self.white_ai_label.grid(row=2, column=0, padx=4, pady=3, sticky="e")
        self.white_ai_menu = tk.OptionMenu(
            setup, self.white_ai_name, "random", "mcts", "minimax"
        )
        self.white_ai_menu.config(
            relief="raised", bd=2, bg="#E8E8E8", activebackground="#D0D0D0"
        )
        self.white_ai_menu.grid(row=2, column=1, padx=4, pady=3, sticky="we")

        self.black_h_label = ttk.Label(setup, text="Black H:")
        self.black_h_label.grid(row=1, column=2, padx=4, pady=3, sticky="e")
        self.black_h_menu = tk.OptionMenu(
            setup, self.black_heuristic, "capture_center", "capture", "center", "rave"
        )
        self.black_h_menu.config(
            relief="raised", bd=2, bg="#E8E8E8", activebackground="#D0D0D0"
        )
        self.black_h_menu.grid(row=1, column=3, padx=4, pady=3, sticky="we")

        self.white_h_label = ttk.Label(setup, text="White H:")
        self.white_h_label.grid(row=2, column=2, padx=4, pady=3, sticky="e")
        self.white_h_menu = tk.OptionMenu(
            setup, self.white_heuristic, "capture_center", "capture", "center", "rave"
        )
        self.white_h_menu.config(
            relief="raised", bd=2, bg="#E8E8E8", activebackground="#D0D0D0"
        )
        self.white_h_menu.grid(row=2, column=3, padx=4, pady=3, sticky="we")

        self.black_d_label = ttk.Label(setup, text="Black D:")
        self.black_d_label.grid(row=1, column=2, padx=4, pady=3, sticky="e")
        self.black_d_menu = tk.OptionMenu(setup, self.black_minimax_depth, 3, 4, 5, 6)
        self.black_d_menu.config(
            relief="raised", bd=2, bg="#E8E8E8", activebackground="#D0D0D0"
        )
        self.black_d_menu.grid(row=1, column=3, padx=4, pady=3, sticky="we")

        self.white_d_label = ttk.Label(setup, text="White D:")
        self.white_d_label.grid(row=2, column=2, padx=4, pady=3, sticky="e")
        self.white_d_menu = tk.OptionMenu(setup, self.white_minimax_depth, 3, 4, 5, 6)
        self.white_d_menu.config(
            relief="raised", bd=2, bg="#E8E8E8", activebackground="#D0D0D0"
        )
        self.white_d_menu.grid(row=2, column=3, padx=4, pady=3, sticky="we")

        actions = ttk.LabelFrame(top, text="Actions", padding=8)
        actions.pack(side="top", fill="x", pady=4)

        for i in range(5):
            actions.columnconfigure(i, weight=1, uniform="act_col")

        RoundedButton(actions, text="New Game", command=self.new_game).grid(
            row=0, column=0, padx=4, pady=2
        )
        RoundedButton(actions, text="Undo", command=self.undo).grid(
            row=0, column=1, padx=4, pady=2
        )
        RoundedButton(actions, text="Pass", command=self.pass_turn).grid(
            row=0, column=2, padx=4, pady=2
        )
        RoundedButton(actions, text="Resign", command=self.resign).grid(
            row=0, column=3, padx=4, pady=2
        )
        RoundedButton(actions, text="Start Match", command=self.start_match).grid(
            row=0, column=4, padx=4, pady=2
        )

        board_px = self.margin * 2 + self.cell * (self.board_size - 1)

        self.canvas = tk.Canvas(
            self,
            width=board_px,
            height=board_px,
            bg="#DAB97A",
            highlightthickness=0,
        )
        self.canvas.pack(side="top", pady=4)
        self.canvas.bind("<Button-1>", self.on_click)

        info = ttk.LabelFrame(self, text="Status", padding=8)
        info.pack(side="top", fill="x", padx=8, pady=4)

        self.status_label = ttk.Label(
            info,
            textvariable=self.status_text,
            justify="left",
            font=("Microsoft YaHei UI", 10),
        )
        self.status_label.pack(side="top", anchor="w")

    @property
    def current_state(self):
        return self.history[-1]["state"]

    @property
    def captures(self):
        rec = self.history[-1]
        return rec["black_caps"], rec["white_caps"]

    def new_game(self):
        game = GameState.new_game(self.board_size)
        self.history = [{"state": game, "black_caps": 0, "white_caps": 0}]
        self.ai_thinking = False
        self.match_started = False
        self.end_reason = ""
        self.game_over_notified = False
        self._refresh()

        self._maybe_schedule_ai_move()

    def _on_mode_change(self, *_):
        # 模式切换时：ai_vs_ai 自动继续，human_vs_ai 等待玩家操作。
        self._update_strategy_controls()
        self._refresh()
        self._maybe_schedule_ai_move()

    def _on_human_color_change(self, *_):
        self._update_strategy_controls()
        self._refresh()
        self._maybe_schedule_ai_move()

    def _on_agent_change(self, *_):
        self._update_strategy_controls()
        self._refresh()

    def _set_widget_visible(self, widget, visible):
        if visible:
            widget.grid()
        else:
            widget.grid_remove()

    def _update_strategy_controls(self):
        is_human_vs_ai = self.play_mode.get() == "human_vs_ai"
        human_is_black = self.human_color.get() == "black"

        show_human_selector = is_human_vs_ai
        show_black_ai_selector = not (is_human_vs_ai and human_is_black)
        show_white_ai_selector = not (is_human_vs_ai and not human_is_black)

        self._set_widget_visible(self.human_label, show_human_selector)
        self._set_widget_visible(self.human_menu, show_human_selector)
        self._set_widget_visible(self.black_ai_label, show_black_ai_selector)
        self._set_widget_visible(self.black_ai_menu, show_black_ai_selector)
        self._set_widget_visible(self.white_ai_label, show_white_ai_selector)
        self._set_widget_visible(self.white_ai_menu, show_white_ai_selector)

        black_is_mcts = self.black_ai_name.get() == "mcts"
        white_is_mcts = self.white_ai_name.get() == "mcts"
        black_is_minimax = self.black_ai_name.get() == "minimax"
        white_is_minimax = self.white_ai_name.get() == "minimax"

        self._set_widget_visible(
            self.black_h_label, show_black_ai_selector and black_is_mcts
        )
        self._set_widget_visible(
            self.black_h_menu, show_black_ai_selector and black_is_mcts
        )
        self._set_widget_visible(
            self.white_h_label, show_white_ai_selector and white_is_mcts
        )
        self._set_widget_visible(
            self.white_h_menu, show_white_ai_selector and white_is_mcts
        )

        self._set_widget_visible(
            self.black_d_label, show_black_ai_selector and black_is_minimax
        )
        self._set_widget_visible(
            self.black_d_menu, show_black_ai_selector and black_is_minimax
        )
        self._set_widget_visible(
            self.white_d_label, show_white_ai_selector and white_is_minimax
        )
        self._set_widget_visible(
            self.white_d_menu, show_white_ai_selector and white_is_minimax
        )

    def start_match(self):
        """显式开始比赛：触发 AI 行棋流程。"""
        if self._is_game_over(self.current_state):
            self.status_text.set("Game is already over. Click New Game first.")
            return
        if getattr(self, "match_started", False):
            self.status_text.set("Match already started.")
            return
        self.match_started = True
        self.status_text.set("Match started.")
        self._maybe_schedule_ai_move()

    def undo(self):
        if self.ai_thinking:
            return
        if len(self.history) <= 1:
            return

        self.history.pop()
        if len(self.history) > 1:
            self.history.pop()

        self._refresh()

    def pass_turn(self):
        if self.ai_thinking:
            return
        if not getattr(self, "match_started", False):
            self.status_text.set("Please click Start Match first.")
            return
        if self._is_game_over(self.current_state):
            return
        if self.play_mode.get() == "ai_vs_ai":
            self.status_text.set("AI vs AI mode is running automatically.")
            return
        if not self._is_human_turn():
            self.status_text.set("It is AI turn. Please wait.")
            return
        self._apply_move(Move.pass_turn())

    def resign(self):
        if self.ai_thinking:
            return
        if not getattr(self, "match_started", False):
            self.status_text.set("Please click Start Match first.")
            return
        if self._is_game_over(self.current_state):
            return
        if self.play_mode.get() == "ai_vs_ai":
            self.status_text.set("AI vs AI mode is running automatically.")
            return
        if not self._is_human_turn():
            self.status_text.set("It is AI turn. Please wait.")
            return
        self._apply_move(Move.resign())

    def on_click(self, event):
        if not getattr(self, "match_started", False):
            self.status_text.set("Please click Start Match first.")
            return
        if self.ai_thinking:
            return
        if self._is_game_over(self.current_state):
            return
        if self.play_mode.get() == "ai_vs_ai":
            self.status_text.set("AI vs AI mode is running automatically.")
            return
        if not self._is_human_turn():
            self.status_text.set("It is AI turn. Please wait.")
            return

        point = self._event_to_point(event.x, event.y)
        if point is None:
            return

        move = Move.play(point)
        if not self.current_state.is_valid_move(move):
            self.status_text.set("Invalid move.")
            return

        self._apply_move(move)

    def _event_to_point(self, x, y):
        col = round((x - self.margin) / self.cell) + 1
        row = round((y - self.margin) / self.cell) + 1

        if row < 1 or row > self.board_size or col < 1 or col > self.board_size:
            return None

        px = self.margin + (col - 1) * self.cell
        py = self.margin + (row - 1) * self.cell
        if abs(x - px) > self.cell * 0.42 or abs(y - py) > self.cell * 0.42:
            return None

        return Point(row, col)

    def _is_human_turn(self):
        player = self.current_state.next_player
        human = Player.black if self.human_color.get() == "black" else Player.white
        return player == human

    def _is_ai_turn(self):
        return not self._is_human_turn()

    def _apply_move(self, move):
        state = self.current_state
        black_caps, white_caps = self.captures

        next_state = state.apply_move(move)

        if move.is_play:
            mover = state.next_player
            captured = self._captured_stones(state, next_state, mover)
            if mover == Player.black:
                black_caps += captured
            else:
                white_caps += captured

        self.history.append(
            {
                "state": next_state,
                "black_caps": black_caps,
                "white_caps": white_caps,
            }
        )

        if self._is_consecutive_pass_end(next_state):
            self.end_reason = "游戏终止：双方连续停一手（pass）。"
            self._show_game_over_dialog(next_state)
        elif self._is_board_full(next_state):
            self.end_reason = "游戏终止：棋盘填满。"
            self._show_game_over_dialog(next_state)
        elif next_state.is_over() and not self.game_over_notified:
            # 保留认输终局提示，避免界面静默结束。
            self.end_reason = "游戏终止：认输。"
            self._show_game_over_dialog(next_state)

        self._refresh()

        self._maybe_schedule_ai_move()

    def _maybe_schedule_ai_move(self):
        if not getattr(self, "match_started", False):
            return
        if self._is_game_over(self.current_state):
            return
        if self.play_mode.get() == "ai_vs_ai":
            self.after(80, self.ai_move)
        elif self._is_ai_turn():
            self.after(80, self.ai_move)

    def _captured_stones(self, before_state, after_state, mover):
        opponent = mover.other
        before_count = self._count_stones(before_state, opponent)
        after_count = self._count_stones(after_state, opponent)
        return max(0, before_count - after_count)

    def _count_stones(self, game_state, player):
        board = game_state.board
        count = 0
        for row in range(1, board.num_rows + 1):
            for col in range(1, board.num_cols + 1):
                if board.get(Point(row, col)) == player:
                    count += 1
        return count

    def ai_move(self):
        if self._is_game_over(self.current_state):
            return
        if self.play_mode.get() != "ai_vs_ai" and not self._is_ai_turn():
            return

        self.ai_thinking = True
        self._refresh()
        self.update_idletasks()

        try:
            move = self._select_ai_move(self.current_state)
        except Exception as exc:
            self.ai_thinking = False
            self.status_text.set(f"AI error: {exc}")
            return

        self.ai_thinking = False
        self._apply_move(move)

    def _select_ai_move(self, game_state):
        player = game_state.next_player
        name = self._current_ai_name(player)

        if name == "random":
            from agents.random_agent import RandomAgent

            return RandomAgent().select_move(game_state)

        if name == "minimax":
            from agents.minimax_agent import MinimaxAgent

            return MinimaxAgent(
                max_depth=self._current_minimax_depth(player)
            ).select_move(game_state)

        from agents.mcts_agent import MCTSAgent

        return MCTSAgent(
            num_rounds=220,
            max_rollout_steps=24,
            time_limit=2.5,
            heuristic=self._current_mcts_heuristic(player),
        ).select_move(game_state)

    def _current_mcts_heuristic(self, player):
        return (
            self.black_heuristic.get()
            if player == Player.black
            else self.white_heuristic.get()
        )

    def _current_minimax_depth(self, player):
        return (
            int(self.black_minimax_depth.get())
            if player == Player.black
            else int(self.white_minimax_depth.get())
        )

    def _current_ai_name(self, player):
        if self.play_mode.get() == "ai_vs_ai":
            return (
                self.black_ai_name.get()
                if player == Player.black
                else self.white_ai_name.get()
            )

        human = Player.black if self.human_color.get() == "black" else Player.white
        if player == human:
            return (
                self.white_ai_name.get()
                if human == Player.black
                else self.black_ai_name.get()
            )

        return (
            self.white_ai_name.get()
            if player == Player.white
            else self.black_ai_name.get()
        )

    def _refresh(self):
        self._draw_board()

        state = self.current_state
        black_caps, white_caps = self.captures

        if self._is_game_over(state):
            winner = self._winner_of_state(state)
            if winner is None:
                turn_text = "Game over: draw"
            else:
                turn_text = f"Game over: {winner.name} wins"

            if self.end_reason:
                turn_text = f"{self.end_reason} {turn_text}"
        else:
            if self.play_mode.get() == "ai_vs_ai":
                turn_text = (
                    f"AI vs AI | Turn: {state.next_player.name}\n"
                    f"Black: {self.black_ai_name.get()} ({self.black_heuristic.get()}) | "
                    f"White: {self.white_ai_name.get()} ({self.white_heuristic.get()})"
                )
            else:
                turn_text = f"Turn: {state.next_player.name}"

        move_number = len(self.history) - 1

        info_lines = [
            f"Move: {move_number} | Mode: {self.play_mode.get()} | Captures B: {black_caps} W: {white_caps}",
            turn_text,
        ]
        if self.ai_thinking:
            info_lines.append("[AI is thinking...]")

        self.status_text.set("\n".join(info_lines))

    def _is_game_over(self, game_state):
        if self._is_consecutive_pass_end(game_state):
            if not self.end_reason:
                self.end_reason = "游戏终止：双方连续停一手（pass）。"
            return True
        if self._is_board_full(game_state):
            if not self.end_reason:
                self.end_reason = "游戏终止：棋盘填满。"
            return True
        if game_state.is_over():
            if not self.end_reason:
                self.end_reason = "游戏终止：认输。"
            return True
        return False

    def _is_board_full(self, game_state):
        board = game_state.board
        for row in range(1, board.num_rows + 1):
            for col in range(1, board.num_cols + 1):
                if board.get(Point(row, col)) is None:
                    return False
        return True

    def _is_consecutive_pass_end(self, game_state):
        if not game_state.is_over() or game_state.last_move is None:
            return False
        if not game_state.last_move.is_pass:
            return False
        prev = game_state.previous_state
        if prev is None or prev.last_move is None:
            return False
        return prev.last_move.is_pass

    def _show_game_over_dialog(self, game_state):
        if self.game_over_notified:
            return

        winner, result, by_resign = self._get_outcome(game_state)
        if winner is None:
            winner_text = "平局"
        else:
            winner_text = "黑方" if winner == Player.black else "白方"

        reason = self.end_reason or "游戏结束"
        if by_resign:
            detail = "胜负依据：认输（不按目数计算胜负）。"
            message = f"{reason}\n胜者：{winner_text}\n{detail}"
        else:
            margin = result.winning_margin
            white_total = result.w + result.komi
            detail = f"计分：黑 {result.b:.1f}，白 {result.w:.1f} + 贴目 {result.komi:.1f} = {white_total:.1f}"
            message = (
                f"{reason}\n胜者：{winner_text}\n胜负差：{margin:.1f} 目\n{detail}"
            )

        messagebox.showinfo("对局结束", message)
        self.game_over_notified = True

    def _winner_of_state(self, game_state):
        winner, _, _ = self._get_outcome(game_state)
        return winner

    def _get_outcome(self, game_state):
        """返回 (winner, result, by_resign)。result 在认输时为 None。"""
        if (
            game_state.last_move is not None
            and game_state.last_move.is_resign
            and game_state.is_over()
        ):
            return game_state.next_player, None, True

        result = compute_game_result(game_state)
        return result.winner, result, False

    def _draw_board(self):
        self.canvas.delete("all")

        n = self.board_size
        for i in range(n):
            x0 = self.margin + i * self.cell
            y0 = self.margin
            x1 = x0
            y1 = self.margin + (n - 1) * self.cell
            self.canvas.create_line(x0, y0, x1, y1, width=2)

        for i in range(n):
            y0 = self.margin + i * self.cell
            x0 = self.margin
            y1 = y0
            x1 = self.margin + (n - 1) * self.cell
            self.canvas.create_line(x0, y0, x1, y1, width=2)

        board = self.current_state.board
        for row in range(1, n + 1):
            for col in range(1, n + 1):
                stone = board.get(Point(row, col))
                if stone is None:
                    continue

                x = self.margin + (col - 1) * self.cell
                y = self.margin + (row - 1) * self.cell
                color = "black" if stone == Player.black else "white"
                edge = "#111111"
                self.canvas.create_oval(
                    x - self.stone_radius,
                    y - self.stone_radius,
                    x + self.stone_radius,
                    y + self.stone_radius,
                    fill=color,
                    outline=edge,
                    width=1.5,
                )


def main():
    app = GoApp(board_size=5)
    app.mainloop()


if __name__ == "__main__":
    main()
