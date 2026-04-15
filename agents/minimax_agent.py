"""
第三小问（选做）：Minimax 智能体

实现 Minimax + Alpha-Beta 剪枝算法，与 MCTS 对比效果。
可选实现，用于对比不同搜索算法的差异。

参考：《深度学习与围棋》第 3 章
"""

from dlgo.gotypes import Player, Point
from dlgo.goboard import GameState, Move
from dlgo.scoring import compute_game_result

__all__ = ["MinimaxAgent"]


class MinimaxAgent:
    """
    Minimax 智能体（带 Alpha-Beta 剪枝）。

    属性：
        max_depth: 搜索最大深度
        evaluator: 局面评估函数
    """

    def __init__(self, max_depth=8, evaluator=None):
        self.max_depth = max_depth
        # 默认评估函数（TODO：学生可替换为神经网络）
        self.evaluator = evaluator or self._default_evaluator
        self.cache = GameResultCache()
        self._root_player = None

    def select_move(self, game_state: GameState) -> Move:
        """
        为当前局面选择最佳棋步。

        Args:
            game_state: 当前游戏状态

        Returns:
            选定的棋步
        """
        # TODO: 实现 Minimax 搜索，调用 minimax 或 alphabeta
        self._root_player = game_state.next_player
        legal_moves = self._get_ordered_moves(game_state)

        if not legal_moves:
            return Move.pass_turn()

        best_move = None
        best_value = float("-inf")

        for move in legal_moves:
            # 对弈中几乎不主动认输，避免搜索被无意义终止干扰。
            if move.is_resign:
                continue

            next_state = game_state.apply_move(move)
            value = self.alphabeta(
                next_state,
                self.max_depth - 1,
                float("-inf"),
                float("inf"),
                maximizing_player=False,
            )

            if value > best_value:
                best_value = value
                best_move = move

        if best_move is None:
            pass_moves = [m for m in legal_moves if m.is_pass]
            return pass_moves[0] if pass_moves else legal_moves[0]
        return best_move

    def minimax(self, game_state, depth, maximizing_player):
        """
        基础 Minimax 算法。

        Args:
            game_state: 当前局面
            depth: 剩余搜索深度
            maximizing_player: 是否在当前层最大化（True=我方）

        Returns:
            该局面的评估值
        """
        # TODO: 实现 Minimax
        # 提示：
        # 1. 终局或 depth=0 时返回评估值
        # 2. 如果是最大化方：取所有子节点最大值
        # 3. 如果是最小化方：取所有子节点最小值
        if depth == 0 or game_state.is_over():
            return self.evaluator(game_state)

        moves = self._get_ordered_moves(game_state)
        if maximizing_player:
            value = float("-inf")
            for move in moves:
                child_value = self.minimax(
                    game_state.apply_move(move), depth - 1, False
                )
                value = max(value, child_value)
            return value

        value = float("inf")
        for move in moves:
            child_value = self.minimax(game_state.apply_move(move), depth - 1, True)
            value = min(value, child_value)
        return value

    def alphabeta(self, game_state, depth, alpha, beta, maximizing_player):
        """
        Alpha-Beta 剪枝优化版 Minimax。

        Args:
            game_state: 当前局面
            depth: 剩余搜索深度
            alpha: 当前最大下界
            beta: 当前最小上界
            maximizing_player: 是否在当前层最大化

        Returns:
            该局面的评估值
        """
        # TODO: 实现 Alpha-Beta 剪枝
        # 提示：在 minimax 基础上添加剪枝逻辑
        # - 最大化方：如果 value >= beta 则剪枝
        # - 最小化方：如果 value <= alpha 则剪枝
        # 使用 (下一手方, 局面哈希, 深度, 极大/极小层) 作为缓存 key。
        zobrist_hash = (
            game_state.next_player,
            game_state.board.zobrist_hash(),
            depth,
            maximizing_player,
        )
        cached = self.cache.get(zobrist_hash)
        if cached is not None:
            return cached["value"]

        if depth == 0 or game_state.is_over():
            value = self.evaluator(game_state)
            self.cache.put(zobrist_hash, depth, value, flag="exact")
            return value

        moves = self._get_ordered_moves(game_state)

        if maximizing_player:
            value = float("-inf")
            for move in moves:
                value = max(
                    value,
                    self.alphabeta(
                        game_state.apply_move(move),
                        depth - 1,
                        alpha,
                        beta,
                        False,
                    ),
                )
                alpha = max(alpha, value)
                if value >= beta:
                    break
            self.cache.put(zobrist_hash, depth, value, flag="exact")
            return value

        value = float("inf")
        for move in moves:
            value = min(
                value,
                self.alphabeta(
                    game_state.apply_move(move),
                    depth - 1,
                    alpha,
                    beta,
                    True,
                ),
            )
            beta = min(beta, value)
            if value <= alpha:
                break
        self.cache.put(zobrist_hash, depth, value, flag="exact")
        return value

    def _default_evaluator(self, game_state):
        """
        默认局面评估函数（简单版本）。

        学生作业：替换为更复杂的评估函数，如：
            - 气数统计
            - 眼位识别
            - 神经网络评估

        Args:
            game_state: 游戏状态

        Returns:
            评估值（正数对我方有利）
        """
        # TODO: 实现简单的启发式评估
        # 示例：子数差 + 气数差
        root_player = self._root_player or game_state.next_player
        opponent = root_player.other

        if game_state.is_over():
            winner = game_state.winner()
            if winner is None:
                return 0.0
            return 1e6 if winner == root_player else -1e6

        board = game_state.board
        my_stones = 0
        opp_stones = 0
        my_liberties = 0
        opp_liberties = 0
        visited_strings = set()

        for row in range(1, board.num_rows + 1):
            for col in range(1, board.num_cols + 1):
                point = Point(row, col)
                stone = board.get(point)
                if stone is None:
                    continue

                if stone == root_player:
                    my_stones += 1
                else:
                    opp_stones += 1

                go_string = board.get_go_string(point)
                sid = id(go_string)
                if sid in visited_strings:
                    continue
                visited_strings.add(sid)

                if go_string.color == root_player:
                    my_liberties += go_string.num_liberties
                elif go_string.color == opponent:
                    opp_liberties += go_string.num_liberties

        # 估值：子数差 + 气数差（轻权重），兼顾地盘倾向。
        score = (my_stones - opp_stones) + 0.15 * (my_liberties - opp_liberties)

        # 作为辅助信号，加入当前局面实地估计差。
        game_result = compute_game_result(game_state)
        territory_term = game_result.b - game_result.w
        if root_player == Player.white:
            territory_term = -territory_term

        return score + 0.05 * territory_term

    def _get_ordered_moves(self, game_state):
        """
        获取排序后的候选棋步（用于优化剪枝效率）。

        好的排序能让 Alpha-Beta 剪掉更多分支。

        Args:
            game_state: 游戏状态

        Returns:
            按启发式排序的棋步列表
        """
        # TODO: 实现棋步排序
        # 提示：优先检查吃子、提子、连络等好棋
        moves = game_state.legal_moves()

        play_moves = [m for m in moves if m.is_play]
        pass_moves = [m for m in moves if m.is_pass]
        resign_moves = [m for m in moves if m.is_resign]

        if not play_moves:
            return pass_moves + resign_moves

        board = game_state.board
        center_r = (board.num_rows + 1) / 2.0
        center_c = (board.num_cols + 1) / 2.0
        opponent = game_state.next_player.other

        opponent_before = 0
        for row in range(1, board.num_rows + 1):
            for col in range(1, board.num_cols + 1):
                if board.get(Point(row, col)) == opponent:
                    opponent_before += 1

        scored = []
        for move in play_moves:
            next_state = game_state.apply_move(move)

            opponent_after = 0
            for row in range(1, board.num_rows + 1):
                for col in range(1, board.num_cols + 1):
                    if next_state.board.get(Point(row, col)) == opponent:
                        opponent_after += 1

            capture_bonus = 1 if opponent_after < opponent_before else 0
            dist_center = abs(move.point.row - center_r) + abs(
                move.point.col - center_c
            )
            scored.append((capture_bonus, -dist_center, move))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        ordered_play = [item[2] for item in scored]
        return ordered_play + pass_moves + resign_moves


class GameResultCache:
    """
    局面缓存（Transposition Table）。

    用 Zobrist 哈希缓存已评估的局面，避免重复计算。
    """

    def __init__(self):
        self.cache = {}

    def get(self, zobrist_hash):
        """获取缓存的评估值。"""
        return self.cache.get(zobrist_hash)

    def put(self, zobrist_hash, depth, value, flag="exact"):
        """
        缓存评估结果。

        Args:
            zobrist_hash: 局面哈希
            depth: 搜索深度
            value: 评估值
            flag: 'exact'/'lower'/'upper'（精确值/下界/上界）
        """
        # TODO: 实现缓存逻辑（考虑深度优先替换策略）
        old = self.cache.get(zobrist_hash)
        if old is None or depth >= old["depth"]:
            self.cache[zobrist_hash] = {
                "depth": depth,
                "value": value,
                "flag": flag,
            }
