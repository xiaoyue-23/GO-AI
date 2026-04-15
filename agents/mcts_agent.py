"""
MCTS (蒙特卡洛树搜索) 智能体模板。

学生作业：完成 MCTS 算法的核心实现。
参考：《深度学习与围棋》第 4 章
"""

import math
import random
import time

from dlgo.gotypes import Player, Point
from dlgo.goboard import GameState, Move
from dlgo.scoring import compute_game_result

__all__ = ["MCTSAgent"]


class MCTSNode:
    """
    MCTS 树节点。


    属性：
        game_state: 当前局面
        parent: 父节点（None 表示根节点）
        children: 子节点列表
        visit_count: 访问次数
        value_sum: 累积价值（胜场数）
        prior: 先验概率（来自策略网络，可选）
    """

    def __init__(self, game_state, parent=None, prior=1.0, move=None):
        self.game_state = game_state
        self.parent = parent
        self.children = []
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior
        self.move = move
        # TODO: 初始化其他必要属性
        # 完全禁止认输（过滤掉 is_resign）
        self.untried_moves = [m for m in game_state.legal_moves() if not m.is_resign]

    @property
    def value(self):
        """计算平均价值 = value_sum / visit_count，防止除零。"""
        # TODO: 实现价值计算
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_leaf(self):
        """是否为叶节点（未展开）。"""
        return len(self.children) == 0

    def is_terminal(self):
        """是否为终局节点。"""
        return self.game_state.is_over()

    def best_child(self, c=1.414):
        """
        选择最佳子节点（UCT 算法）。

        UCT = value + c * sqrt(ln(parent_visits) / visits)

        Args:
            c: 探索常数（默认 sqrt(2)）

        Returns:
            最佳子节点
        """
        # TODO: 实现 UCT 选择
        if not self.children:
            return None

        log_parent = math.log(max(1, self.visit_count))
        best_score = float("-inf")
        best_node = None

        for child in self.children:
            if child.visit_count == 0:
                score = float("inf")
            else:
                # child.value 是从 child 节点“下一手玩家”视角统计，
                # 选择时转换为当前节点玩家视角。
                exploitation = 1.0 - child.value
                exploration = (
                    c * child.prior * math.sqrt(log_parent / child.visit_count)
                )
                score = exploitation + exploration

            if score > best_score:
                best_score = score
                best_node = child

        return best_node

    def expand(self):
        """
        展开节点：为所有合法棋步创建子节点。

        Returns:
            新创建的子节点（用于后续模拟）
        """
        # TODO: 实现节点展开
        if not self.untried_moves:
            return None

        move = self.untried_moves.pop()
        next_state = self.game_state.apply_move(move)
        child = MCTSNode(next_state, parent=self, move=move)
        self.children.append(child)
        return child

    def backup(self, value):
        """
        反向传播：更新从当前节点到根节点的统计。

        Args:
            value: 从当前局面模拟得到的结果（1=胜，0=负，0.5=和）
        """
        # TODO: 实现反向传播
        node = self
        current_value = value
        while node is not None:
            node.visit_count += 1
            node.value_sum += current_value
            # 父节点对应对手视角，价值互补
            current_value = 1.0 - current_value
            node = node.parent


class MCTSAgent:
    """
    MCTS 智能体。

    属性：
        num_rounds: 每次决策的模拟轮数
        temperature: 温度参数（控制探索程度）
    """

    def __init__(
        self,
        num_rounds=1000,
        temperature=1.0,
        max_rollout_steps=24,
        time_limit=9.5,
        heuristic="capture_center",
    ):
        self.num_rounds = num_rounds
        self.temperature = temperature
        self.max_rollout_steps = max_rollout_steps
        self.time_limit = time_limit
        self.heuristic = heuristic
        self._rng = random.Random()
        # RAVE 统计：记录形如 (player, point): [win_sum, visits]
        self.rave_scores = {}

    def select_move(self, game_state: GameState) -> Move:
        """
        为当前局面选择最佳棋步。

        流程：
            1. 创建根节点
            2. 进行 num_rounds 轮模拟：
               a. Selection: 用 UCT 选择路径到叶节点
               b. Expansion: 展开叶节点
               c. Simulation: 随机模拟至终局
               d. Backup: 反向传播结果
            3. 选择访问次数最多的棋步

        Args:
            game_state: 当前游戏状态

        Returns:
            选定的棋步
        """
        # TODO: 实现 MCTS 主循环
        root = MCTSNode(game_state)
        if root.is_terminal():
            return Move.pass_turn()

        start_time = time.time()
        c = 1.414 * max(0.1, float(self.temperature))

        for _ in range(self.num_rounds):
            if time.time() - start_time >= self.time_limit:
                break

            node = root

            # 1) Selection: 沿 UCT 向下直到叶节点或未完全展开节点
            while (
                (not node.is_terminal())
                and (not node.is_leaf())
                and (len(node.untried_moves) == 0)
            ):
                next_node = node.best_child(c=c)
                if next_node is None:
                    break
                node = next_node

            # 2) Expansion: 扩展一个新子节点
            if (not node.is_terminal()) and node.untried_moves:
                expanded = node.expand()
                if expanded is not None:
                    node = expanded

            # 3) Simulation: 从当前节点随机/启发式模拟
            rollout_value = self._simulate(node.game_state)

            # 4) Backup: 回传统计值
            node.backup(rollout_value)

        return self._select_best_move(root)

    def _simulate(self, game_state):
        """
        快速模拟（Rollout）：随机走子至终局。

        【第二小问要求】
        标准 MCTS 使用完全随机走子，但需要实现至少两种优化方法：
        1. 启发式走子策略（如：优先选有气、不自杀、提子的走法）
        2. 限制模拟深度（如：最多走 20-30 步后停止评估）
        3. 其他：快速走子评估（RAVE）、池势启发等

        Args:
            game_state: 起始局面

        Returns:
            从当前玩家视角的结果（1=胜, 0=负, 0.5=和）
        """
        # TODO: 实现快速模拟（含两种优化策略）
        rollout_player = game_state.next_player
        state = game_state
        steps = 0

        # RAVE：记录本局 rollout 中涉及的所有走子 (针对每个玩家只记第一次，即 AMAF 特性)
        rollout_moves = {Player.black: set(), Player.white: set()}

        # 优化策略 1: 深度限制，避免 rollout 过长。
        # 优化策略 2: 启发式走子，优先提子/中心点，减少完全随机噪声。
        while (not state.is_over()) and steps < self.max_rollout_steps:
            legal_moves = state.legal_moves()
            play_moves = [m for m in legal_moves if m.is_play]

            if play_moves:
                move = self._pick_heuristic_move(state, play_moves)
            else:
                # 没有落子时绝对不认输，只选择 pass 停一手
                pass_moves = [m for m in legal_moves if m.is_pass]
                move = pass_moves[0] if pass_moves else Move.pass_turn()

            if move.is_play:
                rollout_moves[state.next_player].add(move.point)

            state = state.apply_move(move)
            steps += 1

        if state.is_over():
            winner = state.winner()
        else:
            # 达到深度上限时，用当前盘面快速估值。
            winner = compute_game_result(state).winner

        # RAVE 统计信息更新（根据结果奖励 rollout 中走过的点）
        for p, points in rollout_moves.items():
            win_val = 1.0 if winner == p else (0.5 if winner is None else 0.0)
            for pt in points:
                key = (p, pt)
                if key not in self.rave_scores:
                    self.rave_scores[key] = [0.0, 0]
                self.rave_scores[key][0] += win_val
                self.rave_scores[key][1] += 1

        if winner is None:
            return 0.5
        return 1.0 if winner == rollout_player else 0.0

    def _select_best_move(self, root):
        """
        根据访问次数选择最佳棋步。

        Args:
            root: MCTS 树根节点

        Returns:
            最佳棋步
        """
        # TODO: 根据访问次数或价值选择
        if not root.children:
            return Move.pass_turn()

        best_child = max(root.children, key=lambda c: (c.visit_count, c.value))
        if best_child.move is None:
            return Move.pass_turn()
        return best_child.move

    def _pick_heuristic_move(self, game_state, play_moves):
        """Rollout 启发式策略（可配置）。"""
        if self.heuristic == "rave":
            return self._pick_rave_move(game_state, play_moves)

        if self.heuristic == "center":
            return self._pick_center_move(game_state, play_moves)

        if self.heuristic == "capture":
            capture_moves = self._capture_moves(game_state, play_moves)
            if capture_moves:
                return self._rng.choice(capture_moves)
            return self._rng.choice(play_moves)

        # 默认 capture_center：优先提子，否则中心偏好。
        return self._pick_capture_center_move(game_state, play_moves)

    def _pick_capture_center_move(self, game_state, play_moves):
        """优先提子，其次偏好中心区域。"""
        capture_moves = self._capture_moves(game_state, play_moves)
        candidates = capture_moves if capture_moves else play_moves
        return self._pick_center_move(game_state, candidates)

    def _pick_center_move(self, game_state, play_moves):
        """中心偏好：优先距离棋盘中心更近的落点。"""
        if not play_moves:
            return Move.pass_turn()

        center_r = (game_state.board.num_rows + 1) / 2.0
        center_c = (game_state.board.num_cols + 1) / 2.0
        scored = []
        for move in play_moves:
            p = move.point
            dist = abs(p.row - center_r) + abs(p.col - center_c)
            scored.append((dist, move))
        scored.sort(key=lambda x: x[0])

        top_k = min(3, len(scored))
        return self._rng.choice([m for _, m in scored[:top_k]])

    def _pick_rave_move(self, game_state, play_moves):
        """RAVE 启发：综合全局历史经验(AMAF)和一些随机性进行走子。"""
        if not play_moves:
            return Move.pass_turn()

        best_moves = []
        best_score = -1.0
        player = game_state.next_player

        for move in play_moves:
            key = (player, move.point)
            # 求该点在所有历史rollout中同色方走过的胜率
            if key in self.rave_scores and self.rave_scores[key][1] > 0:
                score = self.rave_scores[key][0] / self.rave_scores[key][1]
            else:
                score = 0.5  # 尚未积累足够数据时的默认期望

            # 引入极少量的随机扰动，打破平局
            score += self._rng.uniform(0, 0.05)

            if score > best_score:
                best_score = score
                best_moves = [move]
            elif abs(score - best_score) < 1e-5:
                best_moves.append(move)

        if best_moves:
            return self._rng.choice(best_moves)
        return self._rng.choice(play_moves)

    def _capture_moves(self, game_state, play_moves):
        """筛选可立即提子的候选走法。"""
        current_player = game_state.next_player
        opponent = current_player.other

        capture_moves = []
        opponent_before = self._count_stones(game_state, opponent)
        for move in play_moves:
            next_state = game_state.apply_move(move)
            opponent_after = self._count_stones(next_state, opponent)
            if opponent_after < opponent_before:
                capture_moves.append(move)
        return capture_moves

    @staticmethod
    def _count_stones(game_state, player):
        """统计某一方在棋盘上的子数。"""
        board = game_state.board
        count = 0
        for row in range(1, board.num_rows + 1):
            for col in range(1, board.num_cols + 1):
                if board.get(Point(row, col)) == player:
                    count += 1
        return count
