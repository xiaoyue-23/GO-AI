"""
围棋计分模块。

提供地盘计算、对局结果判定等功能。
"""

from collections import namedtuple

from .gotypes import Player, Point


class Territory:
    """
    统计“地、子、中立点（dame）”的容器类。

    territory_map: point -> {黑子/白子/黑地/白地/中立}
    """

    def __init__(self, territory_map):
        self.num_black_territory = 0
        self.num_white_territory = 0
        self.num_black_stones = 0
        self.num_white_stones = 0
        self.num_dame = 0
        self.dame_points = []

        for point, status in territory_map.items():
            if status == Player.black:
                self.num_black_stones += 1
            elif status == Player.white:
                self.num_white_stones += 1
            elif status == 'territory_b':
                self.num_black_territory += 1
            elif status == 'territory_w':
                self.num_white_territory += 1
            elif status == 'dame':
                self.num_dame += 1
                self.dame_points.append(point)


class GameResult(namedtuple('GameResult', 'b w komi')):
    """
    对局结果结构：b=黑方分，w=白方分，komi=贴目。
    """

    @property
    def winner(self):
        """根据双方分数与贴目判断胜者。"""
        if self.b > self.w + self.komi:
            return Player.black
        return Player.white

    @property
    def winning_margin(self):
        """返回胜负差（绝对值）。"""
        w = self.w + self.komi
        return abs(self.b - w)

    def __str__(self):
        """字符串格式化，例如 B+3.5 或 W+2.0。"""
        w = self.w + self.komi
        if self.b > w:
            return 'B+%.1f' % (self.b - w,)
        return 'W+%.1f' % (w - self.b,)


def evaluate_territory(board):
    """
    将棋盘划分为：黑子、白子、黑地、白地、中立点。

    如果一个空点区域被同一种颜色完全包围，则视为该方地盘；
    若边界混色或触边，则视为中立点（dame）。
    """
    status = {}

    for r in range(1, board.num_rows + 1):
        for c in range(1, board.num_cols + 1):
            p = Point(row=r, col=c)
            # 若该点已在之前某个连通区域里处理过，跳过
            if p in status:
                continue

            stone = board.get(p)
            if stone is not None:
                # 有子点，状态为棋子颜色
                status[p] = stone
            else:
                # 空点，收集连通区域和边界颜色
                group, neighbors = _collect_region(p, board)
                if len(neighbors) == 1:
                    # 只有单一颜色边界，视为该方地盘
                    neighbor_stone = neighbors.pop()
                    stone_str = 'b' if neighbor_stone == Player.black else 'w'
                    fill_with = 'territory_' + stone_str
                else:
                    # 边界混色或触边，视为中立点
                    fill_with = 'dame'
                for pos in group:
                    status[pos] = fill_with

    return Territory(status)


def _collect_region(start_pos, board, visited=None):
    """
    从 start_pos 出发，收集同类连通块及其边界集合。

    使用 DFS 遍历与 start_pos 相同类型的连通区域。

    Args:
        start_pos: 起始坐标
        board: 棋盘
        visited: 已访问标记字典

    Returns:
        (连通块点列表, 边界颜色集合)
    """
    if visited is None:
        visited = {}
    if start_pos in visited:
        return [], set()

    all_points = [start_pos]
    all_borders = set()
    visited[start_pos] = True
    here = board.get(start_pos)
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for delta_r, delta_c in deltas:
        next_p = Point(row=start_pos.row + delta_r, col=start_pos.col + delta_c)
        if not board.is_on_grid(next_p):
            continue
        neighbor = board.get(next_p)
        if neighbor == here:
            points, borders = _collect_region(next_p, board, visited)
            all_points += points
            all_borders |= borders
        else:
            all_borders.add(neighbor)

    return all_points, all_borders


def compute_game_result(game_state):
    """
    计算对局结果。

    Args:
        game_state: 游戏状态

    Returns:
        GameResult: 包含黑方分、白方分、贴目的结果对象
    """
    territory = evaluate_territory(game_state.board)
    return GameResult(
        territory.num_black_territory + territory.num_black_stones,
        territory.num_white_territory + territory.num_white_stones,
        komi=2.5,
    )
