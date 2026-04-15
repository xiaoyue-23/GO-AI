"""
围棋基础类型定义模块。

提供 Player（玩家）枚举和 Point（坐标点）类型，
作为整个围棋引擎的基础构建块。
"""

import enum
from collections import namedtuple

__all__ = ["Player", "Point"]


class Player(enum.Enum):
    """
    围棋玩家枚举。

    Attributes:
        black: 黑棋
        white: 白棋
    """

    black = 1
    white = 2

    @property
    def other(self) -> "Player":
        """返回对手玩家。"""
        return Player.black if self == Player.white else Player.white


class Point(namedtuple("Point", "row col")):
    """
    围棋棋盘上的坐标点。

    Attributes:
        row: 行号（从1开始）
        col: 列号（从1开始）

    Note:
        使用 namedtuple 实现不可变性，
        便于作为字典键使用（如 Zobrist 哈希表）。
    """

    def neighbors(self):
        """返回该点的四邻接点（上、下、左、右）。"""
        return [
            Point(self.row - 1, self.col),
            Point(self.row + 1, self.col),
            Point(self.row, self.col - 1),
            Point(self.row, self.col + 1),
        ]

    def __deepcopy__(self, memodict=None):
        """
        深拷贝优化。

        Point 是不可变对象，直接返回自身即可。
        """
        if memodict is None:
            memodict = {}
        return self
