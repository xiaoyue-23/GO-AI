"""
围棋游戏引擎核心模块 (dlgo - Deep Learning and the Game of Go)

本模块提供了围棋游戏的基本功能：
- gotypes: 基础类型定义（Player, Point）
- goboard: 棋盘和游戏规则实现
- scoring: 计分系统
- zobrist: 用于局面哈希的随机数表
"""

__version__ = "0.1.0"

from .gotypes import Player, Point
from .goboard import Board, GameState, Move
from .scoring import compute_game_result, GameResult

__all__ = [
    "Player",
    "Point",
    "Board",
    "GameState",
    "Move",
    "compute_game_result",
    "GameResult",
]
