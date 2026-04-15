"""
对弈测试脚本。

用法：
    python play.py --agent1 mcts --agent2 random --size 5
    python play.py --agent1 minimax --agent2 mcts --size 5 --games 10
"""

import argparse
import random
import time

from dlgo import GameState, Player, Point
from dlgo.goboard import Move


def random_agent(game_state):
    """随机选择合法棋步（第一小问）。"""
    # 使用 agents.random_agent.RandomAgent
    try:
        from agents.random_agent import RandomAgent
        agent = RandomAgent()
        return agent.select_move(game_state)
    except ImportError:
        # 如果文件不存在，使用本地实现
        moves = game_state.legal_moves()
        return random.choice(moves)


def mcts_agent(game_state):
    """MCTS 智能体（占位，学生实现后替换）。"""
    try:
        from agents.mcts_agent import MCTSAgent
        agent = MCTSAgent(num_rounds=100)
        return agent.select_move(game_state)
    except ImportError as e:
        print(f"[WARN] MCTSAgent 未实现或导入错误: {e}")
        return random_agent(game_state)


def minimax_agent(game_state):
    """Minimax 智能体（占位，学生实现后替换）。"""
    try:
        from agents.minimax_agent import MinimaxAgent
        agent = MinimaxAgent(max_depth=3)
        return agent.select_move(game_state)
    except ImportError as e:
        print(f"[WARN] MinimaxAgent 未实现或导入错误: {e}")
        return random_agent(game_state)


AGENTS = {
    "random": random_agent,
    "mcts": mcts_agent,
    "minimax": minimax_agent,
}


def print_board(game_state):
    """打印棋盘（简化版）。"""
    board = game_state.board
    print("  ", end="")
    for c in range(1, board.num_cols + 1):
        print(f"{c:2}", end="")
    print()

    for r in range(1, board.num_rows + 1):
        print(f"{r:2}", end="")
        for c in range(1, board.num_cols + 1):
            stone = board.get(Point(r, c))
            if stone == Player.black:
                print(" X", end="")
            elif stone == Player.white:
                print(" O", end="")
            else:
                print(" .", end="")
        print()


def play_game(agent1_fn, agent2_fn, board_size=9, verbose=True):
    """
    进行一局对弈。

    Returns:
        (winner, move_count, duration_seconds)
    """
    game = GameState.new_game(board_size)
    agents = {
        Player.black: agent1_fn,
        Player.white: agent2_fn,
    }

    move_count = 0
    start_time = time.time()

    while not game.is_over():
        if verbose:
            print(f"\n=== Move {move_count + 1}, {game.next_player.name} ===")
            print_board(game)

        agent_fn = agents[game.next_player]
        move = agent_fn(game)

        if verbose:
            print(f"选择: {move}")

        game = game.apply_move(move)
        move_count += 1

        if move_count > board_size * board_size * 2:
            print("[WARN] 步数过多，强制结束")
            break

    duration = time.time() - start_time
    winner = game.winner()

    if verbose:
        print("\n=== 终局 ===")
        print_board(game)
        if winner:
            print(f"胜者: {winner.name}")
        else:
            print("平局")

    return winner, move_count, duration


def main():
    parser = argparse.ArgumentParser(description="围棋 AI 对弈")
    parser.add_argument(
        "--agent1",
        choices=AGENTS.keys(),
        default="random",
        help="黑方智能体",
    )
    parser.add_argument(
        "--agent2",
        choices=AGENTS.keys(),
        default="random",
        help="白方智能体",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=5,
        help="棋盘大小 (默认 5)",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=1,
        help="对局数 (默认 1)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="静默模式（只显示结果）",
    )

    args = parser.parse_args()

    agent1 = AGENTS[args.agent1]
    agent2 = AGENTS[args.agent2]

    results = {Player.black: 0, Player.white: 0, None: 0}
    total_moves = 0
    total_time = 0

    for i in range(args.games):
        if not args.quiet:
            print(f"\n========== 对局 {i+1}/{args.games} ==========")

        winner, moves, duration = play_game(
            agent1, agent2, args.size, verbose=not args.quiet
        )

        results[winner] += 1
        total_moves += moves
        total_time += duration

    print("\n========== 统计 ==========")
    print(f"对局数: {args.games}")
    print(f"黑方 ({args.agent1}) 胜: {results[Player.black]}")
    print(f"白方 ({args.agent2}) 胜: {results[Player.white]}")
    print(f"平局: {results[None]}")
    print(f"平均步数: {total_moves / args.games:.1f}")
    print(f"平均用时: {total_time / args.games:.2f}s")


if __name__ == "__main__":
    main()
