import numpy as np
from connect6_5.GomokuAndSixGame import GomokuAndSixGame

def print_board(board):
    print("   " + " ".join([f"{i:2}" for i in range(board.shape[1])]))
    for i, row in enumerate(board):
        print(f"{i:2} " + " ".join(['.' if cell == 0 else ('X' if cell == 1 else 'O') for cell in row]))
    print()

def main():
    mode = input("選擇模式：gomoku（五子棋）或 sixstone（六子棋）：").strip()
    game = GomokuAndSixGame(mode=mode)

    board, black_first_done, sixstone_buffer = game.getInitBoard()
    curPlayer = 1  # 1 for black (X), -1 for white (O)

    print_board(board)

    while True:
        print(f"🔴 目前是 {'黑子 (X)' if curPlayer == 1 else '白子 (O)'} 的回合")

        valid_moves = game.getValidMoves(board, curPlayer, black_first_done, sixstone_buffer)
        moves_list = [(idx // game.m, idx % game.m) for idx, v in enumerate(valid_moves[:-1]) if v == 1]

        print(f"可下位置數量：{len(moves_list)}")
        if valid_moves[-1] == 1:
            print("✅ 也可以選擇 PASS（輸入 p）")

        move_input = input("請輸入座標（格式: x y）或 p（PASS）：").strip()
        if move_input.lower() == 'p' and valid_moves[-1] == 1:
            action = game.n * game.m
        else:
            try:
                x, y = map(int, move_input.split())
                action = x * game.m + y
                if valid_moves[action] == 0:
                    print("❌ 這個位置不能下，請重選！")
                    continue
            except:
                print("❌ 輸入格式錯誤，請重試！")
                continue

        board, curPlayer, black_first_done, sixstone_buffer = game.getNextState(
            board, curPlayer, action, black_first_done, sixstone_buffer
        )
        print_board(board)
        # b,bf,sb=game.getCanonicalForm(board,curPlayer,black_first_done,sixstone_buffer)
        # print(b,bf,sb)
        print(board,curPlayer,black_first_done,sixstone_buffer)
        result = game.getGameEnded(board, curPlayer)
        if result != 0:
            if result == 1e-4:
                print("🤝 和局！")
            else:
                winner = '黑子 (X)' if result == 1 else '白子 (O)'
                print(f"🎉 遊戲結束！獲勝者：{winner}")
            break

if __name__ == '__main__':
    main()
