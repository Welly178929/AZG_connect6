import numpy as np
from ArenaConnect6 import ArenaConnect6
from gobang.Connect6Game_copy import Connect6Game
from sixfiveNNet import NNetWrapper as NNet
from MCTS import MCTS
from utils import dotdict

# 1) 初始化遊戲
g = Connect6Game(n=19, win_len=6)

# 2) 定義 Human Connect-6 player
def human_connect6(canonicalBoard):
    # 將 canonicalBoard 換回真實版面並顯示
    board = canonicalBoard  # Connect6Game.display 會自動乘 player
    # Connect6Game.display(board)
    valids = g.getValidMoves(board, 1)
    while True:
        s = input("請輸入落子位置 (格式：row,col)：")
        try:
            x_str, y_str = s.strip().split(',')
            x, y = int(x_str), int(y_str)
            action = x * g.n + y
        except:
            print("格式錯誤，請輸入「行,列」，例如 3,5")
            continue
        if 0 <= x < g.n and 0 <= y < g.n and valids[action]:
            return action
        else:
            print("該位置非法或已被佔用，請重新輸入。")

# 3) 準備 NNet 玩家
nnet = NNet(g)
nnet.load_checkpoint('./0431_sixstone_test/', 'best.h5')
args1 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
mcts = MCTS(g, nnet, args1)
nnet_player = lambda board: np.argmax(mcts.getActionProb(board, temp=0))

# 4) 決定先手／後手
player1 = nnet_player
player2 = human_connect6  # 人類走

# 5) 建立 Arena 並對戰
arena = ArenaConnect6(player1, player2, g, display=Connect6Game.display)
# 例如跑 2 局：每人各一次先手、後手
oneWon, twoWon, draws = arena.playGames(num=2, verbose=True)

print(f"AI 勝 {oneWon} 局；人類勝 {twoWon} 局；平手 {draws} 局")
