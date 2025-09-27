import os
import logging
from Arenav2_0420 import Arena, ArgmaxPlayer
from MCTS import MCTS
from gobang.GobangGame import GobangGame as Game
from gobang.keras.NNET0417 import NNetWrapper

logging.basicConfig(level=logging.INFO)

# 模擬 Args
class Args:
    numMCTSSims = 10
    cpuct = 1.0
    numWorkers = 2  # 測試 multiprocessing 是否可用

args = Args()

def run_test():
    game = Game()
    nnet = NNetWrapper(game)
    nnet.load_checkpoint('./temp0418', 'temp.h5')

    pmcts = MCTS(game, nnet, args)
    nmcts = MCTS(game, nnet, args)

    player1 = ArgmaxPlayer(pmcts)
    player2 = ArgmaxPlayer(nmcts)

    arena = Arena(player1, player2, game)

    print("🎮 正在執行 Arena 多進程測試 ...")
    pwins, nwins, draws = arena.playGames(num=2, num_workers=args.numWorkers, verbose=False)

    print(f"\n✅ 測試完成！結果如下：")
    print(f"  Player1 勝利場數：{pwins}")
    print(f"  Player2 勝利場數：{nwins}")
    print(f"  平手場數       ：{draws}")


if __name__ == "__main__":
    from multiprocessing import freeze_support, set_start_method
    freeze_support()  # ✅ Windows 防卡關必要
    set_start_method("spawn", force=True)  # ✅ Windows 預設 spawn，也可以明確設置
    run_test()
