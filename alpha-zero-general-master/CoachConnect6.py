import logging
import os
import sys
import time
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle
import numpy as np
from tqdm import tqdm
from ArenaConnect6 import ArenaConnect6
from MCTS import MCTS
from multiprocessing import Pool, set_start_method
from connect6_5.keras.sixfiveNNet import NNetWrapper

log = logging.getLogger(__name__)


def self_play_worker(args_serialized):
    import os, time
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 💥 禁用 GPU，讓 worker 使用 CPU
    game, args = args_serialized
    start_time = time.time()

    print(f"[Worker PID {os.getpid()}] Self-play worker started!", flush=True)

    nnet = NNetWrapper(game)
    
    filepath = os.path.join(args.checkpoint, 'temp.h5')
    if os.path.exists(filepath):
        nnet.load_checkpoint(folder=args.checkpoint, filename='temp.h5')
    else:
        print(f"[Worker PID {os.getpid()}] ⚠️ temp.h5 not found, using untrained model.", flush=True)
    ## nnet.load_checkpoint(folder=args.checkpoint, filename='temp.h5')


    mcts = MCTS(game, nnet, args)
    trainExamples = []
    board = game.getInitBoard()
    curPlayer = 1
    episodeStep = 0
    is_second_step = True

    while True:
        episodeStep += 1
        temp = int(episodeStep < args.tempThreshold)
        if episodeStep==1:
            canonicalBoard = game.getCanonicalForm(board, curPlayer)
        pi = mcts.getActionProb(canonicalBoard, temp=temp)


        action = np.random.choice(len(pi), p=pi)
        board, curPlayer = game.getNextState(board, curPlayer, action)

        canonicalBoard = game.getCanonicalForm(board, curPlayer)
        sym = game.getSymmetries(canonicalBoard, pi)
        for b, p in sym:
            trainExamples.append([b, curPlayer, p, None])
        # trainExamples.append([canonicalBoard, curPlayer, pi, None])#if not using data augmentation,only use this line
        
        if is_second_step is True:
            is_second_step = False
            curPlayer = -curPlayer
        else:
            is_second_step = True
        winner = game.getGameEnded(board, curPlayer)
        if winner != 0:
            duration = time.time() - start_time
            print(f"[Worker PID {os.getpid()}] Self-play finished in {duration:.2f} seconds with winner: {winner}", flush=True)
            return [(x[0], x[2], winner * ((-1) ** (x[1] != curPlayer))) for x in trainExamples]


class Coach():
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)
        self.args = args
        self.trainExamplesHistory = []
        self.skipFirstSelfPlay = False ##是否用以產生的訓練資料

    def learn(self):
        set_start_method('spawn', force=True)

        start_iter=self.args.start_iter ## 設置start_iter
        self.skipFirstSelfPlay=self.args.skipFirstSelfPlay
        ## 

        for i in range(start_iter, self.args.numIters + 1):
            log.info(f'🌟 Starting Iteration #{i} ...')
            ## print(self.skipFirstSelfPlay,i)
            if not self.skipFirstSelfPlay:## or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                log.info(f'🎮 Generating {self.args.numEps} self-play episodes using up to {self.args.numWorkers} processes ...')
                print("✅ Multiprocessing will now start...", flush=True)

                mp_start = time.time()

                with Pool(processes=self.args.numWorkers) as pool:
                    episode_args = [(self.game, self.args)] * self.args.numEps
                    results = list(tqdm(pool.imap(self_play_worker, episode_args), total=self.args.numEps))

                mp_end = time.time()
                print(f"✅ All {self.args.numEps} episodes finished in {mp_end - mp_start:.2f} seconds!\n", flush=True)

                for episode in results:
                    iterationTrainExamples += episode

                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing oldest trainExamples entry. Current length: {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)

            self.saveTrainExamples(i - 1)

            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.h5')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.h5')
            pmcts = MCTS(self.game, self.pnet, self.args)


            examples_file = os.path.join(
                self.args.checkpoint,
                f"checkpoint_{i-1}.h5.examples"
            )

            self.nnet.train(trainExamples,examples_file)
            nmcts = MCTS(self.game, self.nnet, self.args)
            # if i == 1:
            #     log.info('✅ First iteration: auto-accepting new model')
            #     self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
            #     self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.h5')
            #     continue
            log.info('⚔️ PITTING AGAINST PREVIOUS VERSION')
            arena = ArenaConnect6(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
            
            pwins, nwins, draws = arena.playGames(num=self.args.arenaCompare,verbose=False)

            win_rate = float(nwins) / (pwins + nwins) if (pwins + nwins) > 0 else 0.0
            with open(os.path.join(self.args.checkpoint, "arena_results.txt"), "a") as f:
                f.write(f"Iter {i:02d}: new={nwins}, prev={pwins}, draw={draws}, win_rate={win_rate:.2f}\n")

            log.info('🏆 NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                log.info('❌ REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.h5')
            else:
                log.info('✅ ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.h5')

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.h5'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        # modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        folder = self.args.load_folder_file[0]
        start_iter=self.args.start_iter - 1
        # examplesFile = modelFile + ".examples"
        examplesFile = os.path.join(folder, 'checkpoint_' + f"{start_iter}" + '.h5' + ".examples")
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("📂 File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info(f'📥 File "{examplesFile}" Loading done!')
            self.skipFirstSelfPlay = True
