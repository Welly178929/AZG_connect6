import numpy as np
from .gobang.Connect6Logic import Board
from Game import Game

class Connect6Game(Game):
    def __init__(self, n=19, win_len=6):
        self.n = n
        self.win_len = win_len
        # self.is_second_step = True

    # ---------------- Game API ----------------
    def getInitBoard(self):
        # self.is_second_step = True
        return np.zeros((self.n, self.n), dtype=np.int8)

    def getBoardSize(self):
        return (self.n, self.n)

    def getActionSize(self):
        return self.n * self.n + 1  # +1 for pass (not used here, but keep API)
    
    def getNextState(self, board, player, action):
        
        # if action == self.n * self.n:
        #     return (board, -player)  # pass —— seldom used in Connect‑6

        b = Board(self.n)
        b.pieces = np.copy(board)
        
        move = (int(action / self.n), action % self.n)
        b.execute_move(move, player)
        
        # if self.is_second_step is True:
        #     self.is_second_step = False
        #     return (b.pieces, -player)
        # else:
        #     self.is_second_step = True
        #     return (b.pieces, player)
        
        return (b.pieces, player)


    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        valids = [0] * self.getActionSize()
        b = Board(self.n)
        b.pieces = np.copy(board)
        legalMoves = b.get_legal_moves(player)
        if len(legalMoves) == 0:
            valids[-1] = 1
            return np.array(valids)
        for x, y in legalMoves:
            valids[self.n * x + y] = 1
        return np.array(valids)

    def getGameEnded(self, board, _player):
         # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = Board(self.n)
        b.pieces = np.copy(board)
        n = self.win_len

        for w in range(self.n):
            for h in range(self.n):
                if (w in range(self.n - n + 1) and board[w][h] != 0 and
                        len(set(board[i][h] for i in range(w, w + n))) == 1):
                    return board[w][h]
                if (h in range(self.n - n + 1) and board[w][h] != 0 and
                        len(set(board[w][j] for j in range(h, h + n))) == 1):
                    return board[w][h]
                if (w in range(self.n - n + 1) and h in range(self.n - n + 1) and board[w][h] != 0 and
                        len(set(board[w + k][h + k] for k in range(n))) == 1):
                    return board[w][h]
                if (w in range(self.n - n + 1) and h in range(n - 1, self.n) and board[w][h] != 0 and
                        len(set(board[w + l][h - l] for l in range(n))) == 1):
                    return board[w][h]
        
        if b.has_legal_moves():
            return 0
        return 1e-4


    def getCanonicalForm(self, board, player):
        return board * player

    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert(len(pi) == self.n**2 + 1)  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return l

    def stringRepresentation(self, board):
        return board.tobytes()

    @staticmethod
    def display(board):
        n = board.shape[0]
        symbols = {0:'.', 1: 'O', -1:'X'}
        print("  ", end="")
        for y in range(n):
            print(f"{y%10}", end=" ")
        print()
        for x in range(n):
            print(f"{x%10} ", end="")
            for y in range(n):
                print(symbols[board[x][y]], end=" ")
            print()
