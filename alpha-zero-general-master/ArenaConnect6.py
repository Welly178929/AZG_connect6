import logging

from tqdm import tqdm

log = logging.getLogger(__name__)


class ArenaConnect6():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def playGame(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0
        is_second_step = True

        # for player in players[0], players[2]:
        #     if hasattr(player, "startGame"):
        #         player.startGame()

        while self.game.getGameEnded(board, curPlayer) == 0:
            it += 1

            action = players[curPlayer + 1](self.game.getCanonicalForm(board, curPlayer))

            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer), 1)

            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0

            # # Notifying the opponent for the move
            # opponent = players[-curPlayer + 1]
            # if hasattr(opponent, "notify"):
            #     opponent.notify(board, action)
            
            if verbose:
                print("Turn Player ", str(curPlayer))

            board, curPlayer = self.game.getNextState(board, curPlayer, action)
            
            if is_second_step is True:
                is_second_step = False
                curPlayer = -curPlayer
            else:
                is_second_step = True
                
            if verbose:
                move = (action // self.game.n, action % self.game.n)
                print("Place on ", move)
                self.game.display(board)
            
        for player in players[0], players[2]:
            if hasattr(player, "endGame"):
                player.endGame()

        if verbose:
            print("Game over !!!")
            winner = self.game.getGameEnded(board, curPlayer)
            if (winner == 1 or winner == -1):
                print("Winner is ", winner)
            self.game.display(board)

        return curPlayer * self.game.getGameEnded(board, curPlayer)

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0
        for _ in tqdm(range(num), desc="Arena.playGames (1)"):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1

        self.player1, self.player2 = self.player2, self.player1

        for _ in tqdm(range(num), desc="Arena.playGames (2)"):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == -1:
                oneWon += 1
            elif gameResult == 1:
                twoWon += 1
            else:
                draws += 1

        return oneWon, twoWon, draws
