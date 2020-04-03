class SinglePlayerArena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """
    def __init__(self, player1, game):
        """
        Input:
            player 1: function that takes board as input, return action
            game: Game object
        """
        self.player1 = player1
        self.game = game

    def playSinglePlayerGame(self):
        """
        Executes one episode of a game.

        Returns:
            1 if game won, -1 if game lost
        """
        player = 1
        board = self.game.getInitBoard()
        while self.game.getGameEnded(board, player) == 0:
            action = self.player1(self.game.getCanonicalForm(board, player))
            board, player = self.game.getNextState(board, player, action)
        return self.game.getGameEnded(board, player)

    def playSinglePlayerGames(self, num):
        wins, losses = 0, 0
        for _ in range(num):
            gameResult = self.playSinglePlayerGame()
            if gameResult == 1:
                wins += 1
            elif gameResult == -1:
                losses += 1
            print('-----', gameResult, '------')
        return wins, losses
