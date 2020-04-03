from collections import deque
from Arena import SinglePlayerArena
from TSPMCTS import TSPMCTS
from tsp.TSPDataset import TSPDataset
import numpy as np
from random import shuffle


class TSPCoach:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, args, game, nnet):
        self.args = args
        self.game = game
        self.nnet = nnet
        self.mcts = TSPMCTS(self.args, self.game, self.nnet)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard,pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        player = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, player)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, player, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, player = self.game.getNextState(board, player, action)

            r = self.game.getGameEnded(board, player)

            if r != 0:
                ex = [(x[0], x[2], r * ((-1) ** (x[1] != player))) for x in trainExamples]
                return ex

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters + 1):
            print('------ ITER ' + str(i) + '------')
            iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

            for eps in range(self.args.numEps):
                print('------ Self Play Episode ' + str(eps) + '------')
                self.mcts = TSPMCTS(self.args, self.game, self.nnet)  # reset search tree
                iterationTrainExamples += self.executeEpisode()

            # save the iteration examples to the history
            self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                self.trainExamplesHistory.pop(0)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)

            # training new network
            if self.args.numEps > 0:
                self.nnet.train(trainExamples)
            nmcts = TSPMCTS(self.args, self.game, self.nnet)

            print('PLAYING GAMES')
            if self.args.arenaCompare:
                arena = SinglePlayerArena(lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
                wins, losses = arena.playSinglePlayerGames(self.args.arenaCompare)
                print('WINS/LOSSES: %d / %d' % (wins, losses))
