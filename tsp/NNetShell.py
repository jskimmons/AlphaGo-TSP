from NeuralNet import NeuralNet


class NNetShell(NeuralNet):

    def __init__(self, game):
        self.game = game
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        pass

    def predict(self, board):
        """
        board: np array with board
        """
        # uniform approach
        # return np.ones(self.game.getActionSize()), 0

        # greedy approach based on weight of edges
        graph, path = board
        cur_node = path[-1]
        pi = graph[cur_node]
        return 1000 - pi, 0


    def save_checkpoint(self, folder, filename):
        pass

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        pass