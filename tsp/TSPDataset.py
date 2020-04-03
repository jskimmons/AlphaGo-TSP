import torch
from torch_geometric.data import Data, InMemoryDataset


class TSPDataset(InMemoryDataset):
    def __init__(self, root, examples, transform=None, pre_transform=None):
        super(TSPDataset, self).__init__(root, transform, pre_transform)
        self.examples = examples
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []  # ['data.pt']

    def download(self):
        pass

    def process(self):
        # preprocess
        data_list = self.convert_boards_to_graphs(self.examples)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def convert_board_to_graph(self, board):
        graph, path = board
        m, n = graph.shape

        edges = []
        for i in range(m):
            for j in range(n):
                if i != j:
                    edges.append((i, j))
        edge_index = torch.tensor(edges, dtype=torch.long).reshape((2, -1))
        x = torch.tensor(self.game.getValidMoves(board, 1), dtype=torch.float).reshape((-1,1))

        data = Data(x=x, edge_index=edge_index)
        return data

    def convert_boards_to_graphs(self, boards):
        data = []
        for board in boards:
            data.append(self.convert_board_to_graph(board))
        return data
