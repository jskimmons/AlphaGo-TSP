import os
import numpy as np
import sys
sys.path.append('../../')
from utils import *
from NeuralNet import NeuralNet

import torch
from torch import nn
import torch.optim as optim

from tsp.TSPGNN import TSPGNN as gnn
from torch_geometric.data import DataLoader, Data

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 20,
    'batch_size': 1,
    'num_channels': 512,
})

class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = gnn()
        self.board_x, self.board_y = game.getBoardSize()
        self.game = game
        self.action_size = game.getActionSize()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        # Create the new dataset here
        data_list = self.convert_boards_to_graphs(examples)
        loader = DataLoader(data_list, batch_size=args.batch_size, shuffle=True)
        losses = []

        print('Beginning Training')
        optimizer = optim.Adam(self.nnet.parameters())
        loss_fn = nn.MSELoss()
        self.nnet.train()
        for i in range(args.epochs):
            for data in loader:
                pred_choices, pred_value = self.nnet(data)
                v_actual = data.v
                pi_actual = data.pi
                l1 = loss_fn(pred_choices, pi_actual)
                l2 = (0.2 * self.loss_v(pred_value, v_actual))
                loss = l1 + l2
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            train_loss = loss
            print('epoch', i, 'training loss', train_loss.item())

    def predict(self, board):
        """
        board: np array with board
        """
        # greedy approach
        # graph, path = board
        # cur_node = path[-1]
        # pi = 1000 - graph[cur_node]

        # preparing input
        board_list = self.convert_boards_to_graphs([board])
        loader = DataLoader(board_list, batch_size=1)
        for batch in loader:
            self.nnet = self.nnet.float()
            self.nnet.eval()
            with torch.no_grad():
                pi, v = self.nnet(batch)
            return pi, v

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2)

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        pass

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        pass

    def convert_board_to_graph(self, example):
        """
        Takes board example and returns the Pytorch Geometric Data object
        """
        if len(example) == 3:  # training example
            board, pi, v = example
        else:  # testing example
            board = example
            pi = []
            v = -1

        graph, path = board
        m, n = graph.shape

        edges = []
        for i in range(m):
            for j in range(n):
                if i != j:
                    edges.append((i, j))
        edge_index = torch.tensor(edges, dtype=torch.long).reshape((2, -1))
        x = torch.tensor(self.game.getValidMoves(board, 1), dtype=torch.float).reshape((-1,1))

        edge_attr = np.zeros(shape=(edge_index.shape[1], 1))
        for i in range(edge_index.shape[1]):
            src = edge_index[:, i][0].item()
            dst = edge_index[:, i][1].item()
            edge_attr[i] = graph[src][dst]

        edge_attr = torch.tensor(edge_attr)

        choices = 1 - x

        data = Data(x=choices, edge_index=edge_index, edge_attr=edge_attr, v=v, pi=torch.tensor(pi))
        return data

    def convert_boards_to_graphs(self, examples):
        """
        Bulk boards to Data objects function
        """
        data = []
        for example in examples:
            data.append(self.convert_board_to_graph(example))
        return data
