import numpy as np
import random
from tsp_solver.greedy import solve_tsp


class TSPEnviornment:
    def __init__(self, nodes):
        self.nodes = nodes
        self.optimal_path_length = -1
        self.solution = []

    def new_board(self, start_node):
        """
        Returns a new game board for TSP
            - A game board is a tuple containing a distance matrix (ndarray) representation of a graph
            - A Path (list) for the current player
        """
        graph = np.zeros(shape=(self.nodes, self.nodes), dtype=np.float32)
        path = [start_node]
        for src in range(self.nodes):
            for dst in range(self.nodes):
                # nodes cannot connect to themselves
                if src != dst:
                    if graph[dst][src] != 0:
                        graph[src][dst] = graph[dst][src]
                    else:
                        graph[src][dst] = random.randint(0, 100)
        # get optimal solution
        self.solution = solve_tsp(graph)
        # print('sol', self.solution)
        self.optimal_path_length = self.get_path_length((graph, self.solution))
        return graph, path

    def visit_node(self, board, dst):
        """
        Visits a given node on the graph and appends the node to the player's path
        """
        board, path = board
        new_board = np.copy(board)
        new_path = path.copy()
        new_path.append(dst)
        return new_board, new_path

    def get_path_length(self, board):
        """
        Returns the length of the cycle selected by the user
        """
        graph, path = board
        path_length = 0
        for i in range(1, len(path)):
            src = path[i-1]
            dst = path[i]
            path_length += graph[src][dst]
        # complete the cycle
        path_length += graph[path[-1]][path[0]]
        return path_length
