"""
Author: Eric P. Nichols
Date: Feb 8, 2008.
Board class.
Board data:
  1=white, -1=black, 0=empty
  first dim is column , 2nd is row:
     pieces[1][7] is the square in column 2,
     at the opposite end of the board in row 8.
Squares are stored and manipulated as (x,y) tuples.
x is the column, y is the row.
"""
import numpy as np


class TSPEnviornment:
    def __init__(self, nodes):
        self.nodes = nodes
        self.optimal_path_length = -1

    def new_board(self, start_node):
        current_node = start_node
        graph = np.zeros(shape=(self.nodes, self.nodes), dtype=np.float32)
        visited_nodes = np.zeros((self.nodes,))
        visited_nodes[start_node] = 1
        for src in range(self.nodes):
            for dst in range(self.nodes):
                # nodes cannot connect to themselves
                if src != dst:
                    if graph[dst][src] != 0:
                        graph[src][dst] = graph[dst][src]
                    else:
                        graph[src][dst] = self.get_random_weight(0, 200)
        # get optimal solution
        self.optimal_path_length = self.compute_solution(graph, 0)
        first_layer = graph
        second_layer = np.tile(visited_nodes, (self.nodes, 1))
        third_layer = np.tile(current_node, (self.nodes, self.nodes))
        return np.stack((first_layer, second_layer, third_layer), axis=0), 0

    def visit_node(self, board, dst):
        board, current_path_length = board
        new_board = np.copy(board)
        new_board[1][:,dst] = 1
        src = int(board[2][0][0])
        current_path_length += new_board[0][src][dst]
        new_board[2][:,:] = dst
        return new_board, current_path_length

    def compute_solution(self, graph, start):
        # implementation of traveling Salesman Problem
        # store all vertex apart from source vertex
        vertex = []
        for i in range(self.nodes):
            if i != start:
                vertex.append(i)

                # store minimum weight Hamiltonian Cycle
        min_path = float('inf')
        while True:
            # store current Path weight(cost)
            current_pathweight = 0
            # compute current path weight
            k = start
            for i in range(len(vertex)):
                current_pathweight += graph[k][vertex[i]]
                k = vertex[i]
            current_pathweight += graph[k][start]
            # update minimum
            min_path = min(min_path, current_pathweight)
            if not self.next_permutation(vertex):
                break
        return min_path

    @staticmethod
    def next_permutation(L):
        n = len(L)
        i = n - 2
        while i >= 0 and L[i] >= L[i + 1]:
            i -= 1
        if i == -1:
            return False
        j = i + 1
        while j < n and L[j] > L[i]:
            j += 1
        j -= 1
        L[i], L[j] = L[j], L[i]
        left = i + 1
        right = n - 1
        while left < right:
            L[left], L[right] = L[right], L[left]
            left += 1
            right -= 1
        return True

    @staticmethod
    def get_random_weight(min_v, max_v):
        return (max_v - min_v) * np.random.random_sample() + min_v
