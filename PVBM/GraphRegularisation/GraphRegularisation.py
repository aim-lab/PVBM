import numpy as np
class TreeReg:
    """A class that store the topology information of a graph in order to remove irregularities"""
    def __init__(self, i, j):
        """
        Initialise a tree node with the root of the tree node coordinate and an empty children node list

        :param i: X axis coodinates of the current tree node
        :type i: int
        :param j: Y axis coodinates of the current tree node
        :type j: int
        """
        self.plot = (i, j)
        self.children = []

    def add_children(self, i, j):
        """
        Instantiate a children node and add it to the list of the children nodes

        :param i: X axis coodinates of the root of the children node
        :type i: int
        :param j: Y axis coodinates of the root of the children node
        :type j: int
        """
        next_tree = TreeReg(i, j)
        self.children.append(next_tree)
        return next_tree

    def recursive_reg(self,A, i, j, n, tree, plot):
        """
        A recursive function that iterate through the continuous graph and store the children of each node as well as its number of ancestor

        :param A: skeleton of the segmentation
        :type A: np.float
        :param i: X axis coodinates of the current node
        :type i: int
        :param j: Y axis coodinates of the current node
        :type j: int
        :param n: number of ancestor of the current node
        :type n: int
        :param tree: current tree node
        :type tree: PVBM.GraphRegularisation.GraphRegularisation.TreeReg
        :param plot: A numpy array where the continuous navigated graph is saved
        :type plot: np.array

        """
        up = (i - 1, j)
        down = (i + 1, j)
        left = (i, j - 1)
        right = (i, j + 1)

        up_left = (i - 1, j - 1)
        up_right = (i - 1, j + 1)
        down_left = (i + 1, j - 1)
        down_right = (i + 1, j + 1)
        points = [up, down, left, right, up_left, up_right, down_left, down_right]
        children = np.sum([A[point] for point in points if
                           (point[0] >= 0 and point[1] >= 0 and point[0] < A.shape[0] and point[1] < A.shape[1])])
        if children >= 1:
            plot[i, j] = 1

        for point in points:
            if point[0] >= 0 and point[0] < A.shape[0] and point[1] < A.shape[1] and point[1] >= 0:
                if A[point] == 1:
                    tree__ = tree.add_children(point[0], point[1])
                    A[point] = 0
                    self.recursive_reg(A, point[0], point[1], n + 1, tree__, plot)

    def print_reg(self,tree, plot):
        """
        A recursive function that correct graphs irregularities by removing children branch which contains less than 10 pixels

        :type tree: PVBM.GraphRegularisation.GraphRegularisation.TreeReg
        :param plot: A numpy array where the corrected graph is saved
        :type plot: np.array

        """
        if len(tree.children) == 0:
            return 1
        else:
            n = 1 + sum([self.print_reg(child, plot) for child in tree.children])
            if n >= 10:
                plot[tree.plot] = 1
            return n