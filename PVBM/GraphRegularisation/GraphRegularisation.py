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
        Iterate through the continuous graph and store the children of each node as well as its number of ancestor.

        Implemented as an explicit depth-first stack walk (rather than Python recursion) so that long,
        densely-connected vessel skeletons do not require deep call stacks, which can exhaust available
        memory on dense, high-resolution segmentations.

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
        def neighbors(i, j):
            return [
                (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1),
                (i - 1, j - 1), (i - 1, j + 1), (i + 1, j - 1), (i + 1, j + 1),
            ]

        def mark_plot(i, j, points):
            children = np.sum([A[point] for point in points if
                               (point[0] >= 0 and point[1] >= 0 and point[0] < A.shape[0] and point[1] < A.shape[1])])
            if children >= 1:
                plot[i, j] = 1

        points = neighbors(i, j)
        mark_plot(i, j, points)
        # Each stack frame mirrors one active call of the recursive version: [i, j, tree, points, next_idx].
        # A child is pushed and fully drained (mirroring an immediate recursive call) before its parent
        # frame resumes scanning the rest of its own points, so the order in which `A` is consumed - and
        # therefore every `children` sum computed along the way - matches the recursive implementation exactly.
        stack = [[i, j, tree, points, 0]]
        while stack:
            frame = stack[-1]
            fi, fj, ftree, fpoints, fidx = frame
            advanced = False
            while fidx < len(fpoints):
                point = fpoints[fidx]
                fidx += 1
                if point[0] >= 0 and point[0] < A.shape[0] and point[1] < A.shape[1] and point[1] >= 0:
                    if A[point] == 1:
                        frame[4] = fidx
                        child_tree = ftree.add_children(point[0], point[1])
                        A[point] = 0
                        child_points = neighbors(point[0], point[1])
                        mark_plot(point[0], point[1], child_points)
                        stack.append([point[0], point[1], child_tree, child_points, 0])
                        advanced = True
                        break
            if not advanced:
                stack.pop()

    def print_reg(self,tree, plot):
        """
        Correct graph irregularities by removing children branches which contain less than 10 pixels.

        Implemented as an explicit iterative post-order walk (rather than Python recursion) for the same
        reason as :meth:`recursive_reg`: tree depth tracks vessel length, and a Python-recursive
        implementation can require very deep call stacks on dense, high-resolution segmentations.

        :type tree: PVBM.GraphRegularisation.GraphRegularisation.TreeReg
        :param plot: A numpy array where the corrected graph is saved
        :type plot: np.array

        """
        values = {}
        stack = [(tree, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                if len(node.children) == 0:
                    values[id(node)] = 1
                else:
                    n = 1 + sum(values[id(child)] for child in node.children)
                    if n >= 10:
                        plot[node.plot] = 1
                    values[id(node)] = n
            else:
                stack.append((node, True))
                for child in node.children:
                    stack.append((child, False))
        return values[id(tree)]
