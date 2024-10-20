import numpy as np
from skimage.morphology import skeletonize,square,dilation

class Tree:
    """A class that store the topology information of a graph relevant for the CRE measurements"""
    def __init__(self, origin, startpoint):
        self.startpoint = startpoint
        self.children = []
        self.parents = []
        self.diameter_list = []
        self.endpoint = None
        self.plot_list = []
        self.origin = origin

    def update(self, diameter, plot):
        """
        Add a diameter and a plot to the list of the stored values

        :param diameter: diameter of the current measurement
        :type diameter: float
        :param plot: plot of the current measurement
        :type plot: np.array
        """
        self.diameter_list.append(diameter)
        self.plot_list.append(plot)

    def finished(self, endpoint):
        """
        Store the endpoint of the blood vessel to the list of the stored values

        :param endpoint: endpoint of the current measurement in x,y coordinate
        :type endpoint: (int,int)
        """
        self.endpoint = endpoint

    def add_children(self, startpoint):
        """
        Add a children node to the topology of the current node

        :param startpoint: startpoint of the new child node in x,y coordinate
        :type startpoint: (int,int)

        :next_tree: the newly initialised child node
        :rtype: PVBM.GraphCentralRetinalEquivalent.Tree
        """
        next_tree = Tree(self.origin, startpoint)
        next_tree.parents.append(self)
        self.children.append(next_tree)
        self.endpoint = startpoint
        return next_tree

    def set_children(self, children):
        """
        Store a children node to the list of the stored values

        :param children: children node to store in the list of the children nodes
        :type children: PVBM.GraphCentralRetinalEquivalent.Tree
        """
        self.children = children

    def remove_children(self):
        """
        Empty the list of the stored children nodes
        """
        self.saved_children = self.children[:]
        self.children = []
        return self

    def print_correct(self, saved_tree, max_d=0, first_tree=0):
        """
        Traverse the tree to find the subtree with the largest average diameter.

        This function computes the average diameter for the current tree node and its children, then identifies and
        returns the subtree with the largest average diameter encountered during the traversal.

        :param saved_tree: The subtree with the largest average diameter found so far.
        :type saved_tree: PVBM.GraphCentralRetinalEquivalent.Tree
        :param max_d: The maximum average diameter encountered so far.
        :type max_d: float
        :param first_tree: The initial tree node being processed.
        :type first_tree: PVBM.GraphCentralRetinalEquivalent.Tree

        :return: A tuple containing:
            - saved_tree (PVBM.GraphCentralRetinalEquivalent.Tree): The subtree with the largest average diameter.
            - max_d (float): The largest average diameter value.
            - first_tree (PVBM.GraphCentralRetinalEquivalent.Tree): The first subtree that met the diameter criteria.
        :rtype: Tuple[PVBM.GraphCentralRetinalEquivalent.Tree, float, PVBM.GraphCentralRetinalEquivalent.Tree]
        """
        if len(self.diameter_list) > 0:
            d = np.mean(self.diameter_list)
            if d > max_d:
                saved_tree = self
                max_d = d
                if first_tree == 0:
                    first_tree = self
        if len(self.children) >= 1:
            return max(
                [(saved_tree, max_d, first_tree)] + [children.print_correct(saved_tree, max_d, first_tree) for children
                                                     in self.children], key=lambda item: item[1])
        else:
            return saved_tree, max_d, first_tree

    def plotable_show_tree(self, l, shape, Toplot = False):
        """
        Compute and optionally visualize the tree's diameter information, then store the results in a list.

        This function computes various statistics about the tree's diameters and, if requested, creates a visualization
        of the tree. The results are stored in a dictionary, which is then appended to the provided list.

        :param l: A list to store the output dictionary containing the tree's diameter statistics and optional visualization.
        :type l: list
        :param shape: The shape of the array used for visualization.
        :type shape: tuple
        :param Toplot: A flag to decide if the visualization should be created and stored (setting it to True uses more RAM).
        :type Toplot: bool

        :return: None
        """
        if len(self.diameter_list) > 0:
            if Toplot:
                p = np.zeros((shape[0], shape[1]))
                tmp = self.plot_list  # [2:-2]
                for i in range(len(tmp)):
                    # print(i)
                    if i % 1 == 0:
                        p += tmp[i]
                tmp2 = np.zeros((shape[0], shape[1], 4))
                tmp2[:, :, 3] = dilation(p, square(3))
                tmp2[:, :, 1] = dilation(p, square(3))
            dico_output = {}
            if Toplot:
                dico_output['plot'] = tmp2
            dico_output['start'] = self.startpoint
            dico_output['Mean diameter'] = np.mean(self.diameter_list)  # [2:-2])
            dico_output['Median diameter'] = np.median(self.diameter_list)  # [2:-2])

            dico_output['Number of Measurement'] = len(self.diameter_list)  # [2:-2])
            dico_output['end'] = self.endpoint
            l.append(dico_output)
            return

    def compute_perpendicular_line(self,A, i, j, pi6, pj6, plotted_tmp, gt):
        """
        Compute the length of a perpendicular line extending from a given point and plot the results if required.

        This function calculates the length of a perpendicular line from a given point (i, j) based on the direction of
        the previous point (pi6, pj6). It optionally updates a provided plot with the calculated line. The function checks
        for boundaries and symmetry to ensure correct line computation.

        :param A: The 2D array representing the area where the perpendicular line is computed.
        :type A: np.array
        :param i: The current x-coordinate.
        :type i: int
        :param j: The current y-coordinate.
        :type j: int
        :param pi6: The previous x-coordinate.
        :type pi6: int
        :param pj6: The previous y-coordinate.
        :type pj6: int
        :param plotted_tmp: A temporary plot holder to be updated with the computed line.
        :type plotted_tmp: np.array or None
        :param gt: The ground truth 2D array used to determine boundary conditions.
        :type gt: np.array

        :return: The total length of the computed perpendicular line or None if a symmetry error is detected.
        :rtype: int or None
        """
        aoustideB = gt

        # Calculate the difference between the current and previous points
        direction = np.array([i - pi6, j - pj6])

        # Normalize the direction vector
        direction = direction / np.linalg.norm(direction)

        # Calculate the normal vector to the direction vector
        normal = np.array([-direction[1], direction[0]])

        # Temporary plot holder (clone of current plot)
        if plotted_tmp is not None:
            plotted_tmp_clone = plotted_tmp.copy()

        # Variables to keep track of distances in the positive and negative directions
        distance_pos = 0
        distance_neg = 0

        cond = True
        while cond:
            # Calculate positions in the positive direction
            pos_i = int(i + (distance_pos + 1) * normal[0])
            pos_j = int(j + (distance_pos + 1) * normal[1])

            # Calculate positions in the negative direction
            neg_i = int(i - (distance_neg + 1) * normal[0])
            neg_j = int(j - (distance_neg + 1) * normal[1])

            if (0 <= pos_i < A.shape[0] and 0 <= pos_j < A.shape[1] and
                    0 <= neg_i < A.shape[0] and 0 <= neg_j < A.shape[1]):

                pos_hit_boundary = aoustideB[pos_i, pos_j] == 1
                neg_hit_boundary = aoustideB[neg_i, neg_j] == 1

                if pos_hit_boundary and neg_hit_boundary:
                    if plotted_tmp is not None:
                        plotted_tmp_clone[pos_i, pos_j] = 1
                        plotted_tmp_clone[neg_i, neg_j] = 1
                    distance_pos += 1
                    distance_neg += 1
                elif pos_hit_boundary:
                    if plotted_tmp is not None:
                        plotted_tmp_clone[pos_i, pos_j] = 1
                    distance_pos += 1
                elif neg_hit_boundary:
                    if plotted_tmp is not None:
                        plotted_tmp_clone[neg_i, neg_j] = 1
                    distance_neg += 1
                else:
                    cond = False
            else:
                cond = False

        # If one side has extended significantly more than the other (symmetric error check)
        if abs(distance_pos - distance_neg) > 1:
            return None

        # If there's no symmetry error, commit the changes to the plot
        if plotted_tmp is not None:
            np.copyto(plotted_tmp, plotted_tmp_clone)
        # plt.imshow(inp[0]/255)
        # plt.imshow(zones/3)
        # plt.imshow(segmentation + gtvs.reshape((1472,1472,1)) + plotted_tmp.reshape((1472,1472,1)))
        # plt.show()
        total_distance = distance_pos + distance_neg
        return total_distance

    # def recursive_CRE(self,A, B, D, i, j, n, pi, pj, pi2, pj2, pi3, pj3, pi4, pj4, pi5, pj5, pi6, pj6, plotted_tmp, i_or,
    #                     j_or, curtree, gt, x_c, y_c, radius_zone_C, p=0, ultimatum=False):
    #     """
    #     Recursively track the topology required to computed CRAE or CRVE equivalent for a given blood vessel graph.
    #
    #     This function traverses a segmented image to compute the diameter of blood vessels and updates the current tree
    #     structure with the computed diameters. The function also handles visualization and ensures that the traversal
    #     respects the boundaries of the given region.
    #
    #     :param A: The array representing the binary blood vessel segmentation.
    #     :type A: np.array
    #     :param B: An auxiliary array used for processing.
    #     :type B: np.array
    #     :param D: Another auxiliary array used for processing.
    #     :type D: np.array
    #     :param i: The current x-coordinate.
    #     :type i: int
    #     :param j: The current y-coordinate.
    #     :type j: int
    #     :param n: The current recursion depth.
    #     :type n: int
    #     :param pi: Previous x-coordinate.
    #     :type pi: int
    #     :param pj: Previous y-coordinate.
    #     :type pj: int
    #     :param pi2: Second previous x-coordinate.
    #     :type pi2: int
    #     :param pj2: Second previous y-coordinate.
    #     :type pi2: int
    #     :param pj2: int
    #     :param pi3: Third previous x-coordinate.
    #     :type pi3: int
    #     :param pj3: Third previous y-coordinate.
    #     :type pi3: int
    #     :param pj3: int
    #     :param pi4: Fourth previous x-coordinate.
    #     :type pi4: int
    #     :param pj4: Fourth previous y-coordinate.
    #     :type pi4: int
    #     :param pj4: int
    #     :param pi5: Fifth previous x-coordinate.
    #     :type pi5: int
    #     :param pj5: Fifth previous y-coordinate.
    #     :type pi5: int
    #     :param pj5: int
    #     :param pi6: Sixth previous x-coordinate.
    #     :type pi6: int
    #     :param pj6: Sixth previous y-coordinate.
    #     :type pi6: int
    #     :param pj6: int
    #     :param plotted_tmp: A temporary plot holder to be updated with the computed line.
    #     :type plotted_tmp: np.array or None
    #     :param i_or: The original x-coordinate of the starting point.
    #     :type i_or: int
    #     :param j_or: The original y-coordinate of the starting point.
    #     :type j_or: int
    #     :param curtree: The current tree structure being updated with diameters.
    #     :type curtree: PVBM.GraphCentralRetinalEquivalent.Tree
    #     :param gt: The ground truth 2D array used to determine boundary conditions.
    #     :type gt: np.array
    #     :param x_c: The x-coordinate of the optic disc center.
    #     :type x_c: int
    #     :param y_c: The y-coordinate of the optic disc center.
    #     :type y_c: int
    #     :param radius_zone_C: The radius of the optic disc zone.
    #     :type radius_zone_C: int
    #     :param p: An auxiliary parameter used for processing.
    #     :type p: int
    #     :param ultimatum: A flag used to determine specific processing conditions.
    #     :type ultimatum: bool
    #
    #     :return: None
    #     """
    #     up = (i - 1, j)
    #     down = (i + 1, j)
    #     left = (i, j - 1)
    #     right = (i, j + 1)
    #
    #     up_left = (i - 1, j - 1)
    #     up_right = (i - 1, j + 1)
    #     down_left = (i + 1, j - 1)
    #     down_right = (i + 1, j + 1)
    #     points = [up, down, left, right, up_left, up_right, down_left, down_right]
    #     children = np.sum([A[point] for point in points])
    #     diameter = 0
    #     if n >= 10:
    #         if pi6 != None:
    #             if plotted_tmp is not None:
    #                 prev_plot = plotted_tmp.copy()
    #             diameter = self.compute_perpendicular_line(A, i, j, pi6, pj6, plotted_tmp, gt)
    #             if diameter is not None:
    #                 p_up = None
    #                 if plotted_tmp is not None:
    #                     p_up = plotted_tmp - prev_plot
    #                 curtree.update(diameter, p_up)
    #     else:
    #         curtree = curtree.add_children((i, j))
    #
    #     if ((x_c - i) ** 2 + (
    #             y_c - j) ** 2) ** 0.5 > radius_zone_C:  # and np.mean(curtree.diameter_list)*6 <= 80  and ultimatum == False:
    #         curtree.finished((i, j))
    #         return
    #
    #     # elif (children==0 or children >1) and len(curtree.diameter_list) > 30: #and ultimatum == True:
    #     #    curtree.finished((i,j))
    #     #    return
    #
    #     elif children == 0:
    #         curtree.finished((i, j))
    #         return
    #
    #     elif children > 1:
    #         if np.mean(curtree.diameter_list) * 6 > 80 and len(curtree.diameter_list) > 30:
    #             # ultimatum = True
    #             todo = 0
    #         curtree = curtree.add_children((i, j))
    #         if plotted_tmp is not None:
    #             plotted_tmp = np.zeros((A.shape[0], A.shape[1]))
    #         n = 0
    #         p += 1
    #
    #     for point in points:
    #         if point[0] >= 0 and point[0] < B.shape[0] and point[1] < B.shape[1] and point[1] >= 0:
    #             if A[point] == 1:
    #                 A[point] = 0
    #                 self.recursive_CRE(A, B, D, point[0], point[1], n + 1, i, j, pi, pj, pi2, pj2, pi3, pj3, pi4, pj4, pi5,
    #                                 pj5, plotted_tmp, i_or, j_or, curtree, gt, x_c, y_c, radius_zone_C, p, ultimatum)


    #Moved to iterative due to stack overflow in c
    def iterative_CRE(self, A, B, D, i, j, n, pi, pj, pi2, pj2, pi3, pj3, pi4, pj4, pi5, pj5, pi6, pj6, plotted_tmp,
                      i_or,
                      j_or, curtree, gt, x_c, y_c, radius_zone_C, p=0, ultimatum=False):
        """
        Iteratively track the topology required to compute CRAE or CRVE equivalent for a given blood vessel graph.
        Equivalent to the recursive_CRE function but uses an iterative approach with a stack.
        """
        # Stack to hold the state of variables for each iteration
        stack = [(i, j, n, pi, pj, pi2, pj2, pi3, pj3, pi4, pj4, pi5, pj5, pi6, pj6, plotted_tmp, i_or, j_or, p,
                  ultimatum, curtree)]

        while stack:
            # Unpack the state including i_or and j_or
            i, j, n, pi, pj, pi2, pj2, pi3, pj3, pi4, pj4, pi5, pj5, pi6, pj6, plotted_tmp, i_or, j_or, p, ultimatum, curtree = stack.pop()

            # Directions to move in
            up = (i - 1, j)
            down = (i + 1, j)
            left = (i, j - 1)
            right = (i, j + 1)

            up_left = (i - 1, j - 1)
            up_right = (i - 1, j + 1)
            down_left = (i + 1, j - 1)
            down_right = (i + 1, j + 1)

            points = [up, down, left, right, up_left, up_right, down_left, down_right]

            # Count children (valid neighboring pixels that are part of the vessel)
            children = np.sum(
                [A[point] for point in points if 0 <= point[0] < B.shape[0] and 0 <= point[1] < B.shape[1]])

            # If recursion depth >= 10, compute the diameter
            # print(i,j,children,pi6,pj6)
            if n >= 10:
                if pi6 is not None:
                    if plotted_tmp is not None:
                        prev_plot = plotted_tmp.copy()
                    diameter = self.compute_perpendicular_line(A, i, j, pi6, pj6, plotted_tmp, gt)
                    if diameter is not None:
                        p_up = None
                        if plotted_tmp is not None:
                            p_up = plotted_tmp - prev_plot
                        curtree.update(diameter, p_up)
            else:
                curtree = curtree.add_children((i, j))

            # Exit conditions based on distance from center
            if ((x_c - i) ** 2 + (y_c - j) ** 2) ** 0.5 > radius_zone_C:
                curtree.finished((i, j))
                continue

            # If no children, finish this branch
            elif children == 0:
                curtree.finished((i, j))
                continue

            # If more than one child, add children and reset state
            elif children > 1:
                if np.mean(curtree.diameter_list) * 6 > 80 and len(curtree.diameter_list) > 30:
                    pass  # Handle special condition if needed
                curtree = curtree.add_children((i, j))
                if plotted_tmp is not None:
                    plotted_tmp = np.zeros((A.shape[0], A.shape[1]))
                n = 0
                p += 1

            # Push neighbors onto the stack for further exploration
            for point in points:
                if 0 <= point[0] < B.shape[0] and 0 <= point[1] < B.shape[1] and A[point] == 1:
                    A[point] = 0
                    # Push the new state onto the stack with i_or and j_or included
                    stack.append(
                        (point[0], point[1], n + 1, i, j, pi, pj, pi2, pj2, pi3, pj3, pi4, pj4, pi5, pj5,
                         plotted_tmp, i_or, j_or, p, ultimatum, curtree))
