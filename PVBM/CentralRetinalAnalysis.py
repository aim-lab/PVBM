import numpy as np
from skimage.morphology import skeletonize
from scipy.signal import convolve2d
from PVBM.helpers.tortuosity import compute_tortuosity
from PVBM.helpers.perimeter import compute_perimeter_
from PVBM.helpers.branching_angle import compute_angles_dictionary
#from PVBM.helpers.far import far
from PVBM.GraphRegularisation.GraphRegularisation import TreeReg
from PVBM.GraphCRE.GraphCentralRetinalEquivalent import Tree
class CREVBMs:
    """A class that can perform geometrical biomarker computation for a fundus image.
    """

    def extract_subgraphs(self,graphs, x_c, y_c):
        """
        Extract B, the a graph where each of the disconnected subgraph is labeled differently and D which contains the euclidian distance graph between each
        point in the graph and the optic disc.

        :param graphs: Original blood vessel segmentation graph
        :type graphs: array
        :param x_c: x axis of the optic disc center
        :type x_c: int
        :param y_c: y axis of the optic disc center
        :type y_c: int

        :return: B,D
        :rtype: tuple
        """
        B = np.zeros_like(graphs)
        D = np.zeros_like(graphs)
        n = 1
        for i in range(graphs.shape[0]):
            for j in range(graphs.shape[1]):
                if B[i, j] == 0 and graphs[i, j] == 1:
                    self.recursive_subgraph(graphs, B, D, i, j, n, x_c, y_c)
                    n += 1
        return B, D

    def recursive_subgraph(self,A, B, D, i, j, n, x_c, y_c):
        """
        Recursively extract the value within B and D.

        :param A: Original blood vessel segmentation graph
        :type A: array
        :param B: A graph where each of the disconnected subgraph is labeled differentl, which is initialized by a zeros matrix and recursively built
        :type B: array
        :param D: Euclidian Distance graph between each point in A and the optic disc center (x_c,y_c), which is initialized by a zeros matrix and recursively built
        :type D: array
        :param i: Current x axis location within the graph
        :type i: int
        :param j: Current y axis location within the graph
        :type j: int
        :param n: Current number of point distance since the optic disc
        :type n: int
        :param x_c: x axis of the optic disc center
        :type x_c: int
        :param y_c: y axis of the optic disc center
        :type y_c: int

        :return: B,D
        :rtype: tuple
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
        for point in points:
            if point[0] >= 0 and point[0] < B.shape[0] and point[1] < B.shape[1] and point[1] >= 0:
                if A[point] == 1:
                    B[point] = n
                    A[point] = 0
                    D[point] = ((y_c - point[0]) ** 2 + (x_c - point[1]) ** 2) ** 0.5
                    self.recursive_subgraph(A, B, D, point[0], point[1], n, x_c, y_c)

    def central_equivalent(self, l,formula):
        """
        Recursively compute the central retinal equivalent given a list of blood vessel measurement and a formula to

        :param l: list of blood vessel measurements
        :type l: List
        :param formula: One of the following formulas: PVBM.CentralRetinalAnalysis.crae_hubbard, PVBM.CentralRetinalAnalysis.crve_hubbard, PVBM.CentralRetinalAnalysis.crae_knudtson or PVBM.CentralRetinalAnalysis.crve_knudtson
        :type formula: python function

        :return: the computed central retinal equivalent VBM
        :rtype: float
        """
        l = np.sort(l[:])
        new_l = []
        pivot =None
        if len(l) == 1:
            return l[0]
        if len(l)%2 !=0:
            idx_pivot = len(l)//2
            pivot =  l[idx_pivot]
        for i in range(len(l)//2):
            new_l.append(formula(l[i],l[-i-1]))
        if pivot != None:
            new_l.append(pivot)
        return self.central_equivalent(new_l,formula)

    def crae_hubbard(self,w1, w2):
        """
        Recursive formulas of the Central Retinal Arteriolar Equivalent according to Hubbard et al.
        :param w1: w1 vessel measurement
        :type w1: float
        :param w2: w2 vessel measurement
        :type w2: float

        :return: the root vessel of w1 and w2 estimation
        :rtype: float
        """
        return (0.87 * w1 ** 2 + 1.01 * w2 ** 2 - 0.22 * w1 * w2 - 10.76) ** 0.5

    def crve_hubbard(self,w1, w2):
        """
        Recursive formulas of the Central Retinal Venular Equivalent according to Hubbard et al.
        :param w1: w1 vessel measurement
        :type w1: float
        :param w2: w2 vessel measurement
        :type w2: float

        :return: the root vessel of w1 and w2 estimation
        :rtype: float
        """
        return (0.72 * w1 ** 2 + 0.91 * w2 ** 2 + 450.05) ** 0.5

    def crae_knudtson(self,w1, w2):
        """
        Recursive formulas of the Central Retinal Arteriolar Equivalent according to Knudtson et al.
        :param w1: w1 vessel measurement
        :type w1: float
        :param w2: w2 vessel measurement
        :type w2: float

        :return: the root vessel of w1 and w2 estimation
        :rtype: float
        """
        return 0.88 * (w1 ** 2 + w2 ** 2) ** 0.5

    def crve_knudtson(self,w1, w2):
        """
        Recursive formulas of the Central Retinal Venular Equivalent according to Knudtson et al.
        :param w1: w1 vessel measurement
        :type w1: float
        :param w2: w2 vessel measurement
        :type w2: float

        :return: the root vessel of w1 and w2 estimation
        :rtype: float
        """
        return 0.95 * (w1 ** 2 + w2 ** 2) ** 0.5


    def apply_roi(self, segmentation, skeleton, zones_ABC):
        """
        Apply a region of interest (ROI) mask to the segmentation and skeleton images.

        :param segmentation: The segmentation image containing binary values within {0, 1}.
        :type segmentation: np.array
        :param skeleton: The skeleton image containing binary values within {0, 1}.
        :type skeleton: np.array
        :param zones_ABC: A mask image used to exclude specific zones, where the second channel defines the exclusion areas.
        :type zones_ABC: np.array

        :return: A tuple containing:
            - The modified segmentation image with the ROI applied.
            - The modified skeleton image with the ROI applied.
        :rtype: Tuple[np.array, np.array]
        """
        zone_A_ = zones_ABC[:, :, 1] / 255
        zone_B_ = zones_ABC[:, :, 0] / 255
        zone_C_ = zones_ABC[:, :, 2] / 255
        roi = (zone_C_ - zone_B_)
        segmentation_roi = (segmentation * roi)
        skeleton_roi = (skeleton * roi)


        return segmentation_roi, skeleton_roi

    def compute_central_retinal_equivalents(self,blood_vessel, skeleton, xc,yc, radius, artery = True, Toplot = False):
        """
        Compute the CRAE or CRVE equivalent for a given blood vessel graph.

        :param blood_vessel: blood_vessel segmentation containing binary values within {0,1}
        :type blood_vessel: np.array
        :param skeleton: blood_vessel segmentation skeleton containing binary values within {0,1}
        :type skeleton: np.array
        :param xc: x axis of the optic disc center
        :type xc: int
        :param yc: y axis of the optic disc center
        :type yc: int
        :param radius: radius in pixel of the optic disc
        :type radius: int
        :param artery: Flag to decide if to use CRAE or CRVE formulas (artery to True means CRAE, and to False means CRVE)
        :type artery: Bool
        :param Toplot: Flag to decide if to store the visualisation element. (Setting it to true use more RAM)
        :type Toplot: Bool


        :return: A tuple containing:
            - A result dictionnary (Dict): Dictionnary containing the computer CRE (-1 if it has failed)
            - plotable_list (List): A summary that contains the topology information required to plot the visualisation (really useful when Toplot is True). Return None if the computation has failed.
        :rtype: Tuple[Dict, List]
        """
        radius_zone_C = int(3 * radius)
        ## Extract the distances graphs
        B, D = self.extract_subgraphs(graphs=skeleton.copy(), x_c=xc, y_c=yc)

        ## Extract the starting points list by navigating through the skeleton graph
        starting_points = np.zeros((skeleton.shape[0], skeleton.shape[1]), dtype=float)
        for i in set(list(B.reshape(-1))) - {0}:
            mask = B == i
            if mask.sum() >= 50:
                min_index = (D * mask + (1 - mask) * 1e10).argmin()
                min_coordinates = np.unravel_index(min_index, D.shape)
                # print(((min_coordinates[0] - yc)**2 + (min_coordinates[1] - xc)**2)**0.5)
                if ((min_coordinates[0] - yc) ** 2 + (min_coordinates[1] - xc) ** 2) ** 0.5 < 20 + 2 * radius:
                    starting_points[min_coordinates[0], min_coordinates[1]] = 1
        starting_point_list = np.argwhere(starting_points == 1)


        ### Cleaning the skeleton graph irregularities
        B = np.zeros((blood_vessel.shape[0], blood_vessel.shape[1]))
        tree_reg_list = []
        plot = np.zeros((blood_vessel.shape[0], blood_vessel.shape[1]))
        for idx_start in starting_point_list:
            tree = TreeReg(idx_start[0], idx_start[1])
            tree.recursive_reg(skeleton.copy(), idx_start[0], idx_start[1], 0, tree, plot)
            tree_reg_list.append(tree)
        plot_list = []
        for tree_reg in tree_reg_list:
            plot = np.zeros((blood_vessel.shape[0], blood_vessel.shape[1]))
            tree_reg.print_reg(tree_reg, plot)
            plot_list.append(plot.copy())
        skoustideB_reg = np.sum(np.array(plot_list), axis=0)


        B = np.zeros_like(skeleton)
        D = np.zeros_like(skeleton)
        plotted_tmp = None if Toplot is False else np.zeros((blood_vessel.shape[0], blood_vessel.shape[1]))

        #####Initialising a dictionary that will be used to store the topology of th graph
        dico = {}

        ####Navigating through the graph to fill the the topology dico
        for idx_start in starting_point_list:
            i, j = idx_start
            tree = Tree((idx_start[0],idx_start[1]),(idx_start[0],idx_start[1]))
            tree.iterative_CRE(skeleton.copy(), B, D, idx_start[0], idx_start[1], 1, None, None, None, None, None,
                            None, None, None, None, None, None, None, plotted_tmp, i, j, tree, blood_vessel, xc, yc,
                            radius_zone_C)

            dico[(idx_start[0], idx_start[1])] = tree

        tree_list = list(dico.values())

        ####Second graph navigation to check consistency of the topology dico (parent vessel is supposed to be bigger than its children).
        final_list = []
        for i in range(len(tree_list)):
            tmp = tree_list[i].print_correct(tree_list[i], max_d=0)
            if type(tmp[2]) is not int:
                if tmp[0].startpoint != tmp[2].startpoint and np.mean(tmp[0].diameter_list) > np.mean(
                        tmp[2].diameter_list) + 2:
                    print('Potential error', i)
                    idx_start = tmp[0].startpoint
                    i, j = idx_start
                    tree = Tree((idx_start[0], idx_start[1]), (idx_start[0], idx_start[1]))
                    tree.iterative_CRE(skoustideB_reg.copy(), B, D, idx_start[0], idx_start[1], 1, None, None, None,
                                    None, None, None, None, None, None, None, None, None, plotted_tmp, i, j, tree, blood_vessel,
                                    xc, yc, radius_zone_C)
                    final_list.append(tree)  # .remove_children())
                else:
                    final_list.append(tmp[2])  # .remove_children())
            else:
                final_list.append(tmp[0])  # .remove_children())

        ### Aggregating the topology dico results
        plotable_list = []
        for tree in final_list:
            tree.plotable_show_tree(plotable_list, blood_vessel.shape, Toplot = Toplot)
        ### Compute and return the CRE VBMs
        if len(plotable_list) != 0:
            blood_vessel_lista = []
            blood_vessel_lista.append(plotable_list[0]['Mean diameter'])
            for j in range(1, len(plotable_list)):
                blood_vessel_lista.append(plotable_list[j]['Mean diameter'])
            if artery:
                craek = self.central_equivalent(np.sort(blood_vessel_lista)[-6:], self.crae_knudtson)
                craeh = self.central_equivalent(np.sort(blood_vessel_lista)[-6:], self.crae_hubbard)
                return ({"craek": craek, "craeh":craeh}, plotable_list)
            else:
                crvek = self.central_equivalent(np.sort(blood_vessel_lista)[-6:], self.crve_knudtson)
                crveh = self.central_equivalent(np.sort(blood_vessel_lista)[-6:], self.crve_hubbard)
                return ({"craek": crvek, "craeh": crveh}, plotable_list)
        else:
            if artery:
                return ({"craek": -1, "craeh": -1}, None)
            else:
                return ({"craek": -1, "craeh": -1}, None)

# if __name__ == "__main__":
#     import numpy as np
#     from skimage.morphology import skeletonize
#     from PIL import Image
#
#     center = (1278,721)
#     radius = 103
#
#     blood_vessel_segmentation_path = '/Users/jonathanfhima/Library/Containers/13E15484-8207-4BD5-BEA5-CB0FAD85FCF7/Data/Documents/test123/segmentation/DR_2_ICDR.jpg'
#     segmentation = np.array(Image.open(blood_vessel_segmentation_path)) / 255  # Open the segmentation
#     segmentation = segmentation[:,:,2]
#     skeleton = skeletonize(segmentation) * 1
#     creVBM = CREVBMs()  # Instanciate a geometrical VBM object
#
#     roi = '/Users/jonathanfhima/Library/Containers/13E15484-8207-4BD5-BEA5-CB0FAD85FCF7/Data/Documents/test123/ROI/DR_2_ICDR.jpg'
#     roi = np.array(Image.open(roi))
#
#     zones_ABC = '/Users/jonathanfhima/Library/Containers/13E15484-8207-4BD5-BEA5-CB0FAD85FCF7/Data/Documents/test123/zones_ABC/DR_2_ICDR.jpg'
#     zones_ABC = np.array(Image.open(zones_ABC))
#
#     segmentation_roi, skeleton_roi = creVBM.apply_roi(
#         segmentation=segmentation,
#         skeleton=skeleton,
#         zones_ABC=zones_ABC,
#     )
#
#     vbms, visual = creVBM.compute_central_retinal_equivalents(
#         blood_vessel=segmentation_roi,
#         skeleton=skeleton_roi,
#         xc=center[0],
#         yc=center[1],
#         radius=radius
#     )