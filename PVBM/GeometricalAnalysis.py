import numpy as np
from skimage.morphology import skeletonize
from scipy.signal import convolve2d
from PVBM.helpers.tortuosity import compute_tortuosity
from PVBM.helpers.perimeter import compute_perimeter_
from PVBM.helpers.branching_angle import compute_angles_dictionary
import warnings

class GeometricalVBMs:
    """
    A class that can perform geometrical biomarker computation for a fundus image.

    .. deprecated:: 2.9.0
       This class will be removed in version 3.0. Use `GeometryAnalysis.GeometricalVBMs` instead.
    """

    def __init__(self):
        super(GeometricalVBMs, self).__init__()

        warnings.warn(
            "The GeometricalVBMs class is deprecated and will be removed in version 3.0. "
            "Use GeometryAnalysis.GeometricalVBMs instead.",
            DeprecationWarning,
            stacklevel=2
        )



    def area(self,segmentation):
        """
    Computes the area of the blood vessels calculated as the total number of pixels in the segmentation,
    and is expressed in pixels^2 (squared pixels).

    :param segmentation: The segmentation is a two-dimensional array (HxW) with binary values (0 or 1).
    :type segmentation: array
    :return: The area of the segmentation.
    :rtype: float
    """
        return np.sum(segmentation)
    
    def compute_particular_points(self,segmentation_skeleton):
        """
    The particular point is the union between the endpoints and the intersection points.
    This function computes the number of endpoints and intersection points of the fundus image vasculature segmentation.

    :param segmentation_skeleton: The skeleton of the segmentation is a two-dimensional array (HxW) with binary values (0 or 1).
    :type segmentation_skeleton: np.ndarray
    :return:
        - The number of endpoints
        - The number of intersection points,
        - An array with endpoint pixel positions
        - An array with intersection point pixel positions.
    
    :rtype: (int, int, np.ndarray, np.ndarray)
    """
        filter_ = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
        neighbours = convolve2d(segmentation_skeleton, filter_, mode="same")

        end_points = neighbours == 11
        inter_points = neighbours >= 13

        end_points_count = np.sum(end_points)
        inter_points_count = np.sum(inter_points)

        return end_points_count, inter_points_count, end_points, inter_points
    
    def compute_tortuosity_length(self,segmentation_skeleton):
        """
    Computes the median tortuosity and the length of the fundus image vasculature segmentation.
    
    :param segmentation_skeleton: The skeleton of the segmentation is a two-dimensional array (HxW) with binary values (0 or 1).
    :type segmentation_skeleton: np.ndarray
    :return:
        - The median tortuosity
        - The overall length (in pixel)
        - A list of chord distance of each blood vessel (in pixel)
        - A list of lengths distance (arc) of each blood vessel (in pixel)
        - A dictionary with connection information.
    
    :rtype: (float, float, list, list, dict)
    """
        median_tor,length,arc,chord,connection_dico = compute_tortuosity(segmentation_skeleton)
        return median_tor,length ,chord, arc,connection_dico

    def compute_perimeter(self,segmentation):
        """
    Computes the perimeter and the border of the fundus image vasculature segmentation.

    :param segmentation: The segmentation is a two-dimensional array (HxW) with binary value (0 or 1).
    :type segmentation: np.ndarray
    :return:
        - The perimeter (in pixel)
        - A matrix containing the edges of the segmentation.
    
    :rtype: (float, np.ndarray)
    """
        segmentation = segmentation
        filter_ = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        derivative = convolve2d(segmentation,filter_, mode="same")
        border = derivative>0
        segmentation_skeleton = skeletonize(np.ascontiguousarray(border))
        perim,_ = compute_perimeter_(segmentation_skeleton.copy())
        return perim,segmentation_skeleton
    
    def compute_branching_angles(self,segmentation_skeleton):
        """
    Computes the mean, standard deviation, and median branching angle of the fundus image vasculature segmentation.

    :param segmentation_skeleton: The skeleton of the segmentation is a two-dimensional array (HxW) with binary values (0 or 1).
    :type segmentation_skeleton: np.ndarray
    :return:
        - The mean of the branching angles (in degrees)
        - The standard deviation of the branching angles (in degrees)
        - The median of the branching angles (in degrees)
        - A dictionary containing all the branching angles with their respective indices in the array as keys
        - A two-dimensional numpy array representing the visualization of the computed centroid of the segmentation skeleton.

    :rtype: (float, float, float, dict, np.ndarray)
    """
        img= segmentation_skeleton
        #return compute_angles_dictionary(img)
        angle_dico,centroid = compute_angles_dictionary(img)
        mean_ba = np.mean(list(angle_dico.values()))
        std_ba = np.std(list(angle_dico.values()))
        median_ba = np.median(list(angle_dico.values()))
        return mean_ba, std_ba, median_ba,angle_dico,centroid
