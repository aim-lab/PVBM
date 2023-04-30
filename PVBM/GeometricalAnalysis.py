import numpy as np
from skimage.morphology import skeletonize
from scipy.signal import convolve2d
from PVBM.helpers.tortuosity import compute_tortuosity
#from PVBM.helpers.perimeter import compute_perimeter
from PVBM.helpers.branching_angle import compute_angles_dictionary
#from PVBM.helpers.far import far

class GeometricalVBMs:
    """A class which holds all required methods to perform
        geometrical biomarker computation.
    """

    def area(self,segmentation):
        """This function compute the overall area of the segmentation.

        :param array segmentation: The segmentation is an two-dimensional array (HxW) with binary value (0 or 1).
        :returns: The area of the segmentation

        :rtype: float
        """
        return np.sum(segmentation)
    
    def compute_particular_points(self,segmentation_skeleton):
        """
        This function compute the number of endpoints and intersection points of the segmentation.

        :param np.ndarray segmentation_skeleton: The skeleton of the segmentation is a two-dimensional array (HxW) with binary values (0 or 1).
        :returns int end_points_count: The number of endpoints 
        :returns int inter_points_count: The number of intersection points 
        :returns np.ndarray end_points: An array where the value at the position (i,j) is True if the pixel is an endpoint, False otherwise
        :returns np.ndarray inter_points: An array where the value at the position (i,j) is True if the pixel is an intersection point, False otherwise
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
        """This function compute the median tortuosity and the lengh of the segmentation.

        :param array segmentation_skeleton: The skeleton of the segmentation is an two-dimensional array (HxW) with binary value (0 or 1).
        :returns float median_tor: The median tortuosity
        :returns float length: The overall normalized length 
        :returns list tor: A list containing the arc-chord ratio of every blood vessels (between two particular points).
        :returns list l: A list containing the non normalized length of every blood vessels (between two particular points).
        :returns dictionary connection_dico: A dictionnary containing in key some particular points and in value the list of the particular points connected to him, with their length. This dictionnary was build by navigating through the skeleton. 

        :rtype: (float,float,List,List,Dictionary)
        """
        median_tor,length,tor,l,connection_dico = compute_tortuosity(segmentation_skeleton)
        return median_tor,length ,tor,l,connection_dico

    def compute_perimeter(self,segmentation):
        """This function compute the perimeter and the border of the segmentation.

            :param array segmentation: The segmentation is an two-dimensional array (HxW) with binary value (0 or 1).
            :returns float perim: The perimeter
            :returns NumpyArray border: The matrix containing the edges of the segmentation.
            :rtype: (float,NumpyArray)
        """
        filter_ = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        derivative = convolve2d(segmentation,filter_, mode="same")
        border = derivative>0
        segmentation_skeleton = skeletonize(np.ascontiguousarray(border.T)).T
        _,perim,_,_,_ = compute_tortuosity(segmentation_skeleton)
        return perim,segmentation_skeleton.T
    
    def compute_branching_angles(self,segmentation_skeleton):
        """This function compute the mean, std and median branching angle of the segmentation.

            :param array segmentation_skeleton: The skeleton of the segmentation is an two-dimensional array (HxW) with binary value (0 or 1).
            :returns float mean_ba: The mean branching angles
            :returns float std_ba: The std branching angles
            :returns float median_ba: The median branching angles
            :returns dictionary angle_dico: A dictionnary containing all the branching angles
            :returns NumpyArray centroid: A numpy array containing a visualisation of the computed centroid

            :rtype: (int,int)
        """
        #img_tmp = np.zeros((segmentation_skeleton.shape[0]+20,segmentation_skeleton.shape[0]+20)) #np.zeros((segmentation_skeleton.shape[0]+20,segmentation_skeleton.shape[0]+20)) 
        #img_tmp[10:-10,10:-10] = segmentation_skeleton

        img= segmentation_skeleton
        #return compute_angles_dictionary(img)
        angle_dico,centroid = compute_angles_dictionary(img)
        mean_ba = np.mean(list(angle_dico.values()))
        std_ba = np.std(list(angle_dico.values()))
        median_ba = np.median(list(angle_dico.values()))
        return mean_ba, std_ba, median_ba,angle_dico,centroid
