import scipy
import skimage.io as skio
import skimage.transform as sktransform
import numpy as np


class MultifractalVBMs:
    """
    A class that can perform multifractal biomarker computation for a fundus vasculature segmentation.


    :param n_dim: Maximum dimension used to calculate Singularity Length.
    :type n_dim: int

    :param n_rotations: Number of rotations performed on the given segmentation from which the optimal run will be chosen.
    :type n_rotations: int

    :param optimize: A boolean value that specify if the computation will be made using several rotated version of the segmentation.
    :type optimize: bool

    :param min_proba: A minimum probability for the occupancy of calculated boxes (boxes with smaller probability will be ignored).
    :type min_proba: float

    :param max_proba: A maximum probability for the occupancy of calculated boxes (boxes with higher probability will be ignored).
    :type max_proba: float
    """
    def __init__(self, n_dim=10, n_rotations=25, optimize=True, min_proba=0.01, maxproba=0.98):
        """
        Constructor

        
        :param n_dim: Maximum dimension used to calculate Singularity Length.
        :type n_dim: int

        :param n_rotations: Number of rotations performed on the given segmentation from which the optimal run will be chosen.
        :type n_rotations: int

        :param optimize: A boolean value that specify if the computation will be made using several rotated version of the segmentation.
        :type optimize: bool

        :param min_proba: A minimum probability for the occupancy of calculated boxes (boxes with smaller probability will be ignored).
        :type min_proba: float

        :param max_proba: A maximum probability for the occupancy of calculated boxes (boxes with higher probability will be ignored).
        :type max_proba: float

        """
        self.n_dim = n_dim
        self.n_rotations = n_rotations
        self.optimize = optimize
        self.box_occupancy_proba = [min_proba, maxproba]

    def compute_multifractals(self, segmentation):
        '''
        Computes the multifractal biomarkers of a given retinal vasculature segmentation, specifically the dimensions D0, D1, D2, and the singularity length.

        :param segmentation: An (N,N) binary numpy array representing the segmentation to be analyzed.
        :type segmentation: np.ndarray
        :return: A numpy array containing the computed biomarkers [D0, D1, D2, Singularity Length].
        :rtype: np.ndarray
        '''

        rotations = self.n_rotations
        if not self.optimize:
            rotations = 1

        image_stats = self.get_q_fractals(segmentation, rotations=rotations, dim_list=[0, 1, 2, -self.n_dim, self.n_dim])
        first_three_dqs = image_stats[:, :3, 0]

        if self.optimize:
            best_rotation_index = self._optimize_dqs(first_three_dqs)
            image_stats_opt = image_stats[best_rotation_index, :, :]
        else:
            image_stats_opt = image_stats.squeeze()

        sing_features = self.get_singularity_length(alpha=image_stats_opt[-2:, 3])
        three_dqs = image_stats_opt[:3, 0]
        return np.hstack((three_dqs, sing_features))

    def get_singularity_length(self, alpha):
        '''
         This method is used to compute singularity length from f(alpha) singularity curve.

        :param alpha: 1D numpy array containing f(alpha) singularity curve values.
        :type alpha: int
        :return: Singularity length of this f(alpha) curve.
        :rtype: float
        '''
        s_len = alpha.max() - alpha.min()
        return s_len

    def get_multi_fractal_dimension(self, segmentation, q):
        '''
        Calculates Q dimension from the given binary segmentation.

        :param segmentation: (N,N) binary numpy array segmentation.
        :type segmentation: np.ndarray
        :param q: q power of multi-fractal D_q which will be calculated from the segmentation.
        :type q: float
        :return: 4-tuple of Multi-fractal calculations:
            * Dq_slope - D_q value for given q.
            * dq_r2 - R^2 of the fit of D_q.
            * fq_slope - f(q) value for given q.
            * alphaq_slope - \alpha(q) value for given q.
        '''
        # Only for 2d binary segmentation
        assert len(segmentation.shape) == 2
        assert (segmentation.max() == 1) and (segmentation.min() == 0)

        # Minimal dimension of segmentation
        p = min(segmentation.shape)
        # create a linear sampling for epsilons
        epsilons = np.linspace(5, int(0.6 * p), 20).astype(np.int64)

        counts = []
        fq_numerators = []
        alpha_q_numerators = []
        for size in epsilons:
            P = self._probability_per_pixel(segmentation, size, occupancy=self.box_occupancy_proba)
            # If all pixels were filtered, try again with no limit on pixel occupancy
            if np.sum(P) == 0:
                P = self._probability_per_pixel(segmentation, size, occupancy=[0.001, 1])

            assert np.linalg.norm(np.sum(P) - 1) <= 0.000001, 'All probabilities must sum to 1'
            p_positive = P[P > 0]
            I = np.sum(p_positive ** q)

            mu = (p_positive ** q) / I
            fq_numerator = np.sum(mu * np.log(mu))
            alpha_q_numerator = np.sum(mu * np.log(p_positive))

            if q == 1:
                dq = - np.sum(p_positive * np.log(p_positive))
            else:
                dq = np.log(I) / (1 - q)

            counts.append(dq)
            fq_numerators.append(fq_numerator)
            alpha_q_numerators.append(alpha_q_numerator)

        # Fit D(q) - the successive log(sizes) with log (counts)
        dq_slope, _, r_value, _, _ = scipy.stats.linregress(-np.log(epsilons), np.array(counts))
        dq_r2 = r_value ** 2
        # Fit f(q)
        fq_slope, _, _, _, _ = scipy.stats.linregress(np.log(epsilons), np.array(fq_numerators))
        # Fit alpha(q)
        alphaq_slope, _, _, _, _ = scipy.stats.linregress(np.log(epsilons), np.array(alpha_q_numerators))

        return dq_slope, dq_r2, fq_slope, alphaq_slope

    def get_q_fractals(self, segmentation, rotations, dim_list):
        '''
        Calculates all multi-fractal dimensions specified in dim_list for rotation number of times.

        :param : (N,N) binary numpy array segmentation.
        :type segmentation: np.ndarray
        :param rotations: number of rotations performed on the given segmentation from which the optimal run will be chosen.
        :type rotations: int
        :param dim_list: Iterable of dimensions to calculate from segmentation.
        :type dim_list: list
        :return: The output of 'get_multi_fractal_dimension' for every rotation and q dimension.
        :rtype: np.ndarray with size (rotations, len(dim_list), 4).
        '''
        angles = np.linspace(0, 360, rotations)

        dqs = []
        for angle in angles:
            print("Angle {}".format(angle))
            rotated_image = sktransform.rotate(segmentation, angle, resize=True, cval=0, mode='constant')
            dq = []
            for q in dim_list:
                print("q {}".format(q))
                dq_value = self.get_multi_fractal_dimension(rotated_image, q)
                dq.append(dq_value)
                print(dq_value)

            dqs.append(dq)

        dqs = np.array(dqs)

        return dqs

    @staticmethod
    def custom_add_reduceat(segmentation, idx, axis=0):
        '''
        Custom numpy.add.reduceat function, twice as fast in the context of binary segmentations.
        Sum a 2d-segmentation along a given axis, in the intervals specified by 'idx'.

        :param segmentation: (N,N) numpy array.
        :type segmentation: np.ndarray
        :param idx: indices along the axis between which the function should sum the values.
        :type idx: np.ndarray
        :param axis: Specifies the dimension of im on which to operate. Must be {0, 1}.
        :type axis: int
        :return: Aggregated segmentation along the given axis.
        :rtype: np.ndarray with size (idx.shape[0], N) for axis=0 or (N, idx.shape[0]) for axis=1.
        '''

        idxs_ext = np.concatenate((idx, np.array([segmentation.shape[axis]])))
        if axis == 0:
            results = np.empty(shape=(idx.size, segmentation.shape[1]))
            for i in range(idx.size):
                results[i, :] = segmentation[idxs_ext[i]:idxs_ext[i + 1], :].sum(axis=axis)
        elif axis == 1:
            results = np.empty(shape=(segmentation.shape[0], idx.size))
            for j in range(idx.size):
                results[:, j] = segmentation[:, idxs_ext[j]:idxs_ext[j + 1]].sum(axis=axis)
        else:
            raise Exception("Axis must be 0 or 1.")

        return results

    def _probability_per_pixel(self, segmentation, k, occupancy):
        '''
        Creates a grid with size k on the segmentation and calculates the occupancy probability for every such box.
        Filters the boxes to be between occupancy[0] <= pixel <= occupancy[1]

        :param segmentation: (N,N) numpy binary array.
        :type segmentation: np.ndarray
        :param k: Grid box size.
        :type k: int
        :param occupancy: a list containing the [min_proba, max_proba]. Boxes with probability smaller than min_proba and larger than max_proba will be ignored.
        :type occupancy: list[float]
        :returns: A probability grid calculated from segmentation with box size k.
        :rtype: np.ndarray with size (math.ceil(N/k), math.ceil(N/k)).
        '''
        # split segmentation to boxes sized k by two steps: first sum along vertical axis and then along horizontal.
        vertical_sum = self.custom_add_reduceat(segmentation, np.arange(0, segmentation.shape[0], k), axis=0)
        horizontal_sum = self.custom_add_reduceat(vertical_sum, np.arange(0, segmentation.shape[1], k), axis=1)
        M = horizontal_sum

        # calculate probability per box
        p = M / (k ** 2)
        # filter by box occupancy
        condition = (p >= occupancy[0]) & (p <= occupancy[1])
        P = M[condition] / np.sum(M[condition])
        return P

    def _optimize_dqs(self, dqs):
        '''
        Finds the best sampling index among all rotations: 
            a sample which satisfies D0>D1>D2 where D0 is the largest.
            if no sampling satisfies D0>D1>D2 returns the sampling index with largest D0.

        :param dqs: ndarray with dqs.shape = (rand_rotations, 3) with the values of D0,D1,D2 for every rotation.
        :type dqs: np.ndarray
        :returns: Finds the best sampling index among all rotations.
        :rtype: int
        '''

        index1 = (dqs[:, 0] - dqs[:, 1]) > 0
        index2 = (dqs[:, 1] - dqs[:, 2]) > 0

        comb_ind = index1 * index2
        full_indx_arr = np.arange(0, dqs.shape[0], 1)
        dqs_subset = dqs[comb_ind, 0]
        subset_index_arr = full_indx_arr[comb_ind]
        if subset_index_arr.size != 0:
            subset_max_index = np.argmax(dqs_subset)
            global_max_index = subset_index_arr[subset_max_index]
        else:
            global_max_index = np.argmax(dqs[:, 0])

        return global_max_index

if __name__ == "__main__":
    import numpy as np
    from skimage.morphology import skeletonize
    from PIL import Image
    import sys
    from PVBM.FractalAnalysis import MultifractalVBMs
    from PVBM.GeometryAnalysis import GeometricalVBMs

    sys.setrecursionlimit(100000)

    center = (645,822)
    radius = 181

    blood_vessel_segmentation_path = '/Users/jonathanfhima/Desktop/Lirot2025/Model2App/LirotAnalysis/testanalysis_Lirotai/segmentation/DRISHTI-GS1-test-3.png'
    segmentation = np.array(Image.open(blood_vessel_segmentation_path)) / 255  # Open the segmentation
    segmentation = segmentation[:,:,2]
    skeleton = skeletonize(segmentation) * 1
    vbms = GeometricalVBMs()  # Instanciate a geometrical VBM object

    roi = '/Users/jonathanfhima/Desktop/Lirot2025/Model2App/LirotAnalysis/testanalysis_Lirotai/ROI/DRISHTI-GS1-test-3.png'
    roi = np.array(Image.open(roi))

    zones_ABC = '/Users/jonathanfhima/Desktop/Lirot2025/Model2App/LirotAnalysis/testanalysis_Lirotai/zones_ABC/DRISHTI-GS1-test-3.png'
    zones_ABC = np.array(Image.open(zones_ABC))

    segmentation, skeleton = vbms.apply_roi(
        segmentation=segmentation,
        skeleton=skeleton,
        zones_ABC=zones_ABC,
        roi = roi,
    )

    vbms, visual = vbms.compute_geomVBMs(
        blood_vessel=segmentation,
        skeleton=skeleton,
        xc=center[0],
        yc=center[1],
        radius=radius
    )

    fractalVBMs = MultifractalVBMs(n_rotations=25, optimize=True, min_proba=0.0001, maxproba=0.9999)
    D0, D1, D2, SL = fractalVBMs.compute_multifractals(segmentation.copy())
    1