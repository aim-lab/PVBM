import scipy
import skimage.io as skio
import skimage.transform as sktransform
import numpy as np


class MultifractalVBMs:
    """A class which holds all required parameters and all the methods to perform
        multifractal calculation on binary images.

    :param int n_dim: maximum dimension used to calculate Singularity Length.
    :param int n_rotations: number of rotations performed on the given image from which the optimal run will be chosen.
    :param bool optimize: if True - an optimal run will be chosen from n_rotations runs, otherwise only 1 angle will be used.
    :param float min_proba: a minimum probability for the occupancy of calculated boxes.
                            Boxes with smaller probability will be ignored.
    :param float max_proba: a maximum probability for the occupancy of calculated boxes.
                            Boxes with higher probability will be ignored.
    Example:
        tar = 'example_arteries_586_LES_AV.png'
        sample_arteries_image = skio.imread(tar)
        f_multi = FundusMultifractal()
        out1 = f_multi.get_fundus_biomarkers(sample_arteries_image)
        print(out1)

        tar = 'sierpinski.png'
        example_monofractal_im = skio.imread(tar) / 255
        f_multi = FundusMultifractal(n_rotations=10)
        out2 = f_multi.get_fundus_biomarkers(example_monofractal_im)
        print(out2)

        tar = 'binhen.tif'
        example_multifractal_im = skio.imread(tar)/255
        f_multi = FundusMultifractal(n_rotations=4)
        out3 = f_multi.get_fundus_biomarkers(example_multifractal_im)
        print(out3)
    """
    def __init__(self, n_dim=10, n_rotations=25, optimize=True, min_proba=0.01, maxproba=0.98):
        """Constractor"""
        self.n_dim = n_dim
        self.n_rotations = n_rotations
        self.optimize = optimize
        self.box_occupancy_proba = [min_proba, maxproba]

    def get_fundus_biomarkers(self, image):
        '''
            This method will be used to calculate the following dimensions: (D0, D1, D2, singularity length).

            :param int image: (N,N) binary numpy array image.
            :returns: A numpy array containing the [D0, D1, D2, Singularity Length] features.
            :rtype: ndarray
        '''
        rotations = self.n_rotations
        if not self.optimize:
            rotations = 1

        image_stats = self.get_q_fractals(image, rotations=rotations, dim_list=[0, 1, 2, -self.n_dim, self.n_dim])
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
            Calculates singularity length from f(alpha) singularity curve.

            :param int alpha: 1D numpy array containing f(alpha) singularity curve values.
            :returns: Singularity length of this f(alpha) curve.
            :rtype: float
        '''
        s_len = alpha.max() - alpha.min()
        return s_len

    def get_multi_fractal_dimension(self, image, q):
        '''
            Calculates Q dimension from the given binary image.

            :param ndarray image: (N,N) binary numpy array image.
            :param float q: q power of multi-fractal D_q which will be calculated from image.
            :returns: 4-tuple of Multi-fractal calculations:
                * Dq_slope - D_q value for given q.
                * dq_r2 - R^2 of the fit of D_q.
                * fq_slope - f(q) value for given q.
                * alphaq_slope - \alpha(q) value for given q.
        '''
        # Only for 2d binary image
        assert len(image.shape) == 2
        assert (image.max() == 1) and (image.min() == 0)

        # Minimal dimension of image
        p = min(image.shape)
        # create a linear sampling for epsilons
        epsilons = np.linspace(5, int(0.6 * p), 20).astype(np.int64)

        counts = []
        fq_numerators = []
        alpha_q_numerators = []
        for size in epsilons:
            P = self._probability_per_pixel(image, size, occupancy=self.box_occupancy_proba)
            # If all pixels were filtered, try again with no limit on pixel occupancy
            if np.sum(P) == 0:
                P = self._probability_per_pixel(image, size, occupancy=[0.001, 1])

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

    def get_q_fractals(self, image, rotations, dim_list):
        '''
            Calculates all multi-fractal dimensions specified in dim_list for rotation number of times.

            :param ndarray image: (N,N) binary numpy array image.
            :param int rotations: number of rotations performed on the given image from which the optimal run will be chosen.
            :param dim_list: Iterable of dimensions to calculate from image.
            :returns: The output of 'get_multi_fractal_dimension' for every rotation and q dimension.
            :rtype: ndarray with size (rotations, len(dim_list), 4).
        '''
        angles = np.linspace(0, 360, rotations)

        dqs = []
        for angle in angles:
            rotated_image = sktransform.rotate(image, angle, resize=True, cval=0, mode='constant')
            dq = []
            for q in dim_list:
                dq_value = self.get_multi_fractal_dimension(rotated_image, q)
                dq.append(dq_value)

            dqs.append(dq)

        dqs = np.array(dqs)

        return dqs

    @staticmethod
    def custom_add_reduceat(image, idx, axis=0):
        '''
            Custom numpy.add.reduceat function, twice as fast in the context of binary images.
            Sum a 2d-image along a given axis, in the intervals specified by 'idx'.

            :param ndarray image: (N,N) numpy array.
            :param ndarray idx: indices along the axis between which the function should sum the values.
            :param int axis: Specifies the dimension of im on which to operate. Must be {0, 1}.
            :returns: Aggregated image along the given axis.
            :rtype: ndarray with size (idx.shape[0], N) for axis=0 or (N, idx.shape[0]) for axis=1.
        '''

        idxs_ext = np.concatenate((idx, np.array([image.shape[axis]])))
        if axis == 0:
            results = np.empty(shape=(idx.size, image.shape[1]))
            for i in range(idx.size):
                results[i, :] = image[idxs_ext[i]:idxs_ext[i + 1], :].sum(axis=axis)
        elif axis == 1:
            results = np.empty(shape=(image.shape[0], idx.size))
            for j in range(idx.size):
                results[:, j] = image[:, idxs_ext[j]:idxs_ext[j + 1]].sum(axis=axis)
        else:
            raise Exception("Axis must be 0 or 1.")

        return results

    def _probability_per_pixel(self, image, k, occupancy):
        '''
            Creates a grid with size k on the image and calculates the occupancy probability for every such box.
            Filters the boxes to be between occupancy[0] <= pixel <= occupancy[1]

            :param ndarray image: (N,N) numpy binary array.
            :param int k: Grid box size.
            :param list[float] occupancy: a list containing the [min_proba, max_proba]. Boxes with probability
                                          smaller than min_proba and larger than max_proba will be ignored.
            :returns: A probability grid calculated from image with box size k.
            :rtype: ndarray with size (math.ceil(N/k), math.ceil(N/k)).
        '''
        # split image to boxes sized k by two steps: first sum along vertical axis and then along horizontal.
        vertical_sum = self.custom_add_reduceat(image, np.arange(0, image.shape[0], k), axis=0)
        horizontal_sum = self.custom_add_reduceat(vertical_sum, np.arange(0, image.shape[1], k), axis=1)
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

            :param ndarray dqs: ndarray with dqs.shape = (rand_rotations, 3) with the values of D0,D1,D2 for every rotation.
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

