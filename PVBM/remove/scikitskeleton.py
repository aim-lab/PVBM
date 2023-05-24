import numpy as np
import os
def _fast_skeletonize(image):
    image = np.array(image)
    print(image.max())
    lut = [0, 0, 0, 1, 0, 0, 1, 3, 0, 0, 3, 1, 1, 0, 1, 3, 0, 0, 0, 0, 0, 0,
       0, 0, 2, 0, 2, 0, 3, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 2, 2, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0,
       0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 2, 0, 0, 0, 3, 1,
       0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 1, 3, 0, 0,
       1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 2, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3,
       0, 1, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0]

    # indices for fast iteration
    nrows = image.shape[0]+2
    ncols = image.shape[1]+2

    # we copy over the image into a larger version with a single pixel border
    # this removes the need to handle border cases below
    _skeleton = np.zeros((nrows, ncols), dtype=np.uint8)
    _skeleton[1:nrows-1, 1:ncols-1] = image > 0

    _cleaned_skeleton = _skeleton.copy()

    skeleton = _skeleton
    cleaned_skeleton = _cleaned_skeleton

    pixel_removed = True

    # the algorithm reiterates the thinning till
    # no further thinning occurred (variable pixel_removed set)
    
    while pixel_removed:
        pixel_removed = False

        # there are two phases, in the first phase, pixels labeled (see below)
        # 1 and 3 are removed, in the second 2 and 3

        # nogil can't iterate through `(True, False)` because it is a Python
        # tuple. Use the fact that 0 is Falsy, and 1 is truthy in C
        # for the iteration instead.
        # for first_pass in (True, False):
        for pass_num in range(2):
            first_pass = (pass_num == 0)
            for row in range(1, nrows-1):
                for col in range(1, ncols-1):
                    # all set pixels ...
                    if skeleton[row, col]:
                        # are correlated with a kernel (coefficients spread around here ...)
                        # to apply a unique number to every possible neighborhood ...

                        # which is used with the lut to find the "connectivity type"

                        neighbors = lut[  1*skeleton[row - 1, col - 1] +   2*skeleton[row - 1, col] +\
                                          4*skeleton[row - 1, col + 1] +   8*skeleton[row, col + 1] +\
                                          16*skeleton[row + 1, col + 1] +  32*skeleton[row + 1, col] +\
                                          64*skeleton[row + 1, col - 1] + 128*skeleton[row, col - 1]]

                        # if the condition is met, the pixel is removed (unset)
                        if ((neighbors == 1 and first_pass) or
                                (neighbors == 2 and not first_pass) or
                                (neighbors == 3)):
                            cleaned_skeleton[row, col] = 0
                            pixel_removed = True

            # once a step has been processed, the original skeleton
            # is overwritten with the cleaned version
            skeleton[:, :] = cleaned_skeleton[:, :]
    out = _skeleton[1:nrows-1, 1:ncols-1].astype(np.uint8).T.reshape(-1) * 255
    
    return out

