import skimage.transform
import numpy as np

def sample(array, coordinates):
    """
    samples the array at given coordinates
    @param array: image array of shape H x W x n or H x W
    @param coordinates: array of shape H x W x 2
    @return:
    """

    # reshape coordinate for skimage
    coordinates = np.transpose(coordinates, axes=[2, 1, 0])

    if array.ndim == 2:
        # only a single color channel. go ahead
        return skimage.transform.warp(array, coordinates, mode='edge')
    elif array.ndim == 3:
        # the last dimension is the channel dimension. We need to sample each channel independently.
        C = array.shape[-1]
        samples_channels = []
        for c in range(C):
            samples_channels.append(skimage.transform.warp(array[:, :, c], coordinates, mode='edge'))
        return np.stack(samples_channels, axis=-1)