import numpy as np
from scipy.ndimage import convolve


def finite_difference(a):
    """
    calculates the gradient of a via finite differences
    @param a:, an array H x W x n with n being different channel intensity values, or H x W
    @return: array H x W x 2, with partial derivative with respect to x and y in the last dimension
    """
    wx = np.array([[1., 0., -1.]])
    wy = wx.T

    if a.ndim == 2:
        # only a single color channel. go ahead
        gx = convolve(a, wx)
        gy = convolve(a, wy)
        return np.stack([gx, gy], axis=-1)
    elif a.ndim == 3:
        # the last dimension is the channel dimension.
        # We calculate the gradient of each channel independently, then add them together.
        C = a.shape[-1]
        grad = finite_difference(a[:, :, 0])
        for c in range(1, C):
            grad += finite_difference(a[:, :, c])
        return grad

if __name__ == "__main__":
    img = np.zeros((5,5))
    img[:, 2] = 1
    print(finite_difference(img))
