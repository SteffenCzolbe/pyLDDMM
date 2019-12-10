import numpy as np
from scipy.ndimage import convolve

class BiharmonicReguarizer:
    def __init__(self, alpha, gamma):
        """
        instantiates the Biharmonic regularizer
        @param alpha: smoothness penalty. Positive number. The higher, the more smoothness will be enforced
        @param gamma: norm penalty. Positive value, so that the operator is non-singular
        """
        self.alpha = alpha
        self.gamma = gamma
        self.A = None

    def L(self, f):
        """
        The Cauchy-Navier operator (Equation 17)
        @param f: an array representing function f, array of dim H x W x 2
        @return: g = L(f), array of dim H x W x 2
        """
        w = np.array([[0.,  1.,  0.],
                      [1., -4.,  1.],
                      [0.,  1.,  0.]])
        dxx = convolve(f[:, :, 0], w)
        dyy = convolve(f[:, :, 1], w)

        dff = np.stack([dxx, dyy], axis= -1)
        g = - self.alpha * dff + self.gamma * f

        return g

    def K(self, g):
        """
        The K = (LL)^-1 operator.
        @param g: an array representing function g, array of dim H x W x 2
        @return: f = K(g) = (LL)^-1 (g), array of dim H x W x 2
        """
        if self.A is None or self.A.shape != g.shape[:-1]:
            # A is not chached. compute A.
            self.A = self.compute_A(g.shape)

        # transform to fourier domain
        G = self.fft2(g)

        # perform operation in fourier space
        F = G / self.A**2

        # transform back to normal domain
        f = self.ifft2(F)
        return f

    def compute_A(self, shape):
        """
        computes the A(k) operator
        @param shape: shape of the input image
        @return: A(k)
        """
        H, W, d = shape
        A = np.zeros((H, W))

        for h in range(H):
            for w in range(W):
                A[h, w] += 2 * self.alpha * ((1 - np.cos(2 * np.pi * h / H)) + (1 - np.cos(2 * np.pi * w / W)))

        A += self.gamma

        # expand dims to match G
        A = np.stack([A, A], axis=-1)
        return A

    def fft2(self, a):
        """
        performs 2d FFT along the first 2 axis of a 3d array
        """
        C = a.shape[2]
        A = np.zeros(a.shape, dtype=np.complex128)
        for c in range(C):
            A[:, :, c] = np.fft.fft2(a[:, :, c])
        return A

    def ifft2(self, A):
        """
        performs 2d iFFT along the first 2 axis of a 3d array
        """
        C = A.shape[2]
        a = np.zeros(A.shape, dtype=np.complex128)
        for c in range(C):
            a[:, :, c] = np.fft.ifft2(A[:, :, c])
        return np.real(a)

if __name__ == '__main__':
    v = np.zeros((5, 5, 2))
    v[2, 2, 0] = 1

    reg = BiharmonicReguarizer(alpha=1, gamma=1)

    print(reg.K(v)[:, :, 0])
