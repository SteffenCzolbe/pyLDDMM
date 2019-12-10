import numpy as np
from pyLDDMM.utils import sampler, grid
from pyLDDMM.utils.grad import finite_difference
from pyLDDMM.regularizer import BiharmonicReguarizer

class LDDMM2D(object):
    """
    2d LDDMM registration
    """

    def register(self, I0, I1, T=32, K=200, sigma=1, alpha=1, gamma = 1, epsilon=0.01):
        """
        registers two images, I0 and I1
        @param I0: image, ndarray of dimension H x W x n
        @param I1: image, ndarray of dimension H x W x n
        @param T: int, simulated discrete time steps
        @param K: int, maximum iterations
        @param sigma: float, sigma for L2 loss. lower values strengthen the L2 loss
        @param alpha: float, smoothness regularization. Higher values regularize stronger
        @param gamma: float, norm penalty. Positive value to ensure injectivity of the regularizer
        @param epsilon: float, learning rate
        @return:
        """

        # set up variables
        self.T = T
        self.K = K
        self.H, self.W = I0.shape[:2]
        self.regularizer = BiharmonicReguarizer(alpha, gamma)
        energies = []

        # define vector fields
        v = np.zeros((T, self.H, self.W, 2))
        dv = np.copy(v)

        # (12): iteration over k
        for k in range(K):

            # (1): Calculate new estimate of velocity
            v -= epsilon * dv

            # (2): Reparameterize
            if k % 10 == 9:
                v = self.reparameterize(v)

            # (3): calculate backward flows
            Phi1 = self.integrate_backward_flow(v)

            # (4): calculate forward flows
            Phi0 = self.integrate_forward_flow(v)

            # (5): push-forward I0
            J0 = self.push_forward(I0, Phi0)

            # (6): pull back I1
            J1 = self.pull_back(I1, Phi1)

            # (7): Calculate image gradient
            dJ0 = self.image_grad(J0)

            # (8): Calculate Jacobian determinant of the transformation
            detPhi1 = self.jacobian_derterminant(Phi1)

            # (9): Calculate the gradient
            for t in range(0, self.T):
                dv[t] = 2*v[t] - self.regularizer.K(2 / sigma**2 * detPhi1[t][:, :, np.newaxis] * dJ0[t] * (J0[t] - J1[t])[:, :, np.newaxis])

            # (10) calculate nor of the gradient, stop if small
            dv_norm = np.linalg.norm(dv)
            if dv_norm < 0.001:
                print(dv_norm)
                print("gradient norm below threshold. stopping")
                break

            # (11): calculate new energy
            E = np.sum([np.linalg.norm(self.regularizer.L(v[t])) for t in range(T)]) \
                + 1 / sigma**2 * np.sum((J0[-1] - I1)**2)
            energies.append(E)

            # (12): iterate k = k+1
            print("iteration {:3d}, energy {:4.2f}".format(k, E))
            # end of for loop block

        # (13): Denote the final velocity field as \hat{v}
        v_hat = v

        # (14): Calculate the length of the path on the manifold
        length = np.sum([np.linalg.norm(self.regularizer.L(v_hat[t])) for t in range(T)])

        return J0[-1], v_hat, energies, length, Phi0, Phi1, J0, J1



    def reparameterize(self, v):
        """
        implements step (2): reparameterization
        @param v:
        @return:
        """
        length = np.sum([np.linalg.norm(self.regularizer.L(v[t])) for t in range(self.T)])
        for t in range(self.T):
            v[t] = length / self.T * v[t] / np.linalg.norm(self.regularizer.L(v[t]))
        return v

    def integrate_backward_flow(self, v):
        """
        implements step (3): Calculation of backward flows
        @return:
        """
        # make identity grid
        x = grid.coordinate_grid((self.H, self.W))

        # create flow
        Phi1 = np.zeros((self.T, self.H, self.W, 2))

        # Phi1_1 is the identity mapping
        Phi1[self.T - 1] = x

        for t in range(self.T-2, -1, -1):
            alpha = self.backwards_alpha(v[t], x)
            Phi1[t] = sampler.sample(Phi1[t + 1], x + alpha)

        return Phi1

    def backwards_alpha(self, v_t, x):
        """
        helper function for step (3): Calculation of backward flows
        @param v_t: the velocity field
        @param x: coordinates
        @return:
        """
        alpha = np.zeros(v_t.shape)
        for i in range(5):
            alpha = sampler.sample(v_t, x + 0.5 * alpha)
        return alpha

    def integrate_forward_flow(self, v):
        """
        implements step (4): Calculation of forward flows
        @return:
        """
        # make identity grid
        x = grid.coordinate_grid((self.H, self.W))

        # create flow
        Phi0 = np.zeros((self.T, self.H, self.W, 2))

        # Phi0_0 is the identity mapping
        Phi0[0] = x

        for t in range(0, self.T-1):
            alpha = self.forward_alpha(v[t], x)
            Phi0[t+1] = sampler.sample(Phi0[t], x - alpha)

        return Phi0

    def forward_alpha(self, v_t, x):
        """
        helper function for step (4): Calculation of forward flows
        @param v_t: the velocity field
        @param x: coordinates
        @return:
        """
        alpha = np.zeros(v_t.shape)
        for i in range(5):
            alpha = sampler.sample(v_t, x - 0.5 * alpha)
        return alpha

    def push_forward(self, I0, Phi0):
        """
        implements step (5): push forward image I0 along flow Phi0
        @param I0: image
        @param Phi0: flow
        @return: sequence of forward pushed images J0
        """
        J0 = np.zeros((self.T,) + I0.shape)

        for t in range(0, self.T):
            J0[t] = sampler.sample(I0, Phi0[t])

        return J0

    def pull_back(self, I1, Phi1):
        """
        implements step (6): pull back image I1 along flow Phi1
        @param I1: image
        @param Phi1: flow
        @return: sequence of back-pulled images J1
        """
        J1 = np.zeros((self.T,) + I1.shape)

        for t in range(self.T-1, -1, -1):
            J1[t] = sampler.sample(I1, Phi1[t])

        return J1

    def image_grad(self, J0):
        """
        implements step (7): Calculate image gradient
        @param J0: sequence of forward pushed images J0
        @return: dJ0: gradients of J0
        """
        dJ0 = np.zeros(J0.shape + (2,))

        for t in range(self.T):
            dJ0[t] = finite_difference(J0[t])

        return dJ0

    def jacobian_derterminant(self, Phi1):
        """
        implements step (8): Calculate Jacobian determinant of the transformation
        @param Phi1: sequence of transformations
        @return: detPhi1: sequence of determinants of J0
        """
        detPhi1 = np.zeros((self.T, self.H, self.W))

        for t in range(self.T):
            # get gradient in x-direction
            dx = finite_difference(Phi1[t, :, :, 0])
            # gradient in y-direction
            dy = finite_difference(Phi1[t, :, :, 1])

            # calculate determinants
            detPhi1[t] = dx[:, :, 0] * dy[:, :, 1] - dx[:, :, 1] * dy[:, :, 0]

        # check injectivity: a function has a differentiable inverse iff the determinant of it's jacobian is positive
        assert detPhi1.min() > 0, "Injectivity violated. Stopping. Try lowering the learning rate (epsilon)."

        return detPhi1



