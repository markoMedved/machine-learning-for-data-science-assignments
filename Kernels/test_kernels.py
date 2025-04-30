import unittest
from unittest.mock import patch
import numpy as np

from hw_kernels import SVR, RBF, Polynomial, \
    KernelizedRidgeRegression


class Linear:
    """An example of a kernel."""

    def __init__(self):
        # here a kernel could set its parameters
        pass

    def __call__(self, A, B):
        """Can be called with vectors or matrices, see the
        comment for test_kernel"""
        return A.dot(B.T)


class KernelTests(unittest.TestCase):

    def setUp(self):
        self.X = np.array([[0, 0],
                           [0, 1],
                           [1, 0],
                           [1, 1]])
        self.y = np.array([0, 0, 1, 1])

    def test_linear(self):
        fitter = KernelizedRidgeRegression(kernel=Linear(), lambda_=0.0001)
        m = fitter.fit(self.X, self.y)
        pred = m.predict(self.X)
        np.testing.assert_almost_equal(pred, self.y, decimal=3)

    def test_rbf(self):
        fitter = KernelizedRidgeRegression(kernel=RBF(sigma=0.5), lambda_=0.0001)
        m = fitter.fit(self.X, self.y)
        pred = m.predict(self.X)
        np.testing.assert_almost_equal(pred, self.y, decimal=3)

    def test_polynomial(self):
        fitter = KernelizedRidgeRegression(kernel=Polynomial(M=2), lambda_=0.0001)
        m = fitter.fit(self.X, self.y)
        pred = m.predict(self.X)
        np.testing.assert_almost_equal(pred, self.y, decimal=3)

    def test_kernel(self):
        """
        Kernel classes should work for vectors (single
        instances; 1D numpy arrays) and matrices (multiple instances; 2D numpy arrays).

        If inputs are:
        - 1D, 1D (two vectors), the result is a number k(x_i,x_j)
        - 1D, 2D (vector and a matrix), the result is a 1D numpy array
        - 2D, 2D the result is a 2D numpy array

        Kernels implemented like this are easy to use.

        All kernels should be implemented without any Python looping.
        Implementing RBF without loops is a bit tricky, because it needs pairwise
        distances. Hint: for vectors a, b you can compute the distance with
        (a-b)^2 = a.dot(a) - 2*a.dot(b) + b.dot(b). This is, with some care,
        vectorizable.
        """
        for kernel in [Linear(), Polynomial(M=3), RBF(sigma=0.2)]:
            number = kernel(self.X[0], self.X[1]) # k
            float(number)  # should not crash
            a1d = kernel(self.X[0], self.X)  # a vector of k
            self.assertTrue(len(a1d.shape) == 1)
            a1d = kernel(self.X, self.X[0])
            self.assertTrue(len(a1d.shape) == 1)
            a2d = kernel(self.X, self.X)  # the K matrix
            self.assertTrue(len(a2d.shape) == 2)


def qp_callargs(call):
    kwargs = call.kwargs.copy()
    names = ["P", "q", "G", "h", "A", "b"]
    for n, v in zip(names, call.args):
        kwargs[n] = v
    return kwargs


class SVRTests(unittest.TestCase):

    def setUp(self):
        self.X = np.array([[0., 0],
                           [0, 1],
                           [1, 0],
                           [1, 1]])
        self.y = np.array([0., 1, 2, 3])

    def test_linear(self):
        fitter = SVR(kernel=Linear(), lambda_=0.0001, epsilon=0.1)
        m = fitter.fit(self.X, self.y)
        pred = m.predict(self.X)
        np.testing.assert_allclose(pred, self.y, atol=0.11)

    def test_rbf(self):
        fitter = SVR(kernel=RBF(sigma=0.5), lambda_=0.0001, epsilon=0.1)
        m = fitter.fit(self.X, self.y)
        pred = m.predict(self.X)
        np.testing.assert_allclose(pred, self.y, atol=0.11)

    def test_polynomial(self):
        fitter = SVR(kernel=Polynomial(M=2), lambda_=0.0001, epsilon=0.1)
        m = fitter.fit(self.X, self.y)
        pred = m.predict(self.X)
        np.testing.assert_allclose(pred, self.y, atol=0.11)

    def test_predictor_get_info(self):
        fitter = SVR(kernel=Polynomial(M=2), lambda_=0.0001, epsilon=0.1)
        m = fitter.fit(self.X, self.y)

        alpha = m.get_alpha()
        np.testing.assert_equal(alpha.shape, (4, 2))  # two alpha for each sample
        # one value in row should be much bigger than the other (the other should be zero)
        np.testing.assert_allclose(alpha.sum(axis=1), alpha.max(axis=1), rtol=1e-5)

        b = m.get_b()
        float(b)

    @patch("cvxopt.solvers.qp")
    def test_enforce_quadratic_programming_order_of_x(self, qp):
        fitter = SVR(kernel=Linear(), lambda_=0.0001, epsilon=0)
        try:
            _ = fitter.fit(self.X, self.y)
        except:
            pass
        usedq = np.array(qp_callargs(qp.call_args)["q"]).flatten()
        np.testing.assert_equal(usedq, [0,  0, -1, 1, -2, 2, -3, 3])

    def test_kernel(self):
        """
        Kernel classes should work for vectors (single
        instances; 1D numpy arrays) and matrices (multiple instances; 2D numpy arrays).

        If inputs are:
        - 1D, 1D (two vectors), the result is a number k(x_i,x_j)
        - 1D, 2D (vector and a matrix), the result is a 1D numpy array
        - 2D, 2D the result is a 2D numpy array

        Kernels implemented like this are easy to use.

        All kernels should be implemented without any Python looping.
        Implementing RBF without loops is a bit tricky, because it needs pairwise
        distances. Hint: for vectors a, b you can compute the distance with
        (a-b)^2 = a.dot(a) - 2*a.dot(b) - b.dot(b). This is, with some care,
        vectorizable.
        """
        for kernel in [Linear(), Polynomial(M=3), RBF(sigma=0.2)]:
            number = kernel(self.X[0], self.X[1]) # k
            float(number)  # should not crash
            a1d = kernel(self.X[0], self.X)  # a vector of k
            self.assertTrue(len(a1d.shape) == 1)
            a1d = kernel(self.X, self.X[0])
            self.assertTrue(len(a1d.shape) == 1)
            a2d = kernel(self.X, self.X)  # the K matrix
            self.assertTrue(len(a2d.shape) == 2)


if __name__ == "__main__":
    unittest.main()
