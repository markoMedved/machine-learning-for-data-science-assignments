import numpy as np
from cvxopt import solvers, matrix
import matplotlib.pyplot as plt


class Linear:
    """An example of a kernel."""

    def __init__(self):
        # here a kernel could set its parameters
        pass

    def __call__(self, A, B):
        """Can be called with vectors or matrices, see the
        comment for test_kernel"""
        return A.dot(B.T)


class Polynomial:
    def __init__(self, M=2, c = 1):
        self.M = M
        self.c = c

    def __call__(self, A, B):
        return (A.dot(B.T) + self.c)**self.M
    
          
class RBF:
    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def __call__(self, A, B):
        A_og = A
        B_og = B
        A = np.atleast_2d(A)
        B = np.atleast_2d(B)

        A_norm = np.sum(A**2, axis=1).reshape(-1, 1)  
        B_norm = np.sum(B**2, axis=1).reshape(1, -1)
        dists_sq = A_norm + B_norm - 2 * A @ B.T 

        K =  np.exp(-dists_sq / (2 * self.sigma**2))

        if A_og.ndim == 1 and B_og.ndim == 1:
            return K[0,0]
        
        elif A_og.ndim == 1 or B_og.ndim == 1:
            return K.flatten()

        return K

class KernelizedRidgeRegression:
    def __init__(self, kernel = Linear(), lambda_ = 0.0001):
        self.kernel = kernel
        self.lambda_ = lambda_

    def fit(self, X, y):
        K = self.kernel(X,X)
        n = K.shape[0]
        alphas = np.linalg.inv(K+ self.lambda_ * np.eye(n)) @ y
        return KernelizedRidgeRegressionPredict(alphas, self.kernel, X)


class KernelizedRidgeRegressionPredict:
    def __init__(self, alphas, kernel, X_train):
        self.alphas = alphas
        self.kernel = kernel
        self.X_train = X_train

    def predict(self, X):
        K = self.kernel(X, self.X_train)
        return K @ self.alphas
    

class SVR:
    def __init__(self, kernel=Linear(), lambda_=0.0001, epsilon=0.1):
        self.kernel = kernel
        self.lambda_ = lambda_
        self.epsilon = epsilon

    def fit(self, X, y):
        K = self.kernel(X,X)
        C = 1/(self.lambda_ + 1e-10)
        n = X.shape[0]
        # Get P
        P = np.zeros((2*n, 2*n))
        for i in range(n):
            for j in range(n):
                k_ij = K[i,j]
                P[2*i, 2*j] = k_ij
                P[2*i +1, 2*j +1] = k_ij
                P[2*i +1, 2*j] = -k_ij
                P[2*i, 2*j + 1] = -k_ij

        P = matrix(P)

        # Get q
        q = np.zeros(2*n)
        for i in range(n):
            q[2 * i] = self.epsilon - y[i]
            q[2 * i +1] = self.epsilon + y[i]

        q = matrix(q)

        # Get G
        G_up = -np.eye(2 *n)
        G_bot = np.eye(2 * n)
        G = matrix(np.vstack([G_up, G_bot]))
        h = matrix(np.hstack([np.zeros(2*n), np.ones(2 * n) * C]))

        # Get A
        A = np.zeros((1, 2 * n))
        for i in range(n):
            A[0,2 * i] = 1
            A[0,2*i + 1] = -1
        
        A = matrix(A)
        b = matrix(0.0)

        solvr = solvers.qp(P, q, G, h, A, b)
        x = np.array(solvr['x']).flatten()
        bias = solvr['y'][0]

        alphas = x[0::2]
        alphas_star = x[1::2]
    
        return SVRPredict(alphas, alphas_star, self.kernel, bias, X)
    
class SVRPredict:
    def __init__(self, alphas, alphas_star, kernel,bias,  X_train):
        self.alphas = alphas
        self.alphas_star = alphas_star
        self.kernel = kernel
        self.X_train = X_train
        self.bias = bias

    def get_alpha(self):
        return np.column_stack((self.alphas, self.alphas_star))
    
    def get_b(self):
        return float(self.bias)

    def predict(self, X):
        K = self.kernel(X, self.X_train)
        return K @ (self.alphas - self.alphas_star) + self.bias
    

    
if __name__ == "__main__":
    # Solve the SVR optimization problem
    pass