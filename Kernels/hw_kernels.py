import numpy as np
from cvxopt import solvers, matrix
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

# TODO: cross validation for all evaluations, finish cross validation for picking lmabda
# TODO: part 3


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
    def __init__(self, kernel = Linear(), lambda_ = 0.001):
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

        options = {'maxiters': 100,'show_progress': False}
        solvr = solvers.qp(P, q, G, h, A, b, options=options)
        x = np.array(solvr['x']).flatten()
        bias = solvr['y'][0]

        self.alphas = x[0::2]
        self.alphas_star = x[1::2]

        threshold = 1e-5
    
        return SVRPredict(self.alphas, self.alphas_star, self.kernel, bias, X)
    
    def support_vectors(self, threshold=1e-5):
        C = 1 / (self.lambda_ + 1e-10)
        support_vector_indices = np.where(
            ((self.alphas > threshold) & (self.alphas < C - threshold)) |
            ((self.alphas_star > threshold) & (self.alphas_star < C - threshold))
        )[0]
        return support_vector_indices
    
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
    # sine = pd.read_csv("sine.csv")
    # X = sine["x"].values.reshape(-1,1)
    # y = sine["y"].values

    # # RBF works nicely
    # fitter = SVR(RBF(sigma=1), epsilon=0.1, lambda_=0.1)
    # model = fitter.fit(X, y)
    # pred = model.predict(X)
    # sv_indices = fitter.support_vectors()
    # X_sv = X[sv_indices]
    # y_sv = y[sv_indices]

    # # Visualize data
    # plt.scatter(X[~np.isin(np.arange(len(X)), sv_indices)], 
    #             y[~np.isin(np.arange(len(X)), sv_indices)], 
    #             label="Non-Support Vectors", color='lightblue')
    # plt.scatter(X, pred, label="Predictions", color='orange')

    # plt.scatter(X[sv_indices], y[sv_indices], label="Support Vectors",
    #              color='lightblue', edgecolor='black', marker='o')  
    # plt.show()

    # fitter = KernelizedRidgeRegression(RBF(sigma=1), lambda_=0.1)
    # model = fitter.fit(X, y)
    # pred = model.predict(X)
    # plt.scatter(X, y)
    # plt.scatter(X,pred)
    # plt.show()

    ## PART2

    # Housing data 
    df = pd.read_csv("housing2r.csv")
    y = df["y"].values
    X = df.drop(columns=["y"]).values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    Ms = np.arange(1,11)


    mse_RR = []
    mse_cv_RR = []
    mse_SVR = []
    mse_cv_SVR = []

    kf = KFold(5, shuffle=True, random_state=42)

    # Polynomial
    for M in Ms:
        # KRR
        fitter = KernelizedRidgeRegression(kernel=Polynomial(M=M), lambda_=1)
        model = fitter.fit(X,y)
        preds = model.predict(X)
        mse_RR.append(mean_squared_error(y, preds))

        fitter = SVR(kernel=Polynomial(M=M), lambda_=1, epsilon=0.1) # TODO set epsilon
        model = fitter.fit(X,y)
        preds = model.predict(X)
        mse_SVR.append(mean_squared_error(y, preds))

        # Cross validation

    
    plt.plot(Ms, mse_RR, label = "Ridge Regression")
    plt.plot(Ms, mse_SVR, label = "SVR")
    plt.ylabel("MSE")
    plt.xlabel("Degree")
    plt.legend()
    plt.show()

    # RBF
    mse_RR = []
    mse_cv_RR = []
    mse_SVR = []
    mse_cv_SVR = []

    sigmas = [0.001,0.01,0.1,1,2,3,4,5,8,10, 100]
    lambdas = [0.001, 0.01, 0.1, 1, 10, 100]

    for sigma in sigmas:
        # lambda = 1
        fitter = KernelizedRidgeRegression(kernel=RBF(sigma=sigma), lambda_=1)
        model = fitter.fit(X,y)
        preds = model.predict(X)
        mse_RR.append(mean_squared_error(y, preds))

        fitter = SVR(kernel=RBF(sigma=sigma), lambda_=1, epsilon=0.1) # TODO set epsilon
        model = fitter.fit(X,y)
        preds = model.predict(X)
        mse_SVR.append(mean_squared_error(y, preds))

        # Cross validation
        for lambda_ in lambdas:
            mse_rr_folds = []
            mse_svr_folds = []

            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # Ridge Regression
                model_rr = KernelizedRidgeRegression(kernel=RBF(sigma=sigma), lambda_=lambda_)
                pred_rr = model_rr.fit(X_train, y_train).predict(X_test)
                mse_rr_folds.append(mean_squared_error(y_test, pred_rr))

                # SVR
                model_svr = SVR(kernel=RBF(sigma=sigma), lambda_=lambda_, epsilon=0.1)
                pred_svr = model_svr.fit(X_train, y_train).predict(X_test)
                mse_svr_folds.append(mean_squared_error(y_test, pred_svr))

        mse_


    plt.xscale('log')
    plt.plot(sigmas, mse_RR, label = "Ridge Regressoin")
    plt.plot(sigmas,mse_SVR, label = "SVR")
    plt.xlabel("sigma")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()
