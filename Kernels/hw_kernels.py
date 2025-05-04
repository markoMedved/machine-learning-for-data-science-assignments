import numpy as np
from cvxopt import solvers, matrix
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

# TODO epsilon setting for part 2
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
    #####################################
    # PART 1 
    #################################

    sine = pd.read_csv("sine.csv")
    X = sine["x"].values.reshape(-1, 1)
    y = sine["y"].values

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    # --- SVR with RBF ---
    svr_rbf = SVR(RBF(sigma=1), epsilon=0.1, lambda_=0.1)
    model_rbf_svr = svr_rbf.fit(X, y)
    pred_rbf_svr = model_rbf_svr.predict(X)
    sv_indices_rbf = svr_rbf.support_vectors()

    ax[0, 0].scatter(X[~np.isin(np.arange(len(X)), sv_indices_rbf)], 
                    y[~np.isin(np.arange(len(X)), sv_indices_rbf)], 
                    color='lightblue', label="Non-Support Vectors")
    ax[0, 0].scatter(X[sv_indices_rbf], y[sv_indices_rbf], 
                    color='lightblue', edgecolor='black', label="Support Vectors")
    ax[0, 0].scatter(X, pred_rbf_svr, color='orange', label="Prediction")
    ax[0, 0].set_title("SVR - RBF Kernel")
    ax[0, 0].legend()

    # --- Kernel Ridge with RBF ---
    rr_rbf = KernelizedRidgeRegression(RBF(sigma=1), lambda_=0.1)
    model_rbf_rr = rr_rbf.fit(X, y)
    pred_rbf_rr = model_rbf_rr.predict(X)

    ax[0, 1].scatter(X, y, label="Data", color='lightblue')
    ax[0, 1].scatter(X, pred_rbf_rr, label="Prediction", color='orange')
    ax[0, 1].set_title("Ridge Regression - RBF Kernel")
    ax[0, 1].legend()

    # --- SVR with Polynomial ---
    svr_poly = SVR(Polynomial(M=3), epsilon=0.1, lambda_=0.1)
    model_poly_svr = svr_poly.fit(X, y)
    pred_poly_svr = model_poly_svr.predict(X)
    sv_indices_poly = svr_poly.support_vectors()

    ax[1, 0].scatter(X[~np.isin(np.arange(len(X)), sv_indices_poly)], 
                    y[~np.isin(np.arange(len(X)), sv_indices_poly)], 
                    color='lightblue', label="Non-Support Vectors")
    ax[1, 0].scatter(X[sv_indices_poly], y[sv_indices_poly], 
                    color='lightblue', edgecolor='black', label="Support Vectors")
    ax[1, 0].scatter(X, pred_poly_svr, color='orange', label="Prediction")
    ax[1, 0].set_title("SVR - Polynomial Kernel")
    ax[1, 0].legend()

    # --- Kernel Ridge with Polynomial ---
    rr_poly = KernelizedRidgeRegression(Polynomial(M=3), lambda_=0.1)
    model_poly_rr = rr_poly.fit(X, y)
    pred_poly_rr = model_poly_rr.predict(X)

    ax[1, 1].scatter(X, y, label="Data", color='lightblue')
    ax[1, 1].scatter(X, pred_poly_rr, label="Prediction", color='orange')
    ax[1, 1].set_title("Ridge Regression - Polynomial Kernel")
    ax[1, 1].legend()

    plt.tight_layout()
    plt.show()

    ########################################
    ## PART2
    #########################################

    # Housing data 
    df = pd.read_csv("housing2r.csv")
    y = df["y"].values
    X = df.drop(columns=["y"]).values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    k = 10
    kf = KFold(k, shuffle=True, random_state=42)

    Ms = np.arange(1,11)
    lambdas = [0.001, 0.01, 0.1, 1, 10, 100]

    mse_RR_pol = []
    mse_cv_RR_pol = []
    mse_SVR_pol = []
    mse_cv_SVR_pol = []

    sv_pol = []
    sv_cv_pol = []


    # Polynomial
    for M in Ms:
        mse_SVR_tmp = []
        mse_RR_tmp = []
        sv = 0
        # lambda = 1
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            fitter = KernelizedRidgeRegression(kernel=Polynomial(M=M), lambda_=1)
            model = fitter.fit(X_train,y_train)
            preds = model.predict(X_test)
            mse_RR_tmp.append(mean_squared_error(y_test, preds))

            fitter = SVR(kernel=Polynomial(M=M), lambda_=1, epsilon=0.1) # TODO set epsilon
            model = fitter.fit(X_train,y_train)
            preds = model.predict(X_test)
            mse_SVR_tmp.append(mean_squared_error(y_test, preds))

            sv += len(fitter.support_vectors())
        
        # Average support vectors in the split
        sv /= k
        sv_pol.append(sv)

        mse_RR_pol.append(np.mean(mse_RR_tmp))
        mse_SVR_pol.append(np.mean(mse_SVR_tmp))

        # Cross validation
        rr_cv_scores = []
        svr_cv_scores = []
        svr_cv_sv_counts = []

        
        for lambda_ in lambdas:
            mse_rr_lmbd, mse_svr_lmbd = [], []
            sv_count = 0

            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                rr = KernelizedRidgeRegression(kernel=Polynomial(M=M), lambda_=lambda_)
                svr = SVR(kernel=Polynomial(M=M), lambda_=lambda_, epsilon=0.1)

                pred_rr = rr.fit(X_train, y_train).predict(X_test)
                pred_svr = svr.fit(X_train, y_train).predict(X_test)

                mse_rr_lmbd.append(mean_squared_error(y_test, pred_rr))
                mse_svr_lmbd.append(mean_squared_error(y_test, pred_svr))

                sv_count += len(svr.support_vectors())

            rr_cv_scores.append(np.mean(mse_rr_lmbd))
            svr_cv_scores.append(np.mean(mse_svr_lmbd))
            svr_cv_sv_counts.append(sv_count / k)  # avg over folds

        mse_cv_RR_pol.append(min(rr_cv_scores))
        mse_cv_SVR_pol.append(min(svr_cv_scores))

        # Get support vector count corresponding to best lambda for SVR
        best_lambda_idx = np.argmin(svr_cv_scores)
        sv_cv_pol.append(svr_cv_sv_counts[best_lambda_idx])


    # RBF
    mse_RR = []
    mse_cv_RR = []
    mse_SVR = []
    mse_cv_SVR = []
    sv_rbf = []         # Support vectors for lambda=1
    sv_cv_rbf = []      # Support vectors for best lambda (CV)

    sigmas = [0.001, 0.01, 0.1, 1, 2, 3, 4, 5, 8, 10, 100]

    for sigma in sigmas:
        mse_SVR_tmp = []
        mse_RR_tmp = []
        sv = 0

        # lambda = 1
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            fitter = KernelizedRidgeRegression(kernel=RBF(sigma=sigma), lambda_=1)
            model = fitter.fit(X_train, y_train)
            preds = model.predict(X_test)
            mse_RR_tmp.append(mean_squared_error(y_test, preds))

            fitter = SVR(kernel=RBF(sigma=sigma), lambda_=1, epsilon=0.1)
            model = fitter.fit(X_train, y_train)
            preds = model.predict(X_test)
            mse_SVR_tmp.append(mean_squared_error(y_test, preds))

            sv += len(fitter.support_vectors())

        mse_RR.append(np.mean(mse_RR_tmp))
        mse_SVR.append(np.mean(mse_SVR_tmp))
        sv_rbf.append(sv / k)

        # Cross validation
        rr_cv_scores = []
        svr_cv_scores = []
        svr_cv_sv_counts = []

        for lambda_ in lambdas:
            mse_rr_folds = []
            mse_svr_folds = []
            sv_count = 0

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

                sv_count += len(model_svr.support_vectors())

            rr_cv_scores.append(np.mean(mse_rr_folds))
            svr_cv_scores.append(np.mean(mse_svr_folds))
            svr_cv_sv_counts.append(sv_count / k)

        mse_cv_RR.append(min(rr_cv_scores))
        mse_cv_SVR.append(min(svr_cv_scores))
        sv_cv_rbf.append(svr_cv_sv_counts[np.argmin(svr_cv_scores)])

    # Plot 
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    # --- Ridge Regression - Polynomial Kernel ---
    ax[0, 0].plot(Ms, mse_RR_pol, label=r"$\lambda = 1$")
    ax[0, 0].plot(Ms, mse_cv_RR_pol, label=r"$\lambda$ chosen with CV")
    ax[0, 0].set_ylabel("MSE", rotation=0, labelpad=15)
    ax[0, 0].set_xlabel("Degree")
    ax[0, 0].set_yscale("log")
    ax[0, 0].legend()
    ax[0, 0].set_title("Ridge Regression - Polynomial Kernel")
    ax[0, 0].grid(True)
    ax[0, 0].yaxis.set_label_position('left')

    # --- SVR - Polynomial Kernel ---
    ax[0, 1].plot(Ms, mse_SVR_pol, label=r"$\lambda = 1$")
    ax[0, 1].plot(Ms, mse_cv_SVR_pol, label=r"$\lambda$ chosen with CV")
    ax[0, 1].set_ylabel("MSE", rotation=0, labelpad=15)
    ax[0, 1].set_xlabel("Degree")
    ax[0, 1].set_yscale("log")
    ax[0, 1].legend()
    ax[0, 1].set_title("SVR - Polynomial Kernel")
    ax[0, 1].grid(True)
    ax[0, 1].yaxis.set_label_position('left')

    # Annotate support vectors on SVR Polynomial Kernel plot
    for i, M in enumerate(Ms):
        ax[0, 1].text(M, mse_SVR_pol[i] * 1.15, f"{sv_pol[i]:.0f}", ha='center', va='bottom', fontsize=8, color='blue')
        ax[0, 1].text(M, mse_cv_SVR_pol[i] * 0.85, f"{sv_cv_pol[i]:.0f}", ha='center', va='top', fontsize=8, color='orange')

    # --- Ridge Regression - RBF Kernel ---
    ax[1, 0].plot(sigmas, mse_RR, label=r"$\lambda = 1$")
    ax[1, 0].plot(sigmas, mse_cv_RR, label=r"$\lambda$ chosen with CV")
    ax[1, 0].set_ylabel("MSE", rotation=0, labelpad=15)
    ax[1, 0].set_xlabel("Sigma")
    ax[1, 0].set_xscale('log')
    ax[1, 0].legend()
    ax[1, 0].set_title("Ridge Regression - RBF Kernel")
    ax[1, 0].grid(True)
    ax[1, 0].yaxis.set_label_position('left')

    # --- SVR - RBF Kernel ---
    ax[1, 1].plot(sigmas, mse_SVR, label=r"$\lambda = 1$")
    ax[1, 1].plot(sigmas, mse_cv_SVR, label=r"$\lambda$ chosen with CV")
    ax[1, 1].set_ylabel("MSE", rotation=0, labelpad=15)
    ax[1, 1].set_xlabel("Sigma")
    ax[1, 1].set_xscale('log')
    ax[1, 1].legend()
    ax[1, 1].set_title("SVR - RBF Kernel")
    ax[1, 1].grid(True)
    ax[1, 1].yaxis.set_label_position('left')

    # Annotate support vectors on SVR RBF Kernel plot
    for i, sigma in enumerate(sigmas):
        ax[1, 1].text(sigma, mse_SVR[i] * 1.02, f"{round(sv_rbf[i]):.0f}", ha='center', va='bottom', fontsize=8, color='blue')
        ax[1, 1].text(sigma, mse_cv_SVR[i] * 0.985, f"{round(sv_cv_rbf[i]):.0f}", ha='center', va='top', fontsize=8, color='orange')


    # Layout adjustments
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.show()





