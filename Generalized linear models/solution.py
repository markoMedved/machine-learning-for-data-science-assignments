from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import unittest
import pandas as pd


class MultinomialLogReg():
    def log_likelihood(self, beta):
        m = len(np.unique(self.y))
        # Get the latent strengths
        u = np.zeros(shape=(m, len(self.X)))
        # Unflatten the beta 
        beta = beta.reshape((m-1, self.X.shape[1]))
        # Multiply with the weights
        u[:m-1] = beta @ self.X.T 
        # Flatten the beta back
        beta = beta.flatten()
        # Reference class to 0s
        u[m-1] = np.zeros(len(self.X))
        # Transpose to get probabilites for each class one row
        u = u.T
        # Apply the softmax
        sum_accross_rows = np.sum(np.exp(u), axis=1).reshape(-1,1)
        sum_accross_rows = np.tile(sum_accross_rows, (1,u.shape[1]))
        probs = np.exp(u) / sum_accross_rows
        # Get tmatrix of targets to be able to just make the product
        y_mat = np.eye(m)[self.y]
        # Get the log likelihood (- to minimize)
        l = -np.sum(np.log(probs) * y_mat)
        return l

    def build(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        self.X = np.c_[np.ones(self.X.shape[0]), self.X]
        m = len(np.unique(self.y))
        # Initialize Beta, (num of classes - 1) * (Num of feautures)
        beta = np.random.rand(m - 1, self.X.shape[1])
        # Get the result with the gradient descent
        result = fmin_l_bfgs_b(self.log_likelihood, beta, approx_grad=True)[0]
        # Reshape the result
        result = result.reshape((m-1, self.X.shape[1]))

        return MultinomialLogRegInference(result)

class MultinomialLogRegInference():
    def __init__(self, beta):
        self.beta = beta

    def predict(self, X):
        X = np.array(X)
        X = np.c_[np.ones(X.shape[0]), X]
        m = self.beta.shape[0] + 1
        # Get the latent strengths
        u = np.zeros(shape=(m, len(X)))
        # Multiply with the weights
        u[:m-1] = self.beta @ X.T 
        # Reference class to 0s
        u[m-1] = np.zeros(len(X))
        # Transpose to get probabilites for each class one row
        u = u.T
        # Apply the softmax
        sum_accross_rows = np.sum(np.exp(u), axis=1).reshape(-1,1)
        sum_accross_rows = np.tile(sum_accross_rows, (1,u.shape[1]))
        probs = np.exp(u) / sum_accross_rows
        # return the probs
   
   
        return probs


def logistic_cdf(x):
    return (1 / (1 + np.exp(-x)))

class OrdinalLogReg():
    def log_likelihood(self, params):
        m = len(np.unique(self.y))
        # Get betas and tresholds
        beta = params[:self.X.shape[1]]
        deltas = params[self.X.shape[1]:]
        # Get the tresholds from deltas
        tresholds = np.concatenate((np.array([-np.inf, 0]), 
                            np.cumsum(deltas),
                            np.array([np.inf]) ))

        # Get the latent strengths
        u = beta @ self.X.T
        l = 0
        for i in range(len(self.y)):
            j = self.y[i]
            l -= np.log(logistic_cdf(tresholds[j + 1] - u[i]) - logistic_cdf(tresholds[j] - u[i]))
        return l

    def build(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        self.X = np.c_[np.ones(self.X.shape[0]), self.X]
        m = len(np.unique(self.y))
        # Initialize Beta, (Num of feautures)
        beta = np.random.rand(self.X.shape[1])
        # Initialize the deltas
        deltas = np.ones(m-2) 
        params = np.concatenate((beta, deltas))
        # Set the bounds, deltas have to be more than 0
        bounds = [(-np.inf, np.inf)] * self.X.shape[1] + [(1e-7, np.inf)] * (m - 2)
        # Get the result with the gradient descent
        result = fmin_l_bfgs_b(self.log_likelihood, params, approx_grad=True, bounds=bounds)[0]
        beta = result[:self.X.shape[1]]
        deltas = result[self.X.shape[1]:]

        return OrdinalLogRegInference(beta, deltas)


class OrdinalLogRegInference():
    def __init__(self, beta, deltas):
        self.beta = beta
        self.deltas = deltas

    def predict(self, X):
        X = np.array(X)
        X = np.c_[np.ones(X.shape[0]), X]
        m = len(self.deltas) + 2
        tresholds = np.concatenate((np.array([-np.inf, 0]), 
                            np.cumsum(self.deltas),
                            np.array([np.inf]) ))
        u = self.beta @ X.T
        pred = np.zeros(shape=(X.shape[0], m))
        for i in range(len(X)):
            for j in range(m): 
                pred[i, j] = logistic_cdf(tresholds[j + 1] - u[i]) - logistic_cdf(tresholds[j] - u[i])

        return pred


class MyTests(unittest.TestCase):
    def setUp(self):
        self.Xs = [np.array([[0, 0],
                           [0, 1],
                           [1, 0],
                           [1, 1],
                           [1, 1]]), 
                    np.array([[0,0,0,1],
                             [2,1,4,5],
                             [3,2,5,7]]),
                    np.array([[1,0],
                             [1,1],
                             [1,2],
                             [2,0],
                             [2,1],
                             [2,2],
                             [3,0],
                             [3,0],
                             [3,0]])]
        self.ys = [np.array([0, 0, 1, 1, 2]),
                   np.array([0,1,2]),
                   np.array([0,0,0,1,1,1,1,1,1])]
    
    def test_multinomial(self):
        for X,y in zip(self.Xs, self.ys):
            train = X[::2], y[::2]
            test = X[1::2], y[1::2]
            l = MultinomialLogReg()
            c = l.build(X, y)
            prob = c.predict(X)
            print(y)
            print(prob)
            #self.assertEqual(prob.shape, (2, 3))
            self.assertTrue((prob <= 1).all())
            self.assertTrue((prob >= 0).all())
            np.testing.assert_almost_equal(prob.sum(axis=1), 1)

    def test_ordinal(self):
        for X,y in zip(self.Xs, self.ys):
            train = X[::2], y[::2]
            test = X[1::2], y[1::2]
            l = OrdinalLogReg()
            c = l.build(X, y)
            prob = c.predict(X)
            print(y)
            print(prob)
            #self.assertEqual(prob.shape, (2, 3))
            self.assertTrue((prob <= 1).all())
            self.assertTrue((prob >= 0).all())
            np.testing.assert_almost_equal(prob.sum(axis=1), 1)



if __name__ == "__main__":
    np.random.seed(42)
    #unittest.main()
    df = pd.read_csv("dataset.csv", sep=";")
    # Encode the data
    encoder = LabelEncoder()
    for col in ["ShotType", "Competition", "PlayerType", "Movement"]:
        df[col] = encoder.fit_transform(df[col])
    # Split into train and test
    train, test = train_test_split(df, test_size=0.3, stratify=df["ShotType"])
    # Get the features and target
    X_train,y_train  = train.drop(columns=["ShotType"]), train["ShotType"]
    X_test, y_test = test.drop(columns=["ShotType"]), test["ShotType"]
    # Scale the data
    scaler = StandardScaler()
    train_scaled = scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    # build the model
    model = MultinomialLogReg()
    model_pred = model.build(X_train, y_train)
    pred = model_pred.predict(X_test)
    pred_classes = np.argmax(pred, axis=1)
    print(classification_report(y_test, pred_classes))
    # Use the other one
    model = OrdinalLogReg()
    model_pred = model.build(X_train, y_train)
    pred = model_pred.predict(X_test)
    pred_classes = np.argmax(pred, axis=1)
    print(classification_report(y_test, pred_classes))


    