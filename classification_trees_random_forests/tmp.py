

import numpy as np

def __get_bs_sample(self, X,y):
    """Gets one bootstrap sample from X and y"""
    X = np.array(X)
    all_rows = range(len(y))
    # Randomly sample rows with replacement
    rows = self.rand.sample(all_rows, len(y))
    oob_rows = list(set(all_rows) - set(rows))

    X_sample = X[rows, :] 
    y_sample = y[rows]
    return X_sample, y_sample, oob_rows