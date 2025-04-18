import numpy as np
import csv

def init_glorot(all_units):
    # Initialize the weights, using glorot initialization
    weights_all_layers = []
    for i in range(len(all_units) - 1):
        scale = np.sqrt(1 / (all_units[i]))
        weights_all_layers.append(
            np.random.normal(loc=0, scale=scale, size=(all_units[i+1], all_units[i]))
        )
    return weights_all_layers

def sigmoid(x):
    return 1 / (1+ np.exp(-x))

def softmax(X, x):
    return np.exp(x)/np.sum(np.exp(X)) 

def forward_pass(weights_all_layers, X):
    # TODO probably also gonna need to save the activations, for GD
    probs = []
    for current_activations in X.copy():
        for i, weights in enumerate(weights_all_layers):
            current_activations = weights @ current_activations
            current_activations = np.array([sigmoid(x) for x in current_activations])

        current_activations = [softmax(current_activations, x_k) for x_k in current_activations]
        probs.append(current_activations)

    return np.array(probs)


class ANNClassification:
    # Units determine the amount of activations in each hidden layer
    def __init__(self, units, lambda_):
        self.units = units
        
    def fit(self,X,y, seed = 42):
        np.random.seed(seed)
    
        # Get classes andd numbers of classes
        classes = np.unique(y)
        num_classes = len(classes)

        # Add input and output layer units
        self.all_units = [len(X[0])]
        self.all_units.extend(self.units)
        self.all_units.append(num_classes)

        # Initialize the weights, using glorot initialization
        weights_all_layers = init_glorot(self.all_units)

        # Forward pass
        probs = forward_pass(weights_all_layers, X)
        print(probs)
        

        

class ANNClassificationPredict:
    pass

class ANNRegression:
    # implement me too, please
    pass


# data reading

def read_tab(fn, adict):
    content = list(csv.reader(open(fn, "rt"), delimiter="\t"))

    legend = content[0][1:]
    data = content[1:]

    X = np.array([d[1:] for d in data], dtype=float)
    y = np.array([adict[d[0]] for d in data])

    return legend, X, y


def doughnut():
    legend, X, y = read_tab("doughnut.tab", {"C1": 0, "C2": 1})
    return X, y


def squares():
    legend, X, y = read_tab("squares.tab", {"C1": 0, "C2": 1})
    return X, y


if __name__ == "__main__":


    # example NN use
    fitter = ANNClassification(units=[3,6,4], lambda_=0)
    X = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=float)
    y = np.array([0, 1, 2])
    model = fitter.fit(X, y)
    predictions = model.predict(X)
    print(predictions)
    np.testing.assert_almost_equal(predictions,
                                   [[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 1]], decimal=3)
