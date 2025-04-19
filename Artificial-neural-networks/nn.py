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
    biases_all_layers = [np.zeros(units) for units in all_units[1:]]
    return weights_all_layers, biases_all_layers

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def softmax(x):
    x = x - np.max(x)  # For numerical stability
    e_x = np.exp(x)
    return e_x / np.sum(e_x)


def forward_pass_one_sample(weights_all_layers, biases_all_layers, x):
    pre_activations = []
    activations = []
    current_activations = x.copy()
    pre_activations.append(current_activations)

    for i, (weights, biases) in enumerate(zip(weights_all_layers, biases_all_layers)):
        # Linear step
        current_activations = weights @ current_activations + biases
        pre_activations.append(current_activations)

        # sigmoid(except in last layer)
        if i < len(weights_all_layers) -1:
            current_activations = sigmoid(current_activations)
            activations.append(current_activations)

    probs = softmax(current_activations)

    return probs, pre_activations, activations
    



class ANNClassification:
    def __init__(self, units, lambda_):
        self.units = units

    def fit(self, X, y, seed = 42, lr = 0.1, epochs = 1, conv_loss = 0.003):
        np.random.seed(seed)

        # Get the number of classes
        num_classes = len(np.unique(y))

        # Add input and output layer units
        self.all_units = [len(X[0])]
        self.all_units.extend(self.units)
        self.all_units.append(num_classes)

        # Initialize the weights 
        weights_all_layers, biases_all_layers = init_glorot(self.all_units)

        # Forward pass and backprop
        gradients_weights_total =  [np.zeros_like(weights) for weights in weights_all_layers]
        gradients_biases_total =  [np.zeros_like(bias) for bias in biases_all_layers]
        total_loss = 0

        for i, (x, y_i) in enumerate(zip(X,y)):

            # get the activations 
            probs, pre_activations, activations = forward_pass_one_sample(weights_all_layers, biases_all_layers, x)
            loss = -np.log(probs[y_i]) / len(y)
            total_loss += loss

            # Gradient for log-loss and softmax
            grad_after_softmax = probs
            grad_after_softmax[y_i] -= 1

            gradients_weights = [np.zeros_like(weights) for weights in weights_all_layers]
            gradients_biases =  [np.zeros_like(bias) for bias in biases_all_layers]

            for k in reversed(range(len(weights_all_layers))):
                



                

           
                


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
    fitter = ANNClassification(units=[3,4], lambda_=0)
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
