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
    return 1 / (1+ np.exp(-x))

def sigmoid_grad(x):
    return sigmoid(x) * (1- sigmoid(x))

def softmax(x):
    e_x = np.exp(x - np.max(x))  # for numerical stability
    return e_x / np.sum(e_x)


def forward_pass_one_sample(weights_all_layers, biases_all_layers, x):
    # Pass forward one sample and return all the activations
    activations = []
    current_activations = x.copy()
    activations.append(current_activations)  # Store input activations

    # Iterate over all layers except the last
    for i, (weights, biases) in enumerate(zip(weights_all_layers, biases_all_layers)):
        current_activations = weights @ current_activations + biases  # Weighted sum (z)
        activations.append(current_activations)  # Store pre-activation values (z)

        # Apply activation (sigmoid) for all layers except the last one
        if i != len(weights_all_layers) - 1:
            current_activations = sigmoid(current_activations)
            activations.append(current_activations)  # Store post-activation values (a)

    # The last layer, no activation function (pre-softmax)
    z_L = current_activations

    # Apply softmax on the last layer's output
    a_L = softmax(z_L)
    activations.append(a_L)  # Store post-softmax (final output probabilities)

    return activations


def backprop(weights_all_layers, biases_all_layers, softmax_grads, activations):
    a_list = activations[::2]   
    z_list = activations[1::2]  
    L = len(weights_all_layers)

    gradients_weights = [np.zeros_like(w) for w in weights_all_layers]
    gradients_biases = [np.zeros_like(b) for b in biases_all_layers]

    current_grads = softmax_grads

    for l in reversed(range(L)):
        a_prev = a_list[l]         
        W = weights_all_layers[l]

        gradients_weights[l] = current_grads.reshape(-1, 1) @ a_prev.reshape(1, -1)
        gradients_biases[l] = current_grads

        if l > 0:
            current_grads = (W.T @ current_grads) * sigmoid_grad(z_list[l-1])

    return gradients_weights, gradients_biases



class ANNClassification:
    # Units determine the amount of activations in each hidden layer
    def __init__(self, units, lambda_):
        self.units = units
        
    def fit(self,X,y, seed = 42, lr= 0.1, epochs = 1, conv_loss=0.003, get_gradients = False):
        np.random.seed(seed)
    
        # Get classes andd numbers of classes
        classes = np.unique(y)
        num_classes = len(classes)

        # Add input and output layer units
        self.all_units = [len(X[0])]
        self.all_units.extend(self.units)
        self.all_units.append(num_classes)

        # Initialize the weights, using glorot initialization
        weights_all_layers, biases_all_layers = init_glorot(self.all_units)

        for epoch in range(epochs):
            # Forward pass and backprop
            gradients_weights_total =  [np.zeros_like(weights) for weights in weights_all_layers]
            gradients_biases_total =  [np.zeros_like(bias) for bias in biases_all_layers]
            total_loss = 0
            for i, (x, y_i) in enumerate(zip(X,y)):

                # get the activations 
                activations = forward_pass_one_sample(weights_all_layers, biases_all_layers, x)
                # get the probs to calculate the log loss
                probs = activations[-1]
                loss = - np.log(probs[y_i]) 
                total_loss += loss

                # Backpropagation
                # loss grad + softmax grad 
                probs = activations[-1]           
                softmax_grads = probs.copy()
                softmax_grads[y_i] -= 1 

                # backprop
                gradients_weights, gradients_bias = backprop(weights_all_layers, biases_all_layers, softmax_grads, activations[:-1])
                
                # Accumulate gradients
                gradients_weights_total = [arr1 + arr2  for arr1, arr2 in zip(gradients_weights, gradients_weights_total)]
                gradients_biases_total = [arr1 + arr2 for arr1,arr2 in zip(gradients_biases_total, gradients_bias) ]
                
            # Descend
            weights_all_layers = [arr1 - lr * arr2 for arr1,arr2 in zip(weights_all_layers, gradients_weights_total)]
            biases_all_layers = [arr1 - lr*arr2 for arr1, arr2 in  zip(biases_all_layers, gradients_biases_total)]
            print(f"loss: {total_loss}, epoch: {epoch}")
            if total_loss < conv_loss:
                print(f"finished in {epoch} epochs")
                break
            
            if get_gradients:
                def loss_with_respect_to_WL(WL):
                    # Replace only the last layer's weights with WL
                    weights_copy = weights_all_layers.copy()
                    weights_copy[-1] = WL
                    loss = 0
                    for x, y_i in zip(X, y):
                        probs = forward_pass_one_sample(weights_copy, biases_all_layers, x)[-1]
                        loss += -np.log(probs[y_i])
                    return loss

                WL = weights_all_layers[-1]
                num_grads_WL = np.zeros_like(WL)
                h = 1e-4
                for i in range(WL.shape[0]):
                    for j in range(WL.shape[1]):
                        WL_plus = WL.copy()
                        WL_minus = WL.copy()
                        WL_plus[i, j] += h
                        WL_minus[i, j] -= h
                        f_plus = loss_with_respect_to_WL(WL_plus)
                        f_minus = loss_with_respect_to_WL(WL_minus)
                        num_grads_WL[i, j] = (f_plus - f_minus) / (2 * h)

                return gradients_weights_total, num_grads_WL


        return ANNClassificationPredict(weights_all_layers, biases_all_layers)
        
      

class ANNClassificationPredict:
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def predict(self, X):
        return np.array([forward_pass_one_sample(self.weights, self.biases, x)[-1] for x in X])
    

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
    # # example NN use
    # fitter = ANNClassification(units=[3,6,4], lambda_=0)
    # X = np.array([
    #     [1, 2, 3],
    #     [4, 5, 6],
    #     [7, 8, 9]
    # ], dtype=float)
    # y = np.array([0, 1, 2])
    # model = fitter.fit(X, y, epochs=100000, lr=0.5)
    # predictions = model.predict(X)
    # print(predictions)
    # np.testing.assert_almost_equal(predictions,
    #                                [[1, 0, 0],
    #                                 [0, 1, 0],
    #                                 [0, 0, 1]], decimal=3)
    

    # Checking gradients
    fitter = ANNClassification(units=[3,6,4], lambda_=0)
    X = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=float)
    y = np.array([0, 1, 2])
    gradients, num_gradients = fitter.fit(X, y, get_gradients=True)
    print(gradients[-1]) 
    print(num_gradients)
    #print(num_gradients)
    # Get numerical gradients




    # doughnut
    # X,y = doughnut()
    # fitter = ANNClassification(units=[10,15,8,6, 6], lambda_=0)
    # model = fitter.fit(X, y, lr=0.02)
    # predictions = model.predict(X)
    # print(predictions)
