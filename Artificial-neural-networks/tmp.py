import numpy as np
import csv


def init_glorot(all_units):
    # Initialize the weights using Glorot (Xavier) normal initialization
    weights_all_layers = []
    for i in range(len(all_units) - 1):
        fan_in = all_units[i]
        fan_out = all_units[i + 1]
        std = 5 * np.sqrt(2 / (fan_in + fan_out))  # Glorot normal
        weights = np.random.normal(loc=0, scale=std, size=(fan_out, fan_in))
        weights_all_layers.append(weights)
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
    
def backprop(weights_all_layers, biases_all_layers, pre_activations, activations, current_activation_grad):
    gradients_weights = [np.zeros_like(weights) for weights in weights_all_layers]
    gradients_biases =  [np.zeros_like(bias) for bias in biases_all_layers]

    # Gradients for linear layers with sigmoids
    for k in reversed(range(len(weights_all_layers))):
        z_k_prev = activations[k - 1] if k != 0 else pre_activations[k]
        W = weights_all_layers[k]

        gradients_weights[k] = current_activation_grad.reshape(-1,1)  @ z_k_prev.reshape(1,-1)
        gradients_biases[k] = current_activation_grad

        if k > 0:
            current_activation_grad = (current_activation_grad @ W) * sigmoid_grad(pre_activations[k])

    return gradients_weights, gradients_biases

class ANNClassification:
    def __init__(self, units, lambda_):
        self.units = units
        self.lambda_ = lambda_

    def fit(self, X, y, seed = 42, lr = 0.1, epochs = 15000, conv_loss = 0.001, get_gradients=False, grad_layer=-1):
        np.random.seed(seed)

        # Get the number of classes, and length of data
        num_classes = len(np.unique(y))
        N = len(y)

        # Add input and output layer units
        self.all_units = [len(X[0])]
        self.all_units.extend(self.units)
        self.all_units.append(num_classes)

        # Initialize the weights 
        weights_all_layers, biases_all_layers = init_glorot(self.all_units)

        for epoch in  range(epochs):
            # Forward pass and backprop
            gradients_weights_total =  [np.zeros_like(weights) for weights in weights_all_layers]
            gradients_biases_total =  [np.zeros_like(bias) for bias in biases_all_layers]
            total_loss = 0
            acc = 0
            for i, (x, y_i) in enumerate(zip(X,y)):

                # get the activations 
                probs, pre_activations, activations = forward_pass_one_sample(weights_all_layers, biases_all_layers, x)
                loss = -np.log(probs[y_i]) / N
                total_loss += loss 
                acc += ( y_i == int(np.argmax(probs)))/N
                

                # Gradient for log-loss and softmax
                grad_after_softmax = probs
                grad_after_softmax[y_i] -= 1

                gradients_weights, gradients_biases = backprop(weights_all_layers, biases_all_layers, pre_activations, activations, grad_after_softmax)
                #print(gradients_weights)
                # Accumulate gradients
                gradients_weights_total = [arr1 +  arr2  for arr1, arr2 in zip(gradients_weights_total, gradients_weights)]
                gradients_biases_total = [arr1 + arr2  for arr1, arr2 in zip(gradients_biases_total, gradients_biases)]

            # Return the gradients
            if get_gradients:
                if epoch == epochs-1:
                    def loss_with_respect_to_WL(WL):
                        weights_copy = weights_all_layers.copy()
                        weights_copy[grad_layer] = WL
                        loss = 0
                        for x, y_i in zip(X, y):
                            probs = forward_pass_one_sample(weights_copy, biases_all_layers, x)[0]
                            loss += -np.log(probs[y_i])
                        return loss

                    def loss_with_respect_to_BL(BL):
                        biases_copy = biases_all_layers.copy()
                        biases_copy[grad_layer] = BL
                        loss = 0
                        for x, y_i in zip(X, y):
                            probs = forward_pass_one_sample(weights_all_layers, biases_copy, x)[0]
                            loss += -np.log(probs[y_i])
                        return loss

                    WL = weights_all_layers[grad_layer]
                    BL = biases_all_layers[grad_layer]

                    num_grads_WL = np.zeros_like(WL)
                    num_grads_BL = np.zeros_like(BL)

                    h = 1e-4

                    # Numerical gradient for weights
                    for i in range(WL.shape[0]):
                        for j in range(WL.shape[1]):
                            WL_plus = WL.copy()
                            WL_minus = WL.copy()
                            WL_plus[i, j] += h
                            WL_minus[i, j] -= h
                            f_plus = loss_with_respect_to_WL(WL_plus)
                            f_minus = loss_with_respect_to_WL(WL_minus)
                            num_grads_WL[i, j] = (f_plus - f_minus) / (2 * h)

                    # Numerical gradient for biases
                    for i in range(BL.shape[0]):
                        BL_plus = BL.copy()
                        BL_minus = BL.copy()
                        BL_plus[i] += h
                        BL_minus[i] -= h
                        f_plus = loss_with_respect_to_BL(BL_plus)
                        f_minus = loss_with_respect_to_BL(BL_minus)
                        num_grads_BL[i] = (f_plus - f_minus) / (2 * h)

                    return gradients_weights_total, num_grads_WL, gradients_biases_total, num_grads_BL

            
            # Descend, ADD L2 regularization in heree only for weights
            weights_all_layers = [arr1 - lr * (arr2 + 2 * self.lambda_ * arr1) for arr1,arr2 in zip(weights_all_layers, gradients_weights_total)]
            biases_all_layers = [arr1 - lr*arr2 for arr1, arr2 in  zip(biases_all_layers, gradients_biases_total)]
            if epoch % 100 == 0:
                print(f"loss: {total_loss}, epoch: {epoch}")
                print(f"acc: {acc}")

            if total_loss < conv_loss:
                print(f"finished in {epoch} epochs")
                break
            
        return ANNClassificationPredict(weights_all_layers, biases_all_layers)
                

class ANNClassificationPredict:
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def predict(self, X):
        return np.array([forward_pass_one_sample(self.weights, self.biases, x)[0] for x in X])
    
           

def forward_pass_one_sample_reg(weights_all_layers, biases_all_layers, x):
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


    return pre_activations, activations


class ANNRegression:
    def __init__(self, units, lambda_):
        self.units = units
        self.lambda_ = lambda_

    def fit(self, X, y, seed = 42, lr = 0.1, epochs = 15000, conv_loss = 0.00001):
        np.random.seed(seed)
        # Length of data
        N = len(y)

        # Add input and output layer units
        self.all_units = [len(X[0])]
        self.all_units.extend(self.units)
        self.all_units.append(1) #The last unit is one NOTE: CHANGE

        # Initialize the weights 
        weights_all_layers, biases_all_layers = init_glorot(self.all_units)

        for epoch in  range(epochs):
            # Forward pass and backprop
            gradients_weights_total =  [np.zeros_like(weights) for weights in weights_all_layers]
            gradients_biases_total =  [np.zeros_like(bias) for bias in biases_all_layers]
            total_loss = 0
            for i, (x, y_i) in enumerate(zip(X,y)):

                # Get the activations
                pre_activations, activations = forward_pass_one_sample_reg(weights_all_layers, biases_all_layers, x)
                # NOTE different loss function
                loss = 2 * (y_i - pre_activations[-1][0])**2 / N
                total_loss += loss
                
                # Gradient for the loss NOTE difference: it is multiplied by 2
                grad_loss = 2 * (pre_activations[-1] - y_i) / N 

                # backprop
                gradients_weights, gradients_biases = backprop(weights_all_layers, biases_all_layers, pre_activations, activations, grad_loss)

                # Accumulate gradients
                gradients_weights_total = [arr1 +  arr2  for arr1, arr2 in zip(gradients_weights_total, gradients_weights)]
                gradients_biases_total = [arr1 + arr2  for arr1, arr2 in zip(gradients_biases_total, gradients_biases)]
            
            # Descend ADD L2 regularization in heree only for weights
            weights_all_layers = [arr1 - lr * (arr2 + 2 * self.lambda_ * arr1) for arr1, arr2 in zip(weights_all_layers, gradients_weights_total)]
            biases_all_layers = [arr1 - lr*arr2 for arr1, arr2 in  zip(biases_all_layers, gradients_biases_total)]

            if epoch % 100 == 0:
                print(f"loss: {total_loss}, epoch: {epoch}")

            if total_loss < conv_loss:
                print(f"finished in {epoch} epochs")
                break

        return ANNRegressionPredict(weights_all_layers, biases_all_layers)
    
    

class ANNRegressionPredict():
    def __init__(self, weigths, biases):
        self.weights_list = weigths
        self.biases = biases

    def weights(self):
        all_weights = []

        for i in range(len(self.weights_list)):
            # Add a column of biases to the weights matrix (biases as last column)
            weight_with_bias = np.hstack([self.weights_list[i], self.biases[i].reshape(-1, 1)]).T
            all_weights.append(weight_with_bias)
        
        return all_weights


    def predict(self, X):
        return np.array([forward_pass_one_sample_reg(self.weights_list, self.biases, x)[0][-1][0] for x in X])


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
    # fitter = ANNClassification(units=[3,4], lambda_=0)
    # X = np.array([
    #     [1, 2, 3],
    #     [4, 5, 6],
    #     [7, 8, 9]
    # ], dtype=float)
    # y = np.array([0, 1, 2])
    # model = fitter.fit(X, y, lr=0.01, epochs=10000)
    # predictions = model.predict(X)
    # print(predictions)
    

    # # Checking gradients
    # fitter = ANNClassification(units=[3,5,5,5,6,4], lambda_=0)
    # X = np.array([
    #     [1, 2, 3],
    #     [4, 5, 6],
    #     [7, 8, 9]
    # ], dtype=float)
    # y = np.array([0, 1, 2])
    # # The layer for which we want to check the gradients
    # grad_layer =1
    # gradients, num_gradients, bias_gradients, num_bias_gradients = fitter.fit(X, y,epochs=100, get_gradients=True, grad_layer=grad_layer)
    # print(gradients[grad_layer]- num_gradients)
    # print(bias_gradients[grad_layer] - num_bias_gradients)

    # doughnut
    X,y = doughnut()
    fitter = ANNClassification(units=[5,5], lambda_=0)
    model = fitter.fit(X, y, lr=0.001, seed=100, epochs=10000, conv_loss=0.02)
    predictions = model.predict(X)
    print(predictions)

    # squares
    X,y = squares()
    fitter = ANNClassification(units=[10,15, 6], lambda_=0)
    model = fitter.fit(X, y, lr=0.001, seed=100, epochs=10000, conv_loss=0.02)
    predictions = model.predict(X)
    print(predictions)