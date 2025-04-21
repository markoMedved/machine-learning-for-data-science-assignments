import numpy as np
import csv
from torch import nn
import torch
from torch import optim
import torch.nn.init as init

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


def init_glorot(all_units):
    # Initialize the weights using Glorot (Xavier) normal initialization
    weights_all_layers = []
    for i in range(len(all_units) - 1):
        fan_in = all_units[i]
        fan_out = all_units[i + 1]
        # NOTE: here we multiply by 3, since otherwise it couldnt completly fit the squares, doughnut
        std = 5* np.sqrt(2 / (fan_in + fan_out))  # Glorot normal
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

def reLU(x):
    return np.maximum(0, x)


def reLU_grad(x):
    return (x > 0).astype(float)
    

def forward_pass_one_sample(weights_all_layers, biases_all_layers, x, activation_functions):
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
            if activation_functions[i] == "reLU":
                current_activations = reLU(current_activations)
            else:
                current_activations = sigmoid(current_activations)
            activations.append(current_activations)
            

    probs = softmax(current_activations)

    return probs, pre_activations, activations
    
def backprop(weights_all_layers, biases_all_layers, pre_activations, activations, current_activation_grad, activation_functions):
    gradients_weights = [np.zeros_like(weights) for weights in weights_all_layers]
    gradients_biases =  [np.zeros_like(bias) for bias in biases_all_layers]

    # Gradients for linear layers with sigmoids
    for k in reversed(range(len(weights_all_layers))):
        z_k_prev = activations[k - 1] if k != 0 else pre_activations[k]
        W = weights_all_layers[k]

        gradients_weights[k] = current_activation_grad.reshape(-1,1)  @ z_k_prev.reshape(1,-1)
        gradients_biases[k] = current_activation_grad

        if k > 0:
            if activation_functions[k-1] == "reLU":
                current_activation_grad = (current_activation_grad @ W) * reLU_grad(pre_activations[k])
            else: 
                current_activation_grad = (current_activation_grad @ W) * sigmoid_grad(pre_activations[k])

    return gradients_weights, gradients_biases

class ANNClassification:
    def __init__(self, units, lambda_, activation_functions = None):
        self.units = units
        self.lambda_ = lambda_
        if activation_functions == None:
            self.activation_functions = ["sig"] * len(units) 
        else:
            self.activation_functions = activation_functions
            if len(activation_functions) != len(units):
                print("Lenght of activation functions does not match units length, the activations functions will defalut to sigmoid")
                self.activation_functions = ["sig"] * len(units) 


    def fit(self, X, y, seed = 42, lr = 1, epochs = 50000, conv_loss = 0.001, get_gradients=False, grad_layer=-1, return_loss=False, verbose=True):
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
        losses = []
        accs = []
        for epoch in range(epochs):
            # Forward pass and backprop
            gradients_weights_total =  [np.zeros_like(weights) for weights in weights_all_layers]
            gradients_biases_total =  [np.zeros_like(bias) for bias in biases_all_layers]
            total_loss = 0
            acc = 0
            for i, (x, y_i) in enumerate(zip(X,y)):

                # get the activations 
                probs, pre_activations, activations = forward_pass_one_sample(weights_all_layers, biases_all_layers, x, self.activation_functions)
                loss = -np.log(probs[y_i]) / N
                total_loss += loss 
                acc += ( y_i == int(np.argmax(probs)))/N
                

                # Gradient for log-loss and softmax
                grad_after_softmax = probs
                grad_after_softmax[y_i] -= 1
                

                gradients_weights, gradients_biases = backprop(weights_all_layers, biases_all_layers, pre_activations, activations, grad_after_softmax, self.activation_functions)

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
                            probs = forward_pass_one_sample(weights_copy, biases_all_layers, x, self.activation_functions)[0]
                            loss += -np.log(probs[y_i])
                        return loss

                    def loss_with_respect_to_BL(BL):
                        biases_copy = biases_all_layers.copy()
                        biases_copy[grad_layer] = BL
                        loss = 0
                        for x, y_i in zip(X, y):
                            probs = forward_pass_one_sample(weights_all_layers, biases_copy, x, self.activation_functions)[0]
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
            if verbose:
                if epoch % 100 == 0 or total_loss > 2:
                    print(f"loss: {total_loss}, acc: {acc}, epoch: {epoch}")

            if total_loss < conv_loss:
                print(f"finished in {epoch} epochs, accurac: {acc}, loss: {total_loss}")
                break
            
            losses.append(total_loss)
            accs.append(acc)

        if return_loss:
            return losses, accs
        
        return ANNClassificationPredict(weights_all_layers, biases_all_layers, self.activation_functions)
                

class ANNClassificationPredict:
    def __init__(self, weights, biases, activation_functions):
        self.weights = weights
        self.biases = biases
        self.activation_functions = activation_functions

    def predict(self, X):
        return np.array([forward_pass_one_sample(self.weights, self.biases, x, self.activation_functions)[0] for x in X])
    
           

def forward_pass_one_sample_reg(weights_all_layers, biases_all_layers, x, activation_functions):
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
            if activation_functions[i] == "reLU":
                current_activations = reLU(current_activations)
            else:
                current_activations = sigmoid(current_activations)
            activations.append(current_activations)


    return pre_activations, activations


class ANNRegression:
    def __init__(self, units, lambda_, activation_functions=None):
        self.units = units
        self.lambda_ = lambda_
        if activation_functions == None:
            self.activation_functions = ["sig"] * len(units) 
        else:
            self.activation_functions = activation_functions
            if len(activation_functions) != len(units):
                print("Lenght of activation functions does not match units length, the activations functions will defalut to sigmoid")
                self.activation_functions = ["sig"] * len(units) 

    def fit(self, X, y, seed = 42, lr = 0.01, epochs = 15000, conv_loss = 0.00001):
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
                pre_activations, activations = forward_pass_one_sample_reg(weights_all_layers, biases_all_layers, x, self.activation_functions)
                # NOTE different loss function
                loss = 2 * (y_i - pre_activations[-1][0])**2 / N
                total_loss += loss
                
                # Gradient for the loss NOTE difference: it is multiplied by 2
                grad_loss = 2 * (pre_activations[-1] - y_i) / N 

                # backprop
                gradients_weights, gradients_biases = backprop(weights_all_layers, biases_all_layers, pre_activations, activations, grad_loss, self.activation_functions)

                # Accumulate gradients
                gradients_weights_total = [arr1 +  arr2  for arr1, arr2 in zip(gradients_weights_total, gradients_weights)]
                gradients_biases_total = [arr1 + arr2  for arr1, arr2 in zip(gradients_biases_total, gradients_biases)]
            
            # Descend ADD L2 regularization in heree only for weights
            weights_all_layers = [arr1 - lr * (arr2 + 2 * self.lambda_ * arr1) for arr1,arr2 in zip(weights_all_layers, gradients_weights_total)]
            biases_all_layers = [arr1 - lr*arr2 for arr1, arr2 in  zip(biases_all_layers, gradients_biases_total)]

            if epoch % 1000 == 0 or total_loss > 2:
                print(f"loss: {total_loss}, epoch: {epoch}")

            if total_loss < conv_loss:
                print(f"finished in {epoch} epochs, loss: {total_loss} ")
                break

        return ANNRegressionPredict(weights_all_layers, biases_all_layers, self.activation_functions)
    
    

class ANNRegressionPredict():
    def __init__(self, weigths, biases, activation_functions):
        self.weights_list = weigths
        self.biases = biases
        self.activation_functions = activation_functions

    def weights(self):
        all_weights = []

        for i in range(len(self.weights_list)):
            # Add a column of biases to the weights matrix (biases as last column)
            weight_with_bias = np.hstack([self.weights_list[i], self.biases[i].reshape(-1, 1)]).T
            all_weights.append(weight_with_bias)
        
        return all_weights


    def predict(self, X):
        return np.array([forward_pass_one_sample_reg(self.weights_list, self.biases, x, self.activation_functions)[0][-1][0] for x in X])



if __name__ == "__main__":
    # example NN use
    fitter = ANNClassification(units=[3,4], lambda_=0, activation_functions=["reLU", "reLU"])
    X_ex = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=float)
    y_ex = np.array([0, 1, 2])
    # model = fitter.fit(X_ex, y_ex, lr=0.01, epochs=10000)
    # predictions = model.predict(X_ex)
    # print(predictions)
    

    # Checking gradients
    fitter = ANNClassification(units=[3,10,50,30,1,4], lambda_=0)
    X_sq,y_sq = squares()
    X_d, y_d = doughnut()
    for X, y in zip([X_sq, X_d, X_ex], [y_sq, y_d, y_ex]):
        # The layer for which we want to check the gradients
        for grad_layer in range(len(fitter.units)):
            print(grad_layer)
            gradients, num_gradients, bias_gradients, num_bias_gradients = fitter.fit(X, y,epochs=1, get_gradients=True, grad_layer=grad_layer)
            diff = gradients[grad_layer]- num_gradients
            diff_bias = bias_gradients[grad_layer] - num_bias_gradients
            #print(diff)
            #print(diff_bias)
            
            for row in diff:
                for df in row:
                    if df > 1e-7:
                        print(f"Weight grad diff is {df}")
            for row in diff:
                for df in row:
                    if df > 1e-7:
                        print(f"Bias grad diff is {df}")

    # # doughnut
    # X,y = doughnut()
    # fitter = ANNClassification(units=[3], lambda_=0.0)
    # model = fitter.fit(X, y, lr=0.01, seed=100, epochs=3000, conv_loss=0.02)
    # predictions = model.predict(X)

    # # squares
    # X,y = squares()
    # fitter = ANNClassification(units=[7], lambda_=0.1)
    # losses, accs = fitter.fit(X, y, lr=0.01, seed=100, epochs=100, conv_loss=0.02, return_loss=True)
    
    

    # # TRY squares with pytorch
    # losses = []
    # X = torch.tensor(X, dtype=torch.float32)
    # y = torch.tensor(y, dtype=torch.long) 
    # model = ANNPytorch(input_dim=X.shape[1],
    #                     output_dim=len(np.unique(y)), units=[5], 
    #                     activations=["sig", "sig"], lambda_=0.0)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=1, weight_decay=model.lambda_)

    # epochs = 3000
    # torch.manual_seed(2)
    # for epoch in range(epochs):
    #     model.train()

    #     logits = model(X)
    #     loss = criterion(logits, y)
    #     acc = (pred == y).float().mean()
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    #     if epoch % 100 == 0:
    #         pred = torch.argmax(logits, dim=1)
    #         print(f"Epoch {epoch} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")

    #     losses.append(loss)
    #     accs.append(acc)



