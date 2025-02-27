import csv
import numpy as np
import random

def read_tab(fn, adict):
    content = list(csv.reader(open(fn, "rt"), delimiter="\t"))

    legend = content[0][1:]
    data = content[1:]

    X = np.array([d[1:] for d in data], dtype=float)
    y = np.array([adict[d[0]] for d in data])

    return legend, X, y


def tki():
    legend, Xt, yt = read_tab("tki-train.tab", {"Bcr-abl": 1, "Wild type": 0})
    _, Xv, yv = read_tab("tki-test.tab", {"Bcr-abl": 1, "Wild type": 0})
    return (Xt, yt), (Xv, yv), legend


def all_columns(X, rand):
    return range(X.shape[1])


def random_sqrt_columns(X, rand: random.Random):
    # Get number of all columns
    ncols = X.shape[1]
    # Get the sqrt for the number of features to take
    ncols_new = int(np.sqrt(ncols))
    # random sampling
    features = rand.sample(range(ncols), ncols_new)
    return features


class Tree:

    def __init__(self, rand=None,
                 get_candidate_columns=all_columns,
                 min_samples=2):
        self.rand = rand  # for replicability
        self.get_candidate_columns = get_candidate_columns  # needed for random forests
        self.min_samples = min_samples # minimal number of samples where the node is still split further




    def __get_class_prob(self, y_current):
        """Returns the probabilities of the classes in the currently evaluated space"""
        length = len(y_current)
        _,classes_counts = np.unique(y_current, return_counts=True)
        classes_probs = [count/length  for count in classes_counts]
        return classes_probs

    def __gini(self, probs):
        """calculates the gini impurity"""
        gini = 1
        for prob in probs:
            gini -= prob**2
        return gini

    def split(self, X_current, y_current,cols, current_constraints = []):
        """Define the split on the currently evaluated space, and the recursively for all the subspaces"""
        X_current = np.array(X_current)
        y_current = np.array(y_current)
       
        output = []

        # Stop if the lenfth reaches 2 or if we only have the same class in the split
        _,unq = np.unique(y_current, return_counts=True)

        if len(y_current) == 0:
            return 0

        if len(y_current) < self.min_samples or len(unq) == 1:

            clss,cls_cnts = np.unique(y_current, return_counts=True)
            cls = clss[cls_cnts.argmax()]
            
            current_constraints.append(cls)
            return current_constraints

        # Initialize stuff
        lowest_cost = -1
        split_feature = 0
        split_treshold = 0

        # Go through all the features
        for i,x in enumerate(np.transpose(X_current)):
           
            if i not in cols:
                continue

            for j,y in enumerate(y_current):
                x_at_y = x[j]
                # The first split of the space
                y_split1 = [yi for yi in y_current[x < x_at_y]]
                y_preds_split1 = self.__get_class_prob(y_split1)

                # The second split of the space
                y_split2 = [yi for yi in y_current[x >= x_at_y]]
                y_preds_split2 = self.__get_class_prob(y_split2)

                # Calculate the cost
                cost = len(y_split1)/len(y_current) * self.__gini(y_preds_split1) + len(y_split2)/len(y_current) * self.__gini(y_preds_split2)
   
                
                # Update the cost, and the best split if lower that lowes_cost
                if lowest_cost == -1 or cost <= lowest_cost:
                    lowest_cost = cost
                    # Update the split feature and the split treshold
                    split_feature = i
                    split_treshold = x_at_y

        # Define the splits made by the optimal feature and treshold combo
        X_new_1 = X_current[X_current.T[split_feature] < split_treshold, :]
        y_new_1 = [yi for yi in y_current[X_current.T[split_feature] < split_treshold]]

        X_new_2 = X_current[X_current.T[split_feature] >= split_treshold, :]
        y_new_2 = [yi for yi in y_current[X_current.T[split_feature] >= split_treshold]]



        # False mean y_current X_current.T[split_feature] < split_treshold
        current_constraints_1 = current_constraints.copy()
        current_constraints_1.append((split_feature, split_treshold, False))

        current_constraints_2 = current_constraints.copy()
        current_constraints_2.append((split_feature, split_treshold, True))

   

        output.extend(self.split(X_current=X_new_1,y_current=y_new_1,cols= cols,current_constraints=current_constraints_1))
        output.extend(self.split(X_current=X_new_2,y_current=y_new_2,cols=cols, current_constraints=current_constraints_2))
   
        # Recuresively, the True and False are used to convey the direction of the inequlity (False means split_feature < split_treshold)
        return output
    
    def build(self, X, y):
            """Builds the TreeModel from the inputed data"""
            X = np.array(X)
            y = np.array(y)

            # Set the random seed for replicability
            

            # Select the appropriate columns (needed for RF)
            cols = self.get_candidate_columns(X, self.rand)
       

            #X = X[:, cols]
   
            # Calculate the splits (split feature, split treshold, </>=)
            split_features_and_splits = self.split(X,y, cols)

            return TreeModel(split_features_and_splits)  # return an object that can do prediction


class TreeModel:

    def __init__(self, split_features_and_splits):
        self.split_features_and_splits = split_features_and_splits

    def predict(self, X):
        y_preds = []
        bol = True
        # Go through all the rows
        for x_row in X:
            # Thorugh the array of specified splits and split classes
            for el in self.split_features_and_splits:

                # if this split was abandoned and we came to the end of it
                if (el ==1 or el == 0) and bol == False:
                    bol = True
                    continue

                # If we came to the end of the split sequence add as prediction
                if el ==1 or el == 0 and bol:
                    y_preds.append(el)
                    break

                # If one of the decisions isn't correct than it isn't the correct split sequence
                if el[2] != (x_row[el[0]] >= el[1]):
                    bol = False

        return y_preds


class RandomForest:

    def __init__(self, rand=None, n=100):
        self.n = n
        self.rand : random.Random = rand
        self.rftree = Tree(min_samples=2, rand=rand, get_candidate_columns=random_sqrt_columns)  # initialize the tree properly

    def __get_bs_sample(self, X,y):
        X = np.array(X)
        rows = self.rand.sample(range(len(y)), len(y))
        X_sample = X[rows, :] 
        y_sample = y[rows]
        return X_sample, y_sample

    def build(self, X, y):

        forest = []
        for i in range(self.n):
            
            X_sample, y_sample = self.__get_bs_sample(X, y)
            
            tree_model = self.rftree.build(X_sample, y_sample)

            forest.append(tree_model)

            
          

        return RFModel(forest)  # return an object that can do prediction


class RFModel:

    def __init__(self, forest):
        self.forest = forest

    def predict(self, X):
        #use majority vote for prediction
        predictions = []

        summation_of_predictions = np.zeros(shape=len(X))

        for tree in self.forest:
            summation_of_predictions += tree.predict(X)
            

        summation_of_predictions /= len(self.forest)
        predictions = np.round(summation_of_predictions)

        return predictions

    def importance(self):
        imps = np.zeros(self.X.shape[1])
        # ...
        return imps



def hw_tree_full(learn, test):
    # Split the data
    X_learn, y_learn = learn
    X_test, y_test = test

    # Initizalize the model 
    tree = Tree(min_samples=2, rand=random.Random(42))
    # build the model
    tree_model = tree.build(X_learn, y_learn)
    # Inference
    y_pred_train = tree_model.predict(X_learn)
    y_pred = tree_model.predict(X_test)

    # Calculate train misclass
    missclass_train = sum(abs(y_pred_train - y_learn)) / len(y_learn)
    SE_train = np.sqrt(missclass_train * (1- missclass_train)/len(y_learn))

    # Calculate missclassification rate
    missclass = sum(abs(y_pred - y_test)) / len(y_test)

    # Standard error
    SE = np.sqrt(missclass * (1- missclass)/len(y_test))

    return (missclass_train, SE_train,missclass, SE)


def hw_randomforests(learn, test):
    # Split the data
    X_learn, y_learn = learn
    X_test, y_test = test

    # Initialize the model
    rf = RandomForest(rand=random.Random(42),n = 100)

    rf_model = rf.build(X_learn, y_learn)
    # Inference
    y_pred_train = rf_model.predict(X_learn)
    y_pred = rf_model.predict(X_test)

    # Calculate train misclass
    missclass_train = sum(abs(y_pred_train - y_learn)) / len(y_learn)
    SE_train = np.sqrt(missclass_train * (1- missclass_train)/len(y_learn))

    # Calculate missclassification rate
    missclass = sum(abs(y_pred - y_test)) / len(y_test)

    # Standard error
    SE = np.sqrt(missclass * (1- missclass)/len(y_test))

    return (missclass_train, SE_train, missclass, SE)


if __name__ == "__main__":
    learn, test, legend = tki()

    misclass_train, SE_train,misclass, SE = hw_tree_full(learn, test)
    print(f"Tree train set missclassification: {misclass_train} +/- {SE_train}") 
    print(f"Tree test set missclassification: {misclass} +/- {SE}")

    misclass_train, SE_train,misclass, SE=hw_randomforests(learn, test)
    print(f"Random forest train set missclassification: {misclass_train} +/- {SE_train}") 
    print(f"Random forest test set missclassification: {misclass} +/- {SE}")



import unittest

class MyTests(unittest.TestCase):
    def test_tree(self):
        X = np.array([[1,2,31,2],
              [3,122,1,7],
              [43,2,5,5],
              [4,5,3,555],
              [32,3,23,2]])

        y = [1,0,1, 1,0]
        tree = Tree()
        self.assertListEqual(tree.build(X,y).predict(X), y)

