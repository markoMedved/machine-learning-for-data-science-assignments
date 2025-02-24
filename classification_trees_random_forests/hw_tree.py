import csv
import numpy as np


def all_columns(X, rand):
    return range(X.shape[1])


def random_sqrt_columns(X, rand):
    c = 3  # select random columns TODO
    return c


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

    def __split(self, X_current, y_current):
        """Define the split on the currently evaluated space, and the recursively for all the subspaces"""

        if len(y_current) < self.min_samples:
            return 

        # Initialize stuff
        lowest_cost = -1
        split_feature = 0
        split_treshold = 0

        # Go through all the features
        for i,x in enumerate(np.transpose(X_current)):
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
                if lowest_cost == -1 or cost < lowest_cost:
                    lowest_cost = cost
                    # Update the split feature and the split treshold
                    split_feature = i
                    split_treshold = x_at_y
        
        # Define the splits made by the optimal feature and treshold combo
        X_new_1 = [xi for xi in X_current[split_feature < split_treshold, :]]
        y_new_1 = [yi for yi in y_current[split_feature < split_treshold]]
        X_new_2 = [xi for xi in X_current[split_feature >= split_treshold, :]]
        y_new_2 = [yi for yi in y_current[split_feature >= split_treshold]]

        # Return the found out split feature and treshold and recursively repeat for the splits
        return (split_feature, split_treshold), self.__split(X_current=X_new_1,y_current=y_new_1), self.__split(X_current=X_new_2,y_current=y_new_2) 



    def build(self, X, y):
        """Builds the TreeModel from the inputed data"""
        X = np.array(X)
        y = np.array(y)

        split_features_and_splits = self.__split(X,y)

        return TreeModel(split_features_and_splits)  # return an object that can do prediction


class TreeModel:

    def __init__(self, split_features_and_splits):
        self.split_features_and_splits = split_features_and_splits

    def __find_closest_tresholds(self, x_row, split_features, split_treshs):
        """Finds the closest thresholds to locate the area that x_row is in, in the space"""
        # the output features and treshs
        split_features_closest = []
        split_treshs_closest = []
        for i, xi in enumerate(x_row):
            # Find the constraints on current feature
            current_smallest_diff = -1
            current_split_features_indices = split_features[split_features == i]
            # If the feature doesnt have a treshold
            if len(current_split_features_indices) == 0:
                break
            # Check differences with all 
            for j in current_split_features_indices:
                diff = xi - split_treshs[j]
                if diff < current_smallest_diff or current_smallest_diff == -1:
                    current_smallest_diff = diff
            split_features_closest
        
            

    def predict(self, X):
        y_preds = np.ones(len(X)) 
        for x_row in X:
            for (split_feature, split_tresh) in self.split_features_and_splits:
                y_pred = 1 if x_row[split_feature] > split_tresh else 0
                y_preds[i] = 
            
        return   


# class RandomForest:

#     def __init__(self, rand=None, n=50):
#         self.n = n
#         self.rand = rand
#         self.rftree = Tree(...)  # initialize the tree properly

#     def build(self, X, y):
#         # ...
#         return RFModel(...)  # return an object that can do prediction


# class RFModel:

#     def __init__(self, ...):
#         # ...

#     def predict(self, X):
#         # ...
#         return predictions

#     def importance(self):
#         imps = np.zeros(self.X.shape[1])
#         # ...
#         return imps


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


if __name__ == "__main__":
    learn, test, legend = tki()

    print("full", hw_tree_full(learn, test))
    #print("random forests", hw_randomforests(learn, test))
