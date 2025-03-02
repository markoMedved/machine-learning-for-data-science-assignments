import csv
import numpy as np
import random
import unittest
import matplotlib.pyplot as plt
from timeit import default_timer as timer


#TODO look at the notes
# TODO make tests

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

    def __gini(self, probs):
        """calculates the gini impurity"""
        gini = 1
        for prob in probs:
            gini -= prob**2
        return gini

    def split(self, X_current, y_current,  current_constraints = []):
        """Define the split on the currently evaluated space, and the recursively for all the subspaces"""
        # Convert tu numpy array 
        X_current = np.array(X_current)
        y_current = np.array(y_current)

        # Select the columns for this split
        cols = self.get_candidate_columns(X_current, self.rand)
       
        # output will be of the form [first reached leaf conditions, first leaf class, second leaf conditions, second leaf class ....]
        # condition is of the form (first var, first var tresh, boolean for </>=)
        output = []
        
        # if somehow the length is zero assign the zero class (no reason just on of the classes), this shouldn't really happen
        if len(list(y_current)) == 0:
            print("empty y")
            #print(current_constraints)
            current_constraints.append(0)
            return current_constraints

        # If all the feature vectors are the same, return the majority class
        tmp = False
        for i  in range(1,len(X_current)):     
            if ((np.array_equal(X_current[i],X_current[i-1])) and y_current[i] != y_current[i-1] ):
                tmp = True
            else:
                tmp = False
                break
        if tmp:
            values, counts = np.unique(y_current, return_counts=True)
            most_frequent = values[np.argmax(counts)]
            current_constraints.extend([most_frequent])
            return current_constraints
            

        # Stop if the length reaches 2 or if we only have the same class in the split
        unq = np.unique(y_current)
        if len(unq) == 1 or len(y_current) < self.min_samples:
            
            #get the class with the higher count (not necessary in case of full tree)
            clss,cls_cnts = np.unique(y_current, return_counts=True)
            cls = clss[cls_cnts.argmax()]
            
            # Append the class to the output
            current_constraints.append(cls)
            return current_constraints

        # Initialize stuff
        lowest_cost = -1
        split_feature = 0
        split_treshold = 0

        length_y = len(y_current)

        # Go through all the features
        for i,x in enumerate(np.transpose(X_current)):
           
           # Only for the selected columns (when doing random forest)
            if i not in cols:
                continue
                
            # Initialize counters for the left and right split 
            cnt1l = 0
            cnt0l =  0
            
            cnt1r = np.count_nonzero(y_current)
            cnt0r = length_y - cnt1r

            # Sort the feature
            sort_indeces = np.argsort(x)
            x = x[sort_indeces]
            y_all = y_current[sort_indeces]
            
            # Go through all the data points
            for j,y in enumerate(y_all[:len(y_all)-1]):
                # Calculate the current treshold to look at
                x_at_y = (x[j] + x[j+1])/2
                
                # If 2 in a row are the same, you can't use this since it doesn't represent the actual splits correctly
                if j > 0 and x[j] == x[j+1]:
                    if y == 0:
                        cnt0r -= 1
                        cnt0l +=1
                    else:
                        cnt1r -= 1
                        cnt1l +=1
                    continue

                # The probabilities first split of the space
                prob1l = 1/2 if (cnt1l + cnt0l)==0 else cnt1l/ (cnt1l + cnt0l)
                prob0l = 1 - prob1l

                # Second split
                prob1r =1/2 if (cnt1r + cnt0r) == 0 else cnt1r /(cnt1r + cnt0r)
                prob0r = 1 - prob1r

                # Calculate the cost by summing both ginis and weighting them 
                cost = (cnt0r+cnt1r)/length_y * self.__gini([prob0r, prob1r]) + (cnt0l + cnt1l)/length_y * self.__gini([prob0l, prob1l])

                # Add/subtract to counters if the corresponding value was next
                if y == 0:
                    cnt0r -= 1
                    cnt0l +=1
                else:
                    cnt1r -= 1
                    cnt1l +=1


                # Update the cost, and the best split if lower that lowes_cost
                if lowest_cost == -1 or cost < lowest_cost:
                    lowest_cost = cost
                    # Update the split feature and the split treshold
                    split_feature = i
                    split_treshold = x_at_y


        # Define the splits made by the optimal feature and treshold combo
        X_new_1 = X_current[X_current.T[split_feature] < split_treshold, :]
        y_new_1 = [yi for yi in y_current[X_current.T[split_feature] < split_treshold]]

        X_new_2 = X_current[X_current.T[split_feature] >= split_treshold, :]
        y_new_2 = [yi for yi in y_current[X_current.T[split_feature] >= split_treshold]]

        # if one of the splits is empty just return
        if len(y_new_1) == 0 or len(y_new_2) == 0:
            values, counts = np.unique(y_current, return_counts=True)
            most_frequent = values[np.argmax(counts)]
            current_constraints.extend([most_frequent])
            return current_constraints



        # False mean y_current X_current.T[split_feature] < split_treshold
        current_constraints_1 = current_constraints.copy()
        current_constraints_1.append((split_feature, split_treshold, False))

        current_constraints_2 = current_constraints.copy()
        current_constraints_2.append((split_feature, split_treshold, True))

   
        # Recuresively extend the output with the conditions
        # The True and False are used to convey the direction of the inequlity (False means split_feature < split_treshold)
        output.extend(self.split(X_current=X_new_1,y_current=y_new_1,current_constraints=current_constraints_1))
        output.extend(self.split(X_current=X_new_2,y_current=y_new_2, current_constraints=current_constraints_2))
   
        return output
    


    
tree = Tree(random.Random(2))
X = [[4,0,10],
     [5,0,10],
     [5,9,10]]

y= [1,0,1]
tree_model = tree.build(X, y)
print(tree_model.split_features_and_splits)
print(tree_model.predict(X))