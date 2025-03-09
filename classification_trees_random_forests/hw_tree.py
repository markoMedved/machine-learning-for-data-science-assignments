import csv
import numpy as np
import random
import unittest
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import json
import re

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
        self.used_features = set()

    def __gini(self, probs):
        """calculates the gini impurity"""
        gini = 1
        for prob in probs:
            gini -= prob**2
        return gini

    def split(self, X_current, y_current):
        """Define the split on the currently evaluated space, and the recursively for all the subspaces"""
        # Convert tu numpy array 
        X_current = np.array(X_current)
        y_current = np.array(y_current)

        # Select the columns for this split
        cols = self.get_candidate_columns(X_current, self.rand)
       
        # output will be a nested list
        output = []
        
        # if somehow the length is zero assign the zero class (no reason just on of the classes), this shouldn't really happen
        if len(list(y_current)) == 0:
            return [0]

        # If all the feature vectors are the same, return the majority class, to prevent infinite recursions
        tmp = False
        for i  in range(1,len(X_current)):     
            if ((np.array_equal(X_current[i],X_current[i-1])) and y_current[i] != y_current[i-1]):
                tmp = True
            else:
                tmp = False
                break
        if tmp:
            values, counts = np.unique(y_current, return_counts=True)
            most_frequent = values[np.argmax(counts)]
            return [most_frequent]
            

        # Stop if the length reaches 2 or if we only have the same class in the split
        unq = np.unique(y_current)
        if len(unq) == 1 or len(y_current) < self.min_samples:
            
            #get the class with the higher count (not necessary in case of full tree)
            clss,cls_cnts = np.unique(y_current, return_counts=True)
            cls = clss[cls_cnts.argmax()]
            return [cls]

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

            # Sort the feature, and labels
            sort_indeces = np.argsort(x)
            x = x[sort_indeces]
            y_all = y_current[sort_indeces]
            
            # Go through all the data points
            for j,y in enumerate(y_all[:len(y_all)-1]):
                # Calculate the current treshold to look at
                x_at_y = (x[j] + x[j+1])/2

                # Continue if the next feature is the same since the cost will not be accurate in this case
                if j > 0 and x[j] == x[j+1]:
                    if y == 0:
                        cnt0r -= 1
                        cnt0l +=1
                    else:
                        cnt1r -= 1
                        cnt1l +=1
                    continue

                # The probabilities for the first split of the space
                prob1l = 1/2 if (cnt1l + cnt0l)==0 else cnt1l/ (cnt1l + cnt0l)
                prob0l = 1 - prob1l

                # Second space
                prob1r =1/2 if (cnt1r + cnt0r) == 0 else cnt1r /(cnt1r + cnt0r)
                prob0r = 1 - prob1r

                # Calculate the cost by summing both ginis and weighting them 
                cost = (cnt0r+cnt1r)/length_y * self.__gini([prob0r, prob1r]) + (cnt0l + cnt1l)/length_y * self.__gini([prob0l, prob1l])

                # Update the counters according to the label value that just passed through
                if y == 0:
                    cnt0r -= 1
                    cnt0l +=1
                else:
                    cnt1r -= 1
                    cnt1l +=1

                # Update the cost, and the best split if lower that lowes_cost (if the same take the first instance of that cost)
                if lowest_cost == -1 or cost < lowest_cost:
                    lowest_cost = cost
                    # Update the split feature and the split treshold
                    split_feature = i
                    split_treshold = x_at_y

        # Add feature to the used features (needed for importance)
        self.used_features.add(split_feature)

        # Define the splits made by the optimal feature and treshold combo
        X_new_1 = X_current[X_current.T[split_feature] <= split_treshold, :]
        y_new_1 = [yi for yi in y_current[X_current.T[split_feature] <= split_treshold]]

        X_new_2 = X_current[X_current.T[split_feature] > split_treshold, :]
        y_new_2 = [yi for yi in y_current[X_current.T[split_feature] > split_treshold]]

        # If a split was left empty, a prediction should be returned instead of causing an infinite loop
        if len(y_new_1) == 0 or len(y_new_2) == 0:
            values, counts = np.unique(y_current, return_counts=True)
            most_frequent = values[np.argmax(counts)]
            return [most_frequent]


        # Recuresively extend the output with the conditions
        # The True and False are used to convey the direction of the inequlity (False means split_feature < split_treshold)
        # The output is in the form of nested looops
        output.append([(split_feature, split_treshold, False),self.split(X_current=X_new_1,y_current=y_new_1)])
        output.append([(split_feature, split_treshold, True),self.split(X_current=X_new_2,y_current=y_new_2)])
   
        return output
    
    def build(self, X, y):
            """Builds the TreeModel from the inputed data"""
            X = np.array(X)
            y = np.array(y)
       
            # Calculate the splits (split feature, split treshold, </>=), and classes for each split
            split_features_and_splits = self.split(X,y)

            return TreeModel(split_features_and_splits, self.used_features)  # Return a tree that can do prediction


class TreeModel:

    def __init__(self, split_features_and_splits, used_features):
        self.split_features_and_splits = split_features_and_splits
        self.used_features = used_features # For faster importance

    def __get_pred(self, row, current_list):
        """Calculate the prediction for one row of the data"""

        while len(current_list) != 1:
            # Seperate into 2 lists for each child node
            l1 = current_list[0]
            # This is the tuple with the (split feature, split treshold, </>=)
            tup1 = l1[0]

            # tup1[2] is True or false (< or >=), tup1[0] = selected feature, tup1[1] = the treshold
            # Choose the correct next node
            if tup1[2] == (row[tup1[0]] > tup1[1]):
                current_list = l1[1]
            else:
                current_list = current_list[1][1]
    
        return current_list[0]
            
    def predict(self, X):
        y_preds = []
        # Go through all the rows
        for x_row in X:
            y_preds.append(self.__get_pred(x_row, self.split_features_and_splits))

        return y_preds


class RandomForest:

    def __init__(self, rand=None, n=100):
        self.n = n # Number of trees
        self.rand : random.Random = rand # Random generator

    def __get_bs_sample(self, X,y):
        """Gets one bootstrap sample from X and y"""
        X = np.array(X)
        all_rows = range(len(y))
        # Randomly sample rows with replacement
        rows = self.rand.choices(all_rows, k=len(y))
        # Select the oob rows
        oob_rows = list(set(all_rows) - set(rows))

        #Return the bootstrap samples, and out-of-the box data
        X_sample = X[rows, :] 
        y_sample = y[rows]
        return X_sample, y_sample, oob_rows

    def build(self, X, y):
        
        # Trees together = forest
        forest = []
        # Out of the bag samples for the forest
        oob_rows_for_each_tree = []
        for i in range(self.n):
            # Get bootstrap sample
            X_sample, y_sample, oob_rows = self.__get_bs_sample(X, y)
            oob_rows_for_each_tree.append(oob_rows)
            # Initialize and build the tree (have to initialize every time, so that the used features are unique for each tree)
            tree = Tree(min_samples=2, rand=self.rand, get_candidate_columns=random_sqrt_columns) 
            tree_model = tree.build(X_sample, y_sample)
            # Append tree to the forest
            forest.append(tree_model)

        return RFModel(forest, X, y, oob_rows_for_each_tree, self.rand)  # Return the prediction model, with all the built trees


class RFModel:

    def __init__(self, forest, X, y, oob_rows_for_each_tree, rand):
        self.forest = forest
        self.X = X
        self.y = y
        self.oob_rows_for_each_tree = oob_rows_for_each_tree # out of the bag indices for all trees
        self.rand = rand # Random generator

    def predict(self, X):
        # Use majority vote for prediction
        all_predictions = np.array([tree.predict(X) for tree in self.forest])
        majority_vote = np.mean(all_predictions, axis=0)    

        return np.round(majority_vote)
    

    def get_first_split_feature(self, X_current, y_current):
        # Same algorith as when building tree, however only return the first feature

        # Go through all the features
        length_y = len(y_current)
        # Initialize stuff
        lowest_cost = -1
        split_feature = 0

        for i,x in enumerate(np.transpose(X_current)):
           
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

                # Second space
                prob1r =1/2 if (cnt1r + cnt0r) == 0 else cnt1r /(cnt1r + cnt0r)
                prob0r = 1 - prob1r

                # Calculate the cost by summing both ginis and weighting them 
                cost = (cnt0r+cnt1r)/length_y * self.__gini([prob0r, prob1r]) + (cnt0l + cnt1l)/length_y * self.__gini([prob0l, prob1l])

                if y == 0:
                    cnt0r -= 1
                    cnt0l +=1
                else:
                    cnt1r -= 1
                    cnt1l +=1

                # Update the cost, and the best split if lower that lowes_cost (if the same take the first instance of that cost)
                if lowest_cost == -1 or cost < lowest_cost:
                    lowest_cost = cost
                    # Update the split feature and the split treshold
                    split_feature = i

        return int(split_feature)
    
    def get_bs_sample(self, X,y):
        """Gets one bootstrap sample from X and y"""
        # Needed for the root features calculation (Repeat)
        X = np.array(X)
        all_rows = range(len(y))
        # Randomly sample rows with replacement
        rows = self.rand.choices(all_rows, k=len(y))

        X_sample = X[rows, :] 
        y_sample = y[rows]
        return X_sample, y_sample
    
    def __gini(self, probs):
        """calculates the gini impurity"""
        # Needed for the root features calculation (Repeat)
        gini = 1
        for prob in probs:
            gini -= prob**2
        return gini
    
    def get_100_roots(self, X,y, n = 100):
        """Get the features that were root features in 100 non-random trees, on bootstraped data"""
        features = np.zeros(shape=(100))
        for i in range(n):
            X_current, y_current = self.get_bs_sample(X,y)
            features[i] = self.get_first_split_feature(X_current, y_current)
        return features


    def importance(self):
        """Calculate the importance of features for RF"""
        len_forest = len(self.forest)
        # Differences in accuracies between og and shuffled features
        acc_diffs  = np.zeros(shape=(len(self.X.T)))

        # Go through the entire forest, also all of the oobs for each tree
        for j, (tree, x_indeces) in enumerate(zip(self.forest,self.oob_rows_for_each_tree)):
            
            # Get only the oob data
            X = self.X[x_indeces]
            targets = self.y[x_indeces]

            # Predict on the original features
            pred_og = tree.predict(X)
            # Calculate acc for the original 
            acc_original = np.mean(pred_og == targets)

            # Through all of the features
            for i in tree.used_features:

                # Permute the feature
                X_shuffled = X.T.copy()
                self.rand.shuffle(X_shuffled[i])
                X_shuffled = X_shuffled.T

                # Calculate the acc with the permutted feature
                pred = tree.predict(X_shuffled)
                acc_shuffled= np.mean(pred == targets)

                # Calculate the difference in accuracy and add to the index for the feature
                acc_diffs[i] += acc_original - acc_shuffled

        # Normalize with the amount of trees
        acc_diffs /= len_forest
        return acc_diffs
    
    def importance3(self):
        len_forest = len(self.forest)
        # Differences in accuracies between og and shuffled features
        acc_diffs  = {}

        # Go through the entire forest, also all of the oobs for each tree
        for m, (tree, x_indeces) in enumerate(zip(self.forest,self.oob_rows_for_each_tree)):
            
            # Get only the oob data
            X = self.X[x_indeces]
            targets = self.y[x_indeces]

            # Predict on the original features
            pred_og = tree.predict(X)
            # Calculate acc for the original 
            acc_original = np.mean(pred_og == targets)

            # Through all of the features triples
            length_feats = len(tree.used_features)
            arr_used_features = np.array(list(tree.used_features))

            for i  in range(length_feats-2):
                for j in range(i+1,length_feats-1):
                    for k in range(j+1,length_feats):
                        feat_i = arr_used_features[i]
                        feat_j  = arr_used_features[j]
                        feat_k = arr_used_features[k]

                        # Permute the features
                        X_shuffled = X.T.copy()
                        self.rand.shuffle(X_shuffled[feat_i])
                        self.rand.shuffle(X_shuffled[feat_j])
                        self.rand.shuffle(X_shuffled[feat_k])
                        X_shuffled = X_shuffled.T

                        # Calculate accuracy with the permuted features
                        pred = tree.predict(X_shuffled)
                        acc_shuffled= np.mean(pred == targets)

                        # Calculate the difference in accuracy and add to the index for the feature
                        feature_lst = tuple(sorted([feat_i, feat_j,feat_k]))
                        
                        if str(feature_lst) in acc_diffs:
                            acc_diffs[str(feature_lst)] += (acc_original - acc_shuffled)/ len_forest

                        # If it doens't exist yet in the dict, add it
                        else:
                            acc_diffs[str(feature_lst)] = (acc_original - acc_shuffled)/ len_forest

        return acc_diffs
    
    def get_tree_importances(self, current_list,depth, n_features):
        """Importances for one tree using structure importance"""
        importances = np.zeros(n_features, dtype=np.float32)
        # If at the leaf return zeros
        if len(current_list) == 1:
            return importances

        # Separate into the 2 lists
        l1 = current_list[0]
        tup1 = l1[0]
        l2 = current_list[1]
        # Importance of a feature at a node is defined like this
        importances[int(tup1[0])] += 1.0/(2**depth)

        # Calculate recursively importances, go into both child nodes
        importances +=  self.get_tree_importances(l2[1], depth+1, n_features) + self.get_tree_importances(l1[1], depth+1, n_features)

        return importances
    
    def importance3_structure(self):
        # Number of features
        n_features = len(self.X.T)
        importances3 = np.zeros(shape=(n_features))
        importances3 = {}
        for tree in self.forest:
            importances = self.get_tree_importances(tree.split_features_and_splits, 1, n_features)
            # Get top 3 features in the forest and sum their importances
            top_3_in_the_tree = np.argsort(importances)[-3:][::-1]
            joint_importance = np.sum(np.sort(importances)[-3:])
        
            feat_i, feat_j,feat_k = top_3_in_the_tree[0], top_3_in_the_tree[1], top_3_in_the_tree[2]

            # Append those 3 features to the dict
            key = str(tuple(sorted([feat_i, feat_j,feat_k])))
            if key in importances3:
                importances3[key] += joint_importance
            else:
                importances3[key] = joint_importance

        return importances3
    

def missclass_fn(y, y_pred, bootstrap_m = 1000, rand: random.Random = random.Random(42)):
    """Calculate missclassification and quantify uncertainty, return missclassification and bootstraped standard error"""
    y_length = len(y)
    errors = np.array(y - y_pred)
    missclass = sum(abs(errors)) / y_length

    # Assuming asymptotic normality
    SE_as_norm = np.std(errors) / np.sqrt(y_length)
    
    # Using bootstrap
    missclass_vec = []
    for i in range(bootstrap_m):
        samp = rand.choices(population=errors, k=y_length)
        missclass_vec.append(np.sum(np.abs(samp)) / y_length)

    SE_bs = np.std(missclass_vec)

    # Return only the bootstrap standard error
    return missclass, SE_bs

def hw_tree_full(learn, test):
    """Test the full tree model"""
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

    # Calculate train misclass, and standard error
    missclass_train, SE_train = missclass_fn(y_learn, y_pred_train)

    # Calculate test misclass, and standard error
    missclass, SE = missclass_fn(y_test, y_pred)


    return (missclass_train, SE_train),(missclass, SE)

def hw_randomforests(learn, test, n = 100):
    """Test the random forest model"""
    # Split the data
    X_learn, y_learn = learn
    X_test, y_test = test

    # Initialize the model
    rf = RandomForest(rand=random.Random(42),n = n)

    rf_model = rf.build(X_learn, y_learn)
    # Inference
    y_pred_train = rf_model.predict(X_learn)
    y_pred = rf_model.predict(X_test)

    # Calculate train misclass, and standard error
    missclass_train, SE_train = missclass_fn(y_learn, y_pred_train)

    # Calculate test misclass, and standard error
    missclass, SE = missclass_fn(y_test, y_pred)

    return (missclass_train, SE_train),(missclass, SE)



def importance_test(learn, n = 100):
    """Try the implemented importance"""
    # Split the data
    X_learn, y_learn = learn

    # Initialize the model
    rf = RandomForest(rand=random.Random(42),n = n)
    rf_model = rf.build(X_learn, y_learn)

    # Calculate the importances
    start_time = timer()
    imp = rf_model.importance()
    stop_time = timer()
    print(f"Importance evaluation took: {stop_time- start_time} seconds")

    # Calculate the root features for 100 non-random trees
    root_features = rf_model.get_100_roots(X_learn, y_learn)
    return np.array(imp), np.array(root_features, dtype=np.int64)


def importance3_test(learn, n = 1000):
    """Try the implemented importance3"""

    # Split the data
    X_learn, y_learn = learn

    # Initialize the model
    rf = RandomForest(rand=random.Random(22),n = n)
    rf_model = rf.build(X_learn, y_learn)

    # Calculate the importances
    start_time = timer()
    imp = rf_model.importance3()
    stop_time = timer()
    print(f"Importance3 evaluation took: {stop_time- start_time} seconds")

    return imp

def importance3_structure_test(learn,n =100):
    # Split the data
    X_learn, y_learn = learn

    # Initialize the model
    rf = RandomForest(rand=random.Random(42),n = n)
    rf_model = rf.build(X_learn, y_learn)
    imp = rf_model.importance3_structure()
    return imp


def missclass_rates_for_number_of_trees(learn, test, n_start, n_stop):
    """Test the random forest for different numbers of trees"""
    # Try all different numbers of trees
    missclassifications_test = []
    SEs_test = []
    missclassifications_train = []
    SEs_train = []
    for n in range(n_start, n_stop+1):
        (missclass_train,SE_train), (missclass_test, SE_test) = hw_randomforests(learn,test,n)
        missclassifications_test.append(missclass_test)
        missclassifications_train.append(missclass_train)
        SEs_test.append(SE_test)
        SEs_train.append(SE_train)

    return np.array(missclassifications_train), np.array(SEs_train), np.array(missclassifications_test), np.array(SEs_test)
    
def plot_missclass_vs_num_of_trees(missclass_test,SEs_test, missclass_train, SEs_train,n_start, n_stop):
    """Plot missclassification for different number of trees"""

    x_axis = range(n_start, n_stop+1)
    plt.plot(x_axis, missclass_train, label = "Train missclassification")
    # Also show the uncertainty
    plt.fill_between(x_axis, missclass_train - SEs_train, missclass_train + SEs_train, alpha = 0.3)

    plt.plot(x_axis, missclass_test, label = "Test missclassification")
    # Also show the uncertainty
    plt.fill_between(x_axis, missclass_test - SEs_test, missclass_test + SEs_test, alpha = 0.3, color = "orange")

    plt.ylabel("Missclassification rate")
    plt.xlabel("Number of trees")
    plt.legend()
    plt.show()



def random_feature(X, rand):
    return [rand.choice(list(range(X.shape[1])))]


class MyTests(unittest.TestCase):
    # test from the other file
    def setUp(self):
        self.X = np.array([[0, 0],
                           [0, 1],
                           [1, 0],
                           [1, 1]])
        
        self.y = np.array([0, 0, 1, 1])
        self.train = self.X[:3], self.y[:3]
        self.test = self.X[3:], self.y[3:]

    # This one fails if the second feature is chosen(since it is a contradiction)
    def test_call_tree(self):
        t = Tree(rand=random.Random(24),
                 get_candidate_columns=random_feature,
                 min_samples=2)
        p = t.build(self.X, self.y)
        pred = p.predict(self.X)
        print(pred, self.y)

    def test_call_randomforest(self):
        rf = RandomForest(rand=random.Random(0),
                          n=20)
        p = rf.build(self.X, self.y)
        pred = p.predict(self.X)
        print(pred, self.y)

    # Constant numbers edge case (doensn't classifiy correctly since you cannot choose correctly here but it doesn't break)
    def test_constant_tree(self):
        X = np.array([[4,0,10],
            [4,0,10],
            [3,9,10]])

        y= np.array([1,0,1])
        tree = Tree()
        tree_model = tree.build(X,y)
        pred = tree_model.predict(X)
        print(pred, y)


    # some random numbers test
    def test_tree(self):
        X = np.array([[1,2,31,2],
              [3,122,1,7],
              [43,2,5,5],
              [4,5,3,555],
              [32,3,23,2]])

        y = np.array([1,0,1, 1,0])
        tree = Tree()
        tree_model = tree.build(X,y)
        pred = tree_model.predict(X)
        print(pred, y)

    def test_another_edge_case(self):
        X = [[4,0,10],
        [5,0,10],
        [5,9,10]]

        y= [1,0,1]
        tree = Tree()
        tree_model = tree.build(X,y)
        pred = tree_model.predict(X)
        print(pred, y)


        

if __name__ == "__main__":
    learn, test, legend = tki()

    # Test the full tree
    start_time = timer()
    (misclass_train, SE_train),(misclass, SE) = hw_tree_full(learn, test)
    stop_time = timer()
    print(f"Tree train set missclassification: {misclass_train} +/- {SE_train}") 
    print(f"Tree test set missclassification: {misclass} +/- {SE}")
    print(f"it took {stop_time - start_time} seconds")
    

    # Test the random forest
    start_time = timer()
    (misclass_train, SE_train),(misclass, SE)=hw_randomforests(learn, test)
    stop_time = timer()
    print(f"Random forest train set missclassification: {misclass_train} +/- {SE_train}") 
    print(f"Random forest test set missclassification: {misclass} +/- {SE}")
    print(f"it took {stop_time - start_time} seconds")

    # Plot missclassification vs number of trees
    n_start,n_stop = 1,100
    missclassifications_train,SEs_train, missclassifications_test,SEs_test = missclass_rates_for_number_of_trees(learn, test,n_start, n_stop)
    plot_missclass_vs_num_of_trees(missclassifications_test,SEs_test,  missclassifications_train,SEs_train,n_start, n_stop)

    # Variable importance 
    imp, root_features = importance_test(learn, 100)
    n_root_features = len(root_features)
    unique_root, counts_root = np.unique(root_features, return_counts=True)
    root_importances = counts_root


    # Plot Variable importance
    only_root_features = np.zeros(len(imp))
    x_axis = list(range(len(imp))) 
    x_axis = [x_ax *2+ 1000 for x_ax in x_axis] # Get the wavelengths

    only_root_features[unique_root] = root_importances
    fig, ax1 = plt.subplots()
    ax1.set_xlim(999,1791)
    ax1.set_ylabel("Permutation based importance",color='#1f77b4')
    ax1.tick_params(axis="y", labelcolor='#1f77b4')
    ax1.set_xlabel("Wavelength [nm]")
    ax1.bar(x_axis, imp,width = 2)


    ax2 = ax1.twinx() 
    ax2.bar(x_axis,only_root_features,alpha=0.5, width = 2, color="orange")
    ax2.set_ylim(-4.53)
    ax2.set_ylabel("Root importance", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")
    

    plt.show()

    # Importance3 test NOTE : THIS TAKES 15-20 MINUTES
    importances3 = importance3_test(learn, 1000)
    with open("jsons/importances3.json" , "w") as f:
        json.dump(importances3, f)

    with open("jsons/importances3.json", "r") as f:
       importances3 = json.load(f)

    # Get top 3 features from both variations
    top3_imp = list(np.argsort(imp)[-3:][::-1])
    top3_imp3 = max(importances3,key=importances3.get)
    print(sorted(importances3.values())[-3:])
   

    # Get the numbers from string
    top3_imp3 = [int(x) for x in re.findall(r'\d+', top3_imp3)][1::2]
    print(top3_imp, top3_imp3)

    # Testing performance of trees on the best 3 variables for singular importance
    X_learn, y_learn = learn
    X_learn = X_learn[:,top3_imp]
    learn_imp = X_learn, y_learn
    X_test, y_test = test
    X_test = X_test[:,top3_imp]
    test_imp = X_test, y_test
    (misclass_train, SE_train),(misclass, SE) = hw_tree_full(learn_imp, test_imp)
    print(f"Tree test set missclassification with top 3 features from importance: {misclass} +/- {SE}")

    # On top 3 variables with importance3()
    X_learn, y_learn = learn
    X_learn = X_learn[:,top3_imp3]
    learn_imp = X_learn, y_learn
    X_test, y_test = test
    X_test = X_test[:,top3_imp3]
    test_imp = X_test, y_test
    (misclass_train, SE_train),(misclass, SE) = hw_tree_full(learn_imp, test_imp)
    print(f"Tree test set missclassification with top 3 features from importance3: {misclass} +/- {SE}")

    # Structure importance, compare with normal importance3
    top5_keys = sorted(importances3, key=importances3.get, reverse=True)[:5]
    top5_keys_ints = []
    for el in top5_keys:
        top5_keys_ints.append([int(x) *2+1000 for x in re.findall(r'\d+', el)][1::2])

    print(top5_keys_ints)

    imp_structure = importance3_structure_test(learn, 1000)
    top5_keys = sorted(imp_structure, key=imp_structure.get, reverse=True)[:5]
    print(imp_structure[top5_keys[0]])
    top5_keys_ints = []
    for el in top5_keys:
        top5_keys_ints.append([int(x) *2+1000 for x in  re.findall(r'\d+', el)][1::2])
    print(top5_keys_ints)

    unittest.main()


