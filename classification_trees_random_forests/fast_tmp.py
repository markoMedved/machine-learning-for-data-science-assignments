import csv
import numpy as np
import random
import unittest
import matplotlib.pyplot as plt
from timeit import default_timer as timer

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
       
        # output will be of the form [first reached leaf conditions, first leaf class, second leaf conditions, second leaf class ....]
        # condition is of the form (first var, first var tresh, boolean for </>=)
        output = []
        
        # if somehow the length is zero assign the zero class (no reason just on of the classes), this shouldn't really happen
        if len(list(y_current)) == 0:
            return [0]

        # If all the feature vectors are the same, return the majority class
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

            # Sort the feature
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

                # The probabilities first split of the space
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

        self.used_features.add(split_feature)

        # Define the splits made by the optimal feature and treshold combo
        X_new_1 = X_current[X_current.T[split_feature] <= split_treshold, :]
        y_new_1 = [yi for yi in y_current[X_current.T[split_feature] < split_treshold]]

        X_new_2 = X_current[X_current.T[split_feature] > split_treshold, :]
        y_new_2 = [yi for yi in y_current[X_current.T[split_feature] > split_treshold]]

        # If a split was left empty, a value should be returned instead of causing an infinite loop
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
            l2 = current_list[1]
            # This is the tuple with the (split feature, split treshold, </>=)
            tup1 = l1[0]

            # tup1[2] is True or false (< or >=), tup1[0] = selected feature, tup1[1] = the treshold
            # Choose the correct next node
            if tup1[2] == (row[tup1[0]] > tup1[1]):
                current_list = l1[1]

            else:
                current_list = l2[1]
    
        return current_list[0]
            
    def predict(self, X):
        y_preds = []
        # Go through all the rows
        for x_row in X:
            y_preds.append(self.__get_pred(x_row, self.split_features_and_splits))

        return np.array(y_preds)


class RandomForest:

    def __init__(self, rand=None, n=100):
        self.n = n # Number of trees
        self.rand : random.Random = rand # Random generator
        # Use only the sqrt(n) number of features
        #self.rftree = Tree(min_samples=2, rand=rand, get_candidate_columns=random_sqrt_columns) 

    def __get_bs_sample(self, X,y):
        """Gets one bootstrap sample from X and y"""
        X = np.array(X)
        all_rows = range(len(y))
        # Randomly sample rows with replacement
        rows = self.rand.choices(all_rows, k=len(y))
        oob_rows = list(set(all_rows) - set(rows))

        X_sample = X[rows, :] 
        y_sample = y[rows]
        return X_sample, y_sample, oob_rows

    def build(self, X, y):
        
        # Trees together = forest
        forest = []
        oob_rows_for_each_tree = []
        for i in range(self.n):
            # Get bootstrap sample
            X_sample, y_sample, oob_rows = self.__get_bs_sample(X, y)
            oob_rows_for_each_tree.append(oob_rows)
            # Build the tree
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
        predictions = []
        
        # Sum up the prediction of all trees
        summation_of_predictions = np.zeros(shape=len(X))
        for tree in self.forest:
            summation_of_predictions += tree.predict(X)
            
        # Get the majority vote by normalizing with the length
        summation_of_predictions /= len(self.forest)
        predictions = np.round(summation_of_predictions)
        return predictions
    

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
        # Return unique ones
        return np.unique(features)


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
        for j, (tree, x_indeces) in enumerate(zip(self.forest,self.oob_rows_for_each_tree)):
            
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
                        acc_diffs[(feat_i, feat_j,feat_k)] += acc_original - acc_shuffled

        # Normalize with the amount of trees
        acc_diffs /= len_forest

        return acc_diffs
    



    


def missclass_fn(y, y_pred, bootstrap_m = 200, rand: random.Random = random.Random(42)):
    """Calculate missclassification and quantify uncertainty"""
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
    tree = Tree(min_samples=2, rand=random.Random(32))
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
    return np.array(imp), np.array(root_features, dtype=int)


def importance3_test(learn, n = 100):
    """Try the implemented importance"""

    # Split the data
    X_learn, y_learn = learn

    # Initialize the model
    rf = RandomForest(rand=random.Random(42),n = n)
    rf_model = rf.build(X_learn, y_learn)

    # Calculate the importances
    imp = rf_model.importance3()


def missclass_rates_for_number_of_trees(learn, test, n_start, n_stop):
    """Test the random forest for different numbers of trees"""
    # Try all different numbers of trees
    missclassifications_test = []
    SEs_test = []
    missclassifications_train = []
    SEs_train = []
    for n in range(n_start, n_stop+1):
        print(n)
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

    plt.plot(x_axis, missclass_test, label = "Test missclassification")
    # Also show the uncertainty
    plt.fill_between(x_axis, missclass_test - SEs_test, missclass_test + SEs_test, alpha = 0.3, color = "orange")

    plt.ylabel("Missclassification rate")
    plt.xlabel("Number of trees")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    learn, test, legend = tki()
    x,y = test
    print(x)



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