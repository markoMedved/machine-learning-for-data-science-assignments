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
    
    def __seperate_into_lists(self,lst):
        indices_of_numbers = [i for i,x in enumerate(lst) if isinstance(x, (np.int64,np.float64))]
        tmp_lst = []
        previous_j = 0
        for i,j in enumerate(indices_of_numbers):
            tmp_lst.append(lst[previous_j:j+1])
            previous_j = j+1

        return tmp_lst
    
    def build(self, X, y):
            """Builds the TreeModel from the inputed data"""
            X = np.array(X)
            y = np.array(y)
       
            # Calculate the splits (split feature, split treshold, </>=), and classes for each split
            split_features_and_splits = self.split(X,y)

            split_features_and_splits = self.__seperate_into_lists(split_features_and_splits)

            return TreeModel(split_features_and_splits)  # Return a tree that can do prediction


class TreeModel:

    def __init__(self, split_features_and_splits):
        self.split_features_and_splits = split_features_and_splits

    def predict(self, X):
        y_preds = []
        # Go through all the rows

        for x_row in X:
            found = False
            # Thorugh the array of specified splits and split classes
            for i,row in enumerate(self.split_features_and_splits):
                for el in row:
                    # If we came to the end of the split sequence add as prediction
                    if el ==1 or el == 0:
                        y_preds.append(el)
                        found = True
                        break
                    
                    # If one of the decisions isn't correct than it isn't the correct split sequence
                    if el[2] != (x_row[el[0]] >= el[1]):
                        break

                if found:
                    break

        return np.array(y_preds)


class RandomForest:

    def __init__(self, rand=None, n=100):
        self.n = n # Number of trees
        self.rand : random.Random = rand # Random generator
        # Use only the sqrt(n) number of features
        self.rftree = Tree(min_samples=2, rand=rand, get_candidate_columns=random_sqrt_columns) 

    def __get_bs_sample(self, X,y):
        """Gets one bootstrap sample from X and y"""
        X = np.array(X)
        # Randomly sample rows with replacement
        rows = self.rand.sample(range(len(y)), len(y))
        X_sample = X[rows, :] 
        y_sample = y[rows]
        return X_sample, y_sample

    def build(self, X, y):
        
        # Trees together = forest
        forest = []
        for i in range(self.n):
            # Get bootstrap sample
            X_sample, y_sample = self.__get_bs_sample(X, y)
            # Build the tree
            tree_model = self.rftree.build(X_sample, y_sample)
            # Append tree to the forest
            forest.append(tree_model)

        return RFModel(forest, X, y)  # Return the prediction model, with all the built trees


class RFModel:

    def __init__(self, forest, X, y):
        self.forest = forest
        self.X = X
        self.y = y

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

    def importance(self):
        len_forest = len(self.forest)
        missclasses_diffs = []
        for i,x_row in enumerate(self.X):
            print(i)
            missclass_current_var_diff = 0
            for tree in self.forest:
                # Permute the feature
                x_row_shuffled = random.shuffle(x_row)
                missclass_original,_ = missclass_fn(self.y,tree.predict(self.X))
                X_shuffled = self.X
                X_shuffled[i] = x_row_shuffled
                missclass_shuffled,_ = missclass_fn(self.y, tree.predict(X_shuffled))
                missclass_current_var_diff += missclass_shuffled - missclass_original

            missclasses_diffs.append(missclass_current_var_diff/len_forest)
            
        return missclasses_diffs

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

    # Variable importance 
    #imp = rf_model.importance()
    #plt.plot(imp)

    return (missclass_train, SE_train),(missclass, SE)

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
    """Calculate missclassification for different number of trees"""
    x_axis = range(n_start, n_stop+1)
    plt.plot(x_axis, missclass_train, label = "Train missclassification")

    plt.plot(x_axis, missclass_test, label = "Test missclassification")
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
    # n_start,n_stop = 1,50
    # missclassifications_train,SEs_train, missclassifications_test,SEs_test = missclass_rates_for_number_of_trees(learn, test,n_start, n_stop)
    # plot_missclass_vs_num_of_trees(missclassifications_test,SEs_test,  missclassifications_train,SEs_train,n_start, n_stop)

    #unittest.main()


