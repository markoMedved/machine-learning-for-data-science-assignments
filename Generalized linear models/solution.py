from sklearn.model_selection import train_test_split,StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler,OneHotEncoder
from sklearn.metrics import  accuracy_score, log_loss
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import norm
import unittest
import pandas as pd
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm





# TODO more tests, explanation of betas plot (also look into log-odds), for the last part try to understand the plots 

class MultinomialLogReg():
    def log_likelihood(self, beta):
        m = len(np.unique(self.y))
        # Get the latent strengths
        u = np.zeros(shape=(m, len(self.X)))
        # Unflatten the beta 
        beta = beta.reshape((m-1, self.X.shape[1]))
        # Multiply with the weights
        u[:m-1] = beta @ self.X.T 
        # Flatten the beta back
        beta = beta.flatten()
        # Reference class to 0s
        u[m-1] = np.zeros(len(self.X))
        # Transpose to get probabilites for each class one row
        u = u.T
        # Apply the softmax
        sum_accross_rows = np.sum(np.exp(u), axis=1).reshape(-1,1)
        sum_accross_rows = np.tile(sum_accross_rows, (1,u.shape[1]))
        probs = np.exp(u) / (sum_accross_rows)
        # Get tmatrix of targets to be able to just make the product
        y_mat = np.eye(m)[self.y]
        # Get the log likelihood (- to minimize)
        l = -np.sum(np.log(probs) * y_mat)
        return l

    def build(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        # Add a column for intercept
        self.X = np.c_[np.ones(self.X.shape[0]), self.X]
        m = len(np.unique(self.y))
        # Initialize Beta, (num of classes - 1) * (Num of feautures)
        beta = np.random.rand(m - 1, self.X.shape[1])
        # Get the result with the gradient descent
        result = fmin_l_bfgs_b(self.log_likelihood, beta, approx_grad=True,
                                maxiter=100000)[0]
        # Reshape the result
        result = result.reshape((m-1, self.X.shape[1]))

        return MultinomialLogRegInference(result)

class MultinomialLogRegInference():
    def __init__(self, beta):
        self.beta = beta

    def predict(self, X):
        X = np.array(X)
        X = np.c_[np.ones(X.shape[0]), X]
        m = self.beta.shape[0] + 1
        # Get the latent strengths
        u = np.zeros(shape=(m, len(X)))
        # Multiply with the weights
        u[:m-1] = self.beta @ X.T 
        # Reference class to 0s
        u[m-1] = np.zeros(len(X))
        # Transpose to get probabilites for each class one row
        u = u.T
        # Apply the softmax
        sum_accross_rows = np.sum(np.exp(u), axis=1).reshape(-1,1)
        sum_accross_rows = np.tile(sum_accross_rows, (1,u.shape[1]))
        probs = np.exp(u) / (sum_accross_rows)
        # return the probs
        return probs
    



def logistic_cdf(x):
    return (1 / (1 + np.exp(-x)))

class OrdinalLogReg():
    def log_likelihood(self, params):
        m = len(np.unique(self.y))
        # Get betas and tresholds
        beta = params[:self.X.shape[1]]
        deltas = params[self.X.shape[1]:]
        # Get the tresholds from deltas
        tresholds = np.concatenate((np.array([-np.inf, 0]), 
                            np.cumsum(deltas),
                            np.array([np.inf]) ))

        # Get the latent strengths
        u = beta @ self.X.T
        l = 0
        for i in range(len(self.y)):
            j = self.y[i]
            l -= np.log(logistic_cdf(tresholds[j + 1] - u[i]) - logistic_cdf(tresholds[j] - u[i]))
        return l

    def build(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        self.X = np.c_[np.ones(self.X.shape[0]), self.X]
        m = len(np.unique(self.y))
        # Initialize Beta, (Num of feautures)
        beta = np.random.rand(self.X.shape[1])
        # Initialize the deltas
        deltas = np.ones(m-2) 
        params = np.concatenate((beta, deltas))
        # Set the bounds, deltas have to be more than 0
        bounds = [(-np.inf, np.inf)] * self.X.shape[1] + [(1, np.inf)] * (m - 2)
        # Get the result with the gradient descent
        result = fmin_l_bfgs_b(self.log_likelihood,
                                params, approx_grad=True, bounds=bounds,
                                 maxiter=100000)[0]
        beta = result[:self.X.shape[1]]
        deltas = result[self.X.shape[1]:]

        return OrdinalLogRegInference(beta, deltas)


class OrdinalLogRegInference():
    def __init__(self, beta, deltas):
        self.beta = beta
        self.deltas = deltas

    def predict(self, X):
        X = np.array(X)
        X = np.c_[np.ones(X.shape[0]), X]
        m = len(self.deltas) + 2
        tresholds = np.concatenate((np.array([-np.inf, 0]), 
                            np.cumsum(self.deltas),
                            np.array([np.inf]) ))
        u = self.beta @ X.T
        pred = np.zeros(shape=(X.shape[0], m))
        for i in range(len(X)):
            for j in range(m): 
                pred[i, j] = logistic_cdf(tresholds[j + 1] - u[i]) - logistic_cdf(tresholds[j] - u[i])

        return pred


class MyTests(unittest.TestCase):
    def setUp(self):
        self.Xs = [np.array([[0, 0],
                           [0, 1],
                           [1, 0],
                           [1, 1],
                           [1, 1]]), 
                    np.array([[0,0,0,1],
                             [2,1,4,5],
                             [3,2,5,7]]),
                    np.array([[1,0],
                             [1,1],
                             [1,2],
                             [2,0],
                             [2,1],
                             [2,2],
                             [3,0],
                             [3,0],
                             [3,0]]),
                np.array([[0], [1], [2], [3]]),
                np.array([[-10],[0], [100]]),
                np.array([[3], [4], [1]])]
        
        self.ys = [np.array([0, 0, 1, 1, 2]),
                   np.array([0,1,2]),
                   np.array([0,0,0,1,1,1,1,1,1]),
                   np.array([0, 1, 2, 2]),
                   np.array([2,1,0]),
                   np.array([0,1,1])]
    
    def test_multinomial(self):
        print("\nTESTING MULTINOMIAL\n")
        for X,y in zip(self.Xs, self.ys):
            train = X[::2], y[::2]
            test = X[1::2], y[1::2]
            l = MultinomialLogReg()
            c = l.build(X, y)
            prob = c.predict(X)
            model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=100000)
            model.fit(X,y)
            preds_sklearn = model.predict_proba(X)
            print(prob)
            print(preds_sklearn)
            print(y)
            #self.assertEqual(prob.shape, (2, 3))
            self.assertTrue((prob <= 1).all())
            self.assertTrue((prob >= 0).all())
            np.testing.assert_almost_equal(prob.sum(axis=1), 1)

    def test_ordinal(self):
        print("\nTESTING ORDINAL\n")
        for X,y in zip(self.Xs, self.ys):
            train = X[::2], y[::2]
            test = X[1::2], y[1::2]
            l = OrdinalLogReg()
            c = l.build(X, y)
            prob = c.predict(X)
            print(y)
            print(prob)
            #self.assertEqual(prob.shape, (2, 3))
            self.assertTrue((prob <= 1).all())
            self.assertTrue((prob >= 0).all())
            np.testing.assert_almost_equal(prob.sum(axis=1), 1)



def log_loss_man(y_true, y_pred_proba, eps=1e-15):
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    return -np.log(y_pred_proba[y_true])


def multinomial_bad_ordinal_good(num_classes=5, num_data_points=500, num_features=5):
    class_len = num_data_points // num_classes
    num_data_points = num_classes * class_len
    targets = []
    rows = []
    for i in range(num_classes):
        for j in range(class_len):
            targets.append(i)
            rows.append(np.random.uniform(i - 0.5, i + 0.5, num_features) + 1.5*np.random.standard_normal(num_features))


    df = pd.DataFrame(data=rows)
    df["target"] = targets
    df= df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

class LinearRegression():
    def build(self, x,y):
        # Add intercept
        y = np.array(y).reshape(-1,1)
        x = np.array(x).reshape(-1,1)
        x = np.c_[np.ones_like(x),x]
        # Use the least-squares formula to get betas
        self.beta = (np.linalg.inv(x.T @ x) @ x.T) @ y
        self.hat = x @(np.linalg.inv(x.T @ x) @ x.T)
        
    def predict(self,x):
        x = np.array(x).reshape(-1,1)
        x = np.c_[np.ones_like(x),x]

        return (self.beta.T @ x.T).flatten()

if __name__ == "__main__":
    np.random.seed(42)
    unittest.main()

    
    ##################################################################
    # Part 1
    ##################################################################
    # Read the data
    df = pd.read_csv("dataset.csv", sep=";")
    df_og =df.copy()
    # Encode the data
    encoder_labs = LabelEncoder()
    df["ShotType"] = encoder_labs.fit_transform(df["ShotType"])
    encoder = OneHotEncoder(sparse_output=False, drop="first")
    for col in ["Competition", "PlayerType", "Movement"]:
        # Fit and transform the column (convert to dense matrix)
        encoded = encoder.fit_transform(df[[col]])
        # Create new column names for the encoded columns
        encoded_columns = encoder.get_feature_names_out([col])
        # Convert the encoded array to a DataFrame with the new column names
        encoded_df = pd.DataFrame(encoded, columns=encoded_columns)
        # Drop the original column and concatenate the new encoded columns
        df = pd.concat([df.drop(columns=[col]), encoded_df], axis=1)

    y = df["ShotType"].copy()
    y.value_counts(), df_og["ShotType"].value_counts()

    X = df.drop(columns = "ShotType")
    y = df["ShotType"]


    cor = X.corr()
    plt.figure(figsize=(16,14))
    mask = np.triu(np.abs(X.corr()))
    sns.heatmap(np.abs(X.corr()),mask=mask, cmap="viridis", annot=True)
    # Set the size of ticks
    plt.xticks(fontsize=15, rotation=45,ha="right")
    plt.yticks(fontsize=15)
    #plt.savefig("report/figures/corr.png")
    df = df.drop(columns=["TwoLegged", "PlayerType_F"])

    ##################################################################
    # Testing if the regressions work, with cross validation
    ##################################################################


    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    mulitnomial_model = MultinomialLogReg()
    ordinal_model = OrdinalLogReg()

    multinomial_log_losses = []
    ordinal_log_losses = []

    multinomial_accs = []
    ordinal_accs = []

    scaler = StandardScaler()

    for train_index, test_index in kf.split(X, y):
        # Split the data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Scaling the relevant features
        scaler.fit(X_train[["Angle", "Distance"]])
        X_train.loc[:, ["Angle", "Distance"]] = scaler.transform(X_train[["Angle", "Distance"]])
        X_test.loc[:, ["Angle", "Distance"]] = scaler.transform(X_test[["Angle", "Distance"]])

        # Train the multinomial model
        multinomial_model_trained = mulitnomial_model.build(X_train, y_train)
        
        # Predict the probabilities and classes for multinomial
        multinomial_preds = multinomial_model_trained.predict(X_test)

        # Train the ordinal model
        ordinal_model_trained = ordinal_model.build(X_train, y_train)

        # Predict the probabilities and classes for ordinal
        ordinal_preds = ordinal_model_trained.predict(X_test)


        # Loop through test set and calculate log loss and accuracy for each data point
        for i, test_point in enumerate(test_index):
            multinomial_log_losses.append(log_loss_man(y_test.iloc[i], multinomial_preds[i]))
            ordinal_log_losses.append(log_loss_man(y_test.iloc[i], ordinal_preds[i]))

            # Accuracy for multinomial and ordinal models
            multinomial_accs.append(accuracy_score(y_test.iloc[i:i+1], np.argmax(multinomial_preds[i:i+1], axis=1)))
            ordinal_accs.append(accuracy_score(y_test.iloc[i:i+1], np.argmax(ordinal_preds[i:i+1], axis=1)))

        # Calculate mean accuracy for multinomial model
    mean_multinomial_acc = np.mean(multinomial_accs)
    print(f"Mean accuracy for Multinomial Logistic Regression: {mean_multinomial_acc} +/- {np.std(multinomial_accs)/np.sqrt(len(multinomial_accs))}")

    # Calculate mean accuracy for ordinal model
    mean_ordinal_acc = np.mean(ordinal_accs)
    print(f"Mean accuracy for Ordinal Logistic Regression: {mean_ordinal_acc}+/- {np.std(ordinal_accs)/np.sqrt(len(multinomial_accs))}")

    # Calculate mean log loss for multinomial and ordinal models
    mean_multinomial_log_loss = np.mean(multinomial_log_losses)
    print(f"Mean log loss for Multinomial Logistic Regression: {mean_multinomial_log_loss}+/- {np.std(multinomial_log_losses)/np.sqrt(len(multinomial_accs))}")

    mean_ordinal_log_loss = np.mean(ordinal_log_losses)
    print(f"Mean log loss for Ordinal Logistic Regression: {mean_ordinal_log_loss}+/- {np.std(ordinal_log_losses)/np.sqrt(len(multinomial_accs))}")
    
    #############################################################
    # PART 2.1
    ############################################################
    np.random.seed(42)
    # Read the data
    df = pd.read_csv("dataset.csv", sep=";")
    df_og =df.copy()
    # Encode the data
    encoder_labs = LabelEncoder()
    df["ShotType"] = encoder_labs.fit_transform(df["ShotType"])
    encoder = OneHotEncoder(sparse_output=False, drop="first")
    for col in ["Competition", "PlayerType", "Movement"]:
        # Fit and transform the column (convert to dense matrix)
        encoded = encoder.fit_transform(df[[col]])
        # Create new column names for the encoded columns
        encoded_columns = encoder.get_feature_names_out([col])
        # Convert the encoded array to a DataFrame with the new column names
        encoded_df = pd.DataFrame(encoded, columns=encoded_columns)
        # Drop the original column and concatenate the new encoded columns
        df = pd.concat([df.drop(columns=[col]), encoded_df], axis=1)

    y = df["ShotType"].copy()
    y.value_counts(), df_og["ShotType"].value_counts()
    X = df.drop(columns = "ShotType")
    y = df["ShotType"]
    # Scaling the data
    scaler = StandardScaler()
    X.loc[:,["Angle", "Distance"]] = scaler.fit_transform(X.loc[:,["Angle", "Distance"]])
    X = X.drop(columns=["TwoLegged", "PlayerType_F"])
    m = 100
    betas = []
    model = MultinomialLogReg()
    for i in tqdm(range(m)):
        X_bs, y_bs = resample(X,y)
        model_pred = model.build(X_bs, y_bs)
        betas.append(model_pred.beta)
    betas_means = np.mean(betas, axis=0)
    betas_stds = np.std(betas, axis=0)

    # Save the results
    np.save("betas_means.npy", betas_means)
    np.save("betas_stds.npy", betas_stds)
    betas_means = np.load("betas_means.npy") 
    betas_stds = np.load("betas_stds.npy") 
    betas_means = np.vstack([betas_means ,np.zeros_like(betas_means[0]) ]) 
    betas_stds = np.vstack([betas_stds ,np.zeros_like(betas_means[0])]) 
    # Define the column names, adding 'intercept' at the beginning
    columns_with_intercept = ["intercept"] + list(X.columns) 
    
    # Create the DataFrame with the intercept column first
    df_betas_means = pd.DataFrame(columns=columns_with_intercept, data=betas_means)
    df_betas_stds = pd.DataFrame(columns=columns_with_intercept, data=betas_stds) 

    labs = [i for i in range(df["ShotType"].nunique())]
    shotTypes_col = encoder_labs.inverse_transform(labs) 
    # HERE you have to set the correct one to 0-s, and also have to shuffle the SE-s
    df_betas_means["ShotType"] = shotTypes_col 
    df_betas_stds["ShotType"] = shotTypes_col
    excluded_col = "ShotType" 
    df_exp = df_betas_means.copy()
    # Get the odds
    for col in df_betas_means.columns:
        if col != excluded_col:
            df_exp[col] = np.exp(df_betas_means[col])
    # PLOTING THE BETAS
    df_betas_filtered = df_betas_means[df_betas_means["ShotType"] != "tip-in"]
    df_stds_filtered = df_betas_stds[df_betas_stds["ShotType"] != "tip-in"]
    df_betas_filtered = df_betas_filtered.rename(columns={"intercept": "Intercept"})

    fig, axes = plt.subplots(2, 3, figsize=(14, 12))
    axes = axes.flatten()

    colors = ['lightblue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'yellow', 'lightgreen']

    for i in range(5):
        curr = df_betas_filtered.iloc[i, :11]
        ax = axes[i]
        se = df_stds_filtered.iloc[i, :11].values

        curr.plot(kind="bar", ax=ax, color=colors[:len(curr)], yerr=se)
        ax.axhline(y = 0, color="black", linestyle = "--", alpha=0.5, linewidth=0.5)
        ax.set_title(df_betas_filtered.iloc[i, 11], fontsize=16)
        ax.set_xticklabels([])  
        ax.set_ylim(-10,20)

    axes[5].axis('off')

    handles = [plt.Line2D([0], [0], color=color, lw=4) for color in colors[:11]]
    labels = df_betas_filtered.columns[:11]

    axes[5].legend(handles, labels, title="Features",title_fontsize=16, loc="center", fontsize=16)

    plt.tight_layout()
    #plt.savefig("report/figures/betas.png")

    ################################################################################
    # Part 2.2
    ###############################################################################
    accs_MN = []
    log_losses_MN = []
    accs_ORD = []
    log_losses_ORD = []

    np.random.seed(23)

    # Generate dataset
    df = multinomial_bad_ordinal_good(5, 200, 5)
    train, test = train_test_split(df, test_size=0.5, stratify=df["target"])
    X_train, y_train = train.drop(columns=["target"]), train["target"]
    X_test, y_test = test.drop(columns=["target"]), test["target"]

    # Multinomial 
    model = MultinomialLogReg()
    model_pred = model.build(X_train, y_train)
    pred_probs = model_pred.predict(X_test)
    pred_classes = np.argmax(pred_probs, axis=1)

    # Accuracy
    correct_preds_MN = (pred_classes == y_test.to_numpy()).astype(int)
    acc_mean_MN = correct_preds_MN.mean()
    acc_se_MN = correct_preds_MN.std(ddof=1) / np.sqrt(len(correct_preds_MN))

    # Log loss
    per_sample_losses_MN = -np.log(pred_probs[np.arange(len(y_test)), y_test])
    log_loss_mean_MN = np.mean(per_sample_losses_MN)
    log_loss_se_MN = np.std(per_sample_losses_MN, ddof=1) / np.sqrt(len(y_test))

    # Ordinal 
    model = OrdinalLogReg()
    model_pred = model.build(X_train, y_train)
    pred_probs_ord = model_pred.predict(X_test)
    pred_classes_ord = np.argmax(pred_probs_ord, axis=1)

    # Accuracy
    correct_preds_ORD = (pred_classes_ord == y_test.to_numpy()).astype(int)
    acc_mean_ORD = correct_preds_ORD.mean()
    acc_se_ORD = correct_preds_ORD.std(ddof=1) / np.sqrt(len(correct_preds_ORD))

    # Log loss
    per_sample_losses_ORD = -np.log(pred_probs_ord[np.arange(len(y_test)), y_test])
    log_loss_mean_ORD = np.mean(per_sample_losses_ORD)
    log_loss_se_ORD = np.std(per_sample_losses_ORD, ddof=1) / np.sqrt(len(y_test))

    # Output 
    print(f"Multinomial Accuracy: {acc_mean_MN:.3f} ± {acc_se_MN:.3f}")
    print(f"Multinomial Log Loss: {log_loss_mean_MN:.3f} ± {log_loss_se_MN:.3f}")
    print(f"Ordinal Accuracy:     {acc_mean_ORD:.3f} ± {acc_se_ORD:.3f}")
    print(f"Ordinal Log Loss:     {log_loss_mean_ORD:.3f} ± {log_loss_se_ORD:.3f}")

    #####################################################################
    # Part 3
    #####################################################################
    ## Plot the regression
    # Get the x ans y 
    df = pd.read_csv("dataset.csv", sep=";")
    y = df["Distance"].values
    x = df["Angle"].values

    model = LinearRegression()
    model.build(x,y)
    preds = model.predict(x)

    plt.figure(figsize=(10,6))
    plt.plot(x, preds, color="red")
    plt.scatter(x,y)
    #plt.show()

    # Get the basic residuals
    response_residuals = y - preds
    # Then we need the pearson residuals and then the quantile residuals
    quantile_residuals = response_residuals  / np.std(response_residuals)
    quantile_residuals = sorted(quantile_residuals)
    N = len(quantile_residuals)
    probabilites = np.linspace(1/(N), N/(N+1), N)
    quantiles_theory = norm.ppf(probabilites)

    # Plot Q-Q plot
    plt.scatter(quantiles_theory, quantile_residuals, label="Quantile Residuals",
                color="white", s=14, edgecolors="k")
    plt.plot(quantiles_theory, quantiles_theory, 'r--', label="y=x(Reference)")
    plt.ylabel("Sample Quantiles")
    plt.xlabel("Theoretical quantiles")
    plt.legend()
    #plt.savefig("report/figures/qq.png")

    # Plot residuals vs. fitted values
    plt.figure(figsize=(8,6))
    plt.scatter(preds,quantile_residuals, color="white", edgecolors="black", s = 3)
    plt.axhline(0, color="grey", linestyle="--")
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    #plt.savefig("report/figures/res_vs_fitted.png")

    # Cooks distance plot
    # Get the hat matrix
    mse = np.mean((response_residuals)**2)
    diags = np.diag(model.hat)
    cook = response_residuals**2 / (2 * mse)  * diags /(1-diags)**2

    plt.figure(figsize=(10,8))
    plt.bar(range(len(cook)), cook)
    plt.ylabel("Cook's distance")
    plt.xlabel("Data points")
    #plt.savefig("report/figures/cook.png")

    # plot the distribution of shot lengths
    plt.figure(figsize=(10, 6)) 
    plt.hist(y, bins=100, color='skyblue', edgecolor='black', alpha=0.7, density=True) 


    plt.xlabel('Distance', fontsize=14)
    plt.ylabel('Probability', fontsize=14)

    plt.grid(True, linestyle='--', alpha=0.7)
    #plt.savefig("report/figures/distr.png")