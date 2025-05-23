# Libraries
library(dplyr)
library(nnet)
library(e1071) 
library(cluster)
library(ggplot2)
library(rpart)

# report bias variance tradeoff
# different parameter in each fold is ok (evaluation of algorithm - so it's ok)

# TODO select k, look again into the hyperparameter selection, other stuff should be correct
# TODO why did you choose the model for the report

# Vprašanja: a je uredu da je v vsakmu foldu drug parameter, a je pr train fold optimization treba razdelit 

# explain what kind of k you choose, why did you choose this model
# leave one out - you go through all the possble models - alternative repetitions of cross validation

########################################################################################################################################
# DATA PREPARATION AND GENERAL FUNCTIONS
########################################################################################################################################

# Read the csv file to a data frame
df <- read.csv("dataset.csv", sep=";", header=TRUE)

# Convert categorical variables to factors
df <- df %>%
  mutate(across(c("ShotType", "Competition", "PlayerType", "Transition", "TwoLegged", "Movement"), as.factor))


# Drop duplicated rows
df <- distinct(df)
# Shuffle df
df <- df[sample(nrow(df)), ]

# Define the baseline classifier
baseline_classifier <- function(y){
  # Take the frequency table and divide it by the number of data points
  bsln <- data.frame(table(y)/length(y[,1]))
  # Change to data frame so that it matches the other models
  bsln <- as.data.frame(t(setNames(bsln$Freq, bsln$ShotType)))
  return(bsln)
}

# Bootstrap algorithm
bootstrap <- function(x, f, m = 500, seed = 42){
  theta <- c()
  set.seed(seed)
  for(i in 1:m){ 
    smp <- sample(x, length(x), replace = TRUE)
    curr <- f(smp)
    theta <- c(theta, curr)
  }
  return (list(ESTIMATE = f(x) , SE = sd(theta)))
  
}


########################################################################################################################################
# IMPLEMENTATION OF STRATIFIED SAMPLING INTO FOLDS
########################################################################################################################################

# Use stratified sampling, because the data set is imbalanced, and it's representative of the DGP
stratified_sampling <- function(df, k, target_col, seed = 42){
  set.seed(seed)
  # Split the data set by classes
  strata <- split(df, df[,target_col])
  # Use this for size uses, since strata var will be changing
  strata_tmp <- strata
  
  # List for storing all the folds of data
  folds <- list()
  
  # Implement stratified sampling
  for(i in  1:k){ 
    
    current_samp <- data.frame()
    # Go through all the classes
    for (j in seq_along(strata)) {
      # Set the length of the sample such that all samples will have equal amount of the class
      len_smp <- as.integer(nrow(strata_tmp[[j]]) / k)
      # Create the rows by sampling the strata for len_smp samples 
      rows <- strata[[j]][sample(nrow(strata[[j]]), len_smp, replace = FALSE), ]
      # Append the rows to the current samp which will at the end include samples from all strata
      current_samp <- rbind(current_samp, rows)
      # Remove the sampled data so that it doesn't repeat
      strata[[j]] <- anti_join(strata[[j]], rows, by = colnames(df))
      
    }
    # Assign the current_samp as one of the folds
    folds[[i]] <- current_samp
    # Shuffle 
    folds[[i]] <- folds[[i]][sample(nrow(folds[[i]])), ]
  }
  
  return(folds)
}

########################################################################################################################################
# FUNCTIONS FOR METRICS
########################################################################################################################################

# Log loss function
log_loss <- function(pred_distr, targets, get_vec = TRUE) {
  # For -inf prevention
  epsilon <- 1e-10
  # Construct a vector of log scores for all samples
  if(get_vec){
    vec <- c()
    # Through all the current data points
    for(i in 1:nrow(targets)){
      prob <- pred_distr[i, ]
      # Calculate the log loss of the current data point, if probability is too low, rather take the epsilon
      vec <- c(vec, -log(max(as.numeric(prob[targets[i, 1]]), epsilon)))
    }
    return(vec)
  }
  # This version is used for parameter optimization, it directly returns the averaged log loss 
  log_scores <- numeric(nrow(targets))
  for(i in 1:nrow(targets)){
    prob <- pred_distr[i, ]
    log_scores[i] <- -log(max(as.numeric(prob[targets[i, 1]]), epsilon))
  }
  loss <- mean(log_scores)
  return(list(log_loss = loss))
}

accuracy <- function(pred_distr, targets) {
  # Get the maximum value in the pred_distr as the prediction 
  preds <- apply(pred_distr, 1, function(prob) names(prob)[which.max(prob)])
  # Get the vector of correct predictions
  corr_pred = preds == targets[, 1]
  return(corr_pred)
  
}

########################################################################################################################################
# TESTING SVM, TUNING COST WITH TRAIN FOLD PERFORMANCE OPTIMIZATION
########################################################################################################################################

#Posible Degree of polynomial
depths <- c(2,3,5,10,15,20)

# For storing all log losses and accuracies
log_losses_tree_all <- c()
accs_tree_all <- c()

for(i in seq_along(folds)){
  # Select the training and test folds
  test <- folds[[i]]
  train <- folds[-i]
  train <- bind_rows(train)
  
  # Get the targets
  targets <- test[target_col]
  # Also get the train targets here, for parameter optimization
  train_targets <- train[target_col]
  
  # # Set the best C to the first val
  
  best_depth <- depths[1]
  best_log_loss <- 1e10
  
  
  # Go through all the possible costs
  for (depth in depths) {
    # Build the svm model, using the polynomial
    tree_model <- rpart(
      ShotType ~ .,
      data = train,
      control = rpart.control(maxdepth = depth,  cp = 0)
    )
    
    
    # Predict and test on the train set to find the best loss on the training set
    
    tree_pred <- predict(tree_model, train, type = "prob")
    
    # Using log-loss for evaluating as criteria for train-fold performance (strictly proper unlike accuracy)
    
    #print(mean(accuracy(tree_pred, train_targets)))
    #print(depth)
    log_loss_tree <- as.numeric(log_loss(tree_pred, train_targets, FALSE)[1])
    
    if (log_loss_tree < best_log_loss) {
      best_log_loss <- log_loss_tree
      best_depth <- depth
    }
  }
  print(best_depth)
  # Eval on the actual test set
  
  tree_model <- rpart(
    ShotType ~ .,
    data = train,
    control = rpart.control(maxdepth = best_depth, cp = 0)
  )
  
  tree_pred <- predict(tree_model, test, type = "prob")
  
  # Calculate accuracy and log-loss
  acc_tree <- accuracy(tree_pred, targets)
  accs_tree_all <- c(accs_tree_all, acc_tree)
  
  log_loss_tree <- log_loss(tree_pred, targets)
  log_losses_tree_all <- c(log_losses_tree_all, log_loss_tree)
  
}

################################
log_losses_tree_nested_all <- c()
accs_tree_nested_all <- c()

for(i in seq_along(folds)){
  # Select the nested training and test folds
  test_nested <- folds_nested[[m]]
  train_nested <- folds_nested[-m]
  train_nested <- bind_rows(train_nested)
  
  # Get the targets
  targets <- test[target_col]
  
  # TODO decide on the inner k
  # k = ?
  k_nested <- 5
  # Again use stratified sampling to get the inner folds
  folds_nested <- stratified_sampling(train, k_nested, target_col)
  # Vector for the log losses using all the different costs
  log_losses_depth <- rep(0, length.out = length(depths))
  
  for(m in seq_along(folds_nested)){
    # Select the nested training and test folds
    test_nested <- folds_nested[[m]]
    train_nested <- folds_nested[-m]
    train_nested <- bind_rows(train_nested)
    
    # Get the targets
    targets_nested <- test_nested[target_col]
    # Try all the different Cs for the inner cross-validation
    for (j in seq_along(depths)){
      tree_model <- rpart(
        ShotType ~ .,
        data = train_nested,
        control = rpart.control(maxdepth = depths[j],cp = 0)
        
      )
      
      # Predict on the nested test set
      tree_pred <- predict(tree_model, test_nested, type = "prob")
      
      # Compute log-loss
      log_loss_tree <- log_loss(tree_pred, targets_nested)
      log_losses_depth[j] <- log_losses_depth[j] + log_loss_tree
    }
    
  }
  # Select the depth with the lowest sum of log losses
  best_depth <- depths[which.min(log_losses_depth)]
  print(best_depth)
  
  # Train the final model on the full training set with the best depth
  tree_model <- rpart(
    ShotType ~ .,
    data = train,
    control = rpart.control(maxdepth = best_depth, cp = 0)
  )
  
  # Predict on the outer test set
  tree_pred <- predict(tree_model, test, type = "prob")
  
  # Compute accuracy and log-loss
  acc_tree <- accuracy(tree_pred, targets)
  accs_tree_nested_all <- c(accs_tree_nested_all, acc_tree)
  
  log_loss_tree <- log_loss(tree_pred, targets)
  log_losses_tree_nested_all <- c(log_losses_tree_nested_all, log_loss_tree)
}

print(mean(log_losses_tree_nested_all))
print(mean(log_losses_tree_all))
print(mean(accs_tree_all))
print(mean(accs_tree_nested_all))

