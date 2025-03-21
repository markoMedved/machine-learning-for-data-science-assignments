# Libraries
library(dplyr)
library(nnet)
library(e1071) 
library(cluster)
library(ggplot2)
# TODO select k, look again into the hyperparameter selection, other stuff should be correct
# TODO Determine the appropriate encoding
# TODO maybe for the training fold optimization it is meant to split the training fold again
# Vprašanja: a je uredu da je v vsakmu foldu drug parameter, a je pr train fold optimization treba razdelit 
########################################################################################################################################
# DATA PREPARATION AND GENERAL FUNCTIONS
########################################################################################################################################

# Read the csv file to a data frame
df <- read.csv("dataset.csv", sep=";", header=TRUE)
# Drop duplicated rows
df <- distinct(df)

# Convert categorical variables to factors
df <- df %>%
  mutate(across(c("ShotType", "Competition", "PlayerType", "Transition", "TwoLegged", "Movement"), as.factor))

# # Scale numeric variables
df <- df %>%
  mutate(across(where(is.numeric), scale))

# Shuffle df
df <- df[sample(nrow(df)), ]


# Bootstrap algorithm
bootstrap <- function(x, f, m = 500, seed = 42){
  theta <- c()
  set.seed(seed)
  for(i in 1:m){ 
    FB <- sample(x, length(x), replace = TRUE)
    curr <- f(FB)
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
  epsilon <- 1e-15
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
degrees <- c(2,3,4,5,6, 8, 10)

# For storing all log losses and accuracies
log_losses_svm_all <- c()
accs_svm_all <- c()

for(i in seq_along(folds)){
  # Select the training and test folds
  test <- folds[[i]]
  train <- folds[-i]
  train <- bind_rows(train)

  # Get the targets
  targets <- test[target_col]
  # Also get the train targets here, for parameter optimization
  train_targets <- train[target_col]

  # Set the best C to the first val
  best_degree <- 2
  # Set the best log-loss really high
  best_log_loss <- 1e10
  # Go through all the possible costs
  for (degree in degrees){
    # Build the svm model, using the polynomial
    svm_model <- svm(train[[target_col]] ~ .,
                     data = train[, colnames(train) != target_col],
                     degree = degree,
                     kernel = "polynomial",
                     probability = TRUE)

    # Predict and test on the train set to find the best loss on the training set
    svm_pred <- predict(svm_model, train, probability = TRUE)
    svm_pred <- attr(svm_pred, "probabilities")

    # Using log-loss for evaluating as criteria for train-fold performance (strictly proper unlike accuracy)
    log_loss_svm <- as.numeric(log_loss(svm_pred, train_targets, FALSE)[1])
    if(log_loss_svm < best_log_loss){
      best_log_loss <- log_loss_svm
      best_degree <- degree
    }
  }
  print(best_degree)
  # Eval on the actual test set
  svm_model <- svm(train[[target_col]] ~ .,
                   data = train[, colnames(train) != target_col],
                   degree = best_degree,
                   kernel = "polynomial",
                   probability = TRUE)

  svm_pred <- predict(svm_model, test, probability = TRUE)
  svm_pred <- attr(svm_pred, "probabilities")

  acc_svm <- accuracy(svm_pred, targets)
  accs_svm_all <- c(accs_svm_all, acc_svm)


  log_loss_svm <- log_loss(svm_pred, targets)
  log_losses_svm_all <- c(log_losses_svm_all, log_loss_svm)

}

########################################################################################################################################
# TESTING SVM, TUNING COST WITH NESTED CROSS VALIDATION
########################################################################################################################################



log_losses_svm_nested_all <- c()
accs_svm_nested_all <- c()

for(i in seq_along(folds)){
  # Select the training and test folds
  test <- folds[[i]]
  train <- folds[-i]
  train <- bind_rows(train)
  
  # Get the targets
  targets <- test[target_col]
  
  # TODO decide on the inner k
  # k = ?
  k_nested = 5
  # Again use stratified sampling to get the inner folds
  folds_nested <- stratified_sampling(train, k_nested, target_col)
  # Vector for the log losses using all the different costs
  log_losses_C <- rep(0, length.out = length(degrees))
  
  for(m in seq_along(folds_nested)){
    # Select the nested training and test folds
    test_nested <- folds_nested[[m]]
    train_nested <- folds_nested[-m]
    train_nested <- bind_rows(train_nested)
    
    # Get the targets
    targets_nested <- test_nested[target_col]
    # Try all the different Cs for the inner cross-validation
    for (j in seq_along(degrees)){
      svm_model <- svm(train_nested[[target_col]] ~ .,
                       data = train_nested[, colnames(train_nested) != target_col],
                       degree = degrees[j],
                       kernel = "polynomial",
                       probability = TRUE,
                       gamma = 0.1,
                       cost = 10)
      
      # Test on the nested inner validation split of the data
      svm_pred <- predict(svm_model, test_nested, probability = TRUE)
      svm_pred <- attr(svm_pred, "probabilities")
      log_loss_svm <- log_loss(svm_pred,  targets_nested, FALSE)$log_loss
      # Sum up the losses along all the folds, at the end the lowest sum is picked for the best C
      log_losses_C[j] <- log_losses_C[j] + log_loss_svm
      # print(degrees[j])
      # print(log_loss_svm)
    }
    
  }
  # Select the C where the sum of log losses is minimal
  best_degree <- degrees[which.min(log_losses_C)]
  print(best_degree)
  # Eval on test
  svm_model <- svm(train_nested[[target_col]] ~ .,
                   data = train_nested[, colnames(train_nested) != target_col],
                   degree = best_degree,
                   kernel = "polynomial",
                   probability = TRUE,
                   gamma = 0.1,
                   cost = 10)
  
  svm_pred <- predict(svm_model, test, probability = TRUE)
  svm_pred <- attr(svm_pred, "probabilities")
  
  acc_svm <- accuracy(svm_pred, targets)
  accs_svm_nested_all <- c(accs_svm_nested_all, acc_svm)
  
  
  # Log score
  log_loss_svm <- log_loss(svm_pred, targets)
  log_losses_svm_nested_all <- c(log_losses_svm_nested_all, log_loss_svm)
  
}

########################################################################################################################################
# RESULTS
########################################################################################################################################
# Eval on the actual test set

# Log losses
print(bootstrap(log_losses_svm_all, mean))
print(bootstrap(log_losses_svm_nested_all, mean))

print(bootstrap(accs_svm_all, mean))
print(bootstrap(accs_svm_nested_all, mean))