# Libraries
library(dplyr)
library(nnet)
library(e1071) 
library(cluster)
library(ggplot2)

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
# BASELINE CLASSIFIER AND LOGISTIC REGRESSION CROSS VALIDATION
########################################################################################################################################

# Use a 10 fold CV, it should be enough - report - look at the paper
k = 10
target_col <- "ShotType"
folds <- stratified_sampling(df, k, target_col)

# Store all log_losses for linear regression
log_losses_log_reg_all <- c()
log_losses_bsln_all <- c()

# Stores TRUE for correct pred and FALSE for incorrect for all data points
accs_log_reg_all <- c()
accs_bsln_all <- c()

for(i in seq_along(folds)){
  # Select the training and test folds
  test <- folds[[i]]
  train <- folds[-i]
  # Change the train fold into a single dataframe
  train <- bind_rows(train)
  
  # Get the targets
  targets <- test[target_col]
  
  ######## Baseline
  bsln <- baseline_classifier(train[target_col])
  
  # Repeat the distribution fo each target("Prediction")
  bsln_pred <- bsln[rep(1, nrow(targets)), ]
  
  # Get the log loss and accuracy vecs for this fold and append them to the entire vector
  log_loss_bsln <- log_loss(bsln_pred, targets)
  log_losses_bsln_all <- c(log_losses_bsln_all, log_loss_bsln)
  
  acc_bsln <- accuracy(bsln_pred, targets)
  accs_bsln_all <- c(accs_bsln_all, acc_bsln)
  
  ####### Logistic regression
  log_reg <- multinom(as.factor(train[[target_col]] ) ~ ., data = train[, colnames(train) != target_col])
  # Predictions
  log_reg_pred <- predict(log_reg, test, type="probs")
  
  # Get the log loss and accuracy vecs for this fold and append them to the entire vector
  log_loss_log_reg <- log_loss(log_reg_pred, targets)
  log_losses_log_reg_all <- c(log_losses_log_reg_all, log_loss_log_reg)
  
  acc_log_reg <- accuracy(log_reg_pred, targets)
  accs_log_reg_all <- c(accs_log_reg_all, acc_log_reg)
  
  
}

########################################################################################################################################
# TESTING SVM, TUNING COST WITH TRAIN FOLD PERFORMANCE OPTIMIZATION
########################################################################################################################################

#Posible Degree of polynomial
degrees <- c(2,3,5,7, 10)

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
                     probability = TRUE,
                     gamma = 0.01,
                     cost = 1)
    
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
                   probability = TRUE,
                   gamma = 0.01,
                   cost = 1)
  
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
  
  # Inner k - slightly smaller
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
                       gamma = 0.01,
                       cost = 1)
      
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
                   gamma = 0.01,
                   cost = 1)
  
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

# Log losses
print(bootstrap(log_losses_bsln_all, mean))
print(bootstrap(log_losses_log_reg_all, mean))
print(bootstrap(log_losses_svm_all, mean))
print(bootstrap(log_losses_svm_nested_all, mean))

# Accuracy results 
print(bootstrap(accs_bsln_all, mean))
print(bootstrap(accs_log_reg_all, mean))
print(bootstrap(accs_svm_all, mean))
print(bootstrap(accs_svm_nested_all, mean))

########################################################################################################################################
# ERROR DISTANCE DEPENDANCE
########################################################################################################################################

# Here the baseline classifier will be excluded 
# Get the entire dataset in the correct order of the folds
all_data <- bind_rows(folds)
# Data frame for the scatter plot
plot_data <- data.frame(
  distance = all_data$Distance,
  log_loss = log_losses_log_reg_all,
  log_loss_svm = log_losses_svm_all,
  log_loss_svm_nested = log_losses_svm_nested_all
)

# Reshape data for faceting
plot_data_long <- plot_data %>%
  tidyr::pivot_longer(
    cols = c(log_loss, log_loss_svm, log_loss_svm_nested),
    names_to = "Model",
    values_to = "LogLoss"
  ) %>%
  mutate(
    Model = factor(Model, levels = c("log_loss", "log_loss_svm", "log_loss_svm_nested"),
                   labels = c("Log-loss (LogReg)", "Log-loss (SVM)", "Log-loss (SVM Nested)"))
  )

# Plot the scatter plot of log loss vs distance,
ggplot(plot_data_long, aes(x = distance, y = LogLoss, color = Model)) +
  geom_point(alpha = 0.6, size = 2) +
  geom_smooth(method = "lm", linetype = "dashed", se = F, color="black") +
  facet_wrap(~ Model) +
  
  scale_color_manual(values = c(
    "Log-loss (LogReg)" = "#E41A1C",     
    "Log-loss (SVM)" = "#377EB8",        
    "Log-loss (SVM Nested)" = "#4DAF4A"  
  )) +
  
  labs(
    title = "Relationship between Distance and Log-loss",
    x = "Distance",
    y = "Log-loss"
  ) +
  ylim(0,6)+
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.title = element_text(face = "bold"),
    strip.text = element_text(face = "bold"),
    legend.position = "none"#  Turn off the legend (unnecesary)
  )

# Bootstrap algorithm
bootstrap_cor <- function(x, y, m = 500, seed = 42){
  theta <- c()
  set.seed(seed)
  for(i in 1:m){ 
    index <- sample(length(x), length(x), replace = TRUE)
    curr <- cor(x[index], y[index])
    theta <- c(theta, curr)
  }
  return (list(ESTIMATE = cor(x,y) , SE = sd(theta)))
  
}

## Computing linear dependance - correlation, of distance and log-loss
print(bootstrap_cor(all_data$Distance, log_losses_log_reg_all))
print(bootstrap_cor(all_data$Distance, log_losses_svm_all))
print(bootstrap_cor(all_data$Distance, log_losses_svm_nested_all))



# This trend makes sense because of the distribution of shottypes with higher distance, visualization below

# Create consistent factor levels for both distributions
all_shot_types <- union(names(table(df$ShotType)), names(table(df[df$Distance > 1.5 * mean(df$Distance), ]$ShotType)))

dist_high <- table(factor(df[df$Distance > 1.5 * mean(df$Distance), ]$ShotType, levels = all_shot_types)) / nrow(df[df$Distance > 1.5 * mean(df$Distance), ])
dist_all <- table(factor(df$ShotType, levels = all_shot_types)) / nrow(df)

# Create the data frame
plot_data <- data.frame(
  ShotType = rep(all_shot_types, 2),
  Probability = c(as.numeric(dist_high), as.numeric(dist_all)),
  Group = rep(c("High Distance", "All Data"), each = length(all_shot_types))
)

# Plot
ggplot(plot_data, aes(x = ShotType, y = Probability, fill = Group)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  scale_fill_manual(values = c("High Distance" = "#E41A1C", "All Data" = "#377EB8")) +
  labs(
    title = "Comparison of Shot Type Distributions",
    x = "Shot Type",
    y = "Probability"
  ) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.title = element_text(face = "bold"),
    legend.title = element_blank()
  )


########################################################################################################################################
# ESTIMATING PERFORMANCE WITH TRUE DISTRIBUTION
########################################################################################################################################

# Basically just use weights on and calculate the new log_loss/accuracy
obs_freq <- table(all_data$Competition) / nrow(all_data)
true_freq <-  c("NBA" = 0.6, "EURO" = 0.1, "SLO1" = 0.1, "U14" = 0.1, "U16" = 0.1)

# Calculate the weight for each row
all_data$weight <- true_freq[all_data$Competition] / obs_freq[all_data$Competition]

# Weighted log-losses
print(bootstrap(log_losses_bsln_all * all_data$weight, mean))
print(bootstrap(log_losses_log_reg_all * all_data$weight, mean))
print(bootstrap(log_losses_svm_all * all_data$weight, mean))
print(bootstrap(log_losses_svm_nested_all * all_data$weight, mean))

# Weighted accuracies
print(bootstrap(accs_bsln_all * all_data$weight, mean))
print(bootstrap(accs_log_reg_all * all_data$weight, mean))
print(bootstrap(accs_svm_all * all_data$weight, mean))
print(bootstrap(accs_svm_nested_all * all_data$weight, mean))


