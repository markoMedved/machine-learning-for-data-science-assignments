# Libraries
library(dplyr)
library(nnet)
library(e1071) 
library(cluster)
library(ggplot2)

# TODO document the code, make up some meaningful plots
# TODO change so that you return the table of losses/accuracies for everything and then calculate the entire log loss/acc at the end, 
# using the bootstrap function (of course also including the standard error)

# Read the csv file to a data frame
df <- read.csv("dataset.csv", sep=";", header=TRUE)
# drop duplicated rows
df <- distinct(df)
# Shuffle df
df <- df[sample(nrow(df)), ]
X <- df[, -1]
y <- df$ShotType

# Define the baseline classifier
baseline_classifier <- function(y){
  bsln <- data.frame(table(y)/length(y[,1]))
  bsln <- as.data.frame(t(setNames(bsln$Freq, bsln$ShotType)))
  return(bsln)
}

# Bootstrap algorithm
bootstrap <- function(x, f, m = 1000, seed = 42){
  theta <- c()
  set.seed(seed)
  for(i in 1:m){ 
    FB <- sample(x, length(x), replace = TRUE)
    curr <- f(FB)
    theta <- c(theta, curr)
  }
  return (SE = sd(theta))
  
}

# Use stratified sampling, because the data set is imbalanced, and it's representative of the DGP
stratified_sampling <- function(df, k, target_col, seed = 42){
  set.seed(seed)
  # Split the data set by classes
  strata <- split(df, df[,target_col])
  strata_tmp <- strata
  
  # lists for storing all the samples
  folds <- list()

  # Implement stratified sampling
  for(i in  1:k){ 

    current_samp <- data.frame()
    for (j in seq_along(strata)) {
      
      len_smp <- as.integer(nrow(strata_tmp[[j]]) / k)

      rows <- strata[[j]][sample(nrow(strata[[j]]), len_smp, replace = FALSE), ]

      current_samp <- rbind(current_samp, rows)
      # Remove the sampled data
      strata[[j]] <- anti_join(strata[[j]], rows, by = colnames(df))
      
    }
    folds[[i]] <- current_samp
    # Shuffle 
    folds[[i]] <- folds[[i]][sample(nrow(folds[[i]])), ]
  }
  
  return(folds)
}

# Log loss function
log_loss <- function(pred_distr, targets, get_vec = FALSE) {
  epsilon <- 1e-15
  log_scores <- numeric(nrow(targets))
  
  if(get_vec){
    vec <- c()
    for(i in 1:nrow(targets)){
      prob <- pred_distr[i, ]
      vec <- c(vec, -log(max(as.numeric(prob[targets[i, 1]]), epsilon)))
    }
    return(vec)
  }
  
  for(i in 1:nrow(targets)){
    prob <- pred_distr[i, ]
    log_scores[i] <- -log(max(as.numeric(prob[targets[i, 1]]), epsilon))
  }
  
  loss <- mean(log_scores)
  SE = bootstrap(log_scores, mean)
  return(list(log_loss = loss, se = SE))
}

accuracy <- function(pred_distr, targets) {
  preds <- apply(pred_distr, 1, function(prob) names(prob)[which.max(prob)])
  corr_pred = preds == targets[, 1]
  acc <- mean(corr_pred)
  se <- bootstrap(corr_pred, mean)
  return(list(accurcy = acc, SE = se))
}

# k = ?
k = 5
target_col <- "ShotType"
folds <- stratified_sampling(df, k, target_col)


# Cross validation
bsln_accs <- c()
bsln_acc_ses <- c()
bsln_log_scores <- c()
bsln_log_ses<- c()
log_reg_accs<- c()
log_reg_acc_ses <- c()
log_reg_log_scores <- c()
log_reg_log_ses <- c()

log_losses_log_reg_all <- c()

for(i in seq_along(folds)){
  # Select the training and test folds
  test <- folds[[i]]
  train <- folds[-i]
  train <- bind_rows(train)

  # Get the targets
  targets <- test[target_col]
  

  ###### Baseline
  bsln <- baseline_classifier(train[target_col])

  # Repeat the distribution fo each target("Prediction")
  bsln_pred <- bsln[rep(1, nrow(targets)), ]

  # Get the log loss and accuracy
  log_loss_bsln <- log_loss(bsln_pred, targets)
  bsln_log_scores[i] <- log_loss_bsln$log_loss
  bsln_log_ses[i] <- log_loss_bsln$se
  
  acc_bsln <- accuracy(bsln_pred, targets)
  bsln_accs[i] <- acc_bsln$accurcy
  bsln_acc_ses[i] <- acc_bsln$SE

  ####### Logistic regression
  log_reg <- multinom(as.factor(train[[target_col]] ) ~ ., data = train[, colnames(train) != target_col])
  # Predictions
  log_reg_pred <- predict(log_reg, test, type="probs")
  
  # Acc 
  acc_log_reg <- accuracy(log_reg_pred, targets)
  log_reg_accs[i] <- acc_log_reg$accurcy
  log_reg_acc_ses[i] <- acc_log_reg$SE

  
  # Log score
  log_loss_log_reg <- log_loss(log_reg_pred, targets)
  log_reg_log_scores[i] <- log_loss_log_reg$log_loss
  log_reg_log_ses[i] <- log_loss_log_reg$se
  
  log_losses_log_reg_all <- c(log_losses_log_reg_all, log_loss(log_reg_pred, targets, TRUE))
   
}

print(bsln_log_scores)
print(bsln_accs)
print(log_reg_log_scores)
print(log_reg_accs)


######### SVM using linear kernel, tuning C
Cs <- c(1e-3,1e-2,1e-1,1) # ADD 10, 100

### Optimizing train fold performance

SVM_accs<- c()
SVM_acc_ses <- c()
SVM_log_scores <- c()
SVM_log_ses <- c()

log_losses_log_svm_all <- c()

for(i in seq_along(folds)){
  # Select the training and test folds
  test <- folds[[i]]
  train <- folds[-i]
  train <- bind_rows(train)
  
  # Get the targets
  targets <- test[target_col]
  # Also get the train targets here
  train_targets <- train[target_col]

  best_C <- 1e-3
  best_log_loss <- 1e10
  for (C in Cs){
    svm_model <- svm(as.factor(train[[target_col]] ) ~ .,
                     data = train[, colnames(train) != target_col]
                     ,kernel = "linear",
                     probability = TRUE,
                     cost = C)
    
    svm_pred <- predict(svm_model, train, probability = TRUE)
    svm_pred <- attr(svm_pred, "probabilities")
    
    # Using log-loss for evaluating as criteria for train-fold performance (strictly proper)
    log_loss_svm <- as.numeric(log_loss(svm_pred, train_targets)[1])
    if(log_loss_svm < best_log_loss){
      best_log_loss <- log_loss_svm
      best_C <- C
    }
  }
  # Eval on test
  svm_model <- svm(as.factor(train[[target_col]] ) ~ .,
                   data = train[, colnames(train) != target_col]
                   ,kernel = "linear",
                   probability = TRUE,
                   cost = best_C)
  
  svm_pred <- predict(svm_model, test, probability = TRUE)
  svm_pred <- attr(svm_pred, "probabilities")

  acc_svm <- accuracy(svm_pred, targets)
  SVM_accs[i] <- acc_svm$accurcy
  SVM_acc_ses[i] <- acc_svm$SE
    
  # Log score
  log_loss_svm <- log_loss(svm_pred, targets)
  SVM_log_scores[i] <- log_loss_svm$log_loss
  SVM_log_ses[i] <- log_loss_svm$se

  log_losses_log_svm_all <- c(log_losses_log_svm_all, log_loss(svm_pred, targets, TRUE))
  
}

print(SVM_accs)
print(SVM_log_scores)


### Nested Cross validation

SVM_nestedCV_accs<- c()
SVM_nestedCV_acc_ses <- c()
SVM_nestedCV_log_scores <- c()
SVM_nestedCV_log_ses <- c()

log_losses_log_svm_nested_all <- c()

for(i in seq_along(folds)){
  # Select the training and test folds
  test <- folds[[i]]
  train <- folds[-i]
  train <- bind_rows(train)
  
  # Get the targets
  targets <- test[target_col]
  # Also get the train targets here
  train_targets <- train[target_col]
  
  # k = ?
  k_nested = 5
  folds_nested <- stratified_sampling(train, k_nested, target_col)
  log_losses_C <- rep(0, length.out = length(Cs))
  
  for(m in seq_along(folds_nested)){
    # Select the nested training and test folds
    test_nested <- folds_nested[[m]]
    train_nested <- folds_nested[-m]
    train_nested <- bind_rows(train_nested)
    
    # Get the targets
    targets_nested <- test_nested[target_col]
    
    for (j in seq_along(Cs)){
      svm_model <- svm(as.factor(train_nested[[target_col]] ) ~ .,
                   data = train_nested[, colnames(train_nested) != target_col]
                   ,kernel = "linear",
                   probability = TRUE,
                   cost = Cs[j])
  
      # Test on the nested validation split of the data
      svm_pred <- predict(svm_model, test_nested, probability = TRUE)
      svm_pred <- attr(svm_pred, "probabilities")
      log_loss_svm <- log_loss(svm_pred,  targets_nested)$log_loss
      # Sum up the losses, at the end the lowest sum is picked
      log_losses_C[j] <- log_losses_C[j] + log_loss_svm
    }
    
  }
  # Select the C where the sum of log losses is minimal
  best_C <- Cs[which.min(log_losses_C)]
  print(best_C)
  # Eval on test
  svm_model <- svm(as.factor(train[[target_col]] ) ~ .,
                   data = train[, colnames(train) != target_col]
                   ,kernel = "linear",
                   probability = TRUE,
                   cost = best_C)
  
  svm_pred <- predict(svm_model, test, probability = TRUE)
  svm_pred <- attr(svm_pred, "probabilities")

  acc_svm <- accuracy(svm_pred, targets)
  SVM_nestedCV_accs[i] <- acc_svm$accurcy
  SVM_nestedCV_acc_ses[i] <- acc_svm$SE
  

  # Log score
  log_loss_svm <- log_loss(svm_pred, targets)
  SVM_nestedCV_log_scores[i] <- log_loss_svm$log_loss
  SVM_nestedCV_log_ses[i] <- log_loss_svm$se
  
  log_losses_log_svm_nested_all <- c(log_losses_log_svm_nested_all, log_loss(svm_pred, targets, TRUE))

}

print(SVM_accs)

print(SVM_log_scores)
print(SVM_nestedCV_log_scores)


################################### 
# PART 2
###################################
# Plot the scatter plot of log loss vs distance, and calculate correlation

# TODO folde morš dt nazaj v en dataframe da boš primerjavo delu (ne vzet df, ker je shufflan use)
all_data <- bind_rows(folds)
plot_data <- data.frame(
  distance = all_data$Distance,
  log_loss = log_losses_log_reg_all,
  log_loss_svm = log_losses_log_svm_all,
  log_loss_svm_nested = log_losses_log_svm_nested_all
)


ggplot(plot_data_long, aes(x = distance, y = LogLoss, color = Model)) +
  geom_point(alpha = 0.6, size = 2) +
  geom_smooth(method = "lm", linetype = "dashed", se = T, color="black") +
  facet_wrap(~ Model) +
  
  scale_color_manual(values = c(
    "Log-loss (LogReg)" = "#E41A1C",     # Red
    "Log-loss (SVM)" = "#377EB8",        # Blue
    "Log-loss (SVM Nested)" = "#4DAF4A"  # Green
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
    strip.text = element_text(face = "bold")
  )

## Computing linear dependance - correlation
cor(plot_data$distance, plot_data$log_loss)
cor(plot_data$distance, plot_data$log_loss_svm)
cor(plot_data$distance, plot_data$log_loss_svm_nested)

#### Probably makes sense that with distance our loss is lower since there is fewer possible shots to take

#################
# ESTIMATING PERFORMANCE WITH TRUE FREQUENCIES
################
# TODO same for accuracy, also add bootstraping here 

# Basically just use weigths on and calculate the new log_loss/accuracy
obs_freq <- table(all_data$Competition) / nrow(all_data)
true_freq <-  c("NBA" = 0.6, "EURO" = 0.1, "SLO1" = 0.1, "U14" = 0.1, "U16" = 0.1)

# Calculate the weight for each row
all_data$weight <- true_freq[all_data$Competition] / obs_freq[all_data$Competition]

# Weighted log-loss
weighted_log_loss_log_reg <- sum(log_losses_log_reg_all * all_data$weight) / sum(all_data$weight)
weighted_log_loss_svm <- sum(log_losses_log_svm_all* all_data$weight) / sum(all_data$weight)
weighted_log_loss_svm_nested <- sum(log_losses_log_svm_nested_all* all_data$weight) / sum(all_data$weight)

