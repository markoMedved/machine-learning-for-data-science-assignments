########################################################################################################################################
# ERROR DISTANCE DEPENDANCE
########################################################################################################################################


# Get the entire dataset in the correct order of the folds
all_data <- bind_rows(folds)

# Data frame for the scatter plot
plot_data <- data.frame(
  distance = all_data$Distance,
  log_loss = log_losses_log_reg_all,
  log_loss_tree = log_losses_tree_all,
  log_loss_tree_nested = log_losses_tree_nested_all,
  log_loss_bsln = log_losses_bsln_all
)

# Reshape data for faceting
plot_data_long <- plot_data %>%
  tidyr::pivot_longer(
    cols = c(log_loss, log_loss_tree, log_loss_tree_nested, log_loss_bsln), # Added log_loss_bsln
    names_to = "Model",
    values_to = "LogLoss"
  ) %>%
  mutate(
    Model = factor(Model, levels = c("log_loss", "log_loss_tree", "log_loss_tree_nested", "log_loss_bsln"),
                   labels = c("Logistic regression", "Decision tree", "Decision tree (Nested CV)", "Baseline")) # Added "Baseline"
  )

ggplot(plot_data_long, aes(x = distance, y = LogLoss, color = Model)) +
  geom_point(alpha = 0.6, size = 3) + # Larger points
  geom_smooth(method = "lm", linetype = "dashed", se = FALSE, color = "black") +
  facet_wrap(~ Model, scales = "free_y") +   # Allow each facet to have its own y-axis scale
  
  scale_color_manual(values = c(
    "Logistic regression" = "#E41A1C",     
    "Decision tree" = "#377EB8",        
    "Decision tree (Nested CV)" = "#4DAF4A",
    "Baseline" = "#984EA3" # Added color for baseline
  )) +
  
  labs(
    x = "Distance",
    y = "Log-loss"
  ) +
  ylim(0,6)+
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 24), # Bigger title
    axis.title = element_text(face = "bold", size = 20),              # Bigger axis titles
    axis.text = element_text(size = 18),                               # Bigger axis labels
    strip.text = element_text(face = "bold", size = 18),              # Bigger facet labels
    legend.position = "none"                                           # Turn off the legend
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
print(bootstrap_cor(all_data$Distance, log_losses_tree_all))
print(bootstrap_cor(all_data$Distance, log_losses_tree_nested_all))
print(bootstrap_cor(all_data$Distance, log_losses_bsln_all))



# This trend makes sense because of the distribution of shot types with higher distance, visualization below

# Create consistent factor levels for all distributions
all_shot_types <- union(names(table(df$ShotType)), names(table(df[df$Distance > quantile(df$Distance, 0.75), ]$ShotType)))

# Low distance distribution (bottom 25% of distances)
dist_low <- table(factor(df[df$Distance < quantile(df$Distance, 0.25), ]$ShotType, levels = all_shot_types)) / nrow(df[df$Distance < quantile(df$Distance, 0.25), ])

# All data distribution
dist_all <- table(factor(df$ShotType, levels = all_shot_types)) / nrow(df)

# High distance distribution (top 25% of distances)
dist_high <- table(factor(df[df$Distance > quantile(df$Distance, 0.75), ]$ShotType, levels = all_shot_types)) / nrow(df[df$Distance > quantile(df$Distance, 0.75), ])
# Create the data frame
plot_data <- data.frame(
  ShotType = rep(all_shot_types, 3),
  Probability = c(as.numeric(dist_low), as.numeric(dist_all), as.numeric(dist_high)),
  Group = factor(rep(c("Low Distance", "All Data", "High Distance"), each = length(all_shot_types)),
                 levels = c("Low Distance", "All Data", "High Distance")) # Force the order here
)

# Plot
ggplot(plot_data, aes(x = ShotType, y = Probability, fill = Group)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  scale_fill_manual(values = c("Low Distance" = "#4DAF4A", "All Data" = "#377EB8", "High Distance" = "#E41A1C")) +
  labs(
    x = "Shot Type",
    y = "Probability"
  ) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 20),  # Larger title
    axis.title = element_text(face = "bold", size = 18),               # Larger axis titles
    axis.text = element_text(size = 18),                                # Larger axis labels
    legend.title = element_text(size = 18, face = "bold"),             # Larger legend title
    legend.text = element_text(size = 18)                              # Larger legend labels
  )


########################################################################################################################################
# ESTIMATING PERFORMANCE WITH TRUE DISTRIBUTION
########################################################################################################################################
all_data$log_losses_bsln_all <- log_losses_bsln_all
all_data$log_losses_log_reg_all <- log_losses_log_reg_all
all_data$log_losses_tree_all <- log_losses_tree_all
all_data$log_losses_tree_nested_all <- log_losses_tree_nested_all
all_data$accs_bsln_all <- accs_bsln_all
all_data$accs_log_reg_all <- accs_log_reg_all
all_data$accs_tree_all <- accs_tree_all
all_data$accs_tree_nested_all <- accs_tree_nested_all



# Implement bootstrap, but repeat differently for each group 
m_base <- 1000 
obs_freq <- table(all_data$Competition) / nrow(all_data)
true_freq <-  c("EURO" = 0.1,"NBA" = 0.6, "SLO1" = 0.1, "U14" = 0.1, "U16" = 0.1)

weights <- true_freq/obs_freq
probs <- weights/ sum(weights)
all_data$probs <- probs[all_data$Competition]

weighted_bootstrap <- function(x,col, f, m = 1000, seed = 42){
  theta <- c()
  set.seed(seed)
  for(i in 1:m){ 
    smp <- sample(x[, col], nrow(x), replace = TRUE, prob = x$probs)
    curr <- f(smp)
    theta <- c(theta, curr)
  }
  return (list(ESTIMATE = f(theta) , SE = sd(theta)))
}


print(weighted_bootstrap(all_data, "log_losses_bsln_all", mean))
print(weighted_bootstrap(all_data, "log_losses_log_reg_all", mean))
print(weighted_bootstrap(all_data, "log_losses_tree_all", mean))
print(weighted_bootstrap(all_data, "log_losses_tree_nested_all", mean))

print(weighted_bootstrap(all_data, "accs_bsln_all", mean))
print(weighted_bootstrap(all_data, "accs_log_reg_all", mean))
print(weighted_bootstrap(all_data, "accs_tree_all", mean))
print(weighted_bootstrap(all_data, "accs_tree_nested_all", mean))


##### PLOTING SHOT TYPE PROBABILITY FOR DIFFERENT COMPETITIONS

# Create consistent factor levels for all distributions
all_shot_types <- union(names(table(df$ShotType)), names(table(df$ShotType)))

# Distribution by competition type
dist_nba <- table(factor(df[df$Competition == "NBA", ]$ShotType, levels = all_shot_types)) / nrow(df[df$Competition == "NBA", ])
dist_euro <- table(factor(df[df$Competition == "EURO", ]$ShotType, levels = all_shot_types)) / nrow(df[df$Competition == "EURO", ])
dist_slo1 <- table(factor(df[df$Competition == "SLO1", ]$ShotType, levels = all_shot_types)) / nrow(df[df$Competition == "SLO1", ])
dist_u14 <- table(factor(df[df$Competition == "U14", ]$ShotType, levels = all_shot_types)) / nrow(df[df$Competition == "U14", ])
dist_u16 <- table(factor(df[df$Competition == "U16", ]$ShotType, levels = all_shot_types)) / nrow(df[df$Competition == "U16", ])

# Create the data frame
plot_data <- data.frame(
  ShotType = rep(all_shot_types, 5),
  Probability = c(as.numeric(dist_nba), as.numeric(dist_euro), as.numeric(dist_slo1), as.numeric(dist_u14), as.numeric(dist_u16)),
  Group = factor(rep(c("NBA", "EURO", "SLO1", "U14", "U16"), each = length(all_shot_types)),
                 levels = c("NBA", "EURO", "SLO1", "U14", "U16")) # Force the order here
)

# Plot
ggplot(plot_data, aes(x = ShotType, y = Probability, fill = Group)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  scale_fill_manual(values = c("NBA" = "#4DAF4A", "EURO" = "#377EB8", "SLO1" = "#E41A1C", "U14" = "#984EA3", "U16" = "#FF7F00")) +
  labs(
    x = "Shot Type",
    y = "Probability"
  ) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 20),  # Larger title
    axis.title = element_text(face = "bold", size = 18),               # Larger axis titles
    axis.text = element_text(size = 18),                                # Larger axis labels
    legend.title = element_text(size = 18, face = "bold"),             # Larger legend title
    legend.text = element_text(size = 18)                              # Larger legend labels
  )



######################
# USING ONLY NBA DATA
######################
nba_data <- all_data[all_data$Competition == "NBA", ]

print(bootstrap(nba_data$log_losses_bsln_all, mean))
print(bootstrap(nba_data$log_losses_log_reg_all, mean))
print(bootstrap(nba_data$log_losses_tree_all, mean))
print(bootstrap(nba_data$log_losses_tree_nested_all, mean))


print(bootstrap(nba_data$accs_bsln_all, mean))
print(bootstrap(nba_data$accs_log_reg_all, mean))
print(bootstrap(nba_data$accs_tree_all, mean))
print(bootstrap(nba_data$accs_tree_nested_all, mean))









