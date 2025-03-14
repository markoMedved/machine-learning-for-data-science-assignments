################################### 
# PART 2
###################################
# Group the data by distance
# Create bins for distance
df$Distance_bin <- cut(df$Distance, breaks = quantile(df$Distance, probs = seq(0, 1, 0.25), na.rm = TRUE), include.lowest = TRUE)




