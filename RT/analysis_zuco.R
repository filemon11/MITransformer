library(Rmisc)
library(tidyverse)
library(stringr)
library(scales)
library(grid)
library(ggpubr)
library(MASS)
library(lme4)
library(stats)
library(modelr)
library(plotrix)
library(mgcv)
library(hexbin)
library(formattable)
library(MuMIn)
library(ggrepel)
library(data.table)
library(comprehenr)
library(ggplot2)
library(dplyr)

rm(list=ls())

args = commandArgs(trailingOnly=TRUE)
corpus <- args[4]

data_dir <- paste("data/", corpus, "_preprocessed_", sep="")
num.models = as.numeric(args[3])
# load data

for (x in 0:(num.models-1)) {
  n <- paste("d1", x, sep = ".")
  assign(n, read.csv(paste(data_dir, args[1], "_", x, ".csv", sep="")))
  n <- paste("d0", x, sep = ".")
  assign(n, read.csv(paste(data_dir, args[2], "_", x, ".csv", sep="")))
}


means <- aggregate(cbind(GPT, surprisal.s, head_distance, first_dependent_distance, left_dependents_distance_sum, left_dependents_count) ~ item + zone,
                   data = d1.0, FUN = mean, na.rm = TRUE)

means$head_distance[means$head_distance > 0] <- 0
means$first_dependent_distance[means$first_dependent_distance > 0] <- 0

# Pearson correlation between candidates for integration costs
cor.test(means$GPT, -means$head_distance, method="pearson")
cor.test(means$GPT, -means$first_dependent_distance, method="pearson")
cor.test(means$GPT, means$left_dependents_distance_sum, method="pearson")
cor.test(means$GPT, means$left_dependents_count, method="pearson")

# Compute a pearson like correlation by measuring the correlation
# between the log probs and the GPT averaged over the readers.
# Plot the data and check that the relation is approx linear.

for (x in 0:(num.models-1)) {
  means <- aggregate(cbind(GPT, surprisal.s) ~ item + zone,
                     data = get(paste("d1", x, sep = ".")), FUN = mean, na.rm = TRUE)

  c <- cor.test(means$GPT, means$surprisal.s, method="pearson")
  assign(paste("corr", x, sep="."), c)
  
  print(paste("Correlation", x))
  print(c$estimate)
  
  pdf(file=paste("results/correlation_", args[1], "_", x, ".pdf", sep=""))
  plot(means$surprisal.s, means$GPT, xlim=c(0,60))+
    abline(lm(means$GPT ~ means$surprisal.s))
  dev.off()
}
print("Mean correlation:")
values <- to_vec(for(i in 0:(num.models-1)) get(paste("corr", i, sep="."))$estimate )
print(mean(values))
print("Standard deviation:")
print(sd(values))


# test head_distance GPT correlation
# cor.test(means$GPT, means$head_distance, method="pearson")
# pdf(file="results/plot_head_distance.pdf")
# plot(means$head_distance, means$GPT, xlim=c(0,60))+
#  abline(lm(means$GPT ~ means$head_distance))
# dev.off()

###############
# Effect of adding surprisal

# TODO

deltalogliks.s <- list()
for (x in 0:(num.models-1)) {
  # ---------------------- Step 1 ----------------------
  # s0 <- lmer(GPT ~ frequency.s + length.s + 
  #                (1|WorkerId) + (1|item),
  #              data=na.omit(get(paste("d1", x, sep = "."))),
  #              REML=FALSE)
  # assign(paste("lme", "s0", x, sep="."), s0)
  # summary(s0)
  # 
  # # ---------------------- Step 2 ----------------------
  # 
  # s1 <- lmer(GPT ~ frequency.s + length.s + surprisal.s +
  #              (1|WorkerId) + (1|item),
  #            data=na.omit(get(paste("d1", x, sep = "."))),
  #            REML=FALSE)
  # assign(paste("lme", "s1", x, sep="."), s1)
  # 
  # summary(s1)
  # 
  # a <- anova(s0, s1)  # Significant?
  # print(paste("Anova", x))
  # print(a)
  # 
  # deltalogliks.s <- append(deltalogliks.s, a$logLik[2]-a$logLik[1])
  # 
  # r.squaredGLMM(s0)
  # r.squaredGLMM(s1)
}
# deltalogliks.s <- do.call("rbind", deltalogliks.s)
# print("Mean Delta LogLik Surprisal effect:")
# mean(deltalogliks.s)
# print("Standard deviation Delta LogLik Surprisal effect:")
# sd(deltalogliks.s)


print("Average governor distance per pos tag")
aggregate(head_distance ~ pos,
          data = d0.0, FUN = mean, na.rm = TRUE)
table(d0.0$pos)

print("Average governor distance per deprel tag")
aggregate(head_distance ~ deprel,
          data = d0.0, FUN = mean, na.rm = TRUE)
table(d0.0$deprel)

print("Average first dependent distance per pos tag")
aggregate(first_dependent_distance ~ pos,
          data = d0.0, FUN = mean, na.rm = TRUE)
table(d0.0$pos)

print("Average first dependent distance per deprel tag")
aggregate(first_dependent_distance ~ deprel,
          data = d0.0, FUN = mean, na.rm = TRUE)
table(d0.0$deprel)

print("Average left dependency distance sum per pos tag")
aggregate(left_dependents_distance_sum ~ pos,
          data = d0.0, FUN = mean, na.rm = TRUE)
table(d0.0$pos)

print("Average left dependency distance sum per deprel tag")
aggregate(left_dependents_distance_sum ~ deprel,
          data = d0.0, FUN = mean, na.rm = TRUE)
table(d0.0$deprel)

print("Average left dependency count sum per pos tag")
aggregate(left_dependents_count ~ pos,
          data = d0.0, FUN = mean, na.rm = TRUE)
table(d0.0$pos)

print("Average left dependency count tag")
aggregate(left_dependents_count ~ deprel,
          data = d0.0, FUN = mean, na.rm = TRUE)
table(d0.0$deprel)

# Histogram of governor distances
pdf(file=paste("results/gov_distance_histogram_", args[1], ".pdf", sep=""))
hist(d0.0$first_dependent_distance,
     breaks=seq(min(d0.0$first_dependent_distance),max(d0.0$first_dependent_distance),l=max(d0.0$first_dependent_distance)-min(d0.0$first_dependent_distance)+1))
dev.off()

# Histogram of first dependent distances
pdf(file=paste("results/gov_first_dependent_distance_histogram_", args[1], ".pdf", sep=""))
hist(d0.0$head_distance,
     breaks=seq(min(d0.0$head_distance),max(d0.0$head_distance),l=max(d0.0$head_distance)-min(d0.0$head_distance)+1))
dev.off()

# Histogram of left dependency distance sum
pdf(file=paste("results/gov_left_dependents_distance_sum_histogram_", args[1], ".pdf", sep=""))
hist(d0.0$left_dependents_distance_sum,
     breaks=seq(min(d0.0$left_dependents_distance_sum),max(d0.0$left_dependents_distance_sum),l=max(d0.0$left_dependents_distance_sum)-min(d0.0$left_dependents_distance_sum)+1))
dev.off()

# Histogram of left dependency count
pdf(file=paste("results/gov_left_dependents_count_histogram_", args[1], ".pdf", sep=""))
hist(d0.0$left_dependents_count,
     breaks=seq(min(d0.0$left_dependents_count),max(d0.0$left_dependents_count),l=max(d0.0$left_dependents_count)-min(d0.0$left_dependents_count)+1))
dev.off()

###############
# Compare against baseline LM
# 
# deltalogliks.m <- list()
# for (x in 0:(num.models-1)) {
#   # TODO: change this so that m0 is only fitted three times and not nine times
#   #for (y in 0:(num.models-1)) {
#   y <- x
#     newdata <- merge(x=get(paste("d1", y, sep = ".")),y=get(paste("d0", x, sep = ".")),
#                           by=c("zone","item","WorkerId"))
#     newdata$frequency.s <- newdata$frequency.s.y
#     newdata$length.s <- newdata$length.s.y
#     newdata$surprisal.s <- newdata$surprisal.s.y
# 
#     # ---------------------- Step 1 ----------------------
#     m0 <- lmer(GPT.y ~ frequency.s + length.s + surprisal.s +
#                  (1|WorkerId) + (1|item),
#                data=na.omit(newdata),
#                REML=FALSE)
#     assign(paste("lme", "m0", x, sep="."), m0)
#     summary(m0)
# 
#     # ---------------------- Step 2 ----------------------
#     m1 = lmer(GPT.y ~ frequency.y + length.y + surprisal.s.y + surprisal.s.x +
#                   (1|WorkerId) + (1|item),
#                 data=na.omit(newdata),
#                 REML=FALSE)
#     assign(paste("lme", "m1", x, y, sep="."), m0)
#     summary(m1)
#     a <- anova(m0, m1)  # Significant?
#     print(paste("Anova", x, y))
#     print(a)
# 
#     deltalogliks.m <- append(deltalogliks.m, a$logLik[2]-a$logLik[1])
#     r.squaredGLMM(m0)
#     r.squaredGLMM(m1)
#   #}
# }
# deltalogliks.m <- do.call("rbind", deltalogliks.m)
# print("Mean Delta LogLik LM improvement effect:")
# print(mean(deltalogliks.m))
# print("Standard deviation Delta LogLik LM improvement effect:")
# print(sd(deltalogliks.m))
# 
# 
# # binned predictions for first_dependent_distance with large bins
# for (x in 0:(num.models-1)) {
#   data <- get(paste("d1", x, sep = "."))
#   data$first_dependent_distance[data$first_dependent_distance <= -10] <- -10
#   data$first_dependent_distance[data$first_dependent_distance <= -5 & data$first_dependent_distance > -10] <- -5
#   data <- data[which(data$first_dependent_distance <= -5), ]
# 
#   data$prs1 <- predict(get(paste("lme", "s1", x, sep = ".")), newdata=data)
# 
#   data$prs1 <- (data$prs1 - data$GPT)^2
# 
#   
#   pr <- aggregate(prs1 ~ first_dependent_distance,
#                   data = data, FUN = mean, na.rm = TRUE)
#   
#   assign(paste("pr", x, sep="."), pr)
# }
# 
# pr <- pr.0
# if (num.models > 1) {
#   for (x in 1:(num.models-1)) {
#     pr$prs1 <- pr$prs1 + get(paste("pr", x, sep="."))$prs1
#   }
# }
# pr$prs1 <- pr$prs1 / num.models
# 
# prst <- pr.0
# prst$prs1 <- (prst$prs1 - pr$prs1)^2
# 
# if (num.models > 1) {
#   for (x in 1:(num.models-1)) {
#     prst$prs1 <- prst$prs1 + (get(paste("pr", x, sep="."))$prs1 - pr$prs1)^2
#   }
# }
# prst$prs1 <- sqGPT(prst$prs1 / num.models)
# 
# print("pr mean")
# print(pr)
# print("pr std")
# print(prst)

# Define the set of pos tags to keep
CONTENT_pos <- c("FW", "MD", "NN", "NNS", "NNP", "NNPS", 
                 "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", 
                 "JJ", "JJR", "JJS")

# Define subset conditions
subset_conditions <- list(
  "first_dependent_distance <= -10"  = function(d) d$first_dependent_distance <= -10,
  "-10 < first_dependent_distance <= -5" = function(d) d$first_dependent_distance > -10 & d$first_dependent_distance <= -5,
  "first_dependent_distance == -4" = function(d) d$first_dependent_distance == -4,
  "first_dependent_distance == -3" = function(d) d$first_dependent_distance == -3,
  "first_dependent_distance == -2" = function(d) d$first_dependent_distance == -2,
  "first_dependent_distance == -1" = function(d) d$first_dependent_distance == -1,
  "first_dependent_distance > 0" = function(d) d$first_dependent_distance > 0
)

# Initialize results storage
results <- data.frame(Subset = character(), Mean_logLik_Without = numeric(), 
                      SD_logLik_Without = numeric(), Mean_logLik_With = numeric(), 
                      SD_logLik_With = numeric(), Mean_delta_logLik = numeric(), 
                      SD_delta_logLik = numeric(), stringsAsFactors = FALSE)

# Loop over subsets
for (subset_name in names(subset_conditions)) {
  
  delta_logLik_list <- c()
  
  # Loop over all datasets d1.0, d1.1, ..., d1.num.models
  for (i in 0:num.models) {
    
    dataset_name <- paste("d1", i, sep = ".")  # Construct dataset name
    if (!exists(dataset_name)) next  # Skip if dataset doesn't exist
    
    data_subset <- subset(get(dataset_name), subset_conditions[[subset_name]](get(dataset_name)))
                          # & pos %in% CONTENT_pos)
    
    if (i == 0) {
      print(paste("Number of elements in ", subset_name))
      print(nrow(aggregate(GPT ~ item + zone,
                           data = data_subset, FUN = mean, na.rm = TRUE)
      ))
    }
    
    # Ensure subset is not empty
    if (nrow(data_subset) > 0) {
      
      # # Fit models
      # model_without_surprisal <- lmer(GPT ~ frequency.s + length.s + (1 | WorkerId) + (1 | item),
      #                            data = data_subset, REML = FALSE)
      # 
      # model_with_surprisal <- lmer(GPT ~ frequency.s + length.s + surprisal.s + (1 | WorkerId) + (1 | item),
      #                         data = data_subset, REML = FALSE)
      # a <- anova(model_without_surprisal, model_with_surprisal)  # Significant?
      # delta_logLik <- as.numeric(a$logLik[2]-a$logLik[1])
      # 
      # # Store delta logLik
      # delta_logLik_list <- c(delta_logLik_list, delta_logLik)
    }
  }
  
  # Compute mean and standard deviation of delta logLik
  if (length(delta_logLik_list) > 0) {
    mean_delta_logLik <- mean(delta_logLik_list)
    sd_delta_logLik <- sd(delta_logLik_list)
    
    # Store results
    results <- rbind(results, data.frame(
      Subset = subset_name,
      Mean_delta_logLik = mean_delta_logLik,
      SD_delta_logLik = sd_delta_logLik
    ))
  }
}

# Print results
print("Refittet models for first_dependent_distance bins")
print(results)

# Initialize results storage
first_dependent_distance_percentages <- data.frame(Subset = character(), Mean_Percentage_first_dependent_correct_1 = numeric(), 
                              SD_Percentage_first_dependent_correct_1 = numeric(), stringsAsFactors = FALSE)

# Loop over subsets defined by first_dependent_distance bins
for (subset_name in names(subset_conditions)) {
  
  percentage_list <- c()  # Store percentages for each model
  
  # Loop over all datasets d1.0, d1.1, ..., d1.num.models
  for (i in 0:num.models) {
    
    dataset_name <- paste("d1", i, sep = ".")  # Construct dataset name
    if (!exists(dataset_name)) next  # Skip if dataset doesn't exist
    
    data_subset <- subset(get(dataset_name), subset_conditions[[subset_name]](get(dataset_name))
                          ) #& pos %in% CONTENT_pos)
    
    if (nrow(data_subset) > 0) {
      # Aggregate by item
      aggregated_data <- aggregate(first_dependent_correct ~ item + zone, data = data_subset, FUN = mean, na.rm = TRUE)
      
      # Count where first_dependent_correct == 1
      count_first_dependent_correct_1 <- sum(aggregated_data$first_dependent_correct == 1.0, na.rm = TRUE)
      
      # Total number of items
      total_items <- nrow(aggregated_data)
      
      # Compute percentage if there are items
      if (total_items > 0) {
        percentage_first_dependent_correct_1 <- (count_first_dependent_correct_1 / total_items) * 100
        percentage_list <- c(percentage_list, percentage_first_dependent_correct_1)
      }
    }
  }
  
  # Compute mean and standard deviation across models
  if (length(percentage_list) > 0) {
    mean_percentage <- mean(percentage_list)
    sd_percentage <- sd(percentage_list)
    
    # Store results
    first_dependent_distance_percentages <- rbind(first_dependent_distance_percentages, data.frame(
      Subset = subset_name,
      Mean_Percentage_first_dependent_correct_1 = mean_percentage,
      SD_Percentage_first_dependent_correct_1 = sd_percentage
    ))
  }
}

# Print results
print("Mean percentage of items where first_dependent_correct == 1 per first_dependent_distance bin (across models)")
print(first_dependent_distance_percentages)


# first_dependent_distance
# Initialize results storage
first_dependent_distance_means <- data.frame(Subset = character(), Mean_Percentage_first_dependent_correct_1 = numeric(), 
                              SD_Percentage_first_dependent_correct_1 = numeric(), stringsAsFactors = FALSE)

# Loop over subsets defined by first_dependent_distance bins
for (subset_name in names(subset_conditions)) {
  
  first_dependent_distance_list <- c()  # Store percentages for each model
  
  # Loop over all datasets d1.0, d1.1, ..., d1.num.models
  for (i in 0:num.models) {
    
    dataset_name <- paste("d1", i, sep = ".")  # Construct dataset name
    if (!exists(dataset_name)) next  # Skip if dataset doesn't exist
    
    data_subset <- subset(get(dataset_name), subset_conditions[[subset_name]](get(dataset_name))
                          ) # & pos %in% CONTENT_pos)
    
    if (nrow(data_subset) > 0) {
      # Aggregate by item
      aggregated_data <- aggregate(first_dependent_distance ~ item + zone, data = data_subset, FUN = mean, na.rm = TRUE)
      
      # Compute mean first_dependent_distance for this model
      mean_first_dependent_distance_model <- mean(aggregated_data$first_dependent_distance, na.rm = TRUE)
      first_dependent_distance_list <- c(first_dependent_distance_list, mean_first_dependent_distance_model)
      
    }
  }
  
  # Compute mean and standard deviation across models
  if (length(first_dependent_distance_list) > 0) {
    mean_first_dependent_distance <- mean(first_dependent_distance_list)
    sd_first_dependent_distance <- sd(first_dependent_distance_list)
    
    # Store results
    first_dependent_distance_means <- rbind(first_dependent_distance_means, data.frame(
      Subset = subset_name,
      Mean_first_dependent_distance = mean_first_dependent_distance,
      SD_first_dependent_distance = sd_first_dependent_distance
    ))
  }
}

# Print results
print("weight on the first_dependent_distance (ideal: 1) (including future as one category)")
print(first_dependent_distance_means)

# Use the first dataset (e.g., d1.0)
dataset_name <- "d1.0"  # Change if needed
if (!exists(dataset_name)) stop("Dataset does not exist!")
data <- get(dataset_name)  

# Compute mean GPT for each (first_dependent_distance, first_dependent_deprel) pair
GPT_means <- data %>%
  group_by(first_dependent_distance, first_dependent_deprel) %>%
  summarise(Mean_GPT = mean(GPT, na.rm = TRUE), SD_GPT = sd(GPT, na.rm = TRUE), .groups = "drop")

# Print results
print("Mean GPT aggregated per raw first_dependent_distance value and first_dependent_deprel")
print(GPT_means)

# Plot results: separate line for each first_dependent_deprel
ggplot(GPT_means, aes(x = first_dependent_distance, y = Mean_GPT, color = as.factor(first_dependent_deprel), group = first_dependent_deprel)) +
  geom_point(size = 2, alpha = 0.6) +  # Scatter plot
  geom_line(alpha = 0.8) +  # Line plot
  geom_errorbar(aes(ymin = Mean_GPT - SD_GPT, ymax = Mean_GPT + SD_GPT), width = 0.2, alpha = 0.5) +  # Error bars
  theme_minimal() +
  labs(title = "Mean GPT as a Function of first_dependent_distance, Split by first_dependent_deprel",
       x = "first_dependent_distance",
       y = "Mean GPT",
       color = "first_dependent_deprel") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-axis labels if necessary

ggsave(paste("results/first_dependent_distance_and_GPT_", args[1], ".pdf", sep=""), width = 20, height = 10)

GPT_means <- data %>%
  group_by(first_dependent_distance) %>%
  summarise(Mean_GPT = mean(GPT, na.rm = TRUE), SD_GPT = sd(GPT, na.rm = TRUE), .groups = "drop")

# Plot results: separate line for each deprel
ggplot(GPT_means, aes(x = first_dependent_distance, y = Mean_GPT)) +
  geom_point(size = 2, color = "blue", alpha = 0.6) +  # Scatter plot
  geom_line(color = "blue", alpha = 0.8) +  # Line plot to connect points
  geom_errorbar(aes(ymin = Mean_GPT - SD_GPT, ymax = Mean_GPT + SD_GPT), width = 0.2, alpha = 0.5) +  # Error bars
  theme_minimal() +
  labs(title = "Mean GPT as a Function of Raw first_dependent_distance",
       x = "first_dependent_distance",
       y = "Mean GPT") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-axis labels if necessary
ggsave(paste("results/first_dependent_distance_and_GPT_raw_", args[1], ".pdf", sep=""), width = 20, height = 10)


# Use the first dataset (e.g., d1.0)
dataset_name <- "d1.0"  # Change if needed
if (!exists(dataset_name)) stop("Dataset does not exist!")
data <- get(dataset_name)  

# Compute mean GPT for each (head_distance, deprel) pair
GPT_means <- data %>%
  group_by(head_distance, deprel) %>%
  summarise(Mean_GPT = mean(GPT, na.rm = TRUE), SD_GPT = sd(GPT, na.rm = TRUE), .groups = "drop")

# Print results
print("Mean GPT aggregated per raw head_distance value and deprel")
print(GPT_means)

# Plot results: separate line for each deprel
ggplot(GPT_means, aes(x = head_distance, y = Mean_GPT, color = as.factor(deprel), group = deprel)) +
  geom_point(size = 2, alpha = 0.6) +  # Scatter plot
  geom_line(alpha = 0.8) +  # Line plot
  geom_errorbar(aes(ymin = Mean_GPT - SD_GPT, ymax = Mean_GPT + SD_GPT), width = 0.2, alpha = 0.5) +  # Error bars
  theme_minimal() +
  labs(title = "Mean GPT as a Function of head_distance, Split by deprel",
       x = "head_distance",
       y = "Mean GPT",
       color = "deprel") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-axis labels if necessary

ggsave(paste("results/headdistance_and_GPT_", args[1], ".pdf", sep=""), width = 20, height = 10)

GPT_means <- data %>%
  group_by(head_distance) %>%
  summarise(Mean_GPT = mean(GPT, na.rm = TRUE), SD_GPT = sd(GPT, na.rm = TRUE), .groups = "drop")

# Plot results
ggplot(GPT_means, aes(x = head_distance, y = Mean_GPT)) +
  geom_point(size = 2, color = "blue", alpha = 0.6) +  # Scatter plot
  geom_line(color = "blue", alpha = 0.8) +  # Line plot to connect points
  geom_errorbar(aes(ymin = Mean_GPT - SD_GPT, ymax = Mean_GPT + SD_GPT), width = 0.2, alpha = 0.5) +  # Error bars
  theme_minimal() +
  labs(title = "Mean GPT as a Function of Raw first_dependent_distance",
       x = "head_distance",
       y = "Mean GPT") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-axis labels if necessary
ggsave(paste("results/headdistance_and_GPT_raw_", args[1], ".pdf", sep=""), width = 20, height = 10)

# # binned predictions
# for (x in 0:(num.models-1)) {
#   data <- get(paste("d1", x, sep = "."))
#   data$head_distance[data$head_distance <= -10 & data$head_distance > -20] <- -10
#   data$head_distance[data$head_distance <= -20 & data$head_distance] <- -20
#   data$head_distance[data$head_distance >= 10 & data$head_distance < 20] <- 10
#   data$head_distance[data$head_distance >= 20 & data$head_distance] <- 20
# 
#   data$prs0 <- predict(get(paste("lme", "m0", x, sep = ".")), newdata=data)
#   data$prs1 <- predict(get(paste("lme", "s1", x, sep = ".")), newdata=data)
# 
#   pr <- aggregate(prs0 ~ head_distance,
#                             data = data, FUN = mean, na.rm = TRUE)
#   pr <- merge(pr, aggregate(prs1 ~ head_distance,
#                              data = data, FUN = mean, na.rm = TRUE),
#               by="head_distance")
#   pr <- merge(pr, aggregate(GPT ~ head_distance,
#                             data = data, FUN = mean, na.rm = TRUE),
#               by="head_distance")
#   pr$diff0 <- pr$prs0 - pr$GPT
#   pr$diff1 <- pr$prs1 - pr$GPT
# 
#   assign(paste("pr", x, sep="."), pr)
# }
# 
# pr <- pr.0
# if (num.models > 1) {
#   for (x in 1:(num.models-1)) {
#     pr$diff0 <- pr$diff0 + get(paste("pr", x, sep="."))$diff0
#     pr$diff1 <- pr$diff1 + get(paste("pr", x, sep="."))$diff1
#     pr$prs0 <- pr$prs0 + get(paste("pr", x, sep="."))$prs0
#     pr$prs1 <- pr$prs1 + get(paste("pr", x, sep="."))$prs1
#   }
# }
# pr$diff0 <- pr$diff0 / num.models
# pr$diff1 <- pr$diff1 / num.models
# pr$prs0 <- pr$prs0 / num.models
# pr$prs1 <- pr$prs1 / num.models
# 
# # Absolute plot
# pdf(file=paste("results/plot_binned_head_distance_abs_", args[1], ".pdf", sep=""))
# 
# labels <- c(expression(phantom(x)<=-20), expression(phantom(x)<=-10), "-9", "-8", "-7", "-6", "-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", expression(phantom(x)>=10), expression(phantom(x)>=20))
# bp <- barplot(as.matrix(t(pr[,2:4])), beside=T, las=2,
#               col=c(adjustcolor("orange", 1.0), adjustcolor("cyan", 1.0), adjustcolor("yellow", 1.0)), axes=TRUE,
#               ylab = "ms", cex.axis=0.3, ylim=c(0, 500), yaxp = c(0, 400, 5),
# )
# bp[1,] <- bp[1,]+1
# text(bp[seq(1, 3*length(labels), 3)], 0, labels, xpd = NA, adj = c(0.5, 10), cex=0.3)
# dev.off()
# 
# # Difference plot
# pdf(file=paste("results/plot_binned_head_distance_diffs_", args[1], ".pdf", sep=""))
# 
# labels <- c(expression(phantom(x)<=-20), expression(phantom(x)<=-10), "-9", "-8", "-7", "-6", "-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", expression(phantom(x)>=10), expression(phantom(x)>=20))
# bp <- barplot(as.matrix(t(pr[,5:6])), beside=T, las=2,
#               col=c(adjustcolor("orange", 1.0), adjustcolor("cyan", 1.0)), axes=TRUE,
#               ylab = "ms", cex.axis=0.3, ylim=c(-10, 10), yaxp = c(-10, 10, 4),
# )
# bp[1,] <- bp[1,]+0.5
# text(bp[seq(1, 2*length(labels), 4)], 0, labels[seq(1, length(labels), 2)], xpd = NA, adj = c(0.5, -60), cex=0.3)
# text(bp[seq(3, 2*length(labels), 4)], 0, labels[seq(2, length(labels), 2)], xpd = NA, adj = c(0.5, -60), cex=0.3)
# dev.off()
# 
# 
# # binned predictions by first dependent distances
# for (x in 0:(num.models-1)) {
#   data <- get(paste("d1", x, sep = "."))
# 
#   data$prs0 <- predict(get(paste("lme", "m0", x, sep = ".")), newdata=data)
#   data$prs1 <- predict(get(paste("lme", "s1", x, sep = ".")), newdata=data)
# 
#   pr <- aggregate(prs0 ~ first_dependent_distance,
#                   data = data, FUN = mean, na.rm = TRUE)
#   pr <- merge(pr, aggregate(prs1 ~ first_dependent_distance,
#                             data = data, FUN = mean, na.rm = TRUE),
#               by="first_dependent_distance")
#   pr <- merge(pr, aggregate(GPT ~ first_dependent_distance,
#                             data = data, FUN = mean, na.rm = TRUE),
#               by="first_dependent_distance")
#   pr$diff0 <- pr$prs0 - pr$GPT
#   pr$diff1 <- pr$prs1 - pr$GPT
# 
#   assign(paste("pr", x, sep="."), pr)
# }
# 
# pr <- pr.0
# if (num.models > 1) {
#   for (x in 1:(num.models-1)) {
#     pr$diff0 <- pr$diff0 + get(paste("pr", x, sep="."))$diff0
#     pr$diff1 <- pr$diff1 + get(paste("pr", x, sep="."))$diff1
#     pr$prs0 <- pr$prs0 + get(paste("pr", x, sep="."))$prs0
#     pr$prs1 <- pr$prs1 + get(paste("pr", x, sep="."))$prs1
#   }
# }
# pr$diff0 <- pr$diff0 / num.models
# pr$diff1 <- pr$diff1 / num.models
# pr$prs0 <- pr$prs0 / num.models
# pr$prs1 <- pr$prs1 / num.models
# 
# # Absolute plot
# pdf(file=paste("results/plot_binned_first_dependent_distance_abs_", args[1], ".pdf", sep=""))
# 
# labels <- pr$first_dependent_distance  #c(expression(phantom(x)<=-20), expression(phantom(x)<=-10), "-9", "-8", "-7", "-6", "-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", expression(phantom(x)>=10), expression(phantom(x)>=20))
# bp <- barplot(as.matrix(t(pr[,2:4])), beside=T, las=2,
#               col=c(adjustcolor("orange", 1.0), adjustcolor("cyan", 1.0), adjustcolor("yellow", 1.0)), axes=TRUE,
#               ylab = "ms", cex.axis=0.3, ylim=c(0, 500), yaxp = c(0, 400, 5),
# )
# bp[1,] <- bp[1,]+1
# text(bp[seq(1, 3*length(labels), 3)], 0, labels, xpd = NA, adj = c(0.5, 10), cex=0.3)
# dev.off()
# 
# # Difference plot
# pdf(file=paste("results/plot_binned_first_dependent_distance_diffs_", args[1], ".pdf", sep=""))
# 
# # labels <- c(expression(phantom(x)<=-20), expression(phantom(x)<=-10), "-9", "-8", "-7", "-6", "-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", expression(phantom(x)>=10), expression(phantom(x)>=20))
# bp <- barplot(as.matrix(t(pr[,5:6])), beside=T, las=2,
#               col=c(adjustcolor("orange", 1.0), adjustcolor("cyan", 1.0)), axes=TRUE,
#               ylab = "ms", cex.axis=0.3, ylim=c(-10, 10), yaxp = c(-10, 10, 4),
# )
# bp[1,] <- bp[1,]+0.5
# text(bp[seq(1, 2*length(labels), 4)], 0, labels[seq(1, length(labels), 2)], xpd = NA, adj = c(0.5, -60), cex=0.3)
# text(bp[seq(3, 2*length(labels), 4)], 0, labels[seq(2, length(labels), 2)], xpd = NA, adj = c(0.5, -60), cex=0.3)
# dev.off()
# 
# 
# # binned predictions by left dependency distance sum
# for (x in 0:(num.models-1)) {
#   data <- get(paste("d1", x, sep = "."))
# 
#   data$prs0 <- predict(get(paste("lme", "m0", x, sep = ".")), newdata=data)
#   data$prs1 <- predict(get(paste("lme", "s1", x, sep = ".")), newdata=data)
# 
#   pr <- aggregate(prs0 ~ left_dependents_distance_sum,
#                   data = data, FUN = mean, na.rm = TRUE)
#   pr <- merge(pr, aggregate(prs1 ~ left_dependents_distance_sum,
#                             data = data, FUN = mean, na.rm = TRUE),
#               by="left_dependents_distance_sum")
#   pr <- merge(pr, aggregate(GPT ~ left_dependents_distance_sum,
#                             data = data, FUN = mean, na.rm = TRUE),
#               by="left_dependents_distance_sum")
#   pr$diff0 <- pr$prs0 - pr$GPT
#   pr$diff1 <- pr$prs1 - pr$GPT
# 
#   assign(paste("pr", x, sep="."), pr)
# }
# 
# pr <- pr.0
# if (num.models > 1) {
#   for (x in 1:(num.models-1)) {
#     pr$diff0 <- pr$diff0 + get(paste("pr", x, sep="."))$diff0
#     pr$diff1 <- pr$diff1 + get(paste("pr", x, sep="."))$diff1
#     pr$prs0 <- pr$prs0 + get(paste("pr", x, sep="."))$prs0
#     pr$prs1 <- pr$prs1 + get(paste("pr", x, sep="."))$prs1
#   }
# }
# pr$diff0 <- pr$diff0 / num.models
# pr$diff1 <- pr$diff1 / num.models
# pr$prs0 <- pr$prs0 / num.models
# pr$prs1 <- pr$prs1 / num.models
# 
# # Absolute plot
# pdf(file=paste("results/plot_binned_left_dependents_distance_sum_abs_", args[1], ".pdf", sep=""))
# 
# labels <- pr$left_dependents_distance_sum  # c(expression(phantom(x)<=-20), expression(phantom(x)<=-10), "-9", "-8", "-7", "-6", "-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", expression(phantom(x)>=10), expression(phantom(x)>=20))
# bp <- barplot(as.matrix(t(pr[,2:4])), beside=T, las=2,
#               col=c(adjustcolor("orange", 1.0), adjustcolor("cyan", 1.0), adjustcolor("yellow", 1.0)), axes=TRUE,
#               ylab = "ms", cex.axis=0.3, ylim=c(0, 500), yaxp = c(0, 400, 5),
# )
# bp[1,] <- bp[1,]+1
# text(bp[seq(1, 3*length(labels), 3)], 0, labels, xpd = NA, adj = c(0.5, 10), cex=0.3)
# dev.off()
# 
# # Difference plot
# pdf(file=paste("results/plot_binned_left_dependents_distance_sum_diffs_", args[1], ".pdf", sep=""))
# 
# # labels <- c(expression(phantom(x)<=-20), expression(phantom(x)<=-10), "-9", "-8", "-7", "-6", "-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", expression(phantom(x)>=10), expression(phantom(x)>=20))
# bp <- barplot(as.matrix(t(pr[,5:6])), beside=T, las=2,
#               col=c(adjustcolor("orange", 1.0), adjustcolor("cyan", 1.0)), axes=TRUE,
#               ylab = "ms", cex.axis=0.3, ylim=c(-10, 10), yaxp = c(-10, 10, 4),
# )
# bp[1,] <- bp[1,]+0.5
# text(bp[seq(1, 2*length(labels), 4)], 0, labels[seq(1, length(labels), 2)], xpd = NA, adj = c(0.5, -60), cex=0.3)
# text(bp[seq(3, 2*length(labels), 4)], 0, labels[seq(2, length(labels), 2)], xpd = NA, adj = c(0.5, -60), cex=0.3)
# dev.off()
# 
# # binned predictions by left dependency count
# for (x in 0:(num.models-1)) {
#   data <- get(paste("d1", x, sep = "."))
# 
#   data$prs0 <- predict(get(paste("lme", "m0", x, sep = ".")), newdata=data)
#   data$prs1 <- predict(get(paste("lme", "s1", x, sep = ".")), newdata=data)
# 
#   pr <- aggregate(prs0 ~ left_dependents_count,
#                   data = data, FUN = mean, na.rm = TRUE)
#   pr <- merge(pr, aggregate(prs1 ~ left_dependents_count,
#                             data = data, FUN = mean, na.rm = TRUE),
#               by="left_dependents_count")
#   pr <- merge(pr, aggregate(GPT ~ left_dependents_count,
#                             data = data, FUN = mean, na.rm = TRUE),
#               by="left_dependents_count")
#   pr$diff0 <- pr$prs0 - pr$GPT
#   pr$diff1 <- pr$prs1 - pr$GPT
# 
#   assign(paste("pr", x, sep="."), pr)
# }
# 
# pr <- pr.0
# if (num.models > 1) {
#   for (x in 1:(num.models-1)) {
#     pr$diff0 <- pr$diff0 + get(paste("pr", x, sep="."))$diff0
#     pr$diff1 <- pr$diff1 + get(paste("pr", x, sep="."))$diff1
#     pr$prs0 <- pr$prs0 + get(paste("pr", x, sep="."))$prs0
#     pr$prs1 <- pr$prs1 + get(paste("pr", x, sep="."))$prs1
#   }
# }
# pr$diff0 <- pr$diff0 / num.models
# pr$diff1 <- pr$diff1 / num.models
# pr$prs0 <- pr$prs0 / num.models
# pr$prs1 <- pr$prs1 / num.models
# 
# # Absolute plot
# pdf(file=paste("results/plot_binned_left_dependents_count_abs_", args[1], ".pdf", sep=""))
# 
# labels <- pr$left_dependents_count  # c(expression(phantom(x)<=-20), expression(phantom(x)<=-10), "-9", "-8", "-7", "-6", "-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", expression(phantom(x)>=10), expression(phantom(x)>=20))
# bp <- barplot(as.matrix(t(pr[,2:4])), beside=T, las=2,
#               col=c(adjustcolor("orange", 1.0), adjustcolor("cyan", 1.0), adjustcolor("yellow", 1.0)), axes=TRUE,
#               ylab = "ms", cex.axis=0.3, ylim=c(0, 500), yaxp = c(0, 400, 5),
# )
# bp[1,] <- bp[1,]+1
# text(bp[seq(1, 3*length(labels), 3)], 0, labels, xpd = NA, adj = c(0.5, 10), cex=0.3)
# dev.off()
# 
# # Difference plot
# pdf(file=paste("results/plot_binned_left_dependents_count_diffs_", args[1], ".pdf", sep=""))
# 
# # labels <- c(expression(phantom(x)<=-20), expression(phantom(x)<=-10), "-9", "-8", "-7", "-6", "-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", expression(phantom(x)>=10), expression(phantom(x)>=20))
# bp <- barplot(as.matrix(t(pr[,5:6])), beside=T, las=2,
#               col=c(adjustcolor("orange", 1.0), adjustcolor("cyan", 1.0)), axes=TRUE,
#               ylab = "ms", cex.axis=0.3, ylim=c(-10, 10), yaxp = c(-10, 10, 4),
# )
# bp[1,] <- bp[1,]+0.5
# text(bp[seq(1, 2*length(labels), 4)], 0, labels[seq(1, length(labels), 2)], xpd = NA, adj = c(0.5, -60), cex=0.3)
# text(bp[seq(3, 2*length(labels), 4)], 0, labels[seq(2, length(labels), 2)], xpd = NA, adj = c(0.5, -60), cex=0.3)
# dev.off()
# 
# # by pos tag
# 
# for (x in 0:(num.models-1)) {
#   data <- get(paste("d1", x, sep = "."))
#   data$prs0 <- predict(get(paste("lme", "m0", x, sep = ".")), newdata=data)
#   data$prs1 <- predict(get(paste("lme", "s1", x, sep = ".")), newdata=data)
#   pr <- aggregate(prs0 ~ pos,
#                   data = data, FUN = mean, na.rm = TRUE)
#   pr <- merge(pr, aggregate(prs1 ~ pos,
#                             data = data, FUN = mean, na.rm = TRUE),
#               by="pos")
#   pr <- merge(pr, aggregate(GPT ~ pos,
#                             data = data, FUN = mean, na.rm = TRUE),
#               by="pos")
#   pr$diff0 <- pr$prs0 - pr$GPT
#   pr$diff1 <- pr$prs1 - pr$GPT
# 
#   assign(paste("pr", x, sep="."), pr)
# }
# 
# pr <- pr.0
# if (num.models > 1) {
#   for (x in 1:(num.models-1)) {
#     pr$diff0 <- pr$diff0 + get(paste("pr", x, sep="."))$diff0
#     pr$diff1 <- pr$diff1 + get(paste("pr", x, sep="."))$diff1
#     pr$prs0 <- pr$prs0 + get(paste("pr", x, sep="."))$prs0
#     pr$prs1 <- pr$prs1 + get(paste("pr", x, sep="."))$prs1
#   }
# }
# pr$diff0 <- pr$diff0 / num.models
# pr$diff1 <- pr$diff1 / num.models
# pr$prs0 <- pr$prs0 / num.models
# pr$prs1 <- pr$prs1 / num.models
# 
# # Absolute plot
# pdf(file=paste("results/plot_binned_pos_abs_", args[1], ".pdf", sep=""))
# 
# bp <- barplot(as.matrix(t(pr[,2:4])), beside=T, las=2,
#               col=c(adjustcolor("orange", 1.0), adjustcolor("cyan", 1.0), adjustcolor("yellow", 1.0)), axes=TRUE,
#               ylab = "ms", cex.axis=0.3, ylim=c(0, 500), yaxp = c(0, 400, 5),
# )
# bp[1,] <- bp[1,]+0.5
# text(bp[seq(1, 3*length(pr$pos), 3)], 0, pr$pos, xpd = NA, adj = c(0, 10), cex=0.5)
# dev.off()
# 
# # Difference plot
# pdf(file=paste("results/plot_binned_pos_diffs_", args[1], ".pdf", sep=""))
# bp <- barplot(as.matrix(t(pr[,5:6])), beside=T, las=2,
#               col=c(adjustcolor("orange", 1.0), adjustcolor("cyan", 1.0)), axes=TRUE,
#               ylab = "ms", cex.axis=0.5, ylim=c(-25,25), yaxp = c(-25, 25, 5),
#               )
# text(bp[seq(1, 2*length(pr$pos), 2)], 0, pr$pos, xpd = NA, adj = c(0, -25), cex=0.5)
# 
# dev.off()
# 
# # by deprel tag
# 
# for (x in 0:(num.models-1)) {
#   data <- get(paste("d1", x, sep = "."))
#   data$prs0 <- predict(get(paste("lme", "m0", x, sep = ".")), newdata=data)
#   data$prs1 <- predict(get(paste("lme", "s1", x, sep = ".")), newdata=data)
#   pr <- aggregate(prs0 ~ deprel,
#                   data = data, FUN = mean, na.rm = TRUE)
#   pr <- merge(pr, aggregate(prs1 ~ deprel,
#                             data = data, FUN = mean, na.rm = TRUE),
#               by="deprel")
#   pr <- merge(pr, aggregate(GPT ~ deprel,
#                             data = data, FUN = mean, na.rm = TRUE),
#               by="deprel")
#   pr$diff0 <- pr$prs0 - pr$GPT
#   pr$diff1 <- pr$prs1 - pr$GPT
# 
#   assign(paste("pr", x, sep="."), pr)
# }
# 
# pr <- pr.0
# if (num.models > 1) {
#   for (x in 1:(num.models-1)) {
#     pr$diff0 <- pr$diff0 + get(paste("pr", x, sep="."))$diff0
#     pr$diff1 <- pr$diff1 + get(paste("pr", x, sep="."))$diff1
#     pr$prs0 <- pr$prs0 + get(paste("pr", x, sep="."))$prs0
#     pr$prs1 <- pr$prs1 + get(paste("pr", x, sep="."))$prs1
#   }
# }
# pr$diff0 <- pr$diff0 / num.models
# pr$diff1 <- pr$diff1 / num.models
# pr$prs0 <- pr$prs0 / num.models
# pr$prs1 <- pr$prs1 / num.models
# 
# # Absolute plot
# pdf(file=paste("results/plot_binned_deprel_abs_", args[1], ".pdf", sep=""))
# 
# bp <- barplot(as.matrix(t(pr[,2:4])), beside=T, las=2,
#               col=c(adjustcolor("orange", 1.0), adjustcolor("cyan", 1.0), adjustcolor("yellow", 1.0)), axes=TRUE,
#               ylab = "ms", cex.axis=0.3, ylim=c(0, 500), yaxp = c(0, 400, 5),
# )
# bp[1,] <- bp[1,]+0.5
# text(bp[seq(1, 3*length(pr$deprel), 6)], 0, pr$deprel[seq(1, length(pr$deprel), 2)], xpd = NA, adj = c(0.5, 5), cex=0.2)
# text(bp[seq(4, 3*length(pr$deprel), 6)], 0, pr$deprel[seq(2, length(pr$deprel), 2)], xpd = NA, adj = c(0.5, 8), cex=0.2)
# dev.off()
# 
# # Difference plot
# pdf(file=paste("results/plot_binned_deprel_diffs_", args[1], ".pdf", sep=""))
# bp <- barplot(as.matrix(t(pr[,5:6])), beside=T, las=2,
#               col=c(adjustcolor("orange", 1.0), adjustcolor("cyan", 1.0)), axes=TRUE,
#               ylab = "ms", cex.axis=0.5, ylim=c(-25,25), yaxp = c(-25, 25, 5),
# )
# bp[1,] <- bp[1,]+0.5
# text(bp[seq(1, 2*length(pr$deprel), 4)], 0, pr$deprel[seq(1, length(pr$deprel), 2)], xpd = NA, adj = c(0.5, -38), cex=0.2)
# text(bp[seq(3, 2*length(pr$deprel), 4)], 0, pr$deprel[seq(2, length(pr$deprel), 2)], xpd = NA, adj = c(0.5, -40), cex=0.2)
# 
# dev.off()