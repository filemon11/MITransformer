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

source("utils.R")


args = commandArgs(trailingOnly=TRUE)
args = c("../data/provo_train_RT_test_preprocessed_exp1_4_0.csv", "GPT", 2)
data_dir <- args[1]
goal <- args[2]
spillover <- as.numeric(args[3])

candidates <- c("surprisal", "attention_entropy")
baseline_predictors <- c("frequency", "length", "position", "word")

# load data

datasets <- list()

data <- read.csv(data_dir)

# Get names of numeric columns
numeric_cols <- names(data)[sapply(data, is.numeric)]
# Subset to numeric columns you want to scale
excluded_vars <- c(goal, "WorkerId", "item", "element", "zone")
cols_to_scale <- setdiff(numeric_cols, excluded_vars)

# Apply cut-off
data <- apply_cutoff(data, goal)

# Remove outliers
data <- remove_outliers(data, c(goal), id_col="WorkerId")

# Apply scaling
data <- data %>%
  mutate(across(all_of(cols_to_scale), scaling_var))

# Compute a pearson like correlation by measuring the correlation
# between the log probs and the rt averaged over the readers.
# Plot the data and check that the relation is approx linear.
compute_mean <- function(datasets, factor1, factor2) {
  corrs <- c()
  for (data in datasets) {
    formula_str <- paste("cbind(", factor1, ",", factor2, ") ~ item + zone")
    means <- aggregate(as.formula(formula_str), data=data, FUN = mean, na.rm = TRUE)
    corr <- cor.test(means[[factor1]], means[[factor2]], method="pearson")
    corrs <- c(corrs, corr$estimate)
  }
  data.frame(Mean = mean(corrs), SD = sd(corrs))
}

get_spillover <- function(predictors, spillover) {
  out <- c()
  for (item in predictors) {
    out <- c(paste(item, ".", spillover, sep=""), out)
  }
  out
}

get_spillover_upto <- function(predictors, spillover) {
  out <- predictors
  if (spillover > 0) {
    for (i in 1:(spillover)) {
      out <- c(out, get_spillover(predictors, i))
    }
  }
  out
}

get_slopes <- function(predictors, slopes_for) {
  out <- c()
  for (sl in slopes_for) {
    out <- c(paste("(", paste(predictors, collapse="+"), "|", sl, ")", sep=""), out)
  }
  out
}

compute_deltalogliks <- function(datasets, to_predict, predict_from, baseline) {
  deltalogliks.s <- c()
  deltaAIC <- c()
  slopes <- get_slopes(c("1"), c("WorkerId"))

  for (data in datasets) {
    data <- as.data.frame(data)

    # ---------------------- Step 1 ----------------------
    formula_str <- paste(to_predict, " ~ ", paste(c(baseline, slopes), collapse=" + "))
    s0 <- lmer(as.formula(formula_str),
               data=na.omit(data),
               REML=FALSE)
    print(paste("Summary for baseline ", paste(baseline, collapse=", "), sep=""))
    print(summary(s0))
    
    # ---------------------- Step 2 ----------------------
    formula_str <- paste(to_predict, " ~ ", paste(c(baseline, slopes, predict_from), collapse=" + "))
    s1 <- lmer(as.formula(formula_str),
               data=na.omit(data),
               REML=FALSE)
    print(paste("Summary for s1 ", paste(predict_from, collapse=", "), sep=""))
    print(summary(s1))

    a <- anova(s0, s1)  # Significant?
    print(paste("Anova for ", to_predict, " and ", paste(predict_from, collapse=", "), " with baseline ", paste(baseline, collapse=", "), sep=""))
    print(a)

    deltalogliks.s <- c(deltalogliks.s, (a$logLik[2]-a$logLik[1]) / nrow(data))
    deltaAIC <- c(deltaAIC, (a$AIC[2]-a$AIC[1]) / nrow(data))
  }
  data.frame(Type = c("DeltaAIC", "DeltaLogLik"), Mean = c(mean(deltaAIC), mean(deltalogliks.s)), SD = c(sd(deltaAIC), sd(deltalogliks.s)))
}


for (goal in goals) {
  for (candidate in candidates) {
    print(paste("Correlation between ", goal, " and ", candidate, sep=""))
    print(compute_mean(datasets, goal, candidate))

    if (spillover > 0) {
      predict_from <- get_spillover_upto(c(candidate, baseline_predictors), spillover-1)
      print(paste("lme DeltaLogLik for ", goal, " and ", candidate,
                  ", improvement of spillover ", spillover, sep=""))
      print(compute_deltalogliks(datasets, goal, get_spillover(c(candidate, baseline_predictors), spillover),
                                  predict_from))
    }

    print(paste("lme DeltaLogLik for ", goal, " and ", candidate,
                ", overall", sep=""))
    print(compute_deltalogliks(datasets, goal,
          get_spillover_upto(candidate, spillover),
          get_spillover_upto(baseline_predictors, spillover)))
    
    for (candidate2 in candidates) {
      if (candidate != candidate2) {

        if (spillover > 0) {
          predict_from <- get_spillover_upto(c(candidate, candidate2, baseline_predictors), spillover-1)
          print(paste("lme DeltaLogLik for ", goal, " and ", candidate, "+", candidate2,
                      ", improvement of spillover ", spillover, sep=""))
          print(compute_deltalogliks(datasets, goal, get_spillover(c(candidate, candidate2, baseline_predictors), spillover),
                                      predict_from))
        }

        print(paste("lme DeltaLogLik for ", goal, " and ", candidate2, " over ", candidate,
                    ", overall", sep=""))
        print(compute_deltalogliks(datasets, goal,
              get_spillover_upto(candidate2, spillover),
              get_spillover_upto(c(baseline_predictors, candidate), spillover)))
      }
    }
  }
  for (candidate1 in candidates) {
    for (candidate2 in candidates) {
      if (candidate1 != candidate2) {

        if (spillover > 0) {
          predict_from <- get_spillover_upto(c(candidate1, candidate2, baseline_predictors), spillover-1)
          print(paste("lme DeltaLogLik for ", goal, " and ", candidate1, "+", candidate2,
                      ", improvement of spillover ", spillover, sep=""))
          print(compute_deltalogliks(datasets, goal, get_spillover(c(candidate1, candidate2, baseline_predictors), spillover),
                                      predict_from))
        }
        
        print(paste("lme DeltaLogLik for ", goal, " and ", candidate1, "+", candidate2,
                    ", overall", sep=""))
        print(compute_deltalogliks(datasets, goal,
              get_spillover_upto(c(candidate1, candidate2), spillover),
              get_spillover_upto(baseline_predictors, spillover)))
      }
    }
  }
}