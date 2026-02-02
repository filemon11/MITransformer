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
corpus <- args[3]
goal <- args[4]
spillover <- as.numeric(args[5])
dir <- args[6]
additionalname <- args[7]

candidates <- c("surprisal", "attention_entropy")
baseline_predictors <- c("frequency", "length", "position", "word")

data_dir <- paste(dir, "/", corpus, "_", additionalname, "_preprocessed_", sep="")

# load data

datasets <- list()

scaling_var <- function(data){
  # The input data is a vector
  data <- as.numeric(data)
  (data - mean(data,na.rm=TRUE))/sd(data,na.rm=TRUE)
}

remove_outliers <- function(data, cols, id_col = "WorkerId", k = 2.5, by = NULL) {
  # data: data.frame
  # cols: numeric columns to trim (e.g., c("FixDur", "RT"))
  # id_col: participant id column
  # k: cutoff in SDs (e.g., 2.5 or 3)
  # by: optional additional grouping columns (e.g., "Condition" or c("Condition","Item"))
  # According to https://link.springer.com/article/10.3758/s13428-023-02137-x
  # with 2.5 being mode in https://www.rd-alliance.org/sites/default/files/Marsden%2C_Thompson%2C_Plonsky_2018_App_Psych_SPR_synthesis.pdf

  stopifnot(all(c(id_col, cols) %in% names(data)))
  if (!is.null(by)) stopifnot(all(by %in% names(data)))

  grp <- c(id_col, by)

  cat("nrows before outlier removal ", nrow(data), "\n")

  # Helper that trims within one group
  trim_one_group <- function(df) {
    keep <- rep(TRUE, nrow(df))

    for (col in cols) {
      x <- df[[col]]

      # Only define bounds using non-missing values
      mu <- mean(x, na.rm = TRUE)
      s  <- sd(x, na.rm = TRUE)

      # If sd is 0 or NA (e.g., too few observations), don't trim this column in this group
      if (is.na(s) || s == 0) next

      lower <- mu - k * s
      upper <- mu + k * s

      # keep NAs (don’t turn missing into “outlier”)
      keep <- keep & (is.na(x) | (x >= lower & x <= upper))
    }

    df[keep, , drop = FALSE]
  }

  # Split-apply-combine without requiring dplyr
  out <- if (length(grp) == 1) {
    do.call(rbind, lapply(split(data, data[[id_col]]), trim_one_group))
  } else {
    # build an interaction key across grouping variables
    key <- interaction(data[grp], drop = TRUE, sep = "___")
    do.call(rbind, lapply(split(data, key), trim_one_group))
  }

  rownames(out) <- NULL
  cat("nrows after outlier removal ", nrow(out), "\n")

  out
}

apply_cutoff <- function(data, goal) {
  cat("nrows before cut-off ", nrow(data), "\n")
  if (goal == "RT") {
    data <- data %>%
      filter(RT>200) %>%
      filter(RT<2000)
    # following modal cutoffs in https://www.rd-alliance.org/sites/default/files/Marsden%2C_Thompson%2C_Plonsky_2018_App_Psych_SPR_synthesis.pdf
  } else {
    data <- data %>%
      filter(goal<1000) %>%
      filter(goal>80)
    # following middle value in https://link.springer.com/article/10.3758/s13428-023-02137-x
    # and 80 ms minimum visual transduction time (majority) in https://link.springer.com/article/10.3758/s13428-023-02137-x
  }
  cat("nrows after cot-off ", nrow(data), "\n")
  data
}


excluded_vars <- c(goal, "WorkerId", "item")
if (as.numeric(args[2]) == 0) {
  data <- read.csv(paste(data_dir, args[1], ".csv", sep=""))
}
else {
  data <- read.csv(paste(data_dir, args[1], "_", x, ".csv", sep=""))
}
# Get names of numeric columns
numeric_cols <- names(data)[sapply(data, is.numeric)]
# Subset to numeric columns you want to scale
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