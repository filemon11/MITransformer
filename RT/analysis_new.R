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
spillover <- as.numeric(args[4])

if (corpus == "frank_SP" || corpus == "naturalstories") {
  corpus_type <- "SP"
} else {
  corpus_type <- "ET"
}

data_dir <- paste("data/", corpus, "_preprocessed_", sep="")
num.models = as.numeric(args[2])
# load data

datasets <- list()

for (x in 0:(num.models-1)) {
  data <- read.csv(paste(data_dir, args[1], "_", x, ".csv", sep=""))
  datasets <- c(datasets, list(data))
}

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

compute_deltalogliks <- function(datasets, to_predict, predict_from, predict_from_baseline) {
  deltalogliks.s <- c()
  deltaAIC <- c()
  if (predict_from_baseline != "") {
    predict_from_baseline <- paste(predict_from_baseline, " +")
  }
  for (data in datasets) {
    # ---------------------- Step 1 ----------------------
    formula_str <- paste(to_predict, "~ frequency + length + ", predict_from_baseline, 
                 " (1|WorkerId) + (1|item)")
    s0 <- lmer(as.formula(formula_str),
               data=na.omit(data),
               REML=FALSE)
    
    # ---------------------- Step 2 ----------------------
    formula_str <- paste(to_predict, "~ frequency + length + ", predict_from_baseline,
                 predict_from, " +
                 (1|WorkerId) + (1|item)")
    s1 <- lmer(as.formula(formula_str),
               data=na.omit(data),
               REML=FALSE)
    
    a <- anova(s0, s1)  # Significant?
    print(paste("Anova for ", to_predict, " and ", predict_from, " with baseline ", predict_from_baseline, sep=""))
    print(a)

    deltalogliks.s <- c(deltalogliks.s, (a$logLik[2]-a$logLik[1]) / nrow(data))
    deltaAIC <- c(deltaAIC, (a$AIC[2]-a$AIC[1]))
  }
  data.frame(Type = c("DeltaAIC", "DeltaLogLik"), Mean = c(mean(deltaAIC), mean(deltalogliks.s)), SD = c(sd(deltaAIC), sd(deltalogliks.s)))
}


if (corpus_type == "SP") {
  goals <- c("RT")
} else if (corpus_type == "ET") {
  goals <- c("FFD", "GPT")
}

candidates <- c("surprisal", "demberg")

for (goal in goals) {
  for (candidate in candidates) {
    print(paste("Correlation between ", goal, " and ", candidate, sep=""))
    print(compute_mean(datasets, goal, candidate))

    predict_from <- candidate
    if (spillover > 0) {
      if (spillover > 1) {
        for (i in 1:(spillover-1)) {
          predict_from <- paste(predict_from, " + ", candidate, ".", i, sep="")
        }
      }
      print(paste("lme DeltaLogLik for ", goal, " and ", candidate,
                  ", improvement of spillover ", spillover, sep=""))
      print(compute_deltalogliks(datasets, goal, paste(candidate, ".", spillover, sep=""),
                                 predict_from))
      
      predict_from <- paste(predict_from, " + ", candidate, ".", spillover, sep="")
    }
    
    print(paste("lme DeltaLogLik for ", goal, " and ", candidate,
                ", overall", sep=""))
    print(compute_deltalogliks(datasets, goal, predict_from, ""))
    
    for (candidate2 in candidates) {
      if (candidate != candidate2) {
        predict_from2 <- candidate2
        
        # TODO: allow different spillover for candidates
        if (spillover > 1) {
          for (i in 1:spillover) {
            predict_from2 <- paste(predict_from2, " + ", candidate2, ".", i, sep="")
          }
        }
        print(paste("lme DeltaLogLik for ", predict_from2, " over ", predict_from,
                    sep=""))
        print(compute_deltalogliks(datasets, goal, predict_from2, predict_from))
      }
    }
  }
}