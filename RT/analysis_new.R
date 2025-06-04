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

compute_deltalogliks <- function(datasets, to_predict) {
  deltalogliks.s <- c()
  for (data in datasets) {
    # ---------------------- Step 1 ----------------------
    formula_str <- paste(to_predict, "~ frequency.s + length.s + 
                 (1|WorkerId) + (1|item)")
    s0 <- lmer(as.formula(formula_str),
               data=na.omit(data),
               REML=FALSE)
    
    # ---------------------- Step 2 ----------------------
    formula_str <- paste(to_predict, "~ frequency.s + length.s + surprisal.s +
                 (1|WorkerId) + (1|item)")
    s1 <- lmer(as.formula(formula_str),
               data=na.omit(data),
               REML=FALSE)
    
    a <- anova(s0, s1)  # Significant?

    deltalogliks.s <- c(deltalogliks.s, (a$logLik[2]-a$logLik[1]) / nrow(data))
  }
  data.frame(Mean = mean(deltalogliks.s), SD = sd(deltalogliks.s))
}


if (corpus_type == "SP") {
  print("Correlation between RT and surprisal")
  print(compute_mean(datasets, "RT", "surprisal.s"))

  print("lme for RT and surprisal")
  print(compute_deltalogliks(datasets, "RT"))

} else if (corpus_type == "ET") {
  print("Correlation between FFD and surprisal")
  print(compute_mean(datasets, "FFD", "surprisal.s"))

  print("Correlation between GPT and surprisal")
  print(compute_mean(datasets, "GPT", "surprisal.s"))
  
  print("lme delta loglik for FFD and surprisal")
  print(compute_deltalogliks(datasets, "FFD"))
  
  print("lme delta loglik for GPT and surprisal")
  print(compute_deltalogliks(datasets, "GPT"))
  
}