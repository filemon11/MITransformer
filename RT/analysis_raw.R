# --- Load Libraries ---
suppressPackageStartupMessages({
  library(tidyverse)
  library(ggpubr)
  library(lme4)
  library(MuMIn)
  library(formattable)
  library(data.table)
  library(ggrepel)
  library(comprehenr)
  library(Rmisc)
  library(stringr)
  library(scales)
  library(grid)
  library(MASS)
  library(modelr)
  library(plotrix)
  library(mgcv)
  library(hexbin)
  library(rlang)
})

rm(list=ls())

funcs <- modules::use("funcs.R")

# --- Args & Data Loading ---
args <- commandArgs(trailingOnly = TRUE)
corpus <- args[2]
data_dir <- paste("data/", corpus, "_preprocessed_", sep="")
data <- read.csv(paste0(data_dir, args[1], ".csv"))


# --- Feature Engineering ---
data <- funcs$prepare_raw_analysis_data(data)

# --- Candidate Predictors ---
candidates <- c("surprisal.s", "demberg", "head", "head_abs", "head_left",
                "fdd", "fdd_abs", "fdd_left", "ldds", "ldc")

candidates_plot <- c("head", "head_left", "fdd", "fdd_left", "ldc")

# --- RT Analysis ---

# Compute mean values per (item, zone)
RT_means <- funcs$compute_means(data, "RT", candidates)

walk(candidates_plot, ~funcs$plot_candidate(data, .x, "RT", "Mean RT", "RT", args[1]))

RT_corr <- funcs$compute_correlations(RT_means, "RT", candidates)

RT_loglik <- funcs$compute_deltaloglik(data, "RT", candidates)

results <- merge(RT_corr, RT_loglik, by = "Candidate")

print(results)

# --- Surprisal Analysis ---
corr_surprisal <- funcs$compute_correlations(RT_means, "surprisal.s", candidates)

print(corr_surprisal)