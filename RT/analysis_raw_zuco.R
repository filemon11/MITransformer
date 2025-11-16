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

rm(list = ls())

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


# --- GPT Analysis ---
gpt_means <- funcs$compute_means(data, "GPT", candidates)

walk(candidates_plot, ~funcs$plot_candidate(data, .x, "GPT", "Mean GPT", "GPT", args[1]))

gpt_corr <- funcs$compute_correlations(gpt_means, "GPT", candidates)

print(gpt_corr)

# --- FFD Analysis ---
ffd_means <- funcs$compute_means(data, "FFD", candidates)

walk(candidates_plot, ~funcs$plot_candidate(data, .x, "FFD", "Mean FFD", "FFD", args[1]))

ffd_corr <- funcs$compute_correlations(ffd_means, "FFD", candidates)

print(ffd_corr)

# --- Surprisal Analysis ---
corr_surprisal <- funcs$compute_correlations(gpt_means, "surprisal.s", candidates)

print(corr_surprisal)