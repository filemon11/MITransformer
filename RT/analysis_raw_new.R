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

if (corpus == "frank_SP" || corpus == "naturalstories") {
  corpus_type <- "SP"
} else {
  corpus_type <- "ET"
}

# --- Feature Engineering ---
data <- funcs$prepare_raw_analysis_data(data)

# --- Candidate Predictors ---
candidates <- c("surprisal.s", "demberg", "head", "head_abs", "head_left",
                "fdd", "fdd_abs", "fdd_left", "ldds", "ldc")

candidates_plot <- c("head", "head_left", "fdd", "fdd_left", "ldc", "demberg")

# --- RT and surprisal Analysis ---

analyse <- function(data, to_predict, candidates, candidates_plot) {
  means <- funcs$compute_means(data, to_predict, candidates)
  walk(candidates_plot, ~funcs$plot_candidate(
    data, .x, to_predict, paste("Mean", to_predict), to_predict, args[1]))
  corr <- funcs$compute_correlations(means, to_predict, candidates)
  loglik <- funcs$compute_deltaloglik(data, to_predict, candidates)
  corr_surprisal <- funcs$compute_correlations(means, "surprisal.s", candidates)
  list(merge(corr, loglik, by = "Candidate"), corr_surprisal)
}

if (corpus_type == "SP") {
  print(analyse(data, "RT", candidates, candidates_plot))
} else {
  print(analyse(data, "FFD", candidates, candidates_plot))
  print(analyse(data, "GPT", candidates, candidates_plot))
}