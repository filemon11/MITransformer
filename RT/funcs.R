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


# --- Functions ---

compute_means <- function(data, measure, candidates) {
  magrittr::`%>%`(magrittr::`%>%` (data,
    dplyr::group_by(item, zone)),
    dplyr::summarise(
      !!measure := mean(.data[[measure]], na.rm = TRUE),
      !!paste0("SD_", measure) := stats::sd(.data[[measure]], na.rm = TRUE),
      across(dplyr::all_of(candidates), ~mean(.x, na.rm = TRUE)),
      .groups = "drop"
    ))
}

plot_candidate <- function(data, candidate, yvar, ylab, filename_prefix, dsname) {
  means_by_candidate <- magrittr::`%>%`(magrittr::`%>%` (data,
    dplyr::group_by(!!rlang::sym(candidate))),
    dplyr::summarise(
      !!paste0("Mean_", yvar) := mean(.data[[yvar]], na.rm = TRUE),
      !!paste0("SD_", yvar) := stats::sd(.data[[yvar]], na.rm = TRUE),
      Mean_surprisal = mean(surprisal.s, na.rm = TRUE),
      SD_surprisal = stats::sd(surprisal.s, na.rm = TRUE),
      .groups = "drop"
    ))
  
  y_mean <- paste0("Mean_", yvar)
  y_sd <- paste0("SD_", yvar)
  
  ggplot2::ggplot(means_by_candidate, ggplot2::aes_string(x = candidate, y = y_mean)) +
    ggplot2::geom_point(size = 2, color = "blue", alpha = 0.6) +
    ggplot2::geom_line(color = "blue", alpha = 0.8) +
    ggplot2::geom_errorbar(ggplot2::aes_string(ymin = paste0(y_mean, " - ", y_sd),
                             ymax = paste0(y_mean, " + ", y_sd)),
                  width = 0.2, alpha = 0.5) +
    ggplot2::theme_minimal() +
    ggplot2::labs(title = paste("Mean", yvar, "as a Function of", candidate),
         x = candidate, y = ylab) +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 1))
  ggplot2::ggsave(paste0("results/", candidate, "_and_", filename_prefix, "_raw_", dsname, ".pdf"),
         width = 20, height = 10)
  
  ggplot2::ggplot(means_by_candidate, ggplot2::aes_string(x = candidate, y = "Mean_surprisal")) +
    ggplot2::geom_point(size = 2, color = "blue", alpha = 0.6) +
    ggplot2::geom_line(color = "blue", alpha = 0.8) +
    ggplot2::geom_errorbar(ggplot2::aes(ymin = Mean_surprisal - SD_surprisal,
                      ymax = Mean_surprisal + SD_surprisal),
                  width = 0.2, alpha = 0.5) +
    ggplot2::theme_minimal() +
    ggplot2::labs(title = paste("Mean surprisal as a Function of", candidate),
         x = candidate, y = "Mean surprisal") +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 1))
  ggplot2::ggsave(paste0("results/", candidate, "_and_surprisal_raw_", dsname, ".pdf"),
         width = 20, height = 10)
}

compute_correlations <- function(means, target, candidates) {
  purrr::map_dfr(candidates, function(cand) {
    correlation <- stats::cor.test(means[[target]], means[[cand]], method = "pearson")$estimate
    data.frame(Candidate = cand, Pearson_Correlation = correlation)
  })
}

compute_deltaloglik <- function(data, target, candidates) {
  formula_str <- stats::as.formula(paste(target, " ~ frequency.s + length.s + (1|WorkerId) + (1|item)"))
  s0 <- lme4::lmer(formula_str,
             data = stats::na.omit(data),
             REML = FALSE)
  purrr::map_dfr(candidates, function(cand) {
    formula_str <- stats::as.formula(paste(target, " ~ frequency.s + length.s +", cand, "+ (1|WorkerId) + (1|item)"))
    s1 <- lme4::lmer(formula_str, data = stats::na.omit(data), REML = FALSE)
    a <- stats::anova(s0, s1)
    deltaloglik <- (a$logLik[2] - a$logLik[1]) / nrow(data)
    data.frame(Candidate = cand, Delta_LogLik = deltaloglik)
  })
}

prepare_raw_analysis_data <- function(data) {
  dplyr::mutate (data,
      head = head_distance,
      head_abs = abs(head),
      head_left = dplyr::if_else(head > 0, 0, head),
      fdd = first_dependent_distance,
      fdd_abs = abs(fdd),
      fdd_left = dplyr::if_else(fdd > 0, 0, fdd),
      ldds = abs(left_dependents_distance_sum),
      ldc = abs(left_dependents_count)
    )
}