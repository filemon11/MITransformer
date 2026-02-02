# R/utils.R

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
      dplyr::filter(RT>200) %>%
      dplyr::filter(RT<2000)
    # following modal cutoffs in https://www.rd-alliance.org/sites/default/files/Marsden%2C_Thompson%2C_Plonsky_2018_App_Psych_SPR_synthesis.pdf
  } else {
    data <- data %>%
      dplyr::filter(.data[[goal]]<1000) %>%
      dplyr::filter(.data[[goal]]>80)
    # following middle value in https://link.springer.com/article/10.3758/s13428-023-02137-x
    # and 80 ms minimum visual transduction time (majority) in https://link.springer.com/article/10.3758/s13428-023-02137-x
  }
  cat("nrows after cut-off ", nrow(data), "\n")
  data
}
