# R/utils.R

scaling_var <- function(x) {
  x <- base::as.numeric(x)
  mu <- base::mean(x, na.rm = TRUE)
  s  <- stats::sd(x, na.rm = TRUE)
  z  <- (x - mu) / s
  base::attr(z, "scaled:center") <- mu
  base::attr(z, "scaled:scale")  <- s
  z
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

apply_cutoff <- function(data, goal, lowerRT = 200, upperRT = 2000, lowerGPT = 80, upperGPT = 3000, lowerOther = 80, upperOther = 1000) {
  cat("nrows before cut-off ", nrow(data), "\n")
  if (goal == "RT") {
    data <- data %>%
      dplyr::filter(RT>lowerRT) %>%
      dplyr::filter(RT<upperRT)
    # following modal cutoffs in https://www.rd-alliance.org/sites/default/files/Marsden%2C_Thompson%2C_Plonsky_2018_App_Psych_SPR_synthesis.pdf
  } else if (goal == "GPT") {
    data <- data %>%
      dplyr::filter(.data[[goal]]<upperGPT) %>%
      dplyr::filter(.data[[goal]]>lowerGPT)
    # following 3*middle value in https://link.springer.com/article/10.3758/s13428-023-02137-x
    # due to GPT being a total reading time
    # and 80 ms minimum visual transduction time (majority) in https://link.springer.com/article/10.3758/s13428-023-02137-x
  } else {
    data <- data %>%
      dplyr::filter(.data[[goal]]<upperOther) %>%
      dplyr::filter(.data[[goal]]>lowerOther)
    # following middle value in https://link.springer.com/article/10.3758/s13428-023-02137-x
    # and 80 ms minimum visual transduction time (majority) in https://link.springer.com/article/10.3758/s13428-023-02137-x
  }
  cat("nrows after cut-off ", nrow(data), "\n")
  data
}

add_lags_scaled <- function(data,
                            lag_cols,
                            group_cols = c(WorkerId, item),
                            n_lags = 3,
                            lag_prefix = "") {
  if (!is.character(group_cols)) {
    # allow bare names too
    group_cols <- vapply(enexpr(group_cols), as_name, character(1))
  }
  
  # 2) add lags for columns (e.g., logp1, logp2, ...)
  out <- data %>%
    group_by(across(all_of(group_cols))) %>%
    mutate(
      across(
        .cols = !!lag_cols,
        .fns  = setNames(
          lapply(seq_len(n_lags), function(k) function(x) lag(x, k)),
          paste0(lag_prefix, seq_len(n_lags))
        ),
        .names = "{.col}{.fn}"
      )
    ) %>%
    ungroup()

  out
}

plot_bins <- function(model,
                      xname,
                      bins = 25,
                      response_transform = c("auto", "none", "exp"),
                      addendum = "",
                      x_backtransform = TRUE,
                      relabel_x_axis = TRUE,
                      data = NULL) {
  
  response_transform <- base::match.arg(response_transform)
  
  # Build sjPlot term string (preserve your addendum behaviour)
  term_string <- xname # base::trimws(base::paste(xname, addendum))
  
  p <- sjPlot::plot_model(
    model,
    type  = "pred",
    terms = term_string
  )
  
  mf <- stats::model.frame(model)
  y  <- stats::model.response(mf)
  
  # --- Locate x column robustly (handles transformed terms) ---
  xcol <- NULL
  if (xname %in% base::names(mf)) {
    xcol <- xname
  } else {
    hits <- base::grep(xname, base::names(mf), value = TRUE)
    if (base::length(hits) == 1L) {
      xcol <- hits
    } else if (base::length(hits) == 0L) {
      base::stop("Could not find xname '", xname, "' in model.frame(model). ",
                 "Available columns: ", base::paste(base::names(mf), collapse = ", "))
    } else {
      base::stop("xname '", xname, "' matched multiple columns in model.frame(model): ",
                 base::paste(hits, collapse = ", "),
                 ". Please pass the exact column name (e.g., one of those matches).")
    }
  }
  
  x <- mf[[xcol]]
  
  # Require numeric for binning
  if (!base::is.numeric(x)) {
    base::stop("Predictor column '", xcol, "' is not numeric; cannot bin.")
  }
  
  # --- Detect scaling metadata (works for scale() and for your scaling_var that sets attributes) ---
  x_center <- base::attr(x, "scaled:center", exact = TRUE)
  x_scale  <- base::attr(x, "scaled:scale",  exact = TRUE)
  
  if ((base::is.null(x_center) || base::is.null(x_scale)) && !base::is.null(data)) {
    if (xname %in% base::names(data)) {
      x_center <- base::attr(data[[xname]], "scaled:center", exact = TRUE)
      x_scale  <- base::attr(data[[xname]], "scaled:scale",  exact = TRUE)
    }
  }
  
  
  has_scale_meta <- x_backtransform &&
    !base::is.null(x_center) &&
    !base::is.null(x_scale) &&
    base::length(x_center) == 1L &&
    base::length(x_scale)  == 1L &&
    base::is.finite(base::as.numeric(x_center)) &&
    base::is.finite(base::as.numeric(x_scale)) &&
    base::as.numeric(x_scale) != 0
  
  # --- Bin on x as used in the model (scaled) ---
  breaks <- base::seq(
    base::min(x, na.rm = TRUE),
    base::max(x, na.rm = TRUE),
    length.out = bins + 1
  )
  mids_scaled <- (breaks[-1] + breaks[-base::length(breaks)]) / 2
  
  # Plot overlay at the same x-scale as the model (scaled),
  # and only RELABEL the axis to original units
  mids_plot <- mids_scaled
  
  # --- Summarise observed y within bins (on model's y-scale), then optionally back-transform y ---
  binned <- dplyr::tibble(x = x, y = y) |>
    dplyr::mutate(
      bin_id = base::as.integer(base::cut(x, breaks = breaks, include.lowest = TRUE))
    ) |>
    dplyr::filter(!base::is.na(bin_id)) |>
    dplyr::group_by(bin_id) |>
    dplyr::summarise(
      n = base::sum(!base::is.na(y)),
      mean_y = base::mean(y, na.rm = TRUE),
      se_y   = stats::sd(y, na.rm = TRUE) / base::sqrt(n),
      tcrit  = stats::qt(0.975, df = base::pmax(n - 1, 1)),
      ci_low = mean_y - tcrit * se_y,
      ci_high= mean_y + tcrit * se_y,
      .groups = "drop"
    ) |>
    dplyr::left_join(
      dplyr::tibble(bin_id = base::seq_along(mids_plot), x_mid = mids_plot),
      by = "bin_id"
    )
  
  # Decide transformation for overlay
  if (response_transform == "auto") {
    # Keep behaviour explicit: default to none unless user sets exp
    response_transform <- "none"
  }
  
  if (response_transform == "exp") {
    binned <- dplyr::mutate(
      binned,
      mean_y = base::exp(mean_y),
      ci_low = base::exp(ci_low),
      ci_high = base::exp(ci_high)
    )
  }
  
  # Optionally relabel x-axis ticks to original units (if we have scaling metadata)
  if (relabel_x_axis && has_scale_meta) {
    # infer original column name from "frequency.s"
    x_unscaled <- base::sub("\\.s$", "", xname)
    
    # original (unscaled) predictor values
    x_orig <- data[[x_unscaled]]
  
    br <- make_nice_breaks_scaled(x_center, x_scale, x_orig, n = 7)
    
    p <- p + ggplot2::scale_x_continuous(
      breaks = br$breaks_scaled,
      labels = function(z) {
        # ignore z; print the precomputed original breaks with appropriate accuracy
        scales::label_number(accuracy = br$step)(br$labels_orig)
      }
    )
    
  }
  
  
  # Overlay binned means + CI
  p <- p +
    ggplot2::geom_errorbar(
      data = binned,
      mapping = ggplot2::aes(x = x_mid, ymin = ci_low, ymax = ci_high),
      inherit.aes = FALSE,
      width = 0
    ) +
    ggplot2::geom_point(
      data = binned,
      mapping = ggplot2::aes(x = x_mid, y = mean_y),
      inherit.aes = FALSE,
      size = 2
    )
  plot(p)
  p
}

make_nice_breaks_scaled <- function(x_center, x_scale, x_orig, n = 6) {
  mu <- base::as.numeric(x_center)
  s  <- base::as.numeric(x_scale)
  
  candidates <- c(1, 0.5, 0.25, 0.2, 0.1, 2, 5, 10)
  
  rng  <- base::range(x_orig, na.rm = TRUE)
  span <- rng[2] - rng[1]
  
  step <- candidates[1]
  for (st in candidates) {
    k <- span / st
    if (base::is.finite(k) && k <= (n - 1)) { step <- st; break }
  }
  
  lo <- base::floor(rng[1] / step) * step
  hi <- base::ceiling(rng[2] / step) * step
  br_orig <- base::seq(lo, hi, by = step)
  
  # ---- NEW: snap to the step grid to avoid 0.50000000002-type values ----
  # decimals needed for the step (e.g., step=0.5 -> 1 decimal; 0.25 -> 2 decimals)
  dec <- base::max(0L, base::ceiling(-base::log10(step)))
  br_orig <- base::round(br_orig / step) * step
  br_orig <- base::round(br_orig, digits = dec + 2L)
  
  br_scaled <- (br_orig - mu) / s
  
  list(breaks_scaled = br_scaled, labels_orig = br_orig, step = step)
}

apply_scaling <- function(data, cols_to_scale) {
  data %>% dplyr::mutate(
    dplyr::across(
      dplyr::all_of(cols_to_scale),
      scaling_var,
      .names = "{.col}.s"
    )
  )
}

spillover_and_scale <- function(data, cols_to_scale, spill=1, group_cols=c("WorkerId", "Text_ID")) {
  # Add spillover
  data <- add_lags_scaled(
    data,
    lag_cols = cols_to_scale,
    group_cols = group_cols,
    n_lags = spill
  )
  
  # Apply scaling
  data <- apply_scaling(data, cols_to_scale)
  
  # Apply scaling for spillover columns
  if (spill > 0) {
    data <- apply_scaling(data, unlist(lapply(seq_len(spill), function(i) paste0(cols_to_scale, i))))
  }
  data
}
