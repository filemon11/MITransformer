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

remove_outliers_na <- function(data, cols, id_col = "WorkerId", k = 2.5, by = NULL) {
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
  
  cat("non-na nrows before outlier removal:", nrow(na.omit(data)), "\n")
  
  # Count NAs before (only for cols we might modify)
  na_before <- sum(vapply(cols, function(col) sum(is.na(data[[col]])), numeric(1)))
  
  # Helper that sets outliers to NA within one group
  trim_one_group_to_na <- function(df) {
    for (col in cols) {
      x <- df[[col]]
      
      # bounds from non-missing
      mu <- mean(x, na.rm = TRUE)
      s  <- sd(x, na.rm = TRUE)
      
      # If sd is 0 or NA (too few obs), skip trimming this column in this group
      if (is.na(s) || s == 0) next
      
      lower <- mu - k * s
      upper <- mu + k * s
      
      # mark outliers (but don't treat NA as outlier)
      is_outlier <- !is.na(x) & (x < lower | x > upper)
      x[is_outlier] <- NA_real_
      
      df[[col]] <- x
    }
    df
  }
  
  # Split-apply-combine without requiring dplyr
  out <- if (length(grp) == 1) {
    pieces <- lapply(split(data, data[[id_col]]), trim_one_group_to_na)
    do.call(rbind, pieces)
  } else {
    key <- interaction(data[grp], drop = TRUE, sep = "___")
    pieces <- lapply(split(data, key), trim_one_group_to_na)
    do.call(rbind, pieces)
  }
  
  rownames(out) <- NULL
  
  na_after <- sum(vapply(cols, function(col) sum(is.na(out[[col]])), numeric(1)))
  cat("values set to NA (across cols):", na_after - na_before, "\n")
  
  # Row count unchanged by design
  cat("non-na nrows after outlier replacement:", nrow(na.omit(out)), "\n")
  
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

apply_cutoff_na <- function(data, goal,
                         lowerRT = 200, upperRT = 2000,
                         lowerGPT = 80, upperGPT = 3000,
                         lowerOther = 80, upperOther = 1000) {
  
  cat("non-na nrows before cut-off:", nrow(na.omit(data)), "\n")
  
  data <- data %>%
    dplyr::mutate(
      !!goal := dplyr::case_when(
        goal == "RT" &
          (.data[[goal]] <= lowerRT | .data[[goal]] >= upperRT) ~ NA_real_,
        
        goal == "GPT" &
          (.data[[goal]] <= lowerGPT | .data[[goal]] >= upperGPT) ~ NA_real_,
        
        goal != "RT" & goal != "GPT" &
          (.data[[goal]] <= lowerOther | .data[[goal]] >= upperOther) ~ NA_real_,
        
        TRUE ~ .data[[goal]]
      )
    )
  
  cat("values set to NA:", sum(is.na(data[[goal]])), "\n")
  cat("non-na nrows after cut-off:", nrow(na.omit(data)), "\n")
  
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


remove_na_sentences <- function(data, goal="unknown", sent_col="item") {
  
  cat("non-na nrows before na sentence removal:", nrow(na.omit(data)), "\n")
  
  data <- data %>%
    dplyr::group_by(.data[[sent_col]]) %>%
    dplyr::mutate(
      .flag_true_in_group = any(as.character(.data[[goal]]) == "True", na.rm = TRUE),
      !!goal := dplyr::if_else(.flag_true_in_group, NA, .data[[goal]])
    ) %>%
    dplyr::ungroup() %>%
    dplyr::select(-.flag_true_in_group)
  
  cat("values set to NA:", sum(is.na(data[[goal]])), "\n")
  cat("non-na nrows after na sentence removal:", nrow(na.omit(data)), "\n")
  
  data
}


load_data2 <- function(
    data_dir, goal="GPT", spill=0,
    scale_to_exclude=c("WorkerId", "item", "element", "Text_ID", "zone", "chunksentence"),
    id_col="WorkerId",
    lag_group_cols=c("WorkerId", "Text_ID")
    ){
  # Does not remove rows but replaces outliers with na so that
  # spillover is aligned.
  
  data <- read.csv(data_dir)
  
  # Get names of numeric columns
  numeric_cols <- names(data)[sapply(data, is.numeric)]
  # Subset to numeric columns you want to scale
  excluded_vars <- c(goal, scale_to_exclude)
  cols_to_scale <- setdiff(numeric_cols, excluded_vars)
  
  # Apply cut-off
  # Provo already comes with a minimal value of 81 for GPT
  data <- apply_cutoff_na(data, goal)
  data <- apply_cutoff_na(data, "frequency", lowerOther=2)
  
  # Remove outliers
  data <- remove_outliers_na(data, c(goal), id_col=id_col)
  
  # Add spillover
  data <- add_lags_scaled(
    data,
    lag_cols = cols_to_scale,
    group_cols = lag_group_cols,
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

## re = object of class ranef.mer
ggCaterpillar <- function(re, QQ=TRUE, likeDotplot=TRUE) {
  # from https://stackoverflow.com/a/16511206
  require(ggplot2)
  f <- function(x) {
    pv   <- attr(x, "postVar")
    cols <- 1:(dim(pv)[1])
    se   <- unlist(lapply(cols, function(i) sqrt(pv[i, i, ])))
    ord  <- unlist(lapply(x, order)) + rep((0:(ncol(x) - 1)) * nrow(x), each=nrow(x))
    pDf  <- data.frame(y=unlist(x)[ord],
                       ci=1.96*se[ord],
                       nQQ=rep(qnorm(ppoints(nrow(x))), ncol(x)),
                       ID=factor(rep(rownames(x), ncol(x))[ord], levels=rownames(x)[ord]),
                       ind=gl(ncol(x), nrow(x), labels=names(x)))
    
    if(QQ) {  ## normal QQ-plot
      p <- ggplot(pDf, aes(nQQ, y))
      p <- p + facet_wrap(~ ind, scales="free")
      p <- p + xlab("Standard normal quantiles") + ylab("Random effect quantiles")
    } else {  ## caterpillar dotplot
      p <- ggplot(pDf, aes(ID, y)) + coord_flip()
      if(likeDotplot) {  ## imitate dotplot() -> same scales for random effects
        p <- p + facet_wrap(~ ind)
      } else {           ## different scales for random effects
        p <- p + facet_grid(ind ~ ., scales="free_y")
      }
      p <- p + xlab("Levels") + ylab("Random effects")
    }
    
    p <- p + theme(legend.position="none")
    p <- p + geom_hline(yintercept=0)
    p <- p + geom_errorbar(aes(ymin=y-ci, ymax=y+ci), width=0, colour="black")
    p <- p + geom_point(aes(size=1.2), colour="blue") 
    return(p)
  }
  lapply(re, f)
}
