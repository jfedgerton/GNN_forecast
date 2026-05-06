# run_ame_baseline.R
# AME (Additive and Multiplicative Effects) baseline comparison for the
# multiplex GNN forecast paper.
#
# Fits Hoff's AME model (via the amen package) to each layer × year of the
# peacesciencer-exported network data. Saves the latent positions per
# (layer, year) so the Python pipeline can compare AME embeddings against
# the GNN embeddings on the same evaluation tasks.
#
# Why this matters: PA reviewers will ask why a GNN is needed if AME exists.
# This script provides the head-to-head comparison data.
#
# Usage:
#   Rscript scripts/run_ame_baseline.R --data-dir data/processed \
#                                       --out-dir outputs/ame_baseline \
#                                       --start-year 1985 --end-year 2025 \
#                                       --emb-dim 32

suppressPackageStartupMessages({
  if (!requireNamespace("amen", quietly = TRUE)) {
    stop("Please install.packages('amen') before running this script.")
  }
  library(amen)
  library(tidyverse)
  library(optparse)
})

set.seed(123)

# ---- CLI ----
option_list <- list(
  make_option("--data-dir", type = "character", default = "data/processed"),
  make_option("--out-dir", type = "character", default = "outputs/ame_baseline"),
  make_option("--start-year", type = "integer", default = 1985),
  make_option("--end-year", type = "integer", default = 2025),
  make_option("--emb-dim", type = "integer", default = 32,
              help = "Latent dimension R in AME ame() call"),
  make_option("--n-iter", type = "integer", default = 5000),
  make_option("--burn", type = "integer", default = 1000),
  make_option("--odens", type = "integer", default = 10,
              help = "Thinning interval")
)
opt <- parse_args(OptionParser(option_list = option_list))

data_dir <- opt$`data-dir`
out_dir <- opt$`out-dir`
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ---- Load layers ----
layer_files <- list(
  defensive_alliances = "layer_alliances_defensive_offensive_undirected.csv",
  igo                 = "layer_igo_shared_undirected.csv",
  trade               = "layer_trade_undirected.csv"
)

# Hardcoded fallback search to mirror the Python loader's discovery logic
candidate_files <- list(
  defensive_alliances = c(
    "layer_defensive_alliances.csv",
    "layer_alliances_defensive_undirected.csv",
    "layer_alliances_defensive_offensive_undirected.csv"
  ),
  igo = c(
    "layer_igo_shared_undirected.csv",
    "layer_igo.csv"
  ),
  trade = c(
    "layer_trade_undirected.csv",
    "layer_trade.csv"
  )
)

# Resolve actual filenames
resolved <- list()
for (ln in names(candidate_files)) {
  for (fn in candidate_files[[ln]]) {
    p <- file.path(data_dir, fn)
    if (file.exists(p)) {
      resolved[[ln]] <- p
      break
    }
  }
}

if (length(resolved) == 0) {
  stop("No layer files found in ", data_dir)
}

cat("Resolved layer files:\n")
for (ln in names(resolved)) {
  cat("  ", ln, "->", resolved[[ln]], "\n")
}

# Build a unified node list across all layers and years in range
all_ccodes <- c()
layer_data <- list()
for (ln in names(resolved)) {
  df <- read_csv(resolved[[ln]], show_col_types = FALSE)
  # Normalize columns
  if ("ccode1" %in% names(df)) df <- rename(df, source_ccode = ccode1)
  if ("ccode2" %in% names(df)) df <- rename(df, target_ccode = ccode2)
  if (!"tie" %in% names(df) && "value" %in% names(df)) df <- rename(df, tie = value)
  df <- df %>% filter(year >= opt$`start-year`, year <= opt$`end-year`)
  layer_data[[ln]] <- df
  all_ccodes <- union(all_ccodes, c(df$source_ccode, df$target_ccode))
}
all_ccodes <- sort(unique(as.integer(all_ccodes)))
n <- length(all_ccodes)
cat("Total nodes across layers:", n, "\n")

# Helper: build dense N×N adjacency for one layer-year
build_adj <- function(df_year, ccodes) {
  N <- length(ccodes)
  A <- matrix(0, nrow = N, ncol = N,
              dimnames = list(as.character(ccodes), as.character(ccodes)))
  for (i in seq_len(nrow(df_year))) {
    s <- as.character(df_year$source_ccode[i])
    t <- as.character(df_year$target_ccode[i])
    if (s %in% rownames(A) && t %in% colnames(A)) {
      A[s, t] <- df_year$tie[i]
      A[t, s] <- df_year$tie[i]  # undirected
    }
  }
  diag(A) <- NA
  A
}

# ---- Fit AME per (layer, year) ----
cat("\nFitting AME per layer-year (R =", opt$`emb-dim`, ")\n")

results_summary <- list()
for (ln in names(resolved)) {
  cat("\n=== Layer:", ln, "===\n")
  df <- layer_data[[ln]]
  years <- sort(unique(df$year))

  layer_dir <- file.path(out_dir, ln)
  dir.create(layer_dir, showWarnings = FALSE, recursive = TRUE)

  for (yr in years) {
    cat("  Year", yr, "...\n")
    df_y <- filter(df, year == yr)
    A <- build_adj(df_y, all_ccodes)

    # Determine the AME model family based on data type
    family <- if (ln == "trade") "nrm" else "bin"

    ame_fit <- tryCatch({
      ame(
        Y       = A,
        family  = family,
        R       = opt$`emb-dim`,
        symmetric = TRUE,
        nscan   = opt$`n-iter`,
        burn    = opt$burn,
        odens   = opt$odens,
        plot    = FALSE,
        print   = FALSE,
        seed    = 123
      )
    }, error = function(e) {
      cat("    AME failed for", ln, yr, ":", conditionMessage(e), "\n")
      NULL
    })

    if (is.null(ame_fit)) next

    # Save latent positions (posterior mean of U)
    U <- ame_fit$U
    if (!is.null(U)) {
      write.csv(
        data.frame(ccode = as.integer(rownames(U)), U),
        file = file.path(layer_dir, sprintf("ame_latent_%d.csv", yr)),
        row.names = FALSE
      )
    }

    results_summary[[length(results_summary) + 1]] <- data.frame(
      layer = ln, year = yr,
      family = family,
      n_nodes = n,
      n_iter = opt$`n-iter`,
      saved = !is.null(U)
    )
  }
}

if (length(results_summary) > 0) {
  summary_df <- bind_rows(results_summary)
  write.csv(summary_df, file.path(out_dir, "ame_fit_summary.csv"), row.names = FALSE)
  cat("\nWrote summary to", file.path(out_dir, "ame_fit_summary.csv"), "\n")
}

cat("\nAME baseline complete. Use compare_ame_to_gnn.py to evaluate.\n")
