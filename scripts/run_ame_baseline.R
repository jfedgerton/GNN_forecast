# run_ame_baseline.R
#
# Fit an Additive and Multiplicative Effects (AME) latent-space model
# on each layer × year, save per-(layer, year) latent positions, and
# write a small fit-summary CSV. Used as the §5.3 baseline against
# which the R-GCN encoder's embeddings are compared.
#
# We fit one bilinear AME per (layer, year) cell using the `amen`
# package. To keep compute manageable we restrict to focal years
# {1985, 1995, 2005, 2015} per layer (one per decade), which is the
# cadence used by the per-year R^2 comparison in §5.3.
#
# Outputs:
#   outputs/ame_baseline/<layer>/ame_latent_<year>.csv
#   outputs/ame_baseline/ame_fit_summary.csv
#
# Usage:
#   Rscript scripts/run_ame_baseline.R [start_year] [end_year]

set.seed(123)

args <- commandArgs(trailingOnly = TRUE)
start_year <- if (length(args) >= 1) as.integer(args[1]) else 1948L
end_year   <- if (length(args) >= 2) as.integer(args[2]) else 2016L

ensure_pkg <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    user_lib <- Sys.getenv("R_LIBS_USER")
    if (nchar(user_lib) == 0) user_lib <- file.path(Sys.getenv("HOME"), "R", "library")
    dir.create(user_lib, showWarnings = FALSE, recursive = TRUE)
    .libPaths(c(user_lib, .libPaths()))
    install.packages(pkg, lib = user_lib, repos = "https://cloud.r-project.org")
  }
}
if (nchar(Sys.getenv("R_LIBS_USER")) > 0) {
  .libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
}
ensure_pkg("amen")
suppressPackageStartupMessages({
  library(amen)
})

cow_path <- "data/processed/cow_state_membership.csv"
if (!file.exists(cow_path)) {
  stop("Need data/processed/cow_state_membership.csv. ",
       "Run scripts/export_cow_membership.R first.")
}
cow <- read.csv(cow_path)
cow$ccode <- as.integer(cow$ccode)
cow$year  <- as.integer(cow$year)

# Layers and their CSVs
layers <- list(
  defensive_alliances = "data/processed/layer_alliances_defensive_offensive_undirected.csv",
  dca                 = "data/processed/layer_dca_undirected.csv",
  fta                 = "data/processed/layer_fta_undirected.csv",
  pta_services        = "data/processed/layer_pta_services_undirected.csv",
  cu                  = "data/processed/layer_cu_undirected.csv"
)

# Focal years (one per decade)
default_focal_years <- c(1985, 1995, 2005, 2015)

out_root <- "outputs/ame_baseline"
dir.create(out_root, showWarnings = FALSE, recursive = TRUE)
summary_rows <- list()

build_year_matrix <- function(layer_df, year, ccodes) {
  sub <- layer_df[layer_df$year == year, ]
  n <- length(ccodes)
  Y <- matrix(0L, nrow = n, ncol = n,
              dimnames = list(as.character(ccodes), as.character(ccodes)))
  if (nrow(sub) == 0) return(Y)
  for (i in seq_len(nrow(sub))) {
    s <- as.character(sub$source_ccode[i])
    t <- as.character(sub$target_ccode[i])
    if (s %in% rownames(Y) && t %in% colnames(Y)) {
      tie <- as.integer(sub$tie[i])
      if (!is.na(tie) && tie == 1) {
        Y[s, t] <- 1L; Y[t, s] <- 1L
      }
    }
  }
  Y
}

for (lname in names(layers)) {
  lpath <- layers[[lname]]
  if (!file.exists(lpath)) {
    cat("SKIP layer", lname, "- file not found:", lpath, "\n")
    next
  }
  cat("=== Layer:", lname, "===\n")
  layer_df <- read.csv(lpath)
  layer_df$source_ccode <- as.integer(layer_df$source_ccode)
  layer_df$target_ccode <- as.integer(layer_df$target_ccode)
  layer_df$year         <- as.integer(layer_df$year)
  available_years <- sort(unique(layer_df$year))
  focal_years <- intersect(default_focal_years, available_years)
  focal_years <- focal_years[focal_years >= start_year & focal_years <= end_year]
  cat("  focal years:", paste(focal_years, collapse = ", "), "\n")

  out_dir <- file.path(out_root, lname)
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

  for (yr in focal_years) {
    ccodes_yr <- cow$ccode[cow$year == yr]
    if (length(ccodes_yr) < 10) {
      cat("  SKIP year", yr, "- too few COW members\n")
      next
    }
    Y <- build_year_matrix(layer_df, yr, ccodes_yr)
    pos_density <- mean(Y == 1L)
    cat("  year", yr, ":", nrow(Y), "x", ncol(Y), " density =",
        round(pos_density * 100, 3), "%\n")
    if (pos_density < 0.001 || pos_density > 0.95) {
      cat("    SKIP - density too extreme for AME\n")
      next
    }
    fit <- tryCatch({
      ame(Y = Y, family = "bin", R = 2, nscan = 1000, burn = 500,
          odens = 25, plot = FALSE, print = FALSE)
    }, error = function(e) { cat("    ERROR fitting AME:", conditionMessage(e), "\n"); NULL })
    if (is.null(fit)) next

    U <- fit$U
    if (is.null(U)) { cat("    no U returned\n"); next }
    if (length(dim(U)) == 3) {
      U_mean <- apply(U, c(1, 2), mean)
    } else {
      U_mean <- U
    }
    out_path <- file.path(out_dir, paste0("ame_latent_", yr, ".csv"))
    out_df <- data.frame(
      ccode = as.integer(rownames(Y)),
      latent_dim_1 = U_mean[, 1],
      latent_dim_2 = U_mean[, 2]
    )
    write.csv(out_df, out_path, row.names = FALSE)
    cat("    wrote", out_path, "\n")

    summary_rows[[length(summary_rows) + 1]] <- data.frame(
      layer = lname, year = yr, n_states = nrow(Y),
      density = pos_density,
      stringsAsFactors = FALSE
    )
  }
}

if (length(summary_rows) > 0) {
  summary_df <- do.call(rbind, summary_rows)
  summary_path <- file.path(out_root, "ame_fit_summary.csv")
  write.csv(summary_df, summary_path, row.names = FALSE)
  cat("\nWrote", summary_path, "\n")
}
cat("\nDone.\n")
