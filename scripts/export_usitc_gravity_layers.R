# export_usitc_gravity_layers.R
#
# Pull binary dyadic agreement layers from the USITC Dynamic Gravity Dataset
# (via the usitcgravity R package), map ISO3 -> COW ccode, restrict to
# (ccode1, ccode2, year) triples where BOTH states are COW system members
# in that year, and write each layer as a standard CSV.
#
# Layers exported:
#   layer_fta_undirected.csv         from agree_fta
#   layer_pta_goods_undirected.csv   from agree_pta_goods
#   layer_cu_undirected.csv          from agree_cu
#   layer_eia_undirected.csv         from agree_eia
#
# Requires:
#   data/processed/cow_state_membership.csv (build with export_cow_membership.R first)
#
# Usage:
#   Rscript scripts/export_usitc_gravity_layers.R [start_year] [end_year]

set.seed(123)

args <- commandArgs(trailingOnly = TRUE)
start_year <- if (length(args) >= 1) as.integer(args[1]) else 1948L
end_year   <- if (length(args) >= 2) as.integer(args[2]) else 2016L

cow_path <- "data/processed/cow_state_membership.csv"
if (!file.exists(cow_path)) {
  stop("COW membership CSV not found: ", cow_path,
       "\nRun scripts/export_cow_membership.R first.")
}

ensure_pkg <- function(pkg, install_args = list()) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat("Installing", pkg, "to user library...\n")
    user_lib <- Sys.getenv("R_LIBS_USER")
    if (nchar(user_lib) == 0) user_lib <- file.path(Sys.getenv("HOME"), "R", "library")
    dir.create(user_lib, showWarnings = FALSE, recursive = TRUE)
    .libPaths(c(user_lib, .libPaths()))
    do.call(install.packages,
            c(list(pkg, lib = user_lib, repos = "https://cloud.r-project.org"),
              install_args))
  }
}
if (nchar(Sys.getenv("R_LIBS_USER")) > 0) {
  .libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
}

ensure_pkg("countrycode")

if (!requireNamespace("usitcgravity", quietly = TRUE)) {
  ensure_pkg("remotes")
  cat("Installing usitcgravity from GitHub...\n")
  remotes::install_github("pachadotdev/usitcgravity", upgrade = "never",
                          lib = Sys.getenv("R_LIBS_USER"))
}

suppressPackageStartupMessages({
  library(usitcgravity)
  library(countrycode)
  library(dplyr)
  library(readr)
})

cat("Loading DGD\n")
dgd <- tryCatch(usitcgravity::gravity,
                error = function(e) tryCatch(usitcgravity::dgd,
                  error = function(e2) stop("Could not load gravity table from usitcgravity")))
cat("  raw rows:", nrow(dgd), "\n")
cat("  columns (first 20):", paste(head(names(dgd), 20), collapse = ", "), "\n")

# Find required columns flexibly (DGD column naming varies by version)
iso_o_col <- intersect(c("iso3_o", "iso3num_o", "iso_o"), names(dgd))[1]
iso_d_col <- intersect(c("iso3_d", "iso3num_d", "iso_d"), names(dgd))[1]
year_col  <- "year"
if (is.null(iso_o_col) || is.null(iso_d_col) || !(year_col %in% names(dgd))) {
  stop("Could not find iso3 origin/destination + year columns in DGD")
}
cat("  using columns:", iso_o_col, "/", iso_d_col, "/", year_col, "\n")

# Year subset
dgd <- dgd[dgd[[year_col]] >= start_year & dgd[[year_col]] <= end_year, ]
cat("  rows in year range:", nrow(dgd), "\n")

# ---- Map ISO3 -> COW ccode ----
cat("Mapping ISO3 -> COW ccode\n")
dgd$source_ccode <- countrycode(dgd[[iso_o_col]], origin = "iso3c",
                                 destination = "cown", warn = FALSE)
dgd$target_ccode <- countrycode(dgd[[iso_d_col]], origin = "iso3c",
                                 destination = "cown", warn = FALSE)
n_before_iso <- nrow(dgd)
dgd <- dgd[!is.na(dgd$source_ccode) & !is.na(dgd$target_ccode), ]
cat("  rows after iso3->ccode (drop unmappable):", nrow(dgd),
    "(", round(100 * nrow(dgd) / n_before_iso, 1), "% kept)\n")

# Drop self-loops
dgd <- dgd[dgd$source_ccode != dgd$target_ccode, ]
cat("  rows after dropping self-loops:", nrow(dgd), "\n")

# ---- Load COW membership and apply dyadic COW filter ----
cat("Loading COW membership panel from", cow_path, "\n")
cow <- read.csv(cow_path)
cow_set <- paste(cow$ccode, cow$year, sep = "_")
cat("  COW (ccode, year) cells:", length(cow_set), "\n")

src_key <- paste(dgd$source_ccode, dgd[[year_col]], sep = "_")
tgt_key <- paste(dgd$target_ccode, dgd[[year_col]], sep = "_")
keep <- src_key %in% cow_set & tgt_key %in% cow_set
n_before_cow <- nrow(dgd)
dgd <- dgd[keep, ]
cat("  rows after COW dyadic filter:", nrow(dgd),
    "(", round(100 * nrow(dgd) / n_before_cow, 1), "% kept)\n")

# Validation: which DGD ccodes never appear in COW?
dgd_ccodes <- unique(c(dgd$source_ccode, dgd$target_ccode))
cow_ccodes <- unique(cow$ccode)
orphans <- setdiff(dgd_ccodes, cow_ccodes)
if (length(orphans) > 0) {
  cat("  NOTE: DGD ccodes not in COW (filtered out earlier):",
      paste(head(orphans, 20), collapse = ", "),
      if (length(orphans) > 20) "..." else "", "\n")
}

# ---- Helper to write one layer CSV ----
write_layer <- function(dgd_df, var, out_path) {
  if (!var %in% names(dgd_df)) {
    cat("  SKIP", var, "(column not present)\n")
    return(invisible(NULL))
  }
  out <- data.frame(
    year         = as.integer(dgd_df[[year_col]]),
    source_ccode = as.integer(dgd_df$source_ccode),
    target_ccode = as.integer(dgd_df$target_ccode),
    tie          = as.integer(ifelse(is.na(dgd_df[[var]]), 0, dgd_df[[var]]))
  )
  out <- out[!duplicated(out[, c("year", "source_ccode", "target_ccode")]), ]
  out <- out[order(out$year, out$source_ccode, out$target_ccode), ]
  write_csv(out, out_path)
  pos <- sum(out$tie == 1)
  cat("  wrote", basename(out_path), "—", nrow(out), "rows,",
      pos, "ties (", round(100 * pos / nrow(out), 2), "% density)\n")
}

dir.create("data/processed", showWarnings = FALSE, recursive = TRUE)

cat("\nWriting COW-filtered layer files\n")
write_layer(dgd, "agree_fta",       "data/processed/layer_fta_undirected.csv")
write_layer(dgd, "agree_pta_goods", "data/processed/layer_pta_goods_undirected.csv")
write_layer(dgd, "agree_cu",        "data/processed/layer_cu_undirected.csv")
write_layer(dgd, "agree_eia",       "data/processed/layer_eia_undirected.csv")

cat("\nDone.\n")
