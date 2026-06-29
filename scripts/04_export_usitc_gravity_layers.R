# export_usitc_gravity_layers.R
#
# Pull binary dyadic agreement layers from the USITC Dynamic Gravity Dataset
# via the usitcgravity R package, map ISO3 -> COW ccode, restrict to
# (ccode1, ccode2, year) triples where BOTH states are COW system members
# in that year, and write each layer as a standard CSV.
#
# Layers exported (pruned 3-layer non-redundant set):
#   layer_fta_undirected.csv          from agree_fta          - trade alignment
#   layer_pta_services_undirected.csv from agree_pta_services - services depth
#   layer_cu_undirected.csv           from agree_cu           - deep integration
#
# Dropped agree_eia (superset of the three above, causes R-GCN relation
# collinearity) and agree_pta_goods (near-duplicate of agree_fta).
#
# IMPORTANT: usitcgravity is a duckdb-backed CONNECTOR, not a data package.
# This script uses the connector API:
#   1. usitcgravity_download() once (~810 MB, cached locally)
#   2. usitcgravity_connect() to get a DBI handle
#   3. DBI::dbGetQuery() to pull just the columns we need with a year filter
#
# Run with R 4.5.0 (not 4.3.1 - older R has ABI mismatches with the
# current cli/vctrs/duckdb binaries on Roar):
#   module load r/4.5.0
#   Rscript scripts/export_usitc_gravity_layers.R [start_year] [end_year]
#
# Requires:
#   data/processed/cow_state_membership.csv (build with export_cow_membership.R first)

set.seed(123)

args <- commandArgs(trailingOnly = TRUE)
start_year <- if (length(args) >= 1) as.integer(args[1]) else 1948L
end_year   <- if (length(args) >= 2) as.integer(args[2]) else 2016L

cow_path <- "data/processed/cow_state_membership.csv"
if (!file.exists(cow_path)) {
  stop("COW membership CSV not found: ", cow_path,
       "\nRun scripts/export_cow_membership.R first.")
}

# ---- Library setup -----------------------------------------------------
user_lib <- Sys.getenv("R_LIBS_USER")
if (nchar(user_lib) == 0) user_lib <- file.path(Sys.getenv("HOME"), "R", "libs")
dir.create(user_lib, showWarnings = FALSE, recursive = TRUE)
.libPaths(c(user_lib, .libPaths()))

ensure_pkg <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat("Installing", pkg, "to", user_lib, "\n")
    install.packages(pkg, lib = user_lib, repos = "https://cloud.r-project.org")
  }
}
ensure_pkg("countrycode")
ensure_pkg("DBI")
ensure_pkg("duckdb")
ensure_pkg("remotes")

if (!requireNamespace("usitcgravity", quietly = TRUE)) {
  cat("Installing usitcgravity from GitHub to", user_lib, "\n")
  remotes::install_github("pachadotdev/usitcgravity", upgrade = "never",
                          lib = user_lib)
}

suppressPackageStartupMessages({
  library(usitcgravity)
  library(countrycode)
  library(DBI)
})

# ---- Make sure the duckdb file is present ------------------------------
cat("=== Ensuring usitcgravity database is downloaded (~810 MB, one-time) ===\n")
usitcgravity_download()

# ---- Connect and inspect schema ----------------------------------------
cat("=== Connecting to usitcgravity duckdb ===\n")
con <- usitcgravity_connect()
on.exit(tryCatch(usitcgravity_disconnect(con), error = function(e) NULL))

tbls <- DBI::dbListTables(con)
cat("  tables:", paste(tbls, collapse = ", "), "\n")

# The gravity table holds the dyadic year-level rows. Tolerate alternate names.
gravity_tbl <- intersect(c("gravity", "dgd", "Gravity"), tbls)[1]
if (is.na(gravity_tbl)) {
  stop("No table named 'gravity' (or 'dgd') in the duckdb. Found: ",
       paste(tbls, collapse = ", "))
}
cat("  gravity table:", gravity_tbl, "\n")

cols <- DBI::dbListFields(con, gravity_tbl)
cat("  gravity columns (first 30):", paste(head(cols, 30), collapse = ", "), "\n")
cat("  total columns:", length(cols), "\n")

# Resolve ISO3 columns flexibly (DGD versions vary)
iso_o_col <- intersect(c("iso3_o", "iso3num_o", "iso_o"), cols)[1]
iso_d_col <- intersect(c("iso3_d", "iso3num_d", "iso_d"), cols)[1]
if (is.na(iso_o_col) || is.na(iso_d_col) || !("year" %in% cols)) {
  stop("Could not find iso3 origin/destination + year columns. Available: ",
       paste(cols, collapse = ", "))
}
cat("  using:", iso_o_col, "/", iso_d_col, "/ year\n")

# Pick agreement columns that are present
needed_agree <- c("agree_fta", "agree_pta_services", "agree_cu")
present_agree <- intersect(needed_agree, cols)
missing_agree <- setdiff(needed_agree, cols)
if (length(missing_agree) > 0) {
  cat("  WARNING: missing agreement columns:",
      paste(missing_agree, collapse = ", "), "\n")
  cat("  agreement-related columns in table:",
      paste(grep("^agree_", cols, value = TRUE), collapse = ", "), "\n")
}
if (length(present_agree) == 0) {
  stop("None of the required agreement columns are in the table.")
}

# ---- Pull only what we need with a SQL year filter ---------------------
sql <- sprintf(
  "SELECT %s AS iso_o, %s AS iso_d, year, %s FROM %s WHERE year BETWEEN %d AND %d",
  iso_o_col, iso_d_col,
  paste(present_agree, collapse = ", "),
  gravity_tbl, start_year, end_year
)
cat("=== Querying ===\n  ", sql, "\n", sep = "")
dgd <- DBI::dbGetQuery(con, sql)
cat("  rows pulled:", nrow(dgd), "\n")

# Disconnect now - rest of work is in-memory
usitcgravity_disconnect(con)
on.exit()

# ---- Map ISO3 -> COW ccode ---------------------------------------------
cat("=== Mapping ISO3 -> COW ccode ===\n")
dgd$source_ccode <- countrycode(dgd$iso_o, origin = "iso3c",
                                 destination = "cown", warn = FALSE)
dgd$target_ccode <- countrycode(dgd$iso_d, origin = "iso3c",
                                 destination = "cown", warn = FALSE)
n_before_iso <- nrow(dgd)
dgd <- dgd[!is.na(dgd$source_ccode) & !is.na(dgd$target_ccode), ]
cat("  rows after iso3->ccode (drop unmappable):", nrow(dgd),
    "(", round(100 * nrow(dgd) / n_before_iso, 1), "% kept)\n")

# Drop self-loops
dgd <- dgd[dgd$source_ccode != dgd$target_ccode, ]
cat("  rows after dropping self-loops:", nrow(dgd), "\n")

# ---- Load COW membership and apply dyadic COW filter -------------------
cat("=== Loading COW membership panel from", cow_path, "===\n")
cow <- read.csv(cow_path)
cow_set <- paste(cow$ccode, cow$year, sep = "_")
cat("  COW (ccode, year) cells:", length(cow_set), "\n")

src_key <- paste(dgd$source_ccode, dgd$year, sep = "_")
tgt_key <- paste(dgd$target_ccode, dgd$year, sep = "_")
keep <- src_key %in% cow_set & tgt_key %in% cow_set
n_before_cow <- nrow(dgd)
dgd <- dgd[keep, ]
cat("  rows after COW dyadic filter:", nrow(dgd),
    "(", round(100 * nrow(dgd) / n_before_cow, 1), "% kept)\n")

dgd_ccodes <- unique(c(dgd$source_ccode, dgd$target_ccode))
cow_ccodes <- unique(cow$ccode)
orphans <- setdiff(dgd_ccodes, cow_ccodes)
if (length(orphans) > 0) {
  cat("  NOTE: DGD ccodes not in COW (already filtered):",
      paste(head(orphans, 20), collapse = ", "),
      if (length(orphans) > 20) "..." else "", "\n")
}

# ---- Helper to write one layer CSV -------------------------------------
write_layer <- function(dgd_df, var, out_path) {
  if (!var %in% names(dgd_df)) {
    cat("  SKIP", var, "(column not present)\n")
    return(invisible(NULL))
  }
  out <- data.frame(
    year         = as.integer(dgd_df$year),
    source_ccode = as.integer(dgd_df$source_ccode),
    target_ccode = as.integer(dgd_df$target_ccode),
    tie          = as.integer(ifelse(is.na(dgd_df[[var]]), 0, dgd_df[[var]]))
  )
  out <- out[!duplicated(out[, c("year", "source_ccode", "target_ccode")]), ]
  out <- out[order(out$year, out$source_ccode, out$target_ccode), ]
  write.csv(out, out_path, row.names = FALSE)
  pos <- sum(out$tie == 1)
  cat("  wrote", basename(out_path), "-", nrow(out), "rows,",
      pos, "ties (", round(100 * pos / nrow(out), 2), "% density)\n")
}

dir.create("data/processed", showWarnings = FALSE, recursive = TRUE)

cat("\n=== Writing COW-filtered layer files (3-layer pruned set) ===\n")
write_layer(dgd, "agree_fta",          "data/processed/layer_fta_undirected.csv")
write_layer(dgd, "agree_pta_services", "data/processed/layer_pta_services_undirected.csv")
write_layer(dgd, "agree_cu",           "data/processed/layer_cu_undirected.csv")

cat("\nDone.\n")
