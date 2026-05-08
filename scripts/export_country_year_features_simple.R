# export_country_year_features_simple.R
#
# Build a per-(country, year) covariate CSV for the GNN node-feature
# pipeline. Uses ONLY base R + already-installed packages (peacesciencer,
# dplyr, readr, tibble) to avoid the tidyverse / optparse dependency.
#
# Usage:
#   Rscript scripts/export_country_year_features_simple.R [start] [end] [out]
#   Rscript scripts/export_country_year_features_simple.R 1816 2025 data/processed/node_features.csv

suppressPackageStartupMessages({
  library(peacesciencer)
  library(dplyr)
  library(readr)
})

set.seed(123)

args <- commandArgs(trailingOnly = TRUE)
start_year <- if (length(args) >= 1) as.integer(args[1]) else 1816L
end_year   <- if (length(args) >= 2) as.integer(args[2]) else 2025L
out_path   <- if (length(args) >= 3) args[3] else "data/processed/node_features.csv"

cat("Building country-year feature panel\n")
cat("  years:", start_year, "to", end_year, "\n")
cat("  out:  ", out_path, "\n\n")

panel <- create_stateyears(system = "cow", subset_years = start_year:end_year)
cat("  base panel rows:", nrow(panel), "\n")

# Try each peacesciencer enricher; skip silently if it fails
try_add <- function(p, fn_name) {
  fn <- tryCatch(get(fn_name, envir = asNamespace("peacesciencer")),
                 error = function(e) NULL)
  if (is.null(fn)) {
    cat("  skipped", fn_name, "(not in this peacesciencer version)\n")
    return(p)
  }
  out <- tryCatch(fn(p),
                  error = function(e) {
                    cat("  WARNING:", fn_name, "failed:", conditionMessage(e), "\n")
                    p
                  })
  cat("  applied", fn_name, "\n")
  out
}

panel <- try_add(panel, "add_nmc")
panel <- try_add(panel, "add_democracy")
panel <- try_add(panel, "add_sim_gdp_pop")

# Capital coordinates (one-time merge if cow_states is exposed)
cap_coords <- tryCatch({
  cs <- peacesciencer::cow_states
  caplat_col <- intersect(c("caplat","cap_lat","capcity_lat","capital_lat"), names(cs))[1]
  caplon_col <- intersect(c("caplong","cap_lon","capcity_lon","capital_lon"), names(cs))[1]
  if (is.null(caplat_col) || is.null(caplon_col)) {
    NULL
  } else {
    out <- cs[, c("ccode", caplat_col, caplon_col)]
    names(out) <- c("ccode", "lat", "lon")
    out <- out[!duplicated(out$ccode), ]
    out
  }
}, error = function(e) NULL)

if (!is.null(cap_coords)) {
  panel <- left_join(panel, cap_coords, by = "ccode")
  cat("  applied capital coordinates\n")
} else {
  panel$lat <- NA_real_
  panel$lon <- NA_real_
}

# Build the standardized output column set (handles missing source columns)
build_log <- function(x) ifelse(is.na(x) | x <= 0, NA_real_, log(x))
get_col   <- function(df, name) if (name %in% names(df)) df[[name]] else rep(NA_real_, nrow(df))

# add_sim_gdp_pop() column names vary across peacesciencer versions:
#   newer (>=1.2): wbgdp2011est, wbpopest          (already log-scale)
#   older (1.0-1.1): sdpest, popest                (already log-scale)
# Try newer first, fall back to older. Report which we found.
gdp_col <- intersect(c("wbgdp2011est", "sdpest"), names(panel))[1]
pop_col <- intersect(c("wbpopest",     "popest"), names(panel))[1]
if (is.na(gdp_col)) {
  cat("  WARNING: no GDP column found in panel (looked for wbgdp2011est, sdpest)\n")
  cat("  available columns:", paste(names(panel), collapse = ", "), "\n")
} else {
  cat("  GDP column found:", gdp_col, "\n")
}
if (is.na(pop_col)) {
  cat("  WARNING: no population column found in panel (looked for wbpopest, popest)\n")
} else {
  cat("  Pop column found:", pop_col, "\n")
}

lp_gdp_vec <- if (!is.na(gdp_col)) panel[[gdp_col]] else rep(NA_real_, nrow(panel))
lp_pop_vec <- if (!is.na(pop_col)) panel[[pop_col]] else rep(NA_real_, nrow(panel))

out <- tibble(
  ccode      = panel$ccode,
  year       = panel$year,
  lp_gdp     = lp_gdp_vec,
  lp_pop     = lp_pop_vec,
  lp_gdppc   = lp_gdp_vec - lp_pop_vec,
  polity2    = get_col(panel, "polity2"),
  cinc       = get_col(panel, "cinc"),
  milex_log  = build_log(get_col(panel, "milex")),
  milper_log = build_log(get_col(panel, "milper")),
  energy_log = build_log(get_col(panel, "pec")),
  lat        = get_col(panel, "lat"),
  lon        = get_col(panel, "lon")
) |>
  arrange(ccode, year)

dir.create(dirname(out_path), showWarnings = FALSE, recursive = TRUE)
write_csv(out, out_path)

cat("\nWrote", nrow(out), "rows to", out_path, "\n")
cat("Columns:", paste(names(out), collapse = ", "), "\n")

# Coverage
coverage <- sapply(setdiff(names(out), c("ccode", "year")),
                   function(c) round(mean(!is.na(out[[c]])), 3))
cat("\nNon-missing coverage by feature:\n")
print(coverage)
