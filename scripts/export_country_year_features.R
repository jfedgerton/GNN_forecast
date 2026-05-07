# export_country_year_features.R
#
# Build a per-(country, year) covariate CSV for the GNN node-feature
# pipeline. Pulls from peacesciencer's bundled COW + WDI + Polity helpers.
#
# Output: data/processed/node_features.csv with columns:
#   ccode, year, lp_gdp, lp_gdppc, lp_pop, polity2, cinc, milex_log,
#   milper_log, energy_log, lat, lon
#
# Missing values are left as NA — the Python loader handles imputation.
#
# Usage:
#   Rscript scripts/export_country_year_features.R \
#       --start-year 1816 --end-year 2025 \
#       --out data/processed/node_features.csv

suppressPackageStartupMessages({
  if (!requireNamespace("peacesciencer", quietly = TRUE)) {
    stop("Please install.packages('peacesciencer') first.")
  }
  if (!requireNamespace("optparse", quietly = TRUE)) {
    stop("Please install.packages('optparse') first.")
  }
  library(peacesciencer)
  library(tidyverse)
  library(optparse)
})

set.seed(123)

option_list <- list(
  make_option("--start-year", type = "integer", default = 1816),
  make_option("--end-year",   type = "integer", default = 2025),
  make_option("--out", type = "character",
              default = "data/processed/node_features.csv")
)
opt <- parse_args(OptionParser(option_list = option_list))

start_year <- opt$`start-year`
end_year   <- opt$`end-year`
out_path   <- opt$out

cat("Building country-year feature panel\n")
cat("  years:", start_year, "to", end_year, "\n")
cat("  out:  ", out_path, "\n\n")

# Build a country-year skeleton from peacesciencer
panel <- create_stateyears(system = "cow", subset_years = start_year:end_year)

# Attach NMC (capabilities + military)
panel <- add_nmc(panel)

# Attach Polity2
panel <- tryCatch(
  add_democracy(panel),
  error = function(e) {
    cat("  WARNING: add_democracy failed:", conditionMessage(e), "\n")
    panel
  }
)

# Attach SDP / GDP from Anders, Fariss, Markowitz
panel <- tryCatch(
  add_sim_gdp_pop(panel),
  error = function(e) {
    cat("  WARNING: add_sim_gdp_pop failed:", conditionMessage(e), "\n")
    panel
  }
)

# Attach capital coordinates (one-time, time-invariant)
panel <- tryCatch(
  add_cow_majors(panel),
  error = function(e) panel
)

cap_coords <- tryCatch(
  cow_states %>%
    select(ccode, capname, capcity_lat = caplat, capcity_lon = caplong) %>%
    distinct(ccode, .keep_all = TRUE),
  error = function(e) tibble(ccode = integer(), capcity_lat = numeric(),
                              capcity_lon = numeric())
)
if (nrow(cap_coords) > 0) {
  panel <- panel %>% left_join(cap_coords, by = "ccode")
}

# Build the standardized output column set
build_log <- function(x) ifelse(is.na(x) | x <= 0, NA_real_, log(x))

out <- panel %>%
  mutate(
    lp_gdp     = if ("sdpest" %in% names(.)) sdpest else NA_real_,
    lp_gdppc   = if ("sdpest" %in% names(.) && "popest" %in% names(.))
                   sdpest - popest else NA_real_,
    lp_pop     = if ("popest" %in% names(.)) popest else NA_real_,
    polity2    = if ("polity2" %in% names(.)) polity2 else NA_real_,
    cinc       = if ("cinc" %in% names(.)) cinc else NA_real_,
    milex_log  = if ("milex" %in% names(.)) build_log(milex) else NA_real_,
    milper_log = if ("milper" %in% names(.)) build_log(milper) else NA_real_,
    energy_log = if ("pec" %in% names(.)) build_log(pec) else NA_real_,
    lat        = if ("capcity_lat" %in% names(.)) capcity_lat else NA_real_,
    lon        = if ("capcity_lon" %in% names(.)) capcity_lon else NA_real_,
  ) %>%
  select(ccode, year, lp_gdp, lp_gdppc, lp_pop, polity2, cinc,
         milex_log, milper_log, energy_log, lat, lon) %>%
  arrange(ccode, year)

dir.create(dirname(out_path), showWarnings = FALSE, recursive = TRUE)
write_csv(out, out_path)

cat("\nWrote", nrow(out), "rows to", out_path, "\n")
cat("Columns:", paste(names(out), collapse = ", "), "\n")

# Quick coverage report
coverage <- sapply(setdiff(names(out), c("ccode", "year")),
                   function(c) round(mean(!is.na(out[[c]])), 3))
cat("\nNon-missing coverage by feature:\n")
print(coverage)
