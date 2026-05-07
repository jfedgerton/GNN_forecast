# export_cow_membership.R
#
# Exports the canonical COW state-year membership panel as a CSV.
# Other scripts (Kinne converter, USITC export, diagnostic) load this
# CSV and filter dyads to (ccode1, ccode2, year) where BOTH states are
# COW members in that year.
#
# Output: data/processed/cow_state_membership.csv with columns:
#   ccode, year, in_cow (always 1 if present), iso3
#
# Usage:
#   Rscript scripts/export_cow_membership.R [start_year] [end_year]
#   Rscript scripts/export_cow_membership.R 1816 2025

set.seed(123)

args <- commandArgs(trailingOnly = TRUE)
start_year <- if (length(args) >= 1) as.integer(args[1]) else 1816L
end_year   <- if (length(args) >= 2) as.integer(args[2]) else 2025L

# Ensure peacesciencer is available
if (!requireNamespace("peacesciencer", quietly = TRUE)) {
  stop("peacesciencer not installed. Run: install.packages('peacesciencer')")
}

# Use countrycode to attach iso3 (helpful for joining USITC data later)
if (!requireNamespace("countrycode", quietly = TRUE)) {
  user_lib <- Sys.getenv("R_LIBS_USER")
  if (nchar(user_lib) == 0) user_lib <- file.path(Sys.getenv("HOME"), "R", "library")
  dir.create(user_lib, showWarnings = FALSE, recursive = TRUE)
  .libPaths(c(user_lib, .libPaths()))
  install.packages("countrycode", lib = user_lib, repos = "https://cloud.r-project.org")
}

suppressPackageStartupMessages({
  library(peacesciencer)
  library(countrycode)
})

cat("Building COW state-year membership panel\n")
cat("  years:", start_year, "to", end_year, "\n")

# create_stateyears with system='cow' returns the canonical ccode-year COW panel.
# Each row is a (ccode, year) where the state was a COW system member.
panel <- create_stateyears(system = "cow", subset_years = start_year:end_year)
cat("  COW state-years in range:", nrow(panel), "\n")
cat("  unique ccodes:", length(unique(panel$ccode)), "\n")

panel$in_cow <- 1L
panel$iso3 <- countrycode(panel$ccode, origin = "cown",
                           destination = "iso3c", warn = FALSE)
panel <- panel[, c("ccode", "year", "in_cow", "iso3")]
panel <- panel[order(panel$ccode, panel$year), ]

# Coverage report
n_with_iso <- sum(!is.na(panel$iso3))
cat("  rows with iso3 mapping:", n_with_iso,
    "(", round(100 * n_with_iso / nrow(panel), 1), "%)\n")
missing_iso_codes <- unique(panel$ccode[is.na(panel$iso3)])
if (length(missing_iso_codes) > 0) {
  cat("  ccodes with no iso3 (will not match USITC data):",
      paste(missing_iso_codes, collapse = ", "), "\n")
}

dir.create("data/processed", showWarnings = FALSE, recursive = TRUE)
out_path <- "data/processed/cow_state_membership.csv"
write.csv(panel, out_path, row.names = FALSE)
cat("Wrote", out_path, "\n")
