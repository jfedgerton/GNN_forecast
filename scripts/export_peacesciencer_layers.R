#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(geosphere)
})

if (!requireNamespace("peacesciencer", quietly = TRUE)) {
  stop("Install peacesciencer first: remotes::install_github('svmiller/peacesciencer')")
}

get_fn <- function(candidates) {
  ns <- asNamespace("peacesciencer")
  for (nm in candidates) {
    if (exists(nm, envir = ns, mode = "function")) {
      return(get(nm, envir = ns))
    }
  }
  NULL
}

start_year <- 1945
end_year <- as.integer(format(Sys.Date(), "%Y"))
out_dir <- "data/processed"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# Base state-year/country table.
create_stateyears_fn <- get_fn(c("create_stateyears", "create_stateyears_cs", "create_dyadyears"))
if (is.null(create_stateyears_fn)) {
  stop("Could not find a state-year constructor in peacesciencer; check package version.")
}

sy <- create_stateyears_fn() %>%
  filter(year >= start_year, year <= end_year) %>%
  distinct(ccode, statename, capital_lat, capital_long)

nodes <- sy %>%
  transmute(
    ccode = as.integer(ccode),
    state_name = statename,
    cap_lat = capital_lat,
    cap_lon = capital_long
  ) %>%
  distinct()

write_csv(nodes, file.path(out_dir, "nodes.csv"))

# Build directed dyad-year grid for extracting ties.
dy <- expand_grid(
  year = start_year:end_year,
  source_ccode = nodes$ccode,
  target_ccode = nodes$ccode
) %>%
  filter(source_ccode != target_ccode)

# Candidate helper wrappers to tolerate peacesciencer version differences.
add_alliance_fn <- get_fn(c("add_alliance_data", "add_cow_alliance", "add_alliance"))
add_igo_fn <- get_fn(c("add_igo_data", "add_s_igo", "add_igo"))
add_trade_fn <- get_fn(c("add_trade_data", "add_cow_trade", "add_trade"))

if (is.null(add_alliance_fn) || is.null(add_igo_fn) || is.null(add_trade_fn)) {
  stop("Could not find one or more tie-construction functions (alliances/IGO/trade) in peacesciencer.")
}

# Alliance layer: offensive OR defensive alliance indicator.
alliances <- dy %>%
  rename(ccode1 = source_ccode, ccode2 = target_ccode) %>%
  add_alliance_fn() %>%
  mutate(
    tie = as.numeric((.data$defense == 1) | (.data$offense == 1)),
    source_ccode = ccode1,
    target_ccode = ccode2
  ) %>%
  select(year, source_ccode, target_ccode, tie)

write_csv(alliances, file.path(out_dir, "layer_alliances_defensive_offensive.csv"))

# Shared IGO layer (count of shared memberships; keep as weighted tie).
igos <- dy %>%
  rename(ccode1 = source_ccode, ccode2 = target_ccode) %>%
  add_igo_fn() %>%
  mutate(
    tie = dplyr::coalesce(.data$igo_joint, .data$igo_shared, 0),
    source_ccode = ccode1,
    target_ccode = ccode2
  ) %>%
  select(year, source_ccode, target_ccode, tie)

write_csv(igos, file.path(out_dir, "layer_igo_shared.csv"))

# Trade layer (flow values, log1p transformed).
trade <- dy %>%
  rename(ccode1 = source_ccode, ccode2 = target_ccode) %>%
  add_trade_fn() %>%
  mutate(
    raw_trade = dplyr::coalesce(.data$trade, .data$flow1, .data$flow2, 0),
    tie = log1p(pmax(raw_trade, 0)),
    source_ccode = ccode1,
    target_ccode = ccode2
  ) %>%
  select(year, source_ccode, target_ccode, tie)

write_csv(trade, file.path(out_dir, "layer_trade.csv"))

message("Export complete in ", out_dir)
