#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(fixest)
})

if (!requireNamespace("peacesciencer", quietly = TRUE)) {
  stop("Install peacesciencer first: remotes::install_github('svmiller/peacesciencer')")
}

# -----------------------
# Settings
# -----------------------
start_year <- 1900
end_year <- as.integer(format(Sys.Date(), "%Y"))
out_dir <- "data/processed"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# -----------------------
# Dyad-years (undirected)
# -----------------------
dy <- peacesciencer::create_dyadyears(system = "cow", directed = "undirected") %>%
  filter(year >= start_year, year <= end_year)

# The peacesciencer dyad-years typically uses ccode1/ccode2 already.
# We’ll keep them and export as "source_ccode/target_ccode" for consistency.
dy <- dy %>%
  transmute(
    year = year,
    ccode1 = as.integer(ccode1),
    ccode2 = as.integer(ccode2)
  ) %>% 
  peacesciencer::add_capital_distance() %>% 
  mutate(capdist = log(capdist))

# -----------------------
# Add alliance data
# -----------------------
dy_all <- dy %>%
  peacesciencer::add_atop_alliance() %>%   # alliance vars include defense/offense indicators
  mutate(
    tie = as.numeric((atop_defense == 1) | (atop_offense == 1)),
    source_ccode = ccode1,
    target_ccode = ccode2
  ) %>%
  select(year, source_ccode, target_ccode, tie, capdist) %>% 
  filter(!is.na(tie))


alliance_model <- fixest::feols(tie ~ capdist | source_ccode + target_ccode + year, data = dy_all)
dy_all_weighted <- dy_all %>%
  mutate(
    tie = tie - predict(alliance_model)
  ) %>%
  select(year, source_ccode, target_ccode, tie)

dy_all <- dy_all %>%
  select(year, source_ccode, target_ccode, tie)

write_csv(dy_all, file.path(out_dir, "layer_alliances_defensive_offensive_undirected.csv"))
write_csv(dy_all_weighted, file.path(out_dir, "layer_alliances_defensive_offensive_undirected_weighted.csv"))

# -----------------------
# Add IGO data (shared)
# -----------------------
dy_igo <- dy %>%
  peacesciencer::add_igos() %>%        # typically creates igo_joint (or similar)
  mutate(
    tie = dyadigos,
    source_ccode = ccode1,
    target_ccode = ccode2
  ) %>%
  select(year, source_ccode, target_ccode, tie, capdist) %>% 
  filter(!is.na(tie))

igo_model <- fixest::feols(tie ~ capdist | source_ccode + target_ccode + year, data = dy_igo)
dy_igo_weighted <- dy_igo %>%
  mutate(
    tie = tie - predict(igo_model)
  ) %>%
  select(year, source_ccode, target_ccode, tie)

dy_igo <- dy_igo %>%
  select(year, source_ccode, target_ccode, tie)

write_csv(dy_igo, file.path(out_dir, "layer_igo_shared_undirected.csv"))
write_csv(dy_igo_weighted, file.path(out_dir, "layer_igo_shared_undirected_weighted.csv"))

# -----------------------
# Add trade data (shared)
# -----------------------
dy_trade <- dy %>%
  peacesciencer::add_cow_trade() %>%        # typically creates igo_joint (or similar)
  mutate(
    tie = smoothflow1 + smoothflow2,
    source_ccode = ccode1,
    target_ccode = ccode2
  ) %>%
  select(year, source_ccode, target_ccode, tie, capdist) %>% 
  filter(!is.na(tie))



trade_model <- fixest::feols(tie ~ capdist | source_ccode + target_ccode + year, data = dy_trade)
dy_trade_weighted <- dy_trade %>%
  mutate(
    tie = tie - predict(trade_model)
  ) %>%
  select(year, source_ccode, target_ccode, tie)

dy_trade <- dy_trade %>%
  select(year, source_ccode, target_ccode, tie)

write_csv(dy_trade, file.path(out_dir, "layer_trade_undirected.csv"))
write_csv(dy_trade_weighted, file.path(out_dir, "layer_trade_undirected_weighted.csv"))

# -----------------------
# Export nodes.csv
# -----------------------
# Extract unique country codes seen across all layer dyads.
all_ccodes <- sort(unique(c(dy$ccode1, dy$ccode2)))
nodes_df <- data.frame(ccode = all_ccodes)

# Add state names from COW state membership data if available.
if (requireNamespace("peacesciencer", quietly = TRUE)) {
  cow_states <- peacesciencer::cow_states %>%
    transmute(ccode = as.integer(ccode), state_name = statenme) %>%
    distinct(ccode, .keep_all = TRUE)
  nodes_df <- nodes_df %>% left_join(cow_states, by = "ccode")
}

# Capital coordinates (cap_lat, cap_lon) are not directly available from
# peacesciencer's dyad-level capdist. Users should add these from an external
# source (e.g., the cshapes package) if needed by network_construction.py.
# Placeholder NA columns are written so the schema is complete.
nodes_df$cap_lat <- NA_real_
nodes_df$cap_lon <- NA_real_

write_csv(nodes_df, file.path(out_dir, "nodes.csv"))

message("Done. Wrote nodes.csv + alliance + IGO + trade layers to: ", out_dir)
