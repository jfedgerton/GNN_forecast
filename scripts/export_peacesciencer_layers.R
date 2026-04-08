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

dy <- dy %>%
  transmute(
    year = year,
    ccode1 = as.integer(ccode1),
    ccode2 = as.integer(ccode2)
  ) %>%
  peacesciencer::add_capital_distance() %>%
  filter(capdist > 0) %>%
  mutate(capdist = log(capdist))

# -----------------------
# Contiguity (for Python-side use)
# -----------------------
dy_contig <- dy %>%
  peacesciencer::add_contiguity() %>%
  mutate(
    contiguous = as.numeric(conttype <= 5 & !is.na(conttype)),
    source_ccode = ccode1,
    target_ccode = ccode2
  ) %>%
  select(year, source_ccode, target_ccode, contiguous) %>%
  distinct()

write_csv(dy_contig, file.path(out_dir, "contiguity.csv"))

# -----------------------
# Offensive alliances
# -----------------------
dy_off <- dy %>%
  peacesciencer::add_atop_alliance() %>%
  mutate(
    tie = as.numeric(atop_offense == 1),
    source_ccode = ccode1,
    target_ccode = ccode2
  ) %>%
  select(year, source_ccode, target_ccode, tie, capdist) %>%
  filter(!is.na(tie))

off_model <- fixest::feols(tie ~ capdist | source_ccode + target_ccode + year, data = dy_off)
dy_off_weighted <- dy_off %>%
  mutate(tie = tie - predict(off_model)) %>%
  select(year, source_ccode, target_ccode, tie)

dy_off <- dy_off %>% select(year, source_ccode, target_ccode, tie)

write_csv(dy_off, file.path(out_dir, "layer_offensive_alliances.csv"))
write_csv(dy_off_weighted, file.path(out_dir, "layer_offensive_alliances_weighted.csv"))

# -----------------------
# Defensive alliances
# -----------------------
dy_def <- dy %>%
  peacesciencer::add_atop_alliance() %>%
  mutate(
    tie = as.numeric(atop_defense == 1),
    source_ccode = ccode1,
    target_ccode = ccode2
  ) %>%
  select(year, source_ccode, target_ccode, tie, capdist) %>%
  filter(!is.na(tie))

def_model <- fixest::feols(tie ~ capdist | source_ccode + target_ccode + year, data = dy_def)
dy_def_weighted <- dy_def %>%
  mutate(tie = tie - predict(def_model)) %>%
  select(year, source_ccode, target_ccode, tie)

dy_def <- dy_def %>% select(year, source_ccode, target_ccode, tie)

write_csv(dy_def, file.path(out_dir, "layer_defensive_alliances.csv"))
write_csv(dy_def_weighted, file.path(out_dir, "layer_defensive_alliances_weighted.csv"))

# -----------------------
# Combined alliances (backward compatibility)
# -----------------------
dy_all <- dy %>%
  peacesciencer::add_atop_alliance() %>%
  mutate(
    tie = as.numeric((atop_defense == 1) | (atop_offense == 1)),
    source_ccode = ccode1,
    target_ccode = ccode2
  ) %>%
  select(year, source_ccode, target_ccode, tie, capdist) %>%
  filter(!is.na(tie))

alliance_model <- fixest::feols(tie ~ capdist | source_ccode + target_ccode + year, data = dy_all)
dy_all_weighted <- dy_all %>%
  mutate(tie = tie - predict(alliance_model)) %>%
  select(year, source_ccode, target_ccode, tie)

dy_all <- dy_all %>% select(year, source_ccode, target_ccode, tie)

write_csv(dy_all, file.path(out_dir, "layer_alliances_defensive_offensive_undirected.csv"))
write_csv(dy_all_weighted, file.path(out_dir, "layer_alliances_defensive_offensive_undirected_weighted.csv"))

# -----------------------
# Add IGO data (shared)
# -----------------------
dy_igo <- dy %>%
  peacesciencer::add_igos() %>%
  mutate(
    tie = dyadigos,
    source_ccode = ccode1,
    target_ccode = ccode2
  ) %>%
  select(year, source_ccode, target_ccode, tie, capdist) %>%
  filter(!is.na(tie))

igo_model <- fixest::feols(tie ~ capdist | source_ccode + target_ccode + year, data = dy_igo)
dy_igo_weighted <- dy_igo %>%
  mutate(tie = tie - predict(igo_model)) %>%
  select(year, source_ccode, target_ccode, tie)

dy_igo <- dy_igo %>% select(year, source_ccode, target_ccode, tie)

write_csv(dy_igo, file.path(out_dir, "layer_igo_shared_undirected.csv"))
write_csv(dy_igo_weighted, file.path(out_dir, "layer_igo_shared_undirected_weighted.csv"))

# -----------------------
# Add trade data (shared)
# -----------------------
dy_trade <- dy %>%
  peacesciencer::add_cow_trade() %>%
  mutate(
    tie = smoothflow1 + smoothflow2,
    source_ccode = ccode1,
    target_ccode = ccode2
  ) %>%
  select(year, source_ccode, target_ccode, tie, capdist) %>%
  filter(!is.na(tie))

trade_model <- fixest::feols(tie ~ capdist | source_ccode + target_ccode + year, data = dy_trade)
dy_trade_weighted <- dy_trade %>%
  mutate(tie = tie - predict(trade_model)) %>%
  select(year, source_ccode, target_ccode, tie)

dy_trade <- dy_trade %>% select(year, source_ccode, target_ccode, tie)

write_csv(dy_trade, file.path(out_dir, "layer_trade_undirected.csv"))
write_csv(dy_trade_weighted, file.path(out_dir, "layer_trade_undirected_weighted.csv"))

# -----------------------
# DCAs (defense cooperation agreements)
# NOTE: DCA data is not in peacesciencer by default. This section creates
# a placeholder that should be replaced with actual DCA data when available.
# See: Kinne (2020) "Defense Cooperation Agreements" for the canonical dataset.
# If you have a DCA CSV, place it at data/raw/dcas.csv with columns:
#   year, ccode1, ccode2, dca (binary)
# -----------------------
dca_raw_path <- file.path("data", "raw", "dcas.csv")
if (file.exists(dca_raw_path)) {
  message("Found DCA raw data at: ", dca_raw_path)
  dy_dca <- read_csv(dca_raw_path) %>%
    transmute(
      year = as.integer(year),
      source_ccode = as.integer(ccode1),
      target_ccode = as.integer(ccode2),
      tie = as.numeric(dca)
    ) %>%
    filter(!is.na(tie))

  # Merge with distance for weighting
  dy_dca_dist <- dy_dca %>%
    left_join(
      dy %>% select(year, ccode1, ccode2, capdist) %>%
        transmute(year, source_ccode = ccode1, target_ccode = ccode2, capdist),
      by = c("year", "source_ccode", "target_ccode")
    ) %>%
    filter(!is.na(capdist))

  if (nrow(dy_dca_dist) > 100) {
    dca_model <- fixest::feols(tie ~ capdist | source_ccode + target_ccode + year, data = dy_dca_dist)
    dy_dca_weighted <- dy_dca_dist %>%
      mutate(tie = tie - predict(dca_model)) %>%
      select(year, source_ccode, target_ccode, tie)
  } else {
    dy_dca_weighted <- dy_dca %>% mutate(tie = 0)
  }

  dy_dca <- dy_dca %>% select(year, source_ccode, target_ccode, tie)

  write_csv(dy_dca, file.path(out_dir, "layer_dca.csv"))
  write_csv(dy_dca_weighted, file.path(out_dir, "layer_dca_weighted.csv"))
  message("Wrote DCA layer.")
} else {
  message("No DCA raw data found at: ", dca_raw_path, ". Skipping DCA layer.")
  message("Place DCA data at data/raw/dcas.csv to include this layer.")
}

# -----------------------
# Export nodes.csv
# -----------------------
all_ccodes <- sort(unique(c(dy$ccode1, dy$ccode2)))
nodes_df <- data.frame(ccode = all_ccodes)

# Add state names from COW state membership data if available.
if (requireNamespace("peacesciencer", quietly = TRUE)) {
  cow_states <- peacesciencer::cow_states %>%
    transmute(ccode = as.integer(ccode), state_name = statenme) %>%
    distinct(ccode, .keep_all = TRUE)
  nodes_df <- nodes_df %>% left_join(cow_states, by = "ccode")
}

# Capital coordinates extracted from the COW capdist dataset.
# The capdist data in peacesciencer contains lat1/lon1 and lat2/lon2 for each dyad.
# We extract unique per-country coordinates from the ccode1 side.
capdist_raw <- tryCatch(
  peacesciencer::cow_capitals,
  error = function(e) NULL
)

if (!is.null(capdist_raw) && "lat" %in% names(capdist_raw)) {
  cap_coords <- capdist_raw %>%
    transmute(ccode = as.integer(ccode), cap_lat = lat, cap_lon = lng) %>%
    group_by(ccode) %>%
    slice_tail(n = 1) %>%
    ungroup()
} else {
  # Fallback: extract from the dyadic capdist dataset
  capdist_dyadic <- tryCatch(
    peacesciencer::cow_cap_dist,
    error = function(e) NULL
  )
  if (!is.null(capdist_dyadic) && "lat1" %in% names(capdist_dyadic)) {
    cap_coords <- capdist_dyadic %>%
      transmute(ccode = as.integer(ccode1), cap_lat = lat1, cap_lon = lon1) %>%
      distinct(ccode, .keep_all = TRUE)
  } else {
    # Last resort: use cshapes or generate from external data
    warning("Could not find capital coordinate data in peacesciencer. ",
            "Attempting to use countrycode package.")
    if (requireNamespace("countrycode", quietly = TRUE)) {
      cap_coords <- data.frame(ccode = all_ccodes) %>%
        mutate(
          iso3c = countrycode::countrycode(ccode, "cown", "iso3c", warn = FALSE)
        )
      # If countrycode doesn't provide lat/lon, fall back to NA
      cap_coords$cap_lat <- NA_real_
      cap_coords$cap_lon <- NA_real_
      warning("Capital coordinates set to NA — populate manually or install cshapes.")
    } else {
      cap_coords <- data.frame(ccode = all_ccodes, cap_lat = NA_real_, cap_lon = NA_real_)
      warning("Capital coordinates set to NA — install peacesciencer with capdist data.")
    }
  }
}

nodes_df <- nodes_df %>% left_join(cap_coords, by = "ccode")

write_csv(nodes_df, file.path(out_dir, "nodes.csv"))

message("Done. Wrote layers to: ", out_dir)
message("Layers exported:")
message("  - offensive_alliances (raw + weighted)")
message("  - defensive_alliances (raw + weighted)")
message("  - alliances_combined (raw + weighted, backward compat)")
message("  - igo_shared (raw + weighted)")
message("  - trade (raw + weighted)")
message("  - contiguity")
if (file.exists(dca_raw_path)) message("  - dca (raw + weighted)")
message("  - nodes.csv")
