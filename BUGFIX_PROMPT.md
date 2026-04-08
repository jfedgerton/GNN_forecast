# GNN_forecast Bug Fix Prompt

Fix the following issues in the GNN_forecast codebase. Work through them in the order listed — the critical items block the pipeline from running at all. Run `PYTHONPATH=src pytest tests/ -v` after each group of fixes to confirm nothing breaks.

---

## CRITICAL (Pipeline will not run without these)

### 1. `src/gnn_forecast/data_layer.py` line 1 — Syntax error

Line 1 reads `git from __future__ import annotations`. Remove the stray `git ` prefix so it reads `from __future__ import annotations`.

### 2. `scripts/export_peacesciencer_layers.R` lines 228-230 — Capital coordinates are NA

The nodes CSV is constructed with `cap_lat <- NA_real_` and `cap_lon <- NA_real_`. Downstream, `network_construction.py` calls `haversine_km()` on these, which produces NaN and crashes `np.linalg.lstsq()`.

Fix: Use peacesciencer's `add_capital_distance()` or the COW capital coordinates data to populate `cap_lat` and `cap_lon` with real values. If that's not available in the peacesciencer pipeline, create a lookup from the `capdist` dataset (which has `lat1`, `lon1`, `lat2`, `lon2`) and merge it onto the nodes dataframe. Every node must have non-NA coordinates.

### 3. `src/gnn_forecast/network_construction.py` lines 31-39 — NaN propagation in left merge

In `residualize_ties_against_distance()`, the left merge can produce NaN in `capital_distance_km` for unmatched edges, which crashes `np.linalg.lstsq()`.

Fix: After the merge, drop rows where `capital_distance_km` is NaN:
```python
merged = merged.dropna(subset=["capital_distance_km"])
```
Also add a check:
```python
if len(merged) == 0:
    raise ValueError("No edges remain after merging with distances — check that edge ccodes match distance ccodes")
```

### 4. `scripts/train_forecast_and_intervene.py` line 82 vs `src/gnn_forecast/data_layer.py` line 49 — Index alignment mismatch

`data_layer.py` creates node indices from **sorted** ccodes. `train_forecast_and_intervene.py` creates `node_to_idx` from **CSV row order**. If `nodes.csv` isn't sorted, embeddings and interventions will be silently misaligned.

Fix in `train_forecast_and_intervene.py`: Sort ccodes before building the index map:
```python
node_ccodes = sorted([int(c) for c in nodes["ccode"].tolist()])
node_to_idx = {cc: i for i, cc in enumerate(node_ccodes)}
```

---

## HIGH (Will cause confusing runtime failures)

### 5. `src/gnn_forecast/network_construction.py` line 15 — Haversine float precision

`np.arcsin(np.sqrt(a))` can receive values slightly > 1.0 due to float error, returning NaN.

Fix: Clamp the argument:
```python
return 2 * r * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))
```

### 6. `src/gnn_forecast/network_construction.py` — No validation of required node columns

`capital_distance_matrix()` expects `ccode`, `cap_lat`, `cap_lon` columns but never validates. Add at the top of the function:
```python
required = {"ccode", "cap_lat", "cap_lon"}
missing = required - set(nodes.columns)
if missing:
    raise ValueError(f"Nodes DataFrame missing required columns: {missing}")
```

### 7. `src/gnn_forecast/interventions.py` lines 48, 51-52 — Missing KeyError handling

`focal_ccode` and partner ccodes are looked up in `node_to_idx` without checks.

Fix: Validate before use:
```python
if focal_ccode not in node_to_idx:
    raise ValueError(f"Focal country code {focal_ccode} not found in node index")
missing_partners = [p for p in partner_ccodes if p not in node_to_idx]
if missing_partners:
    raise ValueError(f"Partner country codes not found in node index: {missing_partners}")
```

### 8. `src/gnn_forecast/interventions.py` ~line 67-69 — Device mismatch risk

The adjacency tensor may be created on a different device than `emb`.

Fix: Ensure the adjacency tensor is on the same device as `emb`:
```python
a = torch.zeros(n, n, device=emb.device)
```
Check all tensor creation calls in this file and make sure they use `device=emb.device`.

### 9. `src/gnn_forecast/models.py` lines 29-30, 42-43, 55-56 — Deferred import error is confusing

If `torch_geometric` is missing, the error is delayed and cryptic.

Fix: Replace the deferred pattern with an immediate raise:
```python
if GCNConv is None:
    raise ImportError("torch_geometric is required for GCNEncoder. Install with: pip install torch_geometric")
```
Do the same for GraphSAGEEncoder and GATEncoder.

---

## MEDIUM (Robustness and correctness)

### 10. `scripts/export_peacesciencer_layers.R` line 33 — log(0) and log(NA)

`mutate(capdist = log(capdist))` will produce `-Inf` if capdist is 0 or NA if capdist is NA.

Fix: Filter or clamp before log:
```r
mutate(capdist = log(pmax(capdist, 1)))  # clamp minimum distance to 1 km
```
Or filter out zero-distance rows (self-loops):
```r
filter(capdist > 0) %>%
mutate(capdist = log(capdist))
```

### 11. `src/gnn_forecast/forecast.py` lines 24-27 — decode_edges() inner product not normalized

Raw inner products grow with embedding dimension, making the threshold meaningless for high-dim embeddings.

Fix: L2-normalize before computing scores:
```python
def decode_edges(emb: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    emb_norm = F.normalize(emb, p=2, dim=1)
    score = emb_norm @ emb_norm.T
    return (score > threshold).float()
```

### 12. `src/gnn_forecast/models.py` — Add dropout to all encoders

Add a `dropout` parameter (default 0.2) and apply it between convolution layers in GCNEncoder, GraphSAGEEncoder, and GATEncoder:
```python
def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
    ...
    self.dropout = nn.Dropout(dropout)

def forward(self, x, edge_index, edge_weight=None):
    x = F.relu(self.c1(x, edge_index, edge_weight))
    x = self.dropout(x)
    return self.c2(x, edge_index, edge_weight)
```

### 13. `scripts/run_research_pipeline.py` — Add year range validation

After parsing arguments, validate:
```python
if start_year > end_year:
    raise ValueError(f"start_year ({start_year}) must be <= end_year ({end_year})")
```

### 14. `src/gnn_forecast/models.py` — Inconsistent edge_weight API

GCNEncoder accepts `edge_weight` but GraphSAGEEncoder and GATEncoder do not. Either add the parameter to all three (passing it through where supported) or document why it differs.

---

## LOW (Code quality and test improvements)

### 15. `tests/test_data_layer.py` line 8 — Fragile fixture path

Change from:
```python
FIXTURE_DIR = Path("tests/fixtures/tiny_processed")
```
To:
```python
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "tiny_processed"
```

### 16. `src/gnn_forecast/forecast.py` — Remove unused ForecastConfig

`ForecastConfig` is defined but never imported or used anywhere. Remove it.

### 17. `pyproject.toml` — Clean up dependencies

- Remove `scikit-learn` if it's not used anywhere in the codebase
- Add `torch_geometric` as an optional dependency:
```toml
[project.optional-dependencies]
geometric = ["torch_geometric"]
```

### 18. `src/gnn_forecast/forecast.py` — Add input validation to rollout_embeddings()

```python
def rollout_embeddings(model, history, steps):
    if history.dim() != 3:
        raise ValueError(f"Expected 3D history tensor [nodes, seq_len, emb_dim], got {history.dim()}D")
```

---

## Verification

After all fixes, run:
```bash
PYTHONPATH=src pytest tests/ -v
```

Then do a dry-run smoke test of the R script (just check it parses):
```bash
Rscript -e 'source("scripts/export_peacesciencer_layers.R", echo=FALSE)' || echo "Check R script syntax"
```
