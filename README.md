# pyURD Python Package

A Python reimplementation of the [URD](https://github.com/farrellja/URD) R package for single-cell trajectory inference, compatible with the [scanpy](https://scanpy.readthedocs.io/) / [AnnData](https://anndata.readthedocs.io/) ecosystem.

## Implemented functions

| Python function | R equivalent | Description |
|---|---|---|
| `flood_pseudotime` | `floodPseudotime` | Probabilistic BFS pseudotime simulations |
| `flood_pseudotime_process` | `floodPseudotimeProcess` | Convert simulations → pseudotime in `adata.obs` |
| `pseudotime_determine_logistic` | `pseudotimeDetermineLogistic` | Fit logistic bias parameters |
| `pseudotime_weight_transition_matrix` | `pseudotimeWeightTransitionMatrix` | Pseudotime-biased transition matrix |
| `simulate_random_walks_from_tips` | `simulateRandomWalksFromTips` | Biased random walks from tip cells |
| `process_random_walks_from_tips` | `processRandomWalksFromTips` | Convert walks → visitation frequencies |
| `load_tip_cells` | `loadTipCells` | Register tip cell membership |
| `build_tree` | `buildTree` | Agglomerative trajectory tree construction |
| `name_segments` | `nameSegments` | Assign human-readable names to segments |
| `plot_tree` | `plotTree` | Dendrogram tree plot |
| `seg_parent` | `segParent` | Parent segment of a given segment |
| `seg_children` | `segChildren` | Direct children of a segment |
| `seg_children_all` | `segChildrenAll` | All descendants of a segment |
| `seg_terminal` | `segTerminal` | Terminal (leaf) segments |
| `save_urd` | *(new)* | Serialise and save AnnData + URD tree to `.h5ad` |
| `load_urd` | *(new)* | Load `.h5ad` and restore the URD tree |
| `make_uns_serialisable` | *(new)* | Convert `adata.uns` to HDF5-safe types |
| `restore_urd` | *(new)* | Restore DataFrames in a loaded URD dict |

> **Note:** Diffusion maps are computed via `sc.tl.diffmap()` in scanpy.  The transition matrix used by pyURD comes from `adata.obsp['connectivities']` (set by `sc.pp.neighbors()`).

## Installation

```bash
# Editable install from the local repo
pip install -e ".[full]"
# or with uv:
uv pip install -e ".[full]"
```

Optional extras:
- `scikit-misc` – required for loess smoothing in `pseudotime_determine_logistic`
- `diptest` – enables the `"preference"` divergence method in `build_tree`
- `joblib` – enables parallel flood simulations (`n_jobs=-1`)

## Quick-start

```python
import scanpy as sc
import pyurd

# Assume adata has already been preprocessed (PCA, etc.)
sc.pp.neighbors(adata, n_neighbors=30, use_rep="X_pca")

# Define root cells (earliest developmental stage)
root_cells = adata.obs_names[adata.obs["stage"] == "HIGH"].tolist()

# 1. Flood pseudotime
floods = pyurd.flood_pseudotime(adata, root_cells, n=50)
pyurd.flood_pseudotime_process(adata, floods, floods_name="pseudotime")

# 2. Bias transition matrix
lp = pyurd.pseudotime_determine_logistic(
    adata, "pseudotime", optimal_cells_forward=20, max_cells_back=40
)
biased_tm = pyurd.pseudotime_weight_transition_matrix(
    adata, "pseudotime", logistic_params=lp
)

# 3. Random walks
walks = pyurd.simulate_random_walks_from_tips(
    adata, tip_group_key="tip_clusters",
    root_cells=root_cells, transition_matrix=biased_tm,
    n_per_tip=25_000,
)
pyurd.process_random_walks_from_tips(adata, walks)

# 4. Build tree
pyurd.load_tip_cells(adata, "tip_clusters")
pyurd.build_tree(adata, pseudotime_key="pseudotime", tips_use=["1", "2"])

# 5. Name and plot
pyurd.name_segments(adata, ["1", "2"], ["Notochord", "Prechordal Plate"],
                    short_names=["Noto", "PCP"])
pyurd.plot_tree(adata, "stage", title="Developmental Stage")

# 6. Save and reload (preserves the full URD tree)
pyurd.save_urd(adata, "output/my_trajectory.h5ad")

# Later, in a fresh session:
adata = pyurd.load_urd("output/my_trajectory.h5ad")
pyurd.plot_tree(adata, "stage")
```

## Data storage in AnnData

All results are stored inside the AnnData object:

| Location | Contents |
|---|---|
| `adata.obs['pseudotime']` | Flood pseudotime values |
| `adata.obs['visitfreq_raw_{tip}']` | Raw walk visit counts per tip |
| `adata.obs['visitfreq_log_{tip}']` | log10 visit counts per tip |
| `adata.obs['segment']` | Segment assignment per cell |
| `adata.obs['node']` | Node assignment per cell |
| `adata.uns['urd']` | All tree structure, layout, and stability data |

## Saving and reloading

`save_urd` / `load_urd` handle the serialisation automatically — pandas DataFrames stored inside `adata.uns['urd']` are converted to HDF5-safe dicts on save, and reconstructed on load:

```python
# Save
pyurd.save_urd(adata, "results/trajectory.h5ad")

# Reload in a fresh session — tree plotting and navigation work immediately
adata = pyurd.load_urd("results/trajectory.h5ad")
pyurd.plot_tree(adata, "celltype")
```

If you need finer control, use `make_uns_serialisable` and `restore_urd` directly.

## Differences from the R package

1. **Diffusion map**: not reimplemented; use `sc.tl.diffmap()`.
2. **Transition matrix source**: uses `adata.obsp['connectivities']` by default.
3. **Parallelism**: flood simulations can be parallelised with `joblib` via the `n_jobs` parameter.
4. **Save/load helpers**: `save_urd` / `load_urd` replace manual serialisation boilerplate.
5. **3D / force-directed layout**: not implemented in this version.
