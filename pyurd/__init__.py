"""pyURD – A Python reimplementation of the URD trajectory inference package.

Compatible with the scanpy / AnnData ecosystem.

Quick-start workflow
--------------------
.. code-block:: python

    import scanpy as sc
    import pyurd

    # 1. Compute diffusion map / neighbourhood graph with scanpy
    sc.pp.neighbors(adata, n_neighbors=30, use_rep="X_pca")
    sc.tl.diffmap(adata)

    # 2. Flood pseudotime from root cells
    root_cells = adata.obs_names[adata.obs["stage"] == "stage_0"].tolist()
    floods = pyurd.flood_pseudotime(adata, root_cells, n=50)
    pyurd.flood_pseudotime_process(adata, floods, floods_name="pseudotime")

    # 3. Determine logistic parameters for biased random walks
    logistic_params = pyurd.pseudotime_determine_logistic(
        adata, "pseudotime", optimal_cells_forward=20, max_cells_back=40
    )

    # 4. Compute pseudotime-biased transition matrix
    biased_tm = pyurd.pseudotime_weight_transition_matrix(
        adata, "pseudotime", logistic_params=logistic_params
    )

    # 5. Simulate random walks from each tip
    walks = pyurd.simulate_random_walks_from_tips(
        adata,
        tip_group_key="tip_clusters",
        root_cells=root_cells,
        transition_matrix=biased_tm,
        n_per_tip=25_000,
    )

    # 6. Process walks into visitation frequencies
    pyurd.process_random_walks_from_tips(adata, walks)

    # 7. Build the tree
    pyurd.load_tip_cells(adata, "tip_clusters")
    pyurd.build_tree(adata, pseudotime_key="pseudotime", tips_use=[1, 2])

    # 8. Name segments and plot
    pyurd.name_segments(adata, ["1", "2"], ["CellType A", "CellType B"])
    pyurd.plot_tree(adata, "stage")
"""

from .flood import (
    flood_pseudotime,
    flood_pseudotime_process,
)
from .diffusion import (
    pseudotime_determine_logistic,
    pseudotime_weight_transition_matrix,
    simulate_random_walk,
    simulate_random_walks_from_tips,
    process_random_walks,
    process_random_walks_from_tips,
)
from .tree import (
    load_tip_cells,
    build_tree,
    name_segments,
    seg_parent,
    seg_children,
    seg_children_all,
    seg_terminal,
)
from .plot import plot_tree
from .utils import make_uns_serialisable, restore_urd, save_urd, load_urd

__all__ = [
    # Flood pseudotime
    "flood_pseudotime",
    "flood_pseudotime_process",
    # Pseudotime weighting & random walks
    "pseudotime_determine_logistic",
    "pseudotime_weight_transition_matrix",
    "simulate_random_walk",
    "simulate_random_walks_from_tips",
    "process_random_walks",
    "process_random_walks_from_tips",
    # Tree construction
    "load_tip_cells",
    "build_tree",
    "name_segments",
    # Tree navigation
    "seg_parent",
    "seg_children",
    "seg_children_all",
    "seg_terminal",
    # Plotting
    "plot_tree",
    # Serialisation helpers
    "make_uns_serialisable",
    "restore_urd",
    "save_urd",
    "load_urd",
]
