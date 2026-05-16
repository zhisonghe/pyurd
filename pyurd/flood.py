"""Flood pseudotime for pyURD.

Equivalent to the R functions:
  - floodBuildTM
  - floodPseudotimeCalc
  - floodPseudotime
  - floodPseudotimeProcess
"""

from __future__ import annotations

from typing import Optional, Union
import warnings

import numpy as np
import pandas as pd
import scipy.sparse
from anndata import AnnData

from .utils import _ensure_urd


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────

def _build_flood_tm(
    adata: AnnData,
    transition_key: str = "connectivities",
) -> scipy.sparse.csr_matrix:
    """Build a normalised transition matrix for flooding.

    Divides the transition matrix by its maximum row-sum so that the
    maximum probability of leaving any cell in one step is 1.

    Parameters
    ----------
    adata
        AnnData object. Must have ``adata.obsp[transition_key]``.
    transition_key
        Key in ``adata.obsp`` for the transition / connectivity matrix.

    Returns
    -------
    scipy.sparse.csr_matrix
        Row-normalised sparse transition matrix.
    """
    if transition_key not in adata.obsp:
        raise KeyError(
            f"Transition matrix '{transition_key}' not found in adata.obsp. "
            "Run sc.pp.neighbors() (or sc.tl.diffmap()) first."
        )
    tm = scipy.sparse.csr_matrix(adata.obsp[transition_key])
    row_sums = np.array(tm.sum(axis=1)).flatten()
    max_sum = row_sums.max()
    if max_sum == 0:
        raise ValueError("Transition matrix has all-zero rows.")
    tm = tm / max_sum
    return tm


def _combine_probs_sparse(submatrix: scipy.sparse.spmatrix) -> np.ndarray:
    """Compute P(visit) = 1 - prod(1 - x) for each row of a sparse matrix.

    Uses the log-sum trick for numerical stability and efficiency:
      1 - prod(1-x) = 1 - exp(sum(log(1-x)))

    Zero entries contribute 0 to the sum (log(1-0)=0), so we operate only
    on stored (nonzero) data.

    Parameters
    ----------
    submatrix
        Sparse matrix (unvisited cells × visited cells).

    Returns
    -------
    np.ndarray of shape (n_unvisited,)
    """
    mat = scipy.sparse.csr_matrix(submatrix, copy=True)
    # Clip values to [0, 1) to avoid log(<=0)
    mat.data = np.clip(mat.data, 0.0, 1.0 - 1e-10)
    mat.data = np.log1p(-mat.data)          # log(1 - x) for nonzero elements
    log_prod = np.array(mat.sum(axis=1)).flatten()   # sum over visited columns
    return 1.0 - np.exp(log_prod)


def _combine_probs_dense(submatrix: np.ndarray) -> np.ndarray:
    """Compute P(visit) = 1 - prod(1 - x) for each row of a dense matrix."""
    mat = np.clip(submatrix, 0.0, 1.0 - 1e-10)
    log_sum = np.log1p(-mat).sum(axis=1)
    return 1.0 - np.exp(log_sum)


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def flood_pseudotime_calc(
    tm_flood: Union[scipy.sparse.spmatrix, np.ndarray],
    cell_names: np.ndarray,
    start_cells: list,
    minimum_cells_flooded: int = 2,
    verbose_freq: int = 0,
) -> np.ndarray:
    """Run a single flood-pseudotime simulation.

    Performs a probabilistic breadth-first search on the transition-
    probability graph starting from *start_cells*.

    Parameters
    ----------
    tm_flood
        Normalised transition matrix (cells × cells).
    cell_names
        Array of cell barcodes matching the rows/columns of *tm_flood*.
    start_cells
        Cell barcodes to initialise as visited (pseudotime 0).
    minimum_cells_flooded
        Stop when fewer than this many new cells are visited in one step.
    verbose_freq
        Print progress every this many steps (0 = silent).

    Returns
    -------
    np.ndarray of shape (n_cells,)
        Step at which each cell was first visited. Unvisited cells get NaN.
    """
    n_cells = len(cell_names)
    cell_to_idx = {c: i for i, c in enumerate(cell_names)}

    pseudotime = np.full(n_cells, np.nan)
    visited_mask = np.zeros(n_cells, dtype=bool)

    # Initialise root cells
    for c in start_cells:
        if c in cell_to_idx:
            idx = cell_to_idx[c]
            pseudotime[idx] = 0
            visited_mask[idx] = True

    is_sparse = scipy.sparse.issparse(tm_flood)
    step = 0
    newly_visited_count = minimum_cells_flooded  # ensure we enter the loop

    while visited_mask.sum() < n_cells - 1 and newly_visited_count >= minimum_cells_flooded:
        step += 1
        if verbose_freq > 0 and step % verbose_freq == 0:
            pct = 100 * visited_mask.sum() / n_cells
            print(f"Flooding step {step} – {pct:.1f}% visited "
                  f"({visited_mask.sum()}/{n_cells})")

        unvisited_idx = np.where(~visited_mask)[0]
        visited_idx = np.where(visited_mask)[0]

        # Transition weights: unvisited → visited (symmetric matrix, so same as visited → unvisited)
        sub = tm_flood[np.ix_(unvisited_idx, visited_idx)] if not is_sparse else \
              tm_flood[unvisited_idx, :][:, visited_idx]

        # Compute visit probabilities
        if is_sparse:
            visit_probs = _combine_probs_sparse(sub)
        else:
            visit_probs = _combine_probs_dense(np.asarray(sub))

        # Bernoulli draw: which unvisited cells get visited this step?
        new_visits = np.random.binomial(1, np.clip(visit_probs, 0.0, 1.0)).astype(bool)
        newly_visited_idx = unvisited_idx[new_visits]
        newly_visited_count = len(newly_visited_idx)

        pseudotime[newly_visited_idx] = step
        visited_mask[newly_visited_idx] = True

    return pseudotime


def flood_pseudotime(
    adata: AnnData,
    root_cells: list,
    n: int = 20,
    minimum_cells_flooded: int = 2,
    tm_flood: Optional[Union[scipy.sparse.spmatrix, np.ndarray]] = None,
    transition_key: str = "connectivities",
    n_jobs: int = 1,
    verbose: bool = False,
) -> pd.DataFrame:
    """Simulate flood pseudotime.

    Runs *n* probabilistic BFS simulations starting from *root_cells* and
    returns the per-simulation visit steps for each cell.  Pass the result
    to :func:`flood_pseudotime_process` to convert it into a pseudotime
    column in *adata*.

    Parameters
    ----------
    adata
        AnnData object with a precomputed transition matrix in
        ``adata.obsp[transition_key]``.
    root_cells
        Cell barcodes that define the root (pseudotime 0).
    n
        Number of simulations.
    minimum_cells_flooded
        Stop a simulation when fewer than this many new cells are visited.
    tm_flood
        Pre-built normalised transition matrix.  If *None*, it is computed
        from ``adata.obsp[transition_key]``.
    transition_key
        Key in ``adata.obsp`` to use when *tm_flood* is *None*.
    n_jobs
        Number of parallel workers.  Requires ``joblib``.
    verbose
        Print progress.

    Returns
    -------
    pd.DataFrame, shape (n_cells, n)
        Columns are simulation indices (1..n); rows are cells.
    """
    if tm_flood is None:
        tm_flood = _build_flood_tm(adata, transition_key)

    cell_names = np.array(adata.obs_names)
    verbose_freq = 10 if verbose else 0

    if n_jobs != 1:
        try:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=n_jobs)(
                delayed(_run_one_flood)(
                    tm_flood, cell_names, root_cells,
                    minimum_cells_flooded, verbose_freq, i, verbose
                )
                for i in range(n)
            )
        except ImportError:
            warnings.warn("joblib not installed; running serially.", stacklevel=2)
            n_jobs = 1

    if n_jobs == 1:
        results = []
        for i in range(n):
            if verbose:
                print(f"Starting flood simulation {i + 1}/{n}")
            results.append(
                flood_pseudotime_calc(
                    tm_flood, cell_names, root_cells,
                    minimum_cells_flooded, verbose_freq,
                )
            )

    floods = pd.DataFrame(
        np.column_stack(results),
        index=adata.obs_names,
        columns=np.arange(1, n + 1),
    )
    return floods


def _run_one_flood(tm_flood, cell_names, root_cells, minimum_cells_flooded,
                   verbose_freq, i, verbose):
    if verbose:
        print(f"Starting flood simulation {i + 1}")
    return flood_pseudotime_calc(tm_flood, cell_names, root_cells,
                                 minimum_cells_flooded, verbose_freq)


def prepare_connectivities(
    adata: AnnData,
    n_neighbors: int = 30,
    n_dcs: int = 10,
    neighbors_key: Optional[str] = None,
    key_added: str = "diffmap_flood",
    run_diffmap: bool = True,
    verbose: bool = True,
) -> str:
    """Build a diffusion-map–based connectivity matrix for flood pseudotime.

    In URD, flood pseudotime propagates through the diffusion-map transition
    graph rather than the raw k-NN graph.  This function:

    1. Optionally runs ``sc.tl.diffmap`` on the existing neighbour graph.
    2. Re-builds a k-NN graph in diffusion-component space via
       ``sc.pp.neighbors(..., use_rep='X_diffmap')``.
    3. Stores the result in ``adata.obsp[f'{key_added}_connectivities']``.

    The returned string can be passed directly as *transition_key* to
    :func:`flood_pseudotime`.

    Parameters
    ----------
    adata
        AnnData object.  Must have a pre-computed neighbour graph in
        ``adata.uns['neighbors']`` (or ``adata.uns[neighbors_key]``) unless
        *run_diffmap* is *False* and ``adata.obsm['X_diffmap']`` already
        exists.
    n_neighbors
        Number of neighbours for the diffusion-map k-NN graph.
    n_dcs
        Number of diffusion components to compute / use.
    neighbors_key
        Key in ``adata.uns`` for the existing neighbour graph used when
        computing the diffusion map.  *None* uses the scanpy default
        (``'neighbors'``).
    key_added
        Prefix for the new connectivity matrix stored in ``adata.obsp``.
        The actual key will be ``f'{key_added}_connectivities'``.
    run_diffmap
        If *True* (default), run ``sc.tl.diffmap`` first.  Set to *False*
        if ``adata.obsm['X_diffmap']`` is already up-to-date.
    verbose
        Print progress messages.

    Returns
    -------
    str
        The ``adata.obsp`` key for the new connectivity matrix, i.e.
        ``f'{key_added}_connectivities'``.  Pass this to
        :func:`flood_pseudotime` as *transition_key*.

    Examples
    --------
    >>> import scanpy as sc
    >>> import pyurd
    >>> sc.pp.neighbors(adata, n_neighbors=30, use_rep="X_pca")
    >>> transition_key = pyurd.prepare_connectivities(adata)
    >>> floods = pyurd.flood_pseudotime(adata, root_cells, transition_key=transition_key)
    """
    try:
        import scanpy as sc
    except ImportError:
        raise ImportError(
            "scanpy is required for prepare_connectivities. "
            "Install it with: pip install scanpy"
        )

    uns_key = neighbors_key if neighbors_key is not None else "neighbors"

    if run_diffmap:
        if uns_key not in adata.uns:
            raise KeyError(
                f"No neighbour graph found in adata.uns['{uns_key}']. "
                "Run sc.pp.neighbors() first."
            )
        if verbose:
            print(f"Computing diffusion map ({n_dcs} components) ...")
        sc.tl.diffmap(adata, n_comps=n_dcs, neighbors_key=neighbors_key)
    else:
        if "X_diffmap" not in adata.obsm:
            raise KeyError(
                "adata.obsm['X_diffmap'] not found and run_diffmap=False. "
                "Either run sc.tl.diffmap() first or set run_diffmap=True."
            )

    if verbose:
        print(
            f"Building k-NN graph in diffusion-map space "
            f"(n_neighbors={n_neighbors}, n_dcs={n_dcs}) ..."
        )
    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        use_rep="X_diffmap",
        n_pcs=n_dcs,
        key_added=key_added,
    )

    transition_key = f"{key_added}_connectivities"
    if verbose:
        print(f"Diffusion-map connectivity stored in adata.obsp['{transition_key}']")
        print(f"Pass transition_key='{transition_key}' to pyurd.flood_pseudotime().")

    return transition_key


def flood_pseudotime_process(
    adata: AnnData,
    floods: pd.DataFrame,
    floods_name: str = "pseudotime",
    max_frac_na: float = 0.4,
    stability_div: int = 10,
) -> None:
    """Process flood simulations into pseudotime and store in *adata*.

    Converts the raw visit-step data returned by :func:`flood_pseudotime`
    into normalised [0, 1] pseudotime values and stores them in
    ``adata.obs[floods_name]``.  Also stores visit-frequency columns
    ``adata.obs['visitfreq_raw_{floods_name}']`` and
    ``adata.obs['visitfreq_log_{floods_name}']``.

    Parameters
    ----------
    adata
        AnnData object.
    floods
        DataFrame returned by :func:`flood_pseudotime`.
    floods_name
        Name for the pseudotime column in ``adata.obs``.
    max_frac_na
        Cells with more than this fraction of NA visits are excluded.
    stability_div
        Number of subsampled stability points to compute.
    """
    _ensure_urd(adata)
    floods_name = str(floods_name)

    # Drop cells with too many NAs
    frac_na = floods.isna().mean(axis=1)
    cells_keep = frac_na[frac_na <= max_frac_na].index
    floods = floods.loc[cells_keep].copy()

    # Normalise each simulation to [0, 1]
    col_max = floods.max(axis=0)
    col_max[col_max == 0] = 1  # avoid division by zero
    floods_norm = floods.div(col_max, axis=1)

    # Stability: compute pseudotime with increasing numbers of simulations
    n_sims = floods_norm.shape[1]
    division_indices = np.ceil(
        np.arange(1, stability_div + 1) / stability_div * n_sims
    ).astype(int)

    stability_pt = {}
    stability_visits = {}
    for div_n in division_indices:
        subset = floods_norm.iloc[:, :div_n]
        stability_pt[div_n] = subset.mean(axis=1)
        stability_visits[div_n] = (~subset.isna()).sum(axis=1)

    # Store stability data
    urd = adata.uns["urd"]
    urd.setdefault("pseudotime_stability", {})[floods_name] = {
        "pseudotime": pd.DataFrame(stability_pt, index=cells_keep),
        "walks_per_cell": pd.DataFrame(stability_visits, index=cells_keep),
    }

    # Final pseudotime = mean across all simulations
    final_pt = floods_norm.mean(axis=1)

    # Visit frequency
    final_visits = (~floods.isna()).sum(axis=1).reindex(adata.obs_names).fillna(0)
    adata.obs[f"visitfreq_raw_{floods_name}"] = final_visits.astype(float)
    adata.obs[f"visitfreq_log_{floods_name}"] = np.log10(final_visits + 1)

    # Pseudotime column
    adata.obs[floods_name] = final_pt.reindex(adata.obs_names)
