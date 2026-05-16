"""Pseudotime-biased transition matrix and random walks for pyURD.

Equivalent to the R functions:
  - pseudotimeDetermineLogistic
  - pseudotimeWeightTransitionMatrix
  - simulateRandomWalk
  - simulateRandomWalksFromTips
  - processRandomWalks
  - processRandomWalksFromTips
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
import scipy.sparse
from anndata import AnnData

from .utils import logistic, _ensure_urd


# ──────────────────────────────────────────────
# Logistic parameter estimation
# ──────────────────────────────────────────────

def pseudotime_determine_logistic(
    adata: AnnData,
    pseudotime_key: str,
    optimal_cells_forward: int,
    max_cells_back: int,
    pseudotime_direction: str = "<",
    do_plot: bool = True,
    verbose: bool = True,
) -> Dict[str, float]:
    """Determine logistic parameters for biasing the transition matrix.

    Fits the inflection point (x0) and slope (k) of a logistic function
    based on the distribution of pseudotime differences between cells that
    are *optimal_cells_forward* and *max_cells_back* apart in pseudotime
    rank order.

    Parameters
    ----------
    adata
        AnnData with ``adata.obs[pseudotime_key]`` populated.
    pseudotime_key
        Column in ``adata.obs`` holding the pseudotime values.
    optimal_cells_forward
        Number of cells in the forward direction at which the logistic
        should reach ~1 (acceptance probability ≈ 1).
    max_cells_back
        Number of cells in the reverse direction at which the logistic
        should reach ~0 (acceptance probability ≈ asymptote).
    pseudotime_direction
        ``"<"`` to bias toward younger (smaller) pseudotime (default),
        ``">"`` to bias toward older pseudotime.
    do_plot
        Plot the resulting logistic curve.
    verbose
        Print the parameter values.

    Returns
    -------
    dict with keys ``"x0"`` and ``"k"``.
    """
    asymptote = 0.01
    pt_vals = adata.obs[pseudotime_key].dropna().values
    if pseudotime_direction == "<":
        pt_sorted = np.sort(pt_vals)[::-1]  # descending: high → low
    elif pseudotime_direction == ">":
        pt_sorted = np.sort(pt_vals)        # ascending
    else:
        raise ValueError("pseudotime_direction must be '<' or '>'")

    n = len(pt_sorted)
    mean_pt_back = float(np.mean(
        pt_sorted[: n - max_cells_back] - pt_sorted[max_cells_back:]
    ))
    mean_pt_forward = float(np.mean(
        pt_sorted[optimal_cells_forward:] - pt_sorted[: n - optimal_cells_forward]
    ))

    x0 = (mean_pt_back + mean_pt_forward) / 2.0
    k = np.log(asymptote) / (x0 - mean_pt_forward)

    if verbose:
        print(f"Mean pseudotime back  (~{max_cells_back} cells): {mean_pt_back:.6f}")
        print(f"Chance of accepted move to equal pseudotime: "
              f"{logistic(0, x0, k):.6f}")
        print(f"Mean pseudotime forward (~{optimal_cells_forward} cells): "
              f"{mean_pt_forward:.6f}")

    if do_plot:
        try:
            import matplotlib.pyplot as plt
            x_range = np.linspace(2 * mean_pt_back, 2 * mean_pt_forward, 200)
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(x_range, logistic(x_range, x0, k))
            ax.axvline(0, color="red")
            ax.axvline(mean_pt_back, color="blue", linestyle="--")
            ax.axvline(mean_pt_forward, color="blue", linestyle="--")
            ax.set_xlabel("Delta pseudotime")
            ax.set_ylabel("Acceptance probability")
            ax.set_title("Logistic bias function")
            plt.tight_layout()
            plt.show()
        except ImportError:
            warnings.warn("matplotlib not available; skipping plot.", stacklevel=2)

    return {"x0": float(x0), "k": float(k)}


# ──────────────────────────────────────────────
# Biased transition matrix
# ──────────────────────────────────────────────

def pseudotime_weight_transition_matrix(
    adata: AnnData,
    pseudotime_key: str,
    logistic_params: Optional[Dict[str, float]] = None,
    x0: Optional[float] = None,
    k: Optional[float] = None,
    pseudotime_direction: str = "<",
    transition_key: str = "connectivities",
    chunk_size: int = 500,
    verbose: bool = False,
) -> np.ndarray:
    """Weight the transition matrix by pseudotime to bias random walks.

    For each pair of cells (i, j) the original transition probability
    T[i, j] is multiplied by ``logistic(pt[j] - pt[i], x0, k)``.  With
    default parameters (``pseudotime_direction='<'`` and a negative *k*)
    this makes transitions toward younger cells more likely and transitions
    toward older cells less likely.

    Parameters
    ----------
    adata
        AnnData with the transition matrix in ``adata.obsp[transition_key]``
        and pseudotime in ``adata.obs[pseudotime_key]``.
    pseudotime_key
        Pseudotime column in ``adata.obs``.
    logistic_params
        Dict with keys ``"x0"`` and ``"k"``, e.g. from
        :func:`pseudotime_determine_logistic`.
    x0, k
        Can be provided directly instead of *logistic_params*.
    pseudotime_direction
        Direction of the bias (see :func:`pseudotime_determine_logistic`).
    transition_key
        Key in ``adata.obsp``.
    chunk_size
        Number of rows to process at a time (to limit peak RAM).
    verbose
        Print progress.

    Returns
    -------
    np.ndarray, shape (n_valid_cells, n_valid_cells)
        Dense biased transition matrix.  Row/column order matches cells
        that have a non-NaN pseudotime value.  Access
        ``adata.obs_names[~adata.obs[pseudotime_key].isna()]`` for the
        corresponding cell barcodes.
    """
    if logistic_params is not None:
        x0 = x0 or logistic_params["x0"]
        k = k or logistic_params["k"]
    if x0 is None or k is None:
        raise ValueError("Provide logistic_params or both x0 and k.")

    pt = adata.obs[pseudotime_key].values.astype(float)
    valid_mask = ~np.isnan(pt)
    pt_valid = pt[valid_mask]
    n = len(pt_valid)

    if verbose:
        print(f"Building biased transition matrix for {n} cells.")

    T = scipy.sparse.csr_matrix(adata.obsp[transition_key])[valid_mask, :][:, valid_mask]

    # Build the biased matrix in row-chunks to control peak memory
    T_biased = np.zeros((n, n), dtype=np.float32)
    T_dense_chunk = T.toarray() if n <= 5000 else None  # small: use dense once

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        if verbose:
            print(f"  Processing rows {start}–{end} / {n}")
        pt_chunk = pt_valid[start:end]
        # delta[i, j] = pt[j] - pt[i] for i in chunk, j in all cells
        delta = pt_valid[np.newaxis, :] - pt_chunk[:, np.newaxis]
        weights = logistic(delta, x0, k).astype(np.float32)
        if T_dense_chunk is not None:
            T_biased[start:end] = weights * T_dense_chunk[start:end]
        else:
            T_biased[start:end] = weights * T[start:end].toarray()

    return T_biased


# ──────────────────────────────────────────────
# Random walk simulation
# ──────────────────────────────────────────────

def simulate_random_walk(
    start_cells: List[str],
    transition_matrix: np.ndarray,
    cell_names: np.ndarray,
    end_cells: List[str],
    n: int = 10_000,
    end_visits: int = 1,
    max_steps: int = 5_000,
    verbose_freq: int = 0,
) -> List[Optional[List[str]]]:
    """Simulate biased random walks from a set of start cells.

    Each walk starts from a randomly chosen cell in *start_cells* and
    hops between cells according to the row-wise probabilities in
    *transition_matrix*.  The walk stops after visiting *end_visits* cells
    from *end_cells* (i.e. the root).

    Parameters
    ----------
    start_cells
        Pool of cells from which walks start (one chosen uniformly at random
        per walk).
    transition_matrix
        Dense or sparse matrix of transition probabilities
        (cells × cells).  Rows are normalised to sum to 1 internally.
    cell_names
        Array of cell barcodes matching the rows/columns of
        *transition_matrix*.
    end_cells
        Cells that terminate a walk (usually the root cells).
    n
        Number of walks to simulate.
    end_visits
        Number of root-cell visits required to stop a walk.
    max_steps
        Abandon walks longer than this (returns *None* for that walk).
    verbose_freq
        Print progress every this many walks.

    Returns
    -------
    list of length *n*
        Each element is a list of cell barcodes visited (including start
        and final root cell), or *None* if the walk was abandoned.
    """
    cell_names = np.asarray(cell_names)
    cell_to_idx = {c: i for i, c in enumerate(cell_names)}
    n_cells = len(cell_names)

    start_idx = np.array([cell_to_idx[c] for c in start_cells if c in cell_to_idx])
    end_set = set(end_cells)

    is_sparse = scipy.sparse.issparse(transition_matrix)

    # Pre-normalise rows so each row sums to 1
    if is_sparse:
        T = scipy.sparse.csr_matrix(transition_matrix, dtype=np.float64)
        row_sums = np.array(T.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        T = T.multiply(1.0 / row_sums[:, np.newaxis])
    else:
        T = np.array(transition_matrix, dtype=np.float64)
        row_sums = T.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        T = T / row_sums

    walks = []
    for i in range(n):
        if verbose_freq > 0 and i % verbose_freq == 0:
            print(f"Starting walk {i + 1}/{n}")

        current_idx = int(np.random.choice(start_idx))
        path = [cell_names[current_idx]]
        stops_in_endzone = 0
        n_steps = 0

        while stops_in_endzone < end_visits:
            if is_sparse:
                row = np.asarray(T.getrow(current_idx).todense()).flatten()
            else:
                row = T[current_idx]

            # Re-normalise to handle any floating-point drift
            row_sum = row.sum()
            if row_sum == 0:
                break
            row = row / row_sum

            current_idx = int(np.random.choice(n_cells, p=row))
            current_cell = cell_names[current_idx]
            path.append(current_cell)

            if current_cell in end_set:
                stops_in_endzone += 1

            n_steps += 1
            if n_steps > max_steps:
                warnings.warn(
                    f"Walk {i + 1} exceeded {max_steps} steps; returning None.",
                    stacklevel=2,
                )
                path = None
                break

        walks.append(path)

    return walks


def simulate_random_walks_from_tips(
    adata: AnnData,
    tip_group_key: str,
    root_cells: List[str],
    transition_matrix: np.ndarray,
    n_per_tip: int = 10_000,
    root_visits: int = 1,
    max_steps: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, List[Optional[List[str]]]]:
    """Simulate biased random walks from every tip in *tip_group_key*.

    Wrapper around :func:`simulate_random_walk` that iterates over all
    unique tip labels in ``adata.obs[tip_group_key]``.

    Parameters
    ----------
    adata
        AnnData object.
    tip_group_key
        Column in ``adata.obs`` containing tip cluster labels.  Cells with
        NaN are ignored.
    root_cells
        Cell barcodes of root cells (stop condition).
    transition_matrix
        Pseudotime-biased transition matrix (from
        :func:`pseudotime_weight_transition_matrix`).
    n_per_tip
        Number of walks per tip.
    root_visits
        Number of root-cell visits to terminate a walk.
    max_steps
        Maximum steps per walk.  Defaults to the number of cells.
    verbose
        Print progress.

    Returns
    -------
    dict mapping tip label → list of walks (as returned by
    :func:`simulate_random_walk`).
    """
    if max_steps is None:
        max_steps = adata.n_obs

    cell_names = np.array(adata.obs_names)
    all_tips = sorted(adata.obs[tip_group_key].dropna().unique().astype(str))

    tip_walks: Dict[str, List] = {}
    for tip in all_tips:
        if verbose:
            print(f"Starting random walks from tip {tip}")
        tip_mask = adata.obs[tip_group_key].astype(str) == tip
        tip_cells = list(adata.obs_names[tip_mask])
        verbose_freq = max(1, n_per_tip // 10) if verbose else 0
        walks = simulate_random_walk(
            start_cells=tip_cells,
            transition_matrix=transition_matrix,
            cell_names=cell_names,
            end_cells=root_cells,
            n=n_per_tip,
            end_visits=root_visits,
            max_steps=max_steps,
            verbose_freq=verbose_freq,
        )
        tip_walks[tip] = walks

    return tip_walks


# ──────────────────────────────────────────────
# Processing walks into visitation frequencies
# ──────────────────────────────────────────────

def process_random_walks(
    adata: AnnData,
    walks: List[Optional[List[str]]],
    walks_name: str,
    n_subsample: int = 10,
    verbose: bool = True,
) -> None:
    """Convert a list of random walk paths into visitation frequency columns.

    Stores ``adata.obs['visitfreq_raw_{walks_name}']`` (raw count) and
    ``adata.obs['visitfreq_log_{walks_name}']`` (log10-transformed).
    Also stores a walk-derived pseudotime
    ``adata.obs['walkpt_{walks_name}']`` (mean relative position in each
    walk, analogous to the R implementation).

    Parameters
    ----------
    adata
        AnnData object.
    walks
        List of walk paths (each a list of cell barcodes, or *None*).
    walks_name
        Suffix used to name the new columns.
    n_subsample
        Number of subsampling points for stability diagnostics.
    verbose
        Print progress.
    """
    _ensure_urd(adata)
    walks_name = str(walks_name)

    # Drop abandoned (None) walks
    valid_walks = [w for w in walks if w is not None]
    if len(valid_walks) == 0:
        warnings.warn(f"No valid walks for tip {walks_name}.", stacklevel=2)
        adata.obs[f"visitfreq_raw_{walks_name}"] = 0.0
        adata.obs[f"visitfreq_log_{walks_name}"] = 0.0
        adata.obs[f"walkpt_{walks_name}"] = np.nan
        return

    # Build a flat DataFrame: cell, relative_position (0-1), walk_index
    records = []
    for wi, walk in enumerate(valid_walks):
        n_steps = len(walk)
        for step, cell in enumerate(walk):
            records.append((cell, (step + 1) / n_steps, wi))

    hops_df = pd.DataFrame(records, columns=["cell", "hops", "walk"])

    # Determine stability sampling points
    n_walks = len(valid_walks)
    division_sizes = np.ceil(
        np.arange(1, n_subsample + 1) / n_subsample * n_walks
    ).astype(int)

    # Compute cumulative walk lengths to slice hops_df
    walk_lengths = np.array([len(w) for w in valid_walks])
    cum_lengths = np.concatenate([[0], np.cumsum(walk_lengths)])

    all_cells = adata.obs_names

    stability_pt_dict: Dict[int, pd.Series] = {}
    stability_visits_dict: Dict[int, pd.Series] = {}

    for div_n in division_sizes:
        if verbose:
            print(f"  Calculating pseudotime with {div_n} walks …")
        rows_to_use = cum_lengths[div_n]
        subset = hops_df.iloc[:rows_to_use]
        visit_freq = subset.groupby("cell")["hops"].count().reindex(all_cells).fillna(0)
        pt_per_cell = subset.groupby("cell")["hops"].mean().reindex(all_cells)
        stability_pt_dict[div_n] = pt_per_cell
        stability_visits_dict[div_n] = visit_freq

    # Store stability diagnostics
    urd = adata.uns["urd"]
    urd.setdefault("pseudotime_stability", {})[walks_name] = {
        "pseudotime": pd.DataFrame(stability_pt_dict, index=all_cells),
        "walks_per_cell": pd.DataFrame(stability_visits_dict, index=all_cells),
    }

    # Final results (all walks)
    final_visits = stability_visits_dict[division_sizes[-1]]
    final_pt = stability_pt_dict[division_sizes[-1]]

    adata.obs[f"visitfreq_raw_{walks_name}"] = final_visits.astype(float)
    adata.obs[f"visitfreq_log_{walks_name}"] = np.log10(final_visits + 1)
    adata.obs[f"walkpt_{walks_name}"] = final_pt


def process_random_walks_from_tips(
    adata: AnnData,
    walks_dict: Dict[str, List[Optional[List[str]]]],
    n_subsample: int = 10,
    verbose: bool = True,
) -> None:
    """Process random walks from all tips into visitation frequencies.

    Calls :func:`process_random_walks` for each tip in *walks_dict* and
    stores all results in *adata*.

    Parameters
    ----------
    adata
        AnnData object.
    walks_dict
        Dict mapping tip label → list of walks (output of
        :func:`simulate_random_walks_from_tips`).
    n_subsample
        Number of subsampling points for stability diagnostics.
    verbose
        Print progress.
    """
    for tip, walks in walks_dict.items():
        if verbose:
            print(f"Processing walks from tip {tip}")
        process_random_walks(adata, walks, walks_name=tip,
                             n_subsample=n_subsample, verbose=verbose)
