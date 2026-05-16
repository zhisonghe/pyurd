"""Tree building, navigation, layout and naming for pyURD.

Equivalent to the R functions:
  - loadTipCells
  - buildTree
  - nameSegments
  + internal helpers: putativeCellsInSegment, allSegmentDivergenceByPseudotime,
    visitDivergenceByPseudotime, divergenceKSVisitation, divergencePreferenceDip,
    pseudotimeBreakpointByStretch, assignCellsToSegments, collapseShortSegments,
    removeUnitarySegments, assignCellsToNodes, treeLayoutDendrogram,
    treeLayoutElaborate, treeLayoutCells,
    segParent, segChildren, segChildrenAll, segTerminal
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
import scipy.stats
from anndata import AnnData

from .utils import preference, _ensure_urd


# ═══════════════════════════════════════════════════════
# Section 1 – Tip cell loading
# ═══════════════════════════════════════════════════════

def load_tip_cells(adata: AnnData, tips_key: str) -> None:
    """Store tip cell memberships in ``adata.uns['urd']``.

    Must be called before :func:`build_tree` when using weighted fusion.

    Parameters
    ----------
    adata
        AnnData object with tip cluster labels in ``adata.obs[tips_key]``.
    tips_key
        Column in ``adata.obs`` containing tip cluster assignments.
        Cells with NaN are ignored.
    """
    _ensure_urd(adata)
    urd = adata.uns["urd"]

    all_tips = sorted(adata.obs[tips_key].dropna().unique().astype(str))
    cells_in_tip: Dict[str, List[str]] = {}
    for tip in all_tips:
        mask = adata.obs[tips_key].astype(str) == tip
        cells_in_tip[tip] = list(adata.obs_names[mask])

    urd["cells_in_tip"] = cells_in_tip
    urd["tips"] = all_tips


# ═══════════════════════════════════════════════════════
# Section 2 – Tree navigation helpers
# ═══════════════════════════════════════════════════════

def seg_parent(urd: dict, segment: str, return_self_if_na: bool = False,
               original_joins: bool = False) -> Optional[str]:
    """Return the parent segment of *segment*."""
    sj = urd["segment_joins_initial"] if original_joins else urd["segment_joins"]
    parents = sj.loc[sj["child"] == segment, "parent"].tolist()
    if parents:
        return parents[0]
    return segment if return_self_if_na else None


def seg_children(urd: dict, segment: str) -> List[str]:
    """Return direct children of *segment*."""
    sj = urd["segment_joins"]
    return sj.loc[sj["parent"] == segment, "child"].tolist()


def seg_children_all(urd: dict, segment: str, include_self: bool = False,
                     original_joins: bool = False) -> List[str]:
    """Return all descendants of *segment* (all depths)."""
    sj = (urd["segment_joins_initial"] if original_joins
          else urd["segment_joins"])
    children: List[str] = []
    frontier = [segment]
    while frontier:
        new_children = sj.loc[sj["parent"].isin(frontier), "child"].tolist()
        children.extend(new_children)
        frontier = new_children
    if include_self:
        children = [segment] + children
    return list(dict.fromkeys(children))  # deduplicate, preserve order


def seg_terminal(urd: dict) -> List[str]:
    """Return all terminal (leaf) segments."""
    sj = urd["segment_joins"]
    return sorted(set(sj["child"]) - set(sj["parent"]))


# ═══════════════════════════════════════════════════════
# Section 3 – Putative cells in segments
# ═══════════════════════════════════════════════════════

def _putative_cells_in_segment(
    adata: AnnData,
    segments: List[str],
    minimum_visits: int,
    visit_threshold: float,
) -> pd.DataFrame:
    """Boolean DataFrame (cells × segments) indicating putative membership."""
    cols = [f"visitfreq_raw_{s}" for s in segments]
    visit_data = adata.obs[cols].values.astype(float)

    max_visit = visit_data.max(axis=1)
    enough_visits = visit_threshold * max_visit
    cell_in_segment = visit_data >= enough_visits[:, np.newaxis]
    pass_min = max_visit >= minimum_visits
    cell_in_segment = cell_in_segment & pass_min[:, np.newaxis]

    return pd.DataFrame(cell_in_segment, index=adata.obs_names, columns=segments)


# ═══════════════════════════════════════════════════════
# Section 4 – Divergence tests
# ═══════════════════════════════════════════════════════

def _divergence_ks(visit_data: pd.DataFrame, windows: np.ndarray,
                   cells_seg1: List[str], cells_seg2: List[str]) -> pd.DataFrame:
    """Per-window KS test on visitation distributions."""
    rows = []
    for win_groups in windows:
        win_groups = set(win_groups.tolist())
        mask = visit_data["pseudotime_group"].isin(win_groups)
        cells_win = visit_data.index[mask].tolist()

        v1 = visit_data.loc[cells_win, "segment_1"].values
        v2 = visit_data.loc[cells_win, "segment_2"].values
        stat, p = scipy.stats.ks_2samp(v1, v2)

        pt_win = visit_data.loc[cells_win, "pseudotime"]
        n1 = len(set(cells_win) & set(cells_seg1))
        n2 = len(set(cells_win) & set(cells_seg2))
        rows.append({
            "p": float(p),
            "mean_pseudotime": float(pt_win.mean()),
            "min_pseudotime": float(pt_win.min()),
            "max_pseudotime": float(pt_win.max()),
            "cells_visited_seg1": n1,
            "cells_visited_seg2": n2,
        })
    return pd.DataFrame(rows)


def _divergence_preference_dip(visit_data: pd.DataFrame,
                                cells_in_windows: List[List[str]],
                                cells_seg1: List[str],
                                cells_seg2: List[str]) -> pd.DataFrame:
    """Per-window Hartigan dip test on signed visitation preference."""
    try:
        import diptest as _diptest
        _dip_available = True
    except ImportError:
        _dip_available = False
        warnings.warn(
            "diptest package not installed; using KS test as fallback for "
            "preference divergence. Install with: pip install diptest",
            stacklevel=4,
        )

    # Signed preference for each cell: (seg1 - seg2) / (seg1 + seg2)
    pref = preference(
        visit_data["segment_1"].values,
        visit_data["segment_2"].values,
        signed=True,
    )
    visit_data = visit_data.copy()
    visit_data["preference"] = pref

    rows = []
    for cells_win in cells_in_windows:
        cells_win_valid = [c for c in cells_win if c in visit_data.index]
        if not cells_win_valid:
            continue
        pt_win = visit_data.loc[cells_win_valid, "pseudotime"]
        n1 = len(set(cells_win_valid) & set(cells_seg1))
        n2 = len(set(cells_win_valid) & set(cells_seg2))
        pref_win = visit_data.loc[cells_win_valid, "preference"].values
        mean_pref = float(np.abs(pref_win).mean()) if len(pref_win) > 0 else 0.0

        if _dip_available and len(pref_win) >= 4:
            _, p = _diptest.diptest(pref_win)
        elif len(pref_win) >= 2:
            # Fallback: KS test against symmetric distribution
            _, p = scipy.stats.ks_2samp(pref_win, -pref_win)
        else:
            p = 1.0

        rows.append({
            "mean_pseudotime": float(pt_win.mean()),
            "min_pseudotime": float(pt_win.min()),
            "max_pseudotime": float(pt_win.max()),
            "cells_visited_seg1": n1,
            "cells_visited_seg2": n2,
            "p": float(p),
            "mean_preference": mean_pref,
        })
    return pd.DataFrame(rows)


def _pseudotime_breakpoint_by_stretch(
    div_pt: pd.DataFrame,
    seg1: str,
    seg2: str,
    visit_data: pd.DataFrame,
    windows: np.ndarray,
    verbose: bool = False,
) -> Optional[float]:
    """Find the pseudotime breakpoint from a window-by-window divergence table."""
    different = div_pt["different"].values.astype(bool)
    # Run-length encoding
    changes = np.concatenate([[True], different[1:] != different[:-1], [True]])
    run_starts = np.where(changes)[0][:-1]
    run_ends = np.where(changes)[0][1:]
    run_lengths = run_ends - run_starts
    run_values = different[run_starts]

    # Build a simple RLE data structure
    rle = pd.DataFrame({
        "lengths": run_lengths,
        "values": run_values,
        "start": run_starts,         # 0-indexed inclusive
        "end": run_ends - 1,         # 0-indexed inclusive
    })
    n_runs = len(rle)

    # Always different → breakpoint at beginning
    if n_runs == 1 and rle.iloc[0]["values"]:
        if verbose:
            print(f"Difference between {seg1} and {seg2} always TRUE "
                  "– breakpoint at beginning.")
        return float(div_pt["min_pseudotime"].min())

    # Always not-different → breakpoint at end (fuse immediately)
    if n_runs == 1 and not rle.iloc[0]["values"]:
        if verbose:
            print(f"Difference between {seg1} and {seg2} always FALSE "
                  "– breakpoint at end.")
        return float(div_pt["max_pseudotime"].max())

    # Exactly two runs: FALSE then TRUE
    if n_runs == 2 and not rle.iloc[0]["values"]:
        boundary_rows = [int(rle.iloc[0]["end"]), int(rle.iloc[1]["start"])]
        overlap_groups = set()
        for r in boundary_rows:
            overlap_groups.update(windows[r].tolist())
        # Cells in the overlap pseudotime groups
        mask = visit_data["pseudotime_group"].isin(overlap_groups)
        if mask.any():
            pt_break = float(visit_data.loc[mask, "pseudotime"].mean())
        else:
            pt_break = float(div_pt["max_pseudotime"].max())
        return pt_break

    # Exactly two runs: TRUE then FALSE (biologically unusual) → NA
    if n_runs == 2 and rle.iloc[0]["values"]:
        if verbose:
            print(f"No obvious breakpoint between {seg1} and {seg2} "
                  "– longest TRUE is upstream of longest FALSE.")
        return None

    # Complex pattern: use longest TRUE and FALSE stretches
    rle_sorted = rle.sort_values("lengths", ascending=False)
    true_start = int(rle_sorted.loc[rle_sorted["values"], "start"].iloc[0])
    false_end = int(rle_sorted.loc[~rle_sorted["values"], "end"].iloc[0])

    if false_end > true_start:
        if verbose:
            warnings.warn(
                f"No obvious breakpoint between {seg1} and {seg2}.",
                stacklevel=3,
            )
        return None

    # Uncertain region between false_end and true_start
    uncertain_rows = list(range(false_end + 1, true_start))
    if not uncertain_rows:
        return None
    group_counts: Dict[int, int] = {}
    for r in uncertain_rows:
        for g in windows[r].tolist():
            group_counts[g] = group_counts.get(g, 0) + 1
    max_count = max(group_counts.values())
    best_groups = {g for g, c in group_counts.items() if c == max_count}
    mask = visit_data["pseudotime_group"].isin(best_groups)
    if mask.any():
        return float(visit_data.loc[mask, "pseudotime"].mean())
    return None


def _visit_divergence_by_pseudotime(
    adata: AnnData,
    pseudotime_key: str,
    seg1: str,
    seg2: str,
    cells_in_segments: pd.DataFrame,
    pseudotime_cuts: int = 80,
    window_size: int = 5,
    pt_min: Optional[float] = None,
    pt_max: Optional[float] = None,
    divergence_method: str = "preference",
    p_thresh: float = 0.01,
    pref_thresh: float = 0.5,
    verbose: bool = False,
) -> dict:
    """Compute visit divergence between two segments across pseudotime windows."""
    cells_seg1 = cells_in_segments.index[cells_in_segments[seg1]].tolist()
    cells_seg2 = cells_in_segments.index[cells_in_segments[seg2]].tolist()
    cells_use = list(dict.fromkeys(cells_seg1 + cells_seg2))

    visit_data = adata.obs.loc[cells_use, [
        f"visitfreq_raw_{seg1}", f"visitfreq_raw_{seg2}", pseudotime_key
    ]].copy()
    visit_data.columns = ["segment_1", "segment_2", "pseudotime"]

    if pt_min is not None:
        visit_data = visit_data[visit_data["pseudotime"] >= pt_min]
    if pt_max is not None:
        visit_data = visit_data[visit_data["pseudotime"] <= pt_max]
    if len(visit_data) == 0:
        return {"cells_considered": 0, "breakpoint": None,
                "cells_in_windows": [], "details": pd.DataFrame()}

    visit_data = visit_data.sort_values("pseudotime")
    n = len(visit_data)
    n_groups = max(round(n / pseudotime_cuts), 1)

    # Assign pseudotime groups (roughly equal-sized bins)
    bin_sizes = np.diff(np.round(
        np.linspace(0, n, n_groups + 1)
    ).astype(int))
    group_labels = np.repeat(np.arange(1, n_groups + 1), bin_sizes)[:n]
    visit_data["pseudotime_group"] = group_labels

    # Sliding window matrix: each row is a window, columns are group ids
    eff_window = min(window_size, n_groups)
    windows = np.lib.stride_tricks.sliding_window_view(
        np.arange(1, n_groups + 1), eff_window
    )  # shape: (n_groups - eff_window + 1, eff_window)

    # Cell lists per window
    cells_in_windows = []
    for win_groups in windows:
        win_set = set(win_groups.tolist())
        mask = visit_data["pseudotime_group"].isin(win_set)
        cells_in_windows.append(visit_data.index[mask].tolist())

    # Divergence calculation
    if divergence_method == "ks":
        div_pt = _divergence_ks(visit_data, windows, cells_seg1, cells_seg2)
        div_pt["p"] = _holm_adjust(div_pt["p"].values)
        div_pt["different"] = div_pt["p"] <= p_thresh
    elif divergence_method == "preference":
        div_pt = _divergence_preference_dip(
            visit_data, cells_in_windows, cells_seg1, cells_seg2
        )
        if "p" not in div_pt.columns:
            return {"cells_considered": len(cells_use), "breakpoint": None,
                    "cells_in_windows": cells_in_windows, "details": div_pt}
        div_pt["p"] = _holm_adjust(div_pt["p"].values)
        div_pt["different"] = (div_pt["p"] <= p_thresh) | \
                               (div_pt["mean_preference"] > pref_thresh)
    else:
        raise ValueError("divergence_method must be 'ks' or 'preference'.")

    if len(div_pt) == 0:
        return {"cells_considered": len(cells_use), "breakpoint": None,
                "cells_in_windows": cells_in_windows, "details": div_pt}

    pt_break = _pseudotime_breakpoint_by_stretch(
        div_pt, seg1, seg2, visit_data, windows, verbose=verbose
    )
    return {
        "cells_considered": len(cells_use),
        "breakpoint": pt_break,
        "cells_in_windows": cells_in_windows,
        "details": div_pt,
    }


def _holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm–Bonferroni multiple testing correction."""
    n = len(p_values)
    if n == 0:
        return p_values
    order = np.argsort(p_values)
    adjusted = np.empty(n)
    cummax = 0.0
    for rank, idx in enumerate(order):
        adj = p_values[idx] * (n - rank)
        cummax = max(cummax, adj)
        adjusted[idx] = min(cummax, 1.0)
    return adjusted


def _all_segment_divergence(
    adata: AnnData,
    pseudotime_key: str,
    segments: List[str],
    seg_pt_limits: pd.DataFrame,
    divergence_method: str,
    pseudotime_cuts: int,
    window_size: int,
    minimum_visits: int,
    visit_threshold: float,
    p_thresh: float,
    cache: Optional[pd.DataFrame],
    cached_details: Optional[dict],
    verbose: bool,
) -> Tuple[pd.DataFrame, dict]:
    """Compute pairwise segment divergences; cache previously-computed pairs."""
    cells_in_segments = _putative_cells_in_segment(
        adata, segments, minimum_visits, visit_threshold
    )
    pairs = list(combinations(segments, 2))
    pair_keys = {(s1, s2): f"{s1}-{s2}" for s1, s2 in pairs}

    # Filter to uncached pairs
    if cache is not None:
        cached_pairs = set(zip(cache["seg_1"], cache["seg_2"]))
        new_pairs = [(s1, s2) for s1, s2 in pairs if (s1, s2) not in cached_pairs]
    else:
        new_pairs = pairs

    new_rows = []
    new_details: dict = {}
    for seg1, seg2 in new_pairs:
        trim_before = max(
            seg_pt_limits.loc[seg1, "start"],
            seg_pt_limits.loc[seg2, "start"],
        )
        trim_after = min(
            seg_pt_limits.loc[seg1, "end"],
            seg_pt_limits.loc[seg2, "end"],
        )
        if verbose:
            print(f"Calculating divergence between {seg1} and {seg2} "
                  f"(Pseudotime {trim_before:.3f} to {trim_after:.3f})")
        result = _visit_divergence_by_pseudotime(
            adata, pseudotime_key, seg1, seg2,
            cells_in_segments=cells_in_segments,
            pseudotime_cuts=pseudotime_cuts,
            window_size=window_size,
            pt_min=trim_before,
            pt_max=trim_after,
            divergence_method=divergence_method,
            p_thresh=p_thresh,
            verbose=verbose,
        )
        key = pair_keys[(seg1, seg2)]
        new_rows.append({
            "seg_1": seg1,
            "seg_2": seg2,
            "pseudotime_breakpoint": result["breakpoint"],
        })
        new_details[key] = result

    new_df = pd.DataFrame(new_rows)

    if cache is not None and len(cache) > 0:
        # Remove stale entries from cache
        old_segs = set(cache["seg_1"]) | set(cache["seg_2"])
        stale = old_segs - set(segments)
        keep_mask = ~(cache["seg_1"].isin(stale) | cache["seg_2"].isin(stale))
        clean_cache = cache[keep_mask].reset_index(drop=True)
        combined = pd.concat([clean_cache, new_df], ignore_index=True)
        combined_details = {
            **{k: v for k, v in (cached_details or {}).items()
               if not any(s in k for s in stale)},
            **new_details,
        }
    else:
        combined = new_df
        combined_details = new_details

    return combined, combined_details


# ═══════════════════════════════════════════════════════
# Section 5 – Cell → segment assignment
# ═══════════════════════════════════════════════════════

def _assign_cells_to_segments(
    adata: AnnData,
    pseudotime_key: str,
    verbose: bool = True,
) -> None:
    """Assign each cell to the segment that visited it most."""
    urd = adata.uns["urd"]
    segments = urd["segments"]
    seg_pt_limits = urd["segment_pseudotime_limits"]

    cols = [f"visitfreq_raw_{s}" for s in segments]
    visit_data = adata.obs[cols].copy().values.astype(float)
    pt = adata.obs[pseudotime_key].values

    # Boost tip cells so they are always assigned to their own tip
    cell_to_idx = {c: i for i, c in enumerate(adata.obs_names)}
    for tip in urd["tips"]:
        tip_cells = urd["cells_in_tip"].get(tip, [])
        if not tip_cells:
            continue
        # Find the segment in the tree that corresponds to this tip
        tip_seg = tip
        while tip_seg not in seg_pt_limits.index:
            parent = seg_parent(urd, tip_seg, original_joins=True)
            if parent is None or parent == tip_seg:
                break
            tip_seg = parent
        if tip_seg not in seg_pt_limits.index:
            continue
        seg_col_idx = segments.index(tip_seg)
        for cell in tip_cells:
            if cell in cell_to_idx:
                ci = cell_to_idx[cell]
                visit_data[ci, seg_col_idx] = visit_data[ci].max() + 1

    # Zero out visitation outside each segment's pseudotime range
    for j, seg in enumerate(segments):
        start = seg_pt_limits.loc[seg, "start"]
        end = seg_pt_limits.loc[seg, "end"]
        out_of_range = (pt > end) | (pt < start)
        visit_data[out_of_range, j] = 0

    # Assign by maximum
    row_max = visit_data.max(axis=1)
    no_visit_mask = row_max == 0
    assigned_seg_idx = visit_data.argmax(axis=1)
    seg_array = np.array(segments)
    cell_assignments = seg_array[assigned_seg_idx].astype(object)
    cell_assignments[no_visit_mask] = None

    n_removed = no_visit_mask.sum()
    if n_removed > 0 and verbose:
        warnings.warn(
            f"{n_removed} cells were not visited by a branch that exists "
            "at their pseudotime and were not assigned.",
            stacklevel=2,
        )

    adata.obs["segment"] = cell_assignments
    urd["cells_in_segment"] = {
        seg: list(adata.obs_names[cell_assignments == seg])
        for seg in segments
    }


# ═══════════════════════════════════════════════════════
# Section 6 – Segment collapsing / pruning
# ═══════════════════════════════════════════════════════

def _collapse_short_segments(
    urd: dict,
    min_cells: int,
    min_pseudotime: float,
    collapse_root: bool = False,
) -> None:
    """Remove segments that are too short (in cells or pseudotime)."""
    sj = urd["segment_joins"]
    seg_pt = urd["segment_pseudotime_limits"]
    cis = urd["cells_in_segment"]
    segments = urd["segments"]
    root_seg = segments[-1]

    n_cells_seg = {s: len(cis.get(s, [])) for s in segments}
    pt_length = {s: abs(seg_pt.loc[s, "end"] - seg_pt.loc[s, "start"])
                 for s in segments if s in seg_pt.index}

    segs_to_remove = set(
        s for s in segments
        if n_cells_seg.get(s, 0) < min_cells
        or pt_length.get(s, 0.0) < min_pseudotime
    )
    if not collapse_root:
        segs_to_remove.discard(root_seg)

    for seg in list(segs_to_remove):
        if seg not in sj["child"].values:
            continue  # already removed
        child_row = sj[sj["child"] == seg]
        if len(child_row) == 0:
            continue
        new_parent = child_row.iloc[0]["parent"]
        parent_rows_mask = sj["parent"] == seg
        sj.loc[parent_rows_mask, "parent"] = new_parent

        # Sync pseudotime breakpoint
        combined_idx = list(child_row.index) + list(sj[parent_rows_mask].index)
        new_pt = sj.loc[combined_idx, "pseudotime"].max()
        new_parent_children_mask = sj["parent"] == new_parent
        sj.loc[new_parent_children_mask, "pseudotime"] = new_pt

        # Remove the row where seg was a child
        sj = sj[sj["child"] != seg].reset_index(drop=True)
        segments = [s for s in segments if s != seg]

        # Update pseudotime limits
        if new_parent in seg_pt.index:
            seg_pt.loc[new_parent, "end"] = new_pt
        for ch in sj.loc[sj["parent"] == new_parent, "child"]:
            if ch in seg_pt.index:
                seg_pt.loc[ch, "start"] = new_pt
        seg_pt = seg_pt.loc[seg_pt.index.isin(segments)]

    urd["segment_joins"] = sj
    urd["segments"] = segments
    urd["segment_pseudotime_limits"] = seg_pt


def _remove_unitary_segments(urd: dict) -> None:
    """Remove segments that have only one child (middleman segments)."""
    sj = urd["segment_joins"]
    seg_pt = urd["segment_pseudotime_limits"]

    parent_counts = sj["parent"].value_counts()
    unitary_parents = set(parent_counts[parent_counts == 1].index)
    unitary_children = sj.loc[sj["parent"].isin(unitary_parents), "child"].tolist()

    for seg in unitary_children:
        child_rows = sj[sj["child"] == seg]
        if len(child_rows) == 0:
            continue
        new_parent = child_rows.iloc[0]["parent"]
        parent_rows_mask = sj["parent"] == seg
        sj.loc[parent_rows_mask, "parent"] = new_parent
        sj = sj[sj["child"] != seg].reset_index(drop=True)
        urd["segments"] = [s for s in urd["segments"] if s != seg]

        # Update pseudotime limits: parent inherits child's end
        if seg in seg_pt.index and new_parent in seg_pt.index:
            seg_pt.loc[new_parent, "end"] = seg_pt.loc[seg, "end"]
        seg_pt = seg_pt.loc[seg_pt.index.isin(urd["segments"])]

    urd["segment_joins"] = sj
    urd["segment_pseudotime_limits"] = seg_pt


# ═══════════════════════════════════════════════════════
# Section 7 – Node assignment and tree layout
# ═══════════════════════════════════════════════════════

def _assign_cells_to_nodes(
    adata: AnnData,
    pseudotime_key: str,
    node_size: int = 100,
) -> None:
    """Divide each segment into nodes and build the edge list."""
    urd = adata.uns["urd"]
    segments = urd["segments"]
    cis = urd["cells_in_segment"]
    sj = urd["segment_joins"]
    pt = adata.obs[pseudotime_key]

    edge_list: List[Tuple[str, str]] = []
    cells_in_nodes: Dict[str, List[str]] = {}
    node_assignments = pd.Series("", index=adata.obs_names, dtype=object)

    for seg in segments:
        seg_cells = cis.get(seg, [])
        if not seg_cells:
            continue
        cell_pt = pt.loc[seg_cells].sort_values()
        n = len(cell_pt)
        n_nodes = max(round(n / node_size), 1)
        node_boundaries = np.round(np.linspace(0, n, n_nodes + 1)).astype(int)

        for node_i in range(n_nodes):
            node_id = f"{seg}-{node_i + 1}"
            node_cells = cell_pt.index[node_boundaries[node_i]:node_boundaries[node_i + 1]].tolist()
            cells_in_nodes[node_id] = node_cells
            node_assignments.loc[node_cells] = node_id

            if node_i > 0:
                edge_list.append((f"{seg}-{node_i}", node_id))

        # Connect last node of this segment to first node of each child
        for child in seg_children(urd, seg):
            edge_list.append((f"{seg}-{n_nodes}", f"{child}-1"))

    adata.obs["node"] = node_assignments
    urd["cells_in_nodes"] = cells_in_nodes
    urd["edge_list"] = edge_list

    node_mean_pt: Dict[str, float] = {}
    node_max_pt: Dict[str, float] = {}
    for node_id, nc in cells_in_nodes.items():
        if nc:
            node_mean_pt[node_id] = float(pt.loc[nc].mean())
            node_max_pt[node_id] = float(pt.loc[nc].max())
        else:
            node_mean_pt[node_id] = np.nan
            node_max_pt[node_id] = np.nan

    urd["node_mean_pseudotime"] = node_mean_pt
    urd["node_max_pseudotime"] = node_max_pt


def _tree_layout_dendrogram(urd: dict) -> None:
    """Assign x-coordinates to segments using a dendrogram algorithm.

    Terminal (leaf) segments get evenly spaced integer x-values.
    Internal segments get x = mean of children.
    """
    sj = urd["segment_joins"]
    segments = urd["segments"]
    terminals = seg_terminal(urd)

    # Sort terminals by their original tip number (numeric if possible)
    try:
        terminals = sorted(terminals, key=lambda s: int(s))
    except ValueError:
        terminals = sorted(terminals)

    x_coords: Dict[str, float] = {}
    for i, t in enumerate(terminals):
        x_coords[t] = float(i)

    # Traverse bottom-up (segments are stored in order of creation, last = root)
    for seg in segments:
        if seg in x_coords:
            continue
        children = seg_children(urd, seg)
        child_xs = [x_coords[c] for c in children if c in x_coords]
        if child_xs:
            x_coords[seg] = float(np.mean(child_xs))
        else:
            x_coords[seg] = 0.0

    segment_layout = pd.DataFrame(
        {"segment": list(x_coords.keys()), "x": list(x_coords.values())}
    ).set_index("segment")
    urd["segment_layout"] = segment_layout


def _tree_layout_elaborate(urd: dict) -> None:
    """Build the full tree edge layout with coordinates for plotting."""
    seg_layout = urd["segment_layout"]
    edge_list = urd["edge_list"]
    node_max_pt = urd["node_max_pseudotime"]
    seg_pt = urd["segment_pseudotime_limits"]

    rows = []
    for n1, n2 in edge_list:
        seg1 = n1.rsplit("-", 1)[0]
        seg2 = n2.rsplit("-", 1)[0]
        x1 = float(seg_layout.loc[seg1, "x"]) if seg1 in seg_layout.index else 0.0
        x2 = float(seg_layout.loc[seg2, "x"]) if seg2 in seg_layout.index else 0.0
        y1 = node_max_pt.get(n1, np.nan)
        y2 = node_max_pt.get(n2, np.nan)

        if np.isnan(y1) and seg1 in seg_pt.index:
            y1 = float(seg_pt.loc[seg1, "start"])
        if np.isnan(y2) and seg2 in seg_pt.index:
            y2 = float(seg_pt.loc[seg2, "end"])

        rows.append({
            "node_1": n1, "node_2": n2,
            "segment_1": seg1, "segment_2": seg2,
            "x1": x1, "y1": y1,
            "x2": x2, "y2": y2,
        })

    # Add the root segment as a stub at the top
    root_seg = urd["segments"][-1]
    root_x = float(seg_layout.loc[root_seg, "x"]) \
              if root_seg in seg_layout.index else 0.0
    root_node = f"{root_seg}-1"
    root_y = node_max_pt.get(root_node, 0.0) or 0.0
    rows.append({
        "node_1": f"{root_seg}-0", "node_2": root_node,
        "segment_1": root_seg, "segment_2": root_seg,
        "x1": root_x, "y1": 0.0,
        "x2": root_x, "y2": root_y,
    })

    tree_layout = pd.DataFrame(rows)

    # Square up edges that cross x-positions (horizontal bar at branchpoint)
    squared = []
    to_drop = []
    for i, row in tree_layout.iterrows():
        if row["x1"] != row["x2"]:
            to_drop.append(i)
            # Horizontal segment at y1
            squared.append({
                "node_1": row["node_1"], "node_2": row["node_2"] + "_h",
                "segment_1": row["segment_1"], "segment_2": row["segment_2"],
                "x1": row["x1"], "y1": row["y1"],
                "x2": row["x2"], "y2": row["y1"],
            })
            # Vertical segment from branchpoint down to y2
            squared.append({
                "node_1": row["node_2"] + "_h", "node_2": row["node_2"],
                "segment_1": row["segment_2"], "segment_2": row["segment_2"],
                "x1": row["x2"], "y1": row["y1"],
                "x2": row["x2"], "y2": row["y2"],
            })

    tree_layout = tree_layout.drop(index=to_drop)
    tree_layout = pd.concat([tree_layout, pd.DataFrame(squared)],
                             ignore_index=True)
    urd["tree_layout"] = tree_layout


def _tree_layout_cells(
    adata: AnnData,
    pseudotime_key: str,
    jitter: float = 0.15,
    jitter_push: float = 0.05,
) -> None:
    """Place cells randomly around their segment's x-position."""
    urd = adata.uns["urd"]
    seg_layout = urd["segment_layout"]
    cells_in_nodes = urd["cells_in_nodes"]
    pt = adata.obs[pseudotime_key]

    rows = []
    for node_id, node_cells in cells_in_nodes.items():
        if not node_cells:
            continue
        seg = node_id.rsplit("-", 1)[0]
        x_seg = float(seg_layout.loc[seg, "x"]) if seg in seg_layout.index else 0.0
        for cell in node_cells:
            y = float(pt.loc[cell]) if cell in pt.index else np.nan
            jitter_val = np.random.uniform(jitter_push, jitter + jitter_push)
            direction = np.random.choice([-1.0, 1.0])
            x = x_seg + jitter_val * direction
            rows.append({"cell": cell, "x": x, "y": y})

    cell_layout = pd.DataFrame(rows).set_index("cell")
    urd["cell_layout"] = cell_layout


# ═══════════════════════════════════════════════════════
# Section 8 – Main build_tree function
# ═══════════════════════════════════════════════════════

def build_tree(
    adata: AnnData,
    pseudotime_key: str,
    tips_use: Optional[List[Union[str, int]]] = None,
    divergence_method: str = "preference",
    weighted_fusion: bool = True,
    use_only_original_tips: bool = True,
    cells_per_pseudotime_bin: int = 80,
    bins_per_pseudotime_window: int = 5,
    minimum_visits: int = 10,
    visit_threshold: float = 0.7,
    p_thresh: float = 0.01,
    min_cells_per_segment: int = 1,
    min_pseudotime_per_segment: float = 0.01,
    dendro_node_size: int = 100,
    dendro_cell_jitter: float = 0.15,
    dendro_cell_dist_to_tree: float = 0.05,
    verbose: bool = True,
) -> None:
    """Build the URD developmental trajectory tree.

    Agglomeratively joins trajectory segments by finding the pseudotime
    at which pairs of segments diverge in their visitation patterns.
    Stores the resulting tree structure and layout in ``adata.uns['urd']``.

    .. warning::
        This function modifies ``adata`` in-place.  Save your object before
        running if you want to re-run with different parameters.

    Parameters
    ----------
    adata
        AnnData with pseudotime in ``adata.obs[pseudotime_key]`` and visit
        frequencies in ``adata.obs['visitfreq_raw_{tip}']`` columns.
        :func:`load_tip_cells` must be called first.
    pseudotime_key
        Column in ``adata.obs`` holding flood pseudotime.
    tips_use
        Tip IDs to include.  If *None*, all tips with a
        ``visitfreq_raw_*`` column are used.
    divergence_method
        ``"ks"`` (Kolmogorov–Smirnov test) or ``"preference"`` (Hartigan
        dip test on signed preference; requires ``diptest`` package).
    weighted_fusion
        Weight the merged visitation by tip cell count.
    use_only_original_tips
        When fusing, combine only original (not internal) tip visitations.
    cells_per_pseudotime_bin
        Approximate cells per pseudotime bin for branchpoint detection.
    bins_per_pseudotime_window
        Width of sliding window in bins.
    minimum_visits
        Minimum walk visits for a cell to be assigned to a segment.
    visit_threshold
        Fraction of the cell's max visitation required for segment membership.
    p_thresh
        P-value threshold for divergence tests.
    min_cells_per_segment
        Segments with fewer cells are collapsed.
    min_pseudotime_per_segment
        Segments shorter than this in pseudotime are collapsed.
    dendro_node_size
        Approximate number of cells per node in the dendrogram.
    dendro_cell_jitter
        Amount of x-jitter applied to cells when placing them on the tree.
    dendro_cell_dist_to_tree
        Minimum x-distance from the tree line for cells.
    verbose
        Print progress.
    """
    _ensure_urd(adata)
    urd = adata.uns["urd"]

    if divergence_method not in ("ks", "preference"):
        raise ValueError("divergence_method must be 'ks' or 'preference'.")

    # Determine tips
    if tips_use is None:
        existing_cols = [c for c in adata.obs.columns if c.startswith("visitfreq_raw_")]
        tips = sorted(
            str(c.replace("visitfreq_raw_", "")) for c in existing_cols
        )
        if verbose:
            print(f"Tips not provided; using: {', '.join(tips)}")
    else:
        tips = [str(t) for t in tips_use]

    urd["tips"] = tips
    urd["pseudotime_key"] = pseudotime_key

    # Store pseudotime used for tree building
    pt = adata.obs[pseudotime_key]

    # Tip sizes for weighted fusion
    tip_sizes: Dict[str, int] = {}
    if weighted_fusion:
        for t in tips:
            tip_sizes[t] = len(urd["cells_in_tip"].get(t, []))

    # Initialise segment pseudotime limits (start, end) for each tip
    seg_pt_limits_dict: Dict[str, Dict[str, float]] = {}
    cells_in_segs_df = _putative_cells_in_segment(
        adata, tips, minimum_visits, visit_threshold
    )

    for t in tips:
        cells_t = cells_in_segs_df.index[cells_in_segs_df[t]].tolist()
        if cells_t:
            pt_t = pt.loc[cells_t].dropna()
            seg_pt_limits_dict[t] = {
                "start": float(pt_t.min()),
                "end": float(pt_t.max()),
            }
        else:
            seg_pt_limits_dict[t] = {"start": 0.0, "end": 0.0}

    seg_pt_limits = pd.DataFrame(seg_pt_limits_dict).T  # index=segments
    urd["segment_pseudotime_limits"] = seg_pt_limits

    # Determine first new segment ID
    try:
        seg_add = str(max(int(t) for t in tips) + 1)
    except ValueError:
        seg_add = "1"

    # Initialise segment_joins (binary format during construction)
    segment_joins_binary: List[Dict] = []
    active_tips = list(tips)

    # Initial divergence computation
    divergence_cache: Optional[pd.DataFrame] = None
    divergence_details_cache: Optional[dict] = None

    divergence_df, divergence_details = _all_segment_divergence(
        adata, pseudotime_key, active_tips,
        seg_pt_limits=seg_pt_limits,
        divergence_method=divergence_method,
        pseudotime_cuts=cells_per_pseudotime_bin,
        window_size=bins_per_pseudotime_window,
        minimum_visits=minimum_visits,
        visit_threshold=visit_threshold,
        p_thresh=p_thresh,
        cache=None,
        cached_details=None,
        verbose=verbose,
    )

    # ── Main tree-building loop ──────────────────────────────────────────
    while len(active_tips) >= 2:
        # Handle case where all breakpoints are NaN → set to 0 (force merge)
        if divergence_df["pseudotime_breakpoint"].isna().all():
            divergence_df["pseudotime_breakpoint"] = 0.0

        fuse_idx = divergence_df["pseudotime_breakpoint"].idxmax()
        seg1 = divergence_df.loc[fuse_idx, "seg_1"]
        seg2 = divergence_df.loc[fuse_idx, "seg_2"]
        pt_break = float(divergence_df.loc[fuse_idx, "pseudotime_breakpoint"])

        if verbose:
            print(f"Joining segments {seg1} and {seg2} at pseudotime "
                  f"{pt_break:.4f} → segment {seg_add}")

        # Record the join in binary format
        segment_joins_binary.append({
            "child_1": seg1, "child_2": seg2,
            "parent": seg_add, "pseudotime": pt_break,
        })

        # Update pseudotime limits
        seg_pt_limits.loc[seg_add] = {
            "start": min(seg_pt_limits.loc[seg1, "start"],
                         seg_pt_limits.loc[seg2, "start"]),
            "end": pt_break,
        }
        seg_pt_limits.loc[seg1, "start"] = pt_break
        seg_pt_limits.loc[seg2, "start"] = pt_break

        # Create visitation for the new merged segment
        # Children = all original tips that descend from seg1 or seg2
        temp_sj = _binary_joins_to_unary(segment_joins_binary)
        desc1 = _seg_children_all_from_binary(temp_sj, seg1, include_self=True)
        desc2 = _seg_children_all_from_binary(temp_sj, seg2, include_self=True)
        all_children = list(dict.fromkeys(desc1 + desc2))
        if use_only_original_tips:
            all_children = [c for c in all_children if c in tips]

        if weighted_fusion and all_children:
            weights = np.array([tip_sizes.get(c, 1) for c in all_children], dtype=float)
            weights /= weights.sum()
            visit_cols = [f"visitfreq_raw_{c}" for c in all_children]
            merged_visits = (
                adata.obs[visit_cols].values * weights[np.newaxis, :]
            ).sum(axis=1)
        elif all_children:
            visit_cols = [f"visitfreq_raw_{c}" for c in all_children]
            merged_visits = adata.obs[visit_cols].values.mean(axis=1)
        else:
            # Fallback: average of the two direct parents
            v1 = adata.obs[f"visitfreq_raw_{seg1}"].values
            v2 = adata.obs[f"visitfreq_raw_{seg2}"].values
            merged_visits = (v1 + v2) / 2.0

        adata.obs[f"visitfreq_raw_{seg_add}"] = merged_visits.astype(float)
        adata.obs[f"visitfreq_log_{seg_add}"] = np.log10(merged_visits + 1)

        # Update active tips
        active_tips = [t for t in active_tips if t not in (seg1, seg2)]
        active_tips.append(seg_add)

        # Increment segment counter
        try:
            seg_add = str(int(seg_add) + 1)
        except ValueError:
            seg_add = str(seg_add) + "a"

        # Recompute divergences (with caching)
        if len(active_tips) >= 2:
            divergence_df, divergence_details = _all_segment_divergence(
                adata, pseudotime_key, active_tips,
                seg_pt_limits=seg_pt_limits,
                divergence_method=divergence_method,
                pseudotime_cuts=cells_per_pseudotime_bin,
                window_size=bins_per_pseudotime_window,
                minimum_visits=minimum_visits,
                visit_threshold=visit_threshold,
                p_thresh=p_thresh,
                cache=divergence_df,
                cached_details=divergence_details,
                verbose=verbose,
            )

    # ── Post-loop: store tree structure ─────────────────────────────────
    # Convert binary joins to unary (parent, child, pseudotime)
    unary_sj = _binary_joins_to_unary(segment_joins_binary)
    urd["segment_joins"] = unary_sj
    urd["segment_joins_initial"] = unary_sj.copy()

    # All unique segments
    all_segs = sorted(
        set(unary_sj["parent"].tolist() + unary_sj["child"].tolist()),
        key=lambda s: int(s) if s.isdigit() else s,
    )
    urd["segments"] = all_segs
    urd["segment_pseudotime_limits"] = seg_pt_limits.loc[all_segs]
    urd["pseudotime_breakpoint_details"] = divergence_details

    # ── Cell assignment and refinement ──────────────────────────────────
    if verbose:
        print("Assigning cells to segments.")
    _assign_cells_to_segments(adata, pseudotime_key, verbose)

    if verbose:
        print("Collapsing short segments.")
    _collapse_short_segments(urd, min_cells_per_segment, min_pseudotime_per_segment)

    if verbose:
        print("Removing unitary segments.")
    _remove_unitary_segments(urd)

    if verbose:
        print("Reassigning cells to segments.")
    _assign_cells_to_segments(adata, pseudotime_key, verbose)

    # Drop cells that were not assigned to any segment
    assigned_mask = adata.obs["segment"].notna()
    if verbose and (~assigned_mask).sum() > 0:
        print(f"Dropping {(~assigned_mask).sum()} unassigned cells from tree data.")

    # ── Nodes and layout ────────────────────────────────────────────────
    if verbose:
        print("Assigning cells to nodes.")
    _assign_cells_to_nodes(adata, pseudotime_key, dendro_node_size)

    if verbose:
        print("Laying out tree.")
    _tree_layout_dendrogram(urd)
    _tree_layout_elaborate(urd)

    if verbose:
        print("Adding cells to tree.")
    _tree_layout_cells(
        adata, pseudotime_key,
        jitter=dendro_cell_jitter,
        jitter_push=dendro_cell_dist_to_tree,
    )

    if verbose:
        print("Done.")


# ═══════════════════════════════════════════════════════
# Section 9 – Segment naming
# ═══════════════════════════════════════════════════════

def name_segments(
    adata: AnnData,
    segments: List[Union[str, int]],
    segment_names: List[str],
    short_names: Optional[List[str]] = None,
    sep: str = "+",
) -> None:
    """Assign human-readable names to tree segments.

    For terminal segments that are combinations of named tips (because
    two tips immediately fused), their name is automatically assembled
    from the constituent tip names joined by *sep*.

    Parameters
    ----------
    adata
        AnnData object with a built tree in ``adata.uns['urd']``.
    segments
        Segment IDs to name.
    segment_names
        Names corresponding to *segments*.
    short_names
        Optional shorter names (used in force-directed layout labels).
    sep
        Separator for combined tip names.
    """
    _ensure_urd(adata)
    urd = adata.uns["urd"]

    segments = [str(s) for s in segments]
    segment_names_list = [str(n) for n in segment_names]
    name_map: Dict[str, str] = dict(zip(segments, segment_names_list))

    short_map: Optional[Dict[str, str]] = None
    if short_names is not None:
        short_map = dict(zip(segments, [str(s) for s in short_names]))

    terminals = seg_terminal(urd)
    named_names: Dict[str, str] = {}
    named_short: Dict[str, str] = {}

    for t in terminals:
        if t in name_map:
            named_names[t] = name_map[t]
            if short_map:
                named_short[t] = short_map.get(t, name_map[t])
        else:
            # Auto-combine from children that are named original tips
            og_children = [
                c for c in seg_children_all(urd, t, original_joins=True)
                if c in name_map
            ]
            if og_children:
                combined = sep.join(sorted(name_map[c] for c in og_children))
                named_names[t] = combined
                if short_map:
                    combined_short = sep.join(
                        sorted(short_map.get(c, name_map[c]) for c in og_children)
                    )
                    named_short[t] = combined_short

    urd["segment_names"] = named_names
    if short_map is not None:
        urd["segment_names_short"] = named_short


# ═══════════════════════════════════════════════════════
# Internal helpers for tree building
# ═══════════════════════════════════════════════════════

def _binary_joins_to_unary(joins_binary: List[Dict]) -> pd.DataFrame:
    """Convert binary join records to a unary parent→child DataFrame."""
    rows = []
    for j in joins_binary:
        rows.append({"parent": j["parent"], "child": j["child_1"],
                     "pseudotime": j["pseudotime"]})
        rows.append({"parent": j["parent"], "child": j["child_2"],
                     "pseudotime": j["pseudotime"]})
    if not rows:
        return pd.DataFrame(columns=["parent", "child", "pseudotime"])
    return pd.DataFrame(rows).reset_index(drop=True)


def _seg_children_all_from_binary(
    sj: pd.DataFrame, segment: str, include_self: bool = False
) -> List[str]:
    """Return all descendants from a unary segment_joins DataFrame."""
    children: List[str] = []
    frontier = [segment]
    while frontier:
        new_ch = sj.loc[sj["parent"].isin(frontier), "child"].tolist()
        children.extend(new_ch)
        frontier = new_ch
    if include_self:
        children = [segment] + children
    return list(dict.fromkeys(children))
