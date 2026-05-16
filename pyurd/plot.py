"""Tree plotting for pyURD.

Equivalent to the R function plotTree.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from anndata import AnnData

from .utils import _ensure_urd


def plot_tree(
    adata: AnnData,
    label: Optional[str] = None,
    label_type: str = "obs",
    title: Optional[str] = None,
    legend: bool = True,
    legend_title: str = "",
    plot_tree: bool = True,
    tree_alpha: float = 1.0,
    tree_size: float = 1.5,
    plot_cells: bool = True,
    cell_alpha: float = 0.25,
    cell_size: float = 1.0,
    label_x: bool = True,
    label_segments: bool = False,
    color_tree: Optional[bool] = None,
    continuous_cmap: str = "viridis",
    discrete_palette: Optional[List[str]] = None,
    color_limits: Optional[Tuple[float, float]] = None,
    symmetric_color_scale: Optional[bool] = None,
    hide_y_ticks: bool = True,
    cells_highlight: Optional[List[str]] = None,
    cells_highlight_alpha: float = 1.0,
    cells_highlight_size: float = 4.0,
    ax=None,
    figsize: Tuple[float, float] = (6, 8),
):
    """Plot the URD developmental trajectory tree (dendrogram style).

    Equivalent to the R function ``plotTree``.

    Parameters
    ----------
    adata
        AnnData object with a built tree in ``adata.uns['urd']``.
    label
        Feature to colour cells/tree by.  Can be a column in
        ``adata.obs`` or a gene name in ``adata.var_names``.
        If *None*, cells are plotted in grey.
    label_type
        ``"obs"`` to look for *label* in ``adata.obs`` first, then
        ``adata.var_names`` (gene expression).  ``"gene"`` to force
        gene expression lookup.
    title
        Plot title.  Defaults to *label*.
    legend
        Show a colour legend.
    legend_title
        Legend title text.
    plot_tree
        Draw the dendrogram skeleton.
    tree_alpha
        Transparency of dendrogram lines.
    tree_size
        Line width of dendrogram.
    plot_cells
        Scatter plot cells on the tree.
    cell_alpha
        Transparency of cell points.
    cell_size
        Size of cell points.
    label_x
        Label the x-axis with terminal segment names.
    label_segments
        Annotate each segment with its ID.
    color_tree
        Colour the tree by *label* (continuous variables).  Default
        *None* auto-detects: True for continuous, False for discrete.
    continuous_cmap
        Matplotlib colormap name for continuous variables.
    discrete_palette
        List of colours for discrete variables.
    color_limits
        (vmin, vmax) for the colour scale.
    symmetric_color_scale
        Centre the colour scale at 0.  Default *None* auto-detects.
    hide_y_ticks
        Hide the pseudotime tick labels on the y-axis.
    cells_highlight
        Cell barcodes to draw on top with *cells_highlight_size*.
    cells_highlight_alpha
        Transparency of highlighted cells.
    cells_highlight_size
        Point size for highlighted cells.
    ax
        Existing Matplotlib axes.  If *None* a new figure is created.
    figsize
        Figure size when *ax* is *None*.

    Returns
    -------
    matplotlib.axes.Axes
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from matplotlib.cm import get_cmap
        from matplotlib.lines import Line2D
    except ImportError:
        raise ImportError(
            "matplotlib is required for plot_tree. "
            "Install with: pip install matplotlib"
        )

    _ensure_urd(adata)
    urd = adata.uns["urd"]

    if "tree_layout" not in urd:
        raise ValueError(
            "Tree layout not found. Run build_tree() first."
        )

    seg_layout = urd["segment_layout"]
    tree_layout = urd["tree_layout"]
    cell_layout = urd.get("cell_layout", pd.DataFrame())
    segment_names = urd.get("segment_names", {})

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if title is None:
        title = label or ""

    # ── Retrieve colour data ────────────────────────────────────────────
    color_data: Optional[pd.Series] = None
    color_discrete: bool = False

    if label is not None:
        if label_type != "gene" and label in adata.obs.columns:
            color_data = adata.obs[label]
            color_discrete = not pd.api.types.is_numeric_dtype(color_data)
        else:
            # Try gene expression
            if label in adata.var_names:
                gene_idx = adata.var_names.get_loc(label)
                raw = adata.X[:, gene_idx]
                if hasattr(raw, "toarray"):
                    raw = raw.toarray()
                expr = np.asarray(raw).flatten()
                color_data = pd.Series(expr, index=adata.obs_names)
                color_discrete = False
            else:
                warnings.warn(
                    f"Label '{label}' not found in adata.obs or adata.var_names.",
                    stacklevel=2,
                )

    # ── Auto-set colour_tree ────────────────────────────────────────────
    if label is None:
        color_tree = False
    elif color_tree is None:
        color_tree = not color_discrete

    # ── Determine colour limits for continuous data ─────────────────────
    vmin, vmax = None, None
    norm = None
    cmap = None

    if color_data is not None and not color_discrete:
        values = color_data.dropna().values.astype(float)
        if color_limits is not None:
            vmin, vmax = color_limits
        else:
            vmin = float(values.min())
            vmax = float(values.max())
            if symmetric_color_scale is None:
                symmetric_color_scale = vmin < 0
            if symmetric_color_scale:
                mv = max(abs(vmin), abs(vmax))
                vmin, vmax = -mv, mv
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = get_cmap(continuous_cmap)

    # ── Plot cells ──────────────────────────────────────────────────────
    if plot_cells and len(cell_layout) > 0:
        common_cells = cell_layout.index.intersection(adata.obs_names)
        cl = cell_layout.loc[common_cells]

        if color_data is not None:
            cd = color_data.loc[common_cells]
            if color_discrete:
                cats = cd.astype("category")
                palette = discrete_palette or _default_palette(len(cats.cat.categories))
                cat_colors = {
                    cat: palette[i % len(palette)]
                    for i, cat in enumerate(cats.cat.categories)
                }
                c_vals = [cat_colors.get(v, "#999999") for v in cd]
            else:
                c_vals = cmap(norm(cd.values.astype(float)))

            if cells_highlight is not None:
                normal_mask = ~cl.index.isin(cells_highlight)
                ax.scatter(
                    cl.loc[normal_mask, "x"], cl.loc[normal_mask, "y"],
                    c=[c_vals[i] for i, in_norm in enumerate(normal_mask) if in_norm],
                    alpha=cell_alpha, s=cell_size ** 2, linewidths=0, rasterized=True,
                )
                hl_mask = cl.index.isin(cells_highlight)
                sc = ax.scatter(
                    cl.loc[hl_mask, "x"], cl.loc[hl_mask, "y"],
                    c=[c_vals[i] for i, in_hl in enumerate(hl_mask) if in_hl],
                    alpha=cells_highlight_alpha, s=cells_highlight_size ** 2,
                    linewidths=0, rasterized=True,
                )
            else:
                sc = ax.scatter(
                    cl["x"], cl["y"],
                    c=c_vals,
                    alpha=cell_alpha, s=cell_size ** 2, linewidths=0, rasterized=True,
                )
        else:
            ax.scatter(
                cl["x"], cl["y"],
                c="#aaaaaa", alpha=cell_alpha, s=cell_size ** 2,
                linewidths=0, rasterized=True,
            )

    # ── Plot tree skeleton ───────────────────────────────────────────────
    if plot_tree and len(tree_layout) > 0:
        tl = tree_layout.dropna(subset=["x1", "y1", "x2", "y2"])
        if color_tree and color_data is not None and not color_discrete:
            # Colour each edge by the mean expression of its end node
            node_mean_expr = _compute_node_mean_expr(urd, color_data)
            for _, row in tl.iterrows():
                expr_val = node_mean_expr.get(row["node_2"], np.nan)
                c = cmap(norm(expr_val)) if not np.isnan(expr_val) else "#555555"
                ax.plot(
                    [row["x1"], row["x2"]], [row["y1"], row["y2"]],
                    color=c, alpha=tree_alpha, linewidth=tree_size,
                    solid_capstyle="butt",
                )
        else:
            for _, row in tl.iterrows():
                ax.plot(
                    [row["x1"], row["x2"]], [row["y1"], row["y2"]],
                    color="black", alpha=tree_alpha, linewidth=tree_size,
                    solid_capstyle="butt",
                )

    # ── Axes formatting ─────────────────────────────────────────────────
    ax.invert_yaxis()
    ax.set_ylabel("Pseudotime")
    ax.set_title(title)

    if hide_y_ticks:
        ax.set_yticks([])
    ax.set_xticks([])

    for spine in ["top", "right", "bottom"]:
        ax.spines[spine].set_visible(False)

    # ── X-axis tip labels ────────────────────────────────────────────────
    if label_x and len(seg_layout) > 0:
        from .tree import seg_terminal
        terminals = seg_terminal(urd)
        for t in terminals:
            if t in seg_layout.index:
                x = float(seg_layout.loc[t, "x"])
                name = segment_names.get(t, t)
                ax.text(
                    x, ax.get_ylim()[0], name,
                    ha="center", va="top", fontsize=8, rotation=45,
                )

    # ── Segment ID labels ─────────────────────────────────────────────
    if label_segments:
        for seg, row in seg_layout.iterrows():
            urd_seg_pt = urd["segment_pseudotime_limits"]
            if seg in urd_seg_pt.index:
                y_mid = (urd_seg_pt.loc[seg, "start"] + urd_seg_pt.loc[seg, "end"]) / 2
                ax.text(row["x"], y_mid, str(seg), ha="center", va="center",
                        fontsize=7, color="red")

    # ── Colour bar / legend ───────────────────────────────────────────
    if legend and color_data is not None:
        if not color_discrete and norm is not None:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label=legend_title or label or "")
        elif color_discrete:
            cats = color_data.astype("category")
            palette = discrete_palette or _default_palette(len(cats.cat.categories))
            handles = [
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=palette[i % len(palette)],
                       markersize=8, label=str(cat))
                for i, cat in enumerate(cats.cat.categories)
            ]
            ax.legend(
                handles=handles, title=legend_title or label or "",
                loc="upper right", frameon=False, fontsize=7,
            )

    return ax


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────

def _compute_node_mean_expr(urd: dict, color_data: pd.Series) -> dict:
    """Compute mean expression per tree node for colouring."""
    node_expr: dict = {}
    for node_id, cells in urd.get("cells_in_nodes", {}).items():
        valid = [c for c in cells if c in color_data.index]
        if valid:
            node_expr[node_id] = float(color_data.loc[valid].mean())
    return node_expr


def _default_palette(n: int) -> List[str]:
    """Return a default colour palette of length *n*."""
    try:
        from matplotlib.cm import tab20, tab10
        if n <= 10:
            return [mcolors.to_hex(tab10(i)) for i in range(n)]
        return [mcolors.to_hex(tab20(i % 20)) for i in range(n)]
    except Exception:
        return ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                "#bcbd22", "#17becf"] * (n // 10 + 1)


def _default_palette(n: int) -> List[str]:  # noqa: F811 (override with import fix)
    try:
        import matplotlib.colors as mcolors
        from matplotlib.cm import tab10, tab20
        if n <= 10:
            return [mcolors.to_hex(tab10(i)) for i in range(n)]
        return [mcolors.to_hex(tab20(i % 20)) for i in range(n)]
    except Exception:
        base = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        return (base * (n // len(base) + 1))[:n]
