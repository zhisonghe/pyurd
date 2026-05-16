"""Utility functions for pyURD."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.sparse

if TYPE_CHECKING:
    from anndata import AnnData


def logistic(x: np.ndarray, x0: float, k: float, c: float = 1.0) -> np.ndarray:
    """Logistic (sigmoid) function.

    Parameters
    ----------
    x : array-like
        Input values.
    x0 : float
        Inflection point (horizontal shift).
    k : float
        Slope (negative k biases toward x < x0).
    c : float
        Maximum value (default 1).

    Returns
    -------
    np.ndarray
    """
    return c / (1.0 + np.exp(-k * (np.asarray(x) - x0)))


def preference(x: np.ndarray, y: np.ndarray, signed: bool = False) -> np.ndarray:
    """Compute visitation preference of x over y.

    preference = (x - y) / (x + y)  if signed
               = |x - y| / (x + y)  otherwise

    Cells where both x and y are 0 get preference 0.

    Parameters
    ----------
    x, y : array-like
        Visitation frequencies from tip 1 and tip 2.
    signed : bool
        Whether to return signed preference.

    Returns
    -------
    np.ndarray
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    denom = x + y
    if signed:
        p = np.where(denom == 0, 0.0, (x - y) / denom)
    else:
        p = np.where(denom == 0, 0.0, np.abs(x - y) / denom)
    return p


def _ensure_urd(adata) -> None:
    """Initialize `adata.uns['urd']` if it does not exist."""
    if "urd" not in adata.uns:
        adata.uns["urd"] = {}
    urd = adata.uns["urd"]
    # Initialise commonly-needed sub-keys
    for key in ("cells_in_tip", "cells_in_segment", "cells_in_nodes",
                "segment_names", "segment_names_short"):
        if key not in urd:
            urd[key] = {}
    for key in ("tips", "segments"):
        if key not in urd:
            urd[key] = []


def make_uns_serialisable(uns: dict) -> dict:
    """Recursively convert ``adata.uns`` values to h5ad-safe types.

    Converts :class:`pandas.DataFrame` and :class:`pandas.Series` objects
    to plain Python dicts/lists and stringifies all dict keys so that
    anndata can write them to HDF5 without errors.

    Parameters
    ----------
    uns
        The ``adata.uns`` dict (or any nested sub-dict) to convert.

    Returns
    -------
    dict
        A new dict safe for ``adata.write_h5ad``.
    """
    out = {}
    for k, v in uns.items():
        k = str(k)
        if isinstance(v, pd.DataFrame):
            df = v.copy()
            df.columns = [str(c) for c in df.columns]
            out[k] = df.reset_index().to_dict(orient="list")
        elif isinstance(v, pd.Series):
            out[k] = v.to_numpy().tolist()
        elif isinstance(v, dict):
            out[k] = make_uns_serialisable(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def save_urd(adata: "AnnData", path: str) -> None:
    """Serialise *adata* (including the URD tree) and write to *path*.

    Wraps :func:`make_uns_serialisable` and ``adata.write_h5ad``.

    Parameters
    ----------
    adata
        Fully annotated AnnData object with ``adata.uns['urd']`` populated.
    path
        Output file path (``*.h5ad``).
    """
    adata_save = adata.copy()
    adata_save.uns = make_uns_serialisable(copy.deepcopy(dict(adata.uns)))
    adata_save.write_h5ad(path)
    print(f"Saved to {path}")


def restore_urd(urd_raw: dict) -> dict:
    """Reconstruct DataFrame objects in a deserialized URD dict.

    Reverses the flattening performed by :func:`make_uns_serialisable`,
    restoring :class:`pandas.DataFrame` objects with the correct index for
    each key that ``plot_tree`` and the tree-navigation helpers expect.

    Parameters
    ----------
    urd_raw
        The ``adata.uns['urd']`` dict as read back from an h5ad file.

    Returns
    -------
    dict
        URD dict with DataFrames restored.
    """
    urd = dict(urd_raw)

    for key, idx_col in [
        ("segment_layout", "segment"),
        ("cell_layout", "cell"),
    ]:
        if key in urd and isinstance(urd[key], dict):
            urd[key] = pd.DataFrame(urd[key]).set_index(idx_col)

    if "segment_pseudotime_limits" in urd and isinstance(
        urd["segment_pseudotime_limits"], dict
    ):
        df = pd.DataFrame(urd["segment_pseudotime_limits"])
        if "index" in df.columns:
            df = df.set_index("index")
        df.index.name = None
        df["start"] = df["start"].astype(float)
        df["end"] = df["end"].astype(float)
        urd["segment_pseudotime_limits"] = df

    for key in ["tree_layout", "segment_joins", "segment_joins_initial"]:
        if key in urd and isinstance(urd[key], dict):
            df = pd.DataFrame(urd[key])
            if "index" in df.columns:
                df = df.drop(columns=["index"])
            urd[key] = df.reset_index(drop=True)

    return urd


def load_urd(path: str) -> "AnnData":
    """Load an h5ad file saved with :func:`save_urd` and restore the URD tree.

    Parameters
    ----------
    path
        Path to the ``.h5ad`` file.

    Returns
    -------
    AnnData
        Object with ``adata.uns['urd']`` fully restored.
    """
    import scanpy as sc

    adata = sc.read_h5ad(path)
    if "urd" in adata.uns:
        adata.uns["urd"] = restore_urd(adata.uns["urd"])
    return adata
