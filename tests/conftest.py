import pytest
import numpy as np
import scanpy as sc
import pandas as pd
import os
import random

random_seed = 12345
np.random.seed(random_seed)
random.seed(random_seed)

cache_fname_adata = "data/pbmc3k.h5ad"
cache_fname_obs = "data/pbmc3k.obs.csv"


@pytest.fixture(scope="module")
def adata_obs(adata):
    """Fixture that returns an anndata object.
    This downloads 5.9 MB of data upon the first call of the function and stores it in ./data/pbmc3k_raw.h5ad.
    Then processed version cached at ./data/pbmc3k.h5ad (not versioned) and .data/pbmc3k.obs.csv (versioned for reference plot consistency).

    To regenerate: rm ./data/pbmc3k.obs.csv
    """
    # cache output of this fixture so we can make baseline figures
    if os.path.exists(cache_fname_obs):
        return pd.read_csv(cache_fname_obs, index_col=0)
    return adata.obs


@pytest.fixture(scope="module")
def adata():
    """Fixture that returns an anndata object.
    This downloads 5.9 MB of data upon the first call of the function and stores it in ./data/pbmc3k_raw.h5ad.
    Then processed version cached at ./data/pbmc3k.h5ad (not versioned).

    To regenerate: rm ./data/pbmc3k.h5ad
    """

    # cache output of this fixture for local test speed
    if os.path.exists(cache_fname_adata):
        return sc.read(cache_fname_adata)

    # Following https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html
    adata = sc.datasets.pbmc3k()

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    mito_genes = adata.var_names.str.startswith("MT-")
    adata.obs["percent_mito"] = np.sum(adata[:, mito_genes].X, axis=1) / np.sum(
        adata.X, axis=1
    )
    adata.obs["n_counts"] = adata.X.sum(axis=1).A1
    adata = adata[adata.obs.n_genes < 2500, :]
    adata = adata[adata.obs.percent_mito < 0.05, :]

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata

    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata = adata[:, adata.var.highly_variable]
    sc.pp.regress_out(adata, ["n_counts", "percent_mito"])
    sc.pp.scale(adata, max_value=10)

    sc.tl.pca(adata, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.umap(adata)
    sc.tl.louvain(adata)

    # these will be out of order
    # TODO: label marker genes and do this the right way with Bindea list
    new_cluster_names = [
        "CD4 T",
        "CD14 Monocytes",
        "B",
        "CD8 T",
        "NK",
        "FCGR3A Monocytes",
        "Dendritic",
        "Megakaryocytes",
    ]
    # TODO:
    # adata.obs['louvain_annotated'] = adata.obs['louvain'].copy()
    # adata.rename_categories("louvain_annotated", new_cluster_names)
    adata.rename_categories("louvain", new_cluster_names)

    # Pull umap into obs
    adata.obs["umap_1"] = adata.obsm["X_umap"][:, 0]
    adata.obs["umap_2"] = adata.obsm["X_umap"][:, 1]

    # Pull a gene value into obs
    adata.obs["CST3"] = adata[:, "CST3"].X

    adata.write(cache_fname_adata, compression="gzip")
    adata.obs.to_csv(cache_fname_obs)

    return adata
