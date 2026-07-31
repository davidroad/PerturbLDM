import numpy as np
import scanpy as sc
import umap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def plot_umap(umap_data, title=None, figsize=(10, 8),
              point_size=1, alpha=0.5, color="blue", save_path=None):
    if isinstance(umap_data, list):
        umap_arr_list = [np.asarray(xx) for xx in umap_data]
        if isinstance(point_size, (int, float)):
            point_size = [point_size] * len(umap_arr_list)
        if isinstance(alpha, (int, float)):
            alpha = [alpha] * len(umap_arr_list)
        if isinstance(color, str):
            color = [color] * len(umap_arr_list)
    else:
        umap_arr_list = [np.asarray(umap_data)]
        point_size = [point_size]
        alpha = [alpha]
        color = [color]

    fig, ax = plt.subplots(figsize=figsize)
    for ii, umap_arr in enumerate(umap_arr_list):
        if umap_arr.ndim != 2 or umap_arr.shape[1] != 2:
            raise ValueError(
                f"umap_data must have shape (n_samples, 2), got {umap_arr.shape}"
            )
        ax.scatter(
            umap_arr[:, 0],
            umap_arr[:, 1],
            s=point_size[ii],
            alpha=alpha[ii],
            c=color[ii],
            rasterized=True,
        )

    ax.set_xlabel("UMAP1", fontsize=8)
    ax.set_ylabel("UMAP2", fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    return fig, ax


def get_hvg(adata_input, n_top_genes=2000):
    sc.pp.highly_variable_genes(adata_input, n_top_genes=n_top_genes, flavor="seurat")
    return adata_input.var["highly_variable"]


def umap_adata_sc(adata):
    if adata.shape[0] >= 500000:
        raise ValueError("exceed limit!")

    adata0 = adata.copy()
    sc.pp.scale(adata0, max_value=10)
    sc.tl.pca(adata0, svd_solver="arpack")
    sc.pp.neighbors(adata0, n_neighbors=15, n_pcs=40)
    sc.tl.umap(adata0, min_dist=0.1, random_state=42)
    return adata0


def get_subset_adata(adata, prop=0.1):
    n_cells = adata.n_obs
    n_sample = int(n_cells * prop)
    random_indices = np.random.choice(adata.obs_names, size=n_sample, replace=False)
    return adata[random_indices].copy()


class SingleCellReducer:
    def __init__(
        self,
        n_pcs: int = 50,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        random_state: int = 42,
        random_state_pca: int = 42,
        max_value: float = 10.0,
        scale_data_flag: bool = True,
    ):
        self.n_pcs = n_pcs
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state
        self.max_value = max_value
        self.random_state_pca = random_state_pca

        self.scaler = None
        self.pca = None
        self.umap_model = None
        self.scale_data_flag = scale_data_flag

    def fit(self, X: np.ndarray):
        if not self.scale_data_flag:
            print("Skipping scaling step as configured.")
            X_scaled = X
        else:
            print("Scaling data.")
            self.scaler = StandardScaler()
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            X_scaled = np.clip(X_scaled, -self.max_value, self.max_value)

        print("Fitting PCA.")
        np.random.seed(self.random_state_pca)
        self.pca = PCA(n_components=self.n_pcs, svd_solver="arpack")
        self.pca.fit(X_scaled)
        X_pca = self.pca.transform(X_scaled)

        print("Fitting UMAP.")
        self.umap_model = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            n_components=2,
            random_state=self.random_state,
        )
        self.umap_model.fit(X_pca)
        X_umap = self.umap_model.transform(X_pca)

        print("Fit complete.")
        return {
            "X_scaled": X_scaled,
            "X_pca": X_pca,
            "X_umap": X_umap,
        }

    def transform(self, X_new: np.ndarray):
        if self.pca is None or self.umap_model is None:
            raise RuntimeError("Please call `fit()` before `transform()`.")

        if not self.scale_data_flag:
            print("Skipping scaling step as configured.")
            X_scaled = X_new
        else:
            if self.scaler is None:
                raise RuntimeError("Scaler not found. Please call `fit()` before `transform()`.")
            print("Scaling new data.")
            X_scaled = self.scaler.transform(X_new)
            X_scaled = np.clip(X_scaled, -self.max_value, self.max_value)

        print("Transforming PCA.")
        np.random.seed(self.random_state_pca)
        X_pca = self.pca.transform(X_scaled)

        print("Transforming UMAP.")
        np.random.seed(self.random_state)
        X_umap = self.umap_model.transform(X_pca)

        print("Transform complete.")
        return {
            "X_scaled": X_scaled,
            "X_pca": X_pca,
            "X_umap": X_umap,
        }
