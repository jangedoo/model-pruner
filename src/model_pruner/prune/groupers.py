import abc

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import DBSCAN, HDBSCAN


class TokenGrouper(abc.ABC):
    """group tokens whose embeddings are similar into clusters.

    Returns a data frame with columns token_id, token, cluster_id. A cluster_id of -1
    marks tokens that are unique (each kept on its own, not merged)."""

    @abc.abstractmethod
    def group(self, token_ids: list[int], tokens: list[str], embedding_weights: torch.Tensor) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, token_ids: list[int], tokens: list[str], embedding_weights: torch.Tensor) -> pd.DataFrame:
        cluster_ids = self.group(token_ids=token_ids, tokens=tokens, embedding_weights=embedding_weights)
        return pd.DataFrame({"token_id": token_ids, "token": tokens, "cluster_id": cluster_ids})


class DBSCANGrouper(TokenGrouper):
    def __init__(self, eps: float = 0.4, min_samples=2, metric="cosine"):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

    def group(self, token_ids: list[int], tokens: list[str], embedding_weights: torch.Tensor):
        cluster_ids = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric=self.metric, n_jobs=-1).fit_predict(
            embedding_weights.numpy()
        )
        return cluster_ids


class HDBSCANGrouper(TokenGrouper):
    def __init__(self, min_cluster_size=2, metric="cosine"):
        self.min_cluster_size = min_cluster_size
        self.metric = metric

    def group(self, token_ids: list[int], tokens: list[str], embedding_weights: torch.Tensor) -> np.ndarray:
        cluster_ids = HDBSCAN(min_cluster_size=self.min_cluster_size, metric=self.metric, n_jobs=-1).fit_predict(
            embedding_weights.numpy()
        )
        return cluster_ids
