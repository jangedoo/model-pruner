import abc

import torch


class ClusterReducer(abc.ABC):
    """reduce one cluster of tokens to a single representative embedding.

    Receives the token_ids and tokens belonging to a cluster and returns
    (representative_token_id, embedding) — the many tokens collapse onto this embedding."""

    @abc.abstractmethod
    def __call__(
        self, token_ids: list[int], tokens: list[str], embedding_weights: torch.Tensor
    ) -> tuple[int, torch.Tensor]:
        raise NotImplementedError()


class ReduceToFirst(ClusterReducer):
    def __call__(
        self, token_ids: list[int], tokens: list[str], embedding_weights: torch.Tensor
    ) -> tuple[int, torch.Tensor]:
        representative_token_id = token_ids[0]
        return representative_token_id, embedding_weights[representative_token_id]


class ReduceToMean(ClusterReducer):
    def __call__(
        self, token_ids: list[int], tokens: list[str], embedding_weights: torch.Tensor
    ) -> tuple[int, torch.Tensor]:
        representative_token_id = token_ids[0]
        return representative_token_id, embedding_weights[token_ids].mean(dim=0)
