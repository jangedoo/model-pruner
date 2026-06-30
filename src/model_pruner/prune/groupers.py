import abc
import logging

import datasets
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


class TokenGrouper(abc.ABC):
    """group tokens whose embeddings are similar into clusters.

    Returns a data frame with columns token_id, token, cluster_id. A cluster_id of -1
    marks tokens that are unique (each kept on its own, not merged)."""

    @abc.abstractmethod
    def group(
        self,
        token_ids: list[int],
        tokens: list[str],
        embedding_weights: torch.Tensor,
        model: SentenceTransformer,
        tokenizer: PreTrainedTokenizerFast,
    ) -> np.ndarray:
        raise NotImplementedError

    def __call__(
        self,
        token_ids: list[int],
        tokens: list[str],
        embedding_weights: torch.Tensor,
        model: SentenceTransformer,
        tokenizer: PreTrainedTokenizerFast,
    ) -> pd.DataFrame:
        cluster_ids = self.group(
            token_ids=token_ids,
            tokens=tokens,
            embedding_weights=embedding_weights,
            model=model,
            tokenizer=tokenizer,
        )
        return pd.DataFrame(
            {"token_id": token_ids, "token": tokens, "cluster_id": cluster_ids}
        )


class DBSCANGrouper(TokenGrouper):
    def __init__(self, eps: float = 0.4, min_samples=2, metric="cosine"):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

    def group(
        self,
        token_ids: list[int],
        tokens: list[str],
        embedding_weights: torch.Tensor,
        model: SentenceTransformer,
        tokenizer: PreTrainedTokenizerFast,
    ):
        # for miltilingual-e5-small, the embedding_weights has more rows than number of tokens
        # there seems to be 37 extra rows. so we only select the embeddings for the given token ids
        token_embeddings = embedding_weights[token_ids]
        cluster_ids = DBSCAN(
            eps=self.eps, min_samples=self.min_samples, metric=self.metric, n_jobs=-1
        ).fit_predict(token_embeddings.numpy())
        return cluster_ids


class DomainGrouper(TokenGrouper):
    def __init__(
        self,
        dataset_name: str,
        subset: str | None,
        split: str,
        max_rows: int = 100000,
        text_column: str = "text",
    ):
        self.corpus_ds = (
            datasets.load_dataset(
                dataset_name,
                name=subset,
                split=split,
                streaming=True,
            )
            .shuffle(buffer_size=min(max_rows, 10000), seed=101)
            .select_columns([text_column])
        )
        if text_column != "text":
            self.corpus_ds = self.corpus_ds.rename_column(text_column, "text")
        self.max_rows = max_rows
        self.out_of_domain_cluster_id = 999_999_999

    def group(
        self,
        token_ids: list[int],
        tokens: list[str],
        embedding_weights: torch.Tensor,
        model: SentenceTransformer,
        tokenizer: PreTrainedTokenizerFast,
    ) -> np.ndarray:
        self.domain_token_ids = set(tokenizer.all_special_ids)
        print(
            f"initial domain token ids using special tokens ids of the tokenizer: {self.domain_token_ids}"
        )

        for batch in self.corpus_ds.take(self.max_rows).batch(batch_size=128):
            token_ids_list = [
                e.ids for e in tokenizer.backend_tokenizer.encode_batch(batch["text"])
            ]
            for ids in token_ids_list:
                self.domain_token_ids.update(ids)

        print(f"domain tokens computed. total {len(self.domain_token_ids):,} tokens")

        # if any of the token_id is not in domain token ids, then we assign them to same cluster_id
        cluster_ids = []
        for token_id in token_ids:
            if token_id not in self.domain_token_ids:
                cluster_ids.append(self.out_of_domain_cluster_id)
            else:
                # we'll use token_id as cluster_id
                cluster_ids.append(token_id)
        return np.array(cluster_ids)

    @classmethod
    def english_domain(cls, max_rows=100_000):
        return cls(
            dataset_name="HuggingFaceFW/fineweb-edu",
            subset="sample-10BT",
            split="train",
            max_rows=max_rows,
        )


class DomainPlusClusterGrouper(TokenGrouper):
    def __init__(self, domain_grouper: DomainGrouper, cluster_grouper: DBSCANGrouper):
        self.domain_grouper = domain_grouper
        self.cluster_grouper = cluster_grouper

    def group(
        self,
        token_ids: list[int],
        tokens: list[str],
        embedding_weights: torch.Tensor,
        model: SentenceTransformer,
        tokenizer: PreTrainedTokenizerFast,
    ) -> np.ndarray:
        domain_clusters_df = self.domain_grouper(
            token_ids=token_ids,
            tokens=tokens,
            embedding_weights=embedding_weights,
            model=model,
            tokenizer=tokenizer,
        )
        # we only need to cluster the tokens which are not out_of_domain_cluster_id
        out_of_domain_cluster_id = self.domain_grouper.out_of_domain_cluster_id

        # this df[token, token_id, cluster_id] contains tokens which are in-domain
        in_domain_clusters_df = domain_clusters_df.query(
            "cluster_id != @out_of_domain_cluster_id"
        )
        print(
            f"only clustering {len(in_domain_clusters_df):,} tokens which are in-domain"
        )

        in_domain_token_ids = in_domain_clusters_df["token_id"].tolist()
        in_domain_tokens = in_domain_clusters_df["token"].tolist()

        semantic_cluster_ids = self.cluster_grouper(
            token_ids=in_domain_token_ids,
            tokens=in_domain_tokens,
            embedding_weights=embedding_weights,
            model=model,
            tokenizer=tokenizer,
        )["cluster_id"].values

        final_cluster_ids = domain_clusters_df["cluster_id"].to_numpy().copy()
        in_domain_mask = final_cluster_ids != out_of_domain_cluster_id
        final_cluster_ids[in_domain_mask] = semantic_cluster_ids
        return final_cluster_ids
