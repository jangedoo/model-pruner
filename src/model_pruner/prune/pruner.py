import itertools
from typing import cast

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from model_pruner.prune.groupers import TokenGrouper
from model_pruner.prune.reducers import ClusterReducer
from model_pruner.pruning_tokenizer import PrunedTokenizer


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def seuqential_id_allocator(reserved_ids=None):
    reserved_ids = set(map(int, reserved_ids or []))
    for i in itertools.count(0):
        if i not in reserved_ids:
            yield i


def get_embeddings_layer(model: SentenceTransformer) -> torch.nn.Embedding:
    if model.transformers_model is None:
        raise Exception(f"model has no transformers_model")

    return model.transformers_model.get_input_embeddings()  # type: ignore


def get_embedding_weights(model: SentenceTransformer) -> torch.Tensor:
    with torch.no_grad():
        embedding_weights: torch.Tensor = get_embeddings_layer(model=model).weight.detach().cpu()  # type: ignore
        return embedding_weights


def reduce_clusters(
    clusters_df: pd.DataFrame,
    special_token_ids: list | set,
    embedding_weights: torch.Tensor,
    reducer: ClusterReducer,
):

    special_token_ids = set(special_token_ids)
    old_token_id_to_pruned_token_id = {}
    pruned_token_id_to_embedding = {}
    id_allocator = seuqential_id_allocator()
    # assign new sequential ids to special tokens first (sorted for determinism).
    # we cannot keep special tokens at their original IDs because high-ID specials
    # (e.g. Gemma's <image_soft_token> at 255999/256000) would create gaps after
    # the regular vocab is compacted down.
    for old_id in sorted(special_token_ids):
        new_id = next(id_allocator)
        old_token_id_to_pruned_token_id[old_id] = new_id
        pruned_token_id_to_embedding[new_id] = embedding_weights[old_id]
    for _, row in clusters_df.iterrows():
        all_token_ids = list(row["token_ids"])
        # only consider tokens which are not special tokens since we already remapped them
        token_ids = []
        tokens = []
        for i, token_id in enumerate(all_token_ids):
            if token_id not in special_token_ids:
                token_ids.append(token_id)
                tokens.append(row["tokens"][i])

        if len(token_ids) == 0:
            continue

        # if this cluster of tokens is "valid" for pruning
        is_valid = row["is_valid"]
        if is_valid:
            # multiple old token ids collapse into a single pruned_token_id
            pruned_token_id = next(id_allocator)
            _, cluster_embedding = reducer(token_ids=token_ids, tokens=tokens, embedding_weights=embedding_weights)
            for old_token_id in token_ids:
                old_token_id_to_pruned_token_id[old_token_id] = pruned_token_id

            pruned_token_id_to_embedding[pruned_token_id] = cluster_embedding
        else:
            # all tokens in this clusters are unique but we still need to assign new "pruned" token id
            for old_token_id in token_ids:
                pruned_token_id = next(id_allocator)
                old_token_id_to_pruned_token_id[old_token_id] = pruned_token_id
                pruned_token_id_to_embedding[pruned_token_id] = embedding_weights[old_token_id]

    # validate that pruned token ids are sequential i.e. without any gaps
    if set(pruned_token_id_to_embedding.keys()) != set(range(len(pruned_token_id_to_embedding))):
        raise Exception(f"pruned token ids are not sequential. There are gaps")

    pruned_embedding_weights = torch.stack(
        [pruned_token_id_to_embedding[i] for i in range(len(pruned_token_id_to_embedding))]
    )

    return old_token_id_to_pruned_token_id, pruned_embedding_weights


def update_model_token_embeddings(model: SentenceTransformer, new_embeddings: torch.Tensor):
    new_vocab_size = new_embeddings.shape[0]
    if model.transformers_model is None:
        raise Exception("transformers_model attribute of SentenceTransformer is None")

    model.transformers_model.resize_token_embeddings(new_vocab_size)
    with torch.no_grad():
        get_embeddings_layer(model=model).weight.copy_(new_embeddings)  # type: ignore


def prune(
    model: str | SentenceTransformer,
    grouper: TokenGrouper,
    reducer: ClusterReducer,
    tokens_to_preserve: list | None = None,
):
    if isinstance(model, str):
        print(f"Loading model {model}")
        model = SentenceTransformer(model)

    embedding_weights = get_embedding_weights(model=model)
    print(f"Retrieved embedding weights with shape {embedding_weights.shape}")

    tokenizer = model.tokenizer
    token_ids = list(range(tokenizer.vocab_size))
    tokens = [tokenizer.convert_ids_to_tokens(i) for i in token_ids]

    print(f"Tokenizer has {len(token_ids):,} vocab size")

    df = grouper(
        token_ids=token_ids, tokens=tokens, embedding_weights=embedding_weights, model=model, tokenizer=tokenizer
    )
    required_cols = {"token_id", "token", "cluster_id"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Grouper must return a data frame with at least these columns: {required_cols} but got columns {df.columns}"
        )
    clusters_df = (
        df.groupby("cluster_id", as_index=False)
        .agg(token_ids=("token_id", list), tokens=("token", list), cluster_size=("token", len))
        .sort_values("cluster_size", ascending=False)
    )

    print(f"{len(clusters_df):,} clusters were formed")

    tokens_to_preserve = tokens_to_preserve or []
    tokens_to_preserve = [*tokenizer.special_tokens_map.values(), *tokens_to_preserve]
    print(f"Any clusters with the following tokens will not be merged. {tokens_to_preserve}")

    def is_valid(row):
        tokens = row["tokens"]
        return row["cluster_id"] != -1 and not any(preserve in tokens for preserve in tokens_to_preserve)

    clusters_df["is_valid"] = clusters_df.apply(is_valid, axis=1)

    valid_clusters_df = clusters_df.query("is_valid == True")
    n_orig_tokens = tokenizer.vocab_size
    n_tokens_to_merge = valid_clusters_df["cluster_size"].sum()
    n_new_tokens = len(valid_clusters_df) + clusters_df.query("is_valid == False")["cluster_size"].sum()
    pct_of_original = n_new_tokens / n_orig_tokens
    pct_reduction = (n_orig_tokens - n_new_tokens) / n_orig_tokens

    print(
        f"original tokens = {n_orig_tokens:,}, tokens to merge = {n_tokens_to_merge:,}, after merging = {n_new_tokens:,}, pct reduction = {pct_reduction*100:.3}%, pct of original = {pct_of_original*100:.3}%"
    )

    old_token_id_to_pruned_token_id, pruned_embedding_weights = reduce_clusters(
        clusters_df=clusters_df,
        # embeddinggemma-300m has a special token <image_soft_token> with id=262144
        # but it has embedding weight of 262144, 768. it seems tokenizer can have special tokens which
        # are not looked up in embedding table. so we just make sure every id is < the embeddings table
        special_token_ids=[id for id in tokenizer.all_special_ids if id < embedding_weights.shape[0]],
        embedding_weights=embedding_weights,
        reducer=reducer,
    )

    print(f"pruning successful. new vocab size: {pruned_embedding_weights.shape[0]:,}")

    print("updating embedding weights")
    orig_params_count = count_params(model)
    update_model_token_embeddings(model, pruned_embedding_weights)
    pruned_params_count = count_params(model)
    print(
        f"orig params = {orig_params_count:,}, new params = {pruned_params_count:,}, pct of original = {pruned_params_count / orig_params_count * 100:.3}%, reduction = {(orig_params_count - pruned_params_count) / orig_params_count * 100:.3}%"
    )

    print("updating tokenizer")
    # this is a bit hacky. not sure if there is a better way
    pruned_tokenizer = cast(PrunedTokenizer, model[0].processor)
    pruned_tokenizer.__class__ = PrunedTokenizer
    pruned_tokenizer.set_id_map(old_token_id_to_pruned_token_id)
    return model
