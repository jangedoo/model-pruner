import argparse
import dataclasses
import json
from typing import Any

import datasets
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import (
    EmbeddingSimilarityEvaluator,
    NanoBEIREvaluator,
    SimilarityFunction,
)

# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

SUPPORTED_DATASETS = ["stsb", "nanobeir", "nanobeir_ne"]


def _build_stsb_evaluator(split: str = "test") -> EmbeddingSimilarityEvaluator:
    ds = datasets.load_dataset("sentence-transformers/stsb", split=split)
    return EmbeddingSimilarityEvaluator(
        sentences1=ds["sentence1"],
        sentences2=ds["sentence2"],
        scores=ds["score"],
        main_similarity=SimilarityFunction.COSINE,
        name="stsb",
    )


def _build_nanobeir_evaluator() -> NanoBEIREvaluator:
    return NanoBEIREvaluator()


def _build_nanobeir_ne_evaluator() -> NanoBEIREvaluator:
    return NanoBEIREvaluator(dataset_id="jangedoo/NanoBEIR-ne")


def _build_evaluators(dataset_names: list[str]) -> dict[str, Any]:
    evaluators = {}
    for name in dataset_names:
        if name == "stsb":
            evaluators["stsb"] = _build_stsb_evaluator()
        elif name == "nanobeir":
            evaluators["nanobeir"] = _build_nanobeir_evaluator()
        elif name == "nanobeir_ne":
            evaluators["nanobeir_ne"] = _build_nanobeir_ne_evaluator()
        else:
            raise ValueError(f"Unknown dataset '{name}'. Supported: {SUPPORTED_DATASETS}")
    return evaluators


# ---------------------------------------------------------------------------
# Model stats
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ModelStats:
    vocab_size: int
    num_parameters: int
    embedding_parameters: int
    embedding_size_mb: float  # word embedding matrix only, float32


def _model_stats(model: SentenceTransformer) -> ModelStats:
    vocab_size = model.transformers_model.config.vocab_size  # type: ignore[union-attr]
    num_parameters = sum(p.numel() for p in model.parameters())
    emb = model.transformers_model.get_input_embeddings()  # type: ignore[union-attr]
    emb_params = emb.weight.numel()
    emb_size_mb = emb_params * 4 / 1024 / 1024  # float32
    return ModelStats(
        vocab_size=vocab_size,
        num_parameters=num_parameters,
        embedding_parameters=emb_params,
        embedding_size_mb=round(emb_size_mb, 2),
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _run_evaluators(model: SentenceTransformer, evaluators: dict[str, Any]) -> dict[str, dict[str, float]]:
    results = {}
    for name, evaluator in evaluators.items():
        raw = model.evaluate(evaluator)
        results[name] = raw
    return results


def _relative(base_val: float, pruned_val: float) -> float:
    """Pruned score as a fraction of base (base = 1.0)."""
    if base_val == 0:
        return float("nan")
    return round(pruned_val / base_val, 4)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class PruningReport:
    base_model: str
    pruned_model: str
    base_stats: ModelStats
    pruned_stats: ModelStats
    base_metrics: dict[str, dict[str, float]]
    pruned_metrics: dict[str, dict[str, float]]

    # ---- computed properties ----

    @property
    def vocab_reduction_pct(self) -> float:
        return round((1 - self.pruned_stats.vocab_size / self.base_stats.vocab_size) * 100, 2)

    @property
    def param_reduction_pct(self) -> float:
        return round((1 - self.pruned_stats.num_parameters / self.base_stats.num_parameters) * 100, 2)

    @property
    def embedding_reduction_pct(self) -> float:
        return round((1 - self.pruned_stats.embedding_parameters / self.base_stats.embedding_parameters) * 100, 2)

    @property
    def embedding_saving_mb(self) -> float:
        return round(self.base_stats.embedding_size_mb - self.pruned_stats.embedding_size_mb, 2)

    def relative_metrics(self) -> dict[str, dict[str, float]]:
        """For each metric, pruned / base (so base = 1.0)."""
        rel: dict[str, dict[str, float]] = {}
        for dataset, base_scores in self.base_metrics.items():
            rel[dataset] = {}
            pruned_scores = self.pruned_metrics.get(dataset, {})
            for metric, base_val in base_scores.items():
                pruned_val = pruned_scores.get(metric, float("nan"))
                rel[dataset][metric] = _relative(base_val, pruned_val)
        return rel

    def to_dict(self) -> dict:
        return {
            "base_model": self.base_model,
            "pruned_model": self.pruned_model,
            "stats": {
                "vocab_size": {
                    "base": self.base_stats.vocab_size,
                    "pruned": self.pruned_stats.vocab_size,
                    "reduction_pct": self.vocab_reduction_pct,
                },
                "num_parameters": {
                    "base": self.base_stats.num_parameters,
                    "pruned": self.pruned_stats.num_parameters,
                    "reduction_pct": self.param_reduction_pct,
                },
                "embedding_parameters": {
                    "base": self.base_stats.embedding_parameters,
                    "pruned": self.pruned_stats.embedding_parameters,
                    "reduction_pct": self.embedding_reduction_pct,
                    "saving_mb": self.embedding_saving_mb,
                },
            },
            "metrics": {
                "base": self.base_metrics,
                "pruned": self.pruned_metrics,
                "relative": self.relative_metrics(),
            },
        }

    def print_summary(self) -> None:
        d = self.to_dict()
        print("\n" + "=" * 60)
        print("PRUNING EVALUATION REPORT")
        print("=" * 60)
        print(f"  Base   : {self.base_model}")
        print(f"  Pruned : {self.pruned_model}")

        print("\n── Model size ──────────────────────────────────────────────")
        s = d["stats"]
        print(
            f"  Vocab size       : {s['vocab_size']['base']:>10,}  →  {s['vocab_size']['pruned']:>10,}  (-{s['vocab_size']['reduction_pct']}%)"
        )
        print(
            f"  Parameters       : {s['num_parameters']['base']:>10,}  →  {s['num_parameters']['pruned']:>10,}  (-{s['num_parameters']['reduction_pct']}%)"
        )
        print(
            f"  Embedding params : {s['embedding_parameters']['base']:>10,}  →  {s['embedding_parameters']['pruned']:>10,}  (-{s['embedding_parameters']['reduction_pct']}%)"
        )
        print(
            f"  Embedding size   : {self.base_stats.embedding_size_mb:>9.1f} MB  →  {self.pruned_stats.embedding_size_mb:>9.1f} MB  (-{self.embedding_saving_mb} MB)"
        )

        print("\n── Metrics (base → pruned  [relative]) ─────────────────────")
        rel = self.relative_metrics()
        for dataset in self.base_metrics:
            print(f"\n  [{dataset}]")
            for metric, base_val in self.base_metrics[dataset].items():
                pruned_val = self.pruned_metrics.get(dataset, {}).get(metric, float("nan"))
                ratio = rel[dataset].get(metric, float("nan"))
                print(f"    {metric:<40s} {base_val:.4f}  →  {pruned_val:.4f}  [rel: {ratio:.4f}]")
        print("=" * 60 + "\n")


def evaluate(
    base_model: str | SentenceTransformer,
    pruned_model: str | SentenceTransformer,
    dataset_names: list[str] | None = None,
    trust_remote_code: bool = True,
) -> PruningReport:
    """Compare a base model against its pruned counterpart.

    Args:
        base_model: HF model id or loaded SentenceTransformer.
        pruned_model: Local path, HF repo id, or loaded SentenceTransformer.
        dataset_names: Which benchmark datasets to run. Defaults to ["stsb"].
        trust_remote_code: Passed to SentenceTransformer when loading from a path/repo.

    Returns:
        PruningReport with raw metrics, relative scores, and model size stats.
    """
    if dataset_names is None:
        dataset_names = ["stsb"]

    if isinstance(base_model, str):
        base_model_id = base_model
        base_model = SentenceTransformer(base_model)
    else:
        base_model_id = getattr(base_model, "model_name_or_path", "base")

    if isinstance(pruned_model, str):
        pruned_model_id = pruned_model
        pruned_model = SentenceTransformer(pruned_model, trust_remote_code=trust_remote_code)
    else:
        pruned_model_id = getattr(pruned_model, "model_name_or_path", "pruned")

    evaluators = _build_evaluators(dataset_names)

    base_metrics = _run_evaluators(base_model, evaluators)
    pruned_metrics = _run_evaluators(pruned_model, evaluators)

    return PruningReport(
        base_model=base_model_id,
        pruned_model=pruned_model_id,
        base_stats=_model_stats(base_model),
        pruned_stats=_model_stats(pruned_model),
        base_metrics=base_metrics,
        pruned_metrics=pruned_metrics,
    )


class PrunedQueryBaseDocumentModel:
    """this is a simple wrapper with uses base_model to encode document and pruned_model
    to encode query. this is supposed to be used with SentenceTransformer's evaluators.
    This class replicates typical scenario where a batch job with original model computes
    the document embeddings where latency is not that important and the pruned model is
    used to encode user's query on the fly."""

    def __init__(self, base_model: SentenceTransformer, pruned_model: SentenceTransformer):
        self.base_model = base_model
        self.pruned_model = pruned_model
        self.transformers_model = self.pruned_model.transformers_model
        self.tokenizer = self.pruned_model.tokenizer
        self.processor = self.pruned_model.processor
        self.parameters = self.pruned_model.parameters

    @property
    def model_card_data(self):
        return self.pruned_model.model_card_data

    def encode(self, inputs, *args, **kwargs):
        print("encoding inputs using pruned model")
        return self.pruned_model.encode(inputs, *args, **kwargs)

    def encode_document(self, inputs, *args, **kwargs):
        print("encoding documents using base model")
        return self.base_model.encode_document(inputs, *args, **kwargs)

    def encode_query(self, inputs, *args, **kwargs):
        print("encoding queries using pruned model")
        return self.pruned_model.encode_query(inputs, *args, **kwargs)

    def evaluate(self, *args, **kwargs):
        return self.pruned_model.evaluate(*args, **kwargs)
