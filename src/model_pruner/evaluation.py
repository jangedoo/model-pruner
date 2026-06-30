import dataclasses
import gc
import logging
from typing import Any, Literal

import datasets
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import (
    EmbeddingSimilarityEvaluator,
    NanoBEIREvaluator,
    SimilarityFunction,
)
from tqdm import tqdm

logger = logging.getLogger(__name__)

Precision = Literal["float32", "bf16", "fp16"]


SUPPORTED_DATASETS = ["stsb", "nanobeir", "nanobeir_ne"]


def _build_stsb_evaluator(
    split: str = "test",
    *,
    show_progress_bar: bool = True,
    batch_size: int | None = None,
    precision: Precision | None = None,
    truncate_dim: int | None = None,
) -> EmbeddingSimilarityEvaluator:
    ds = datasets.load_dataset("sentence-transformers/stsb", split=split)
    kwargs: dict[str, Any] = {
        "sentences1": ds["sentence1"],
        "sentences2": ds["sentence2"],
        "scores": ds["score"],
        "main_similarity": SimilarityFunction.COSINE,
        "name": "stsb",
        "show_progress_bar": show_progress_bar,
    }
    if batch_size is not None:
        kwargs["batch_size"] = batch_size
    if precision is not None:
        kwargs["precision"] = precision
    if truncate_dim is not None:
        kwargs["truncate_dim"] = truncate_dim
    return EmbeddingSimilarityEvaluator(**kwargs)


def _build_nanobeir_evaluator(
    *,
    show_progress_bar: bool = True,
    batch_size: int | None = None,
    truncate_dim: int | None = None,
) -> NanoBEIREvaluator:
    kwargs: dict[str, Any] = {
        "show_progress_bar": show_progress_bar,
    }
    if batch_size is not None:
        kwargs["batch_size"] = batch_size
    if truncate_dim is not None:
        kwargs["truncate_dim"] = truncate_dim
    return NanoBEIREvaluator(**kwargs)


def _build_nanobeir_ne_evaluator(
    *,
    show_progress_bar: bool = True,
    batch_size: int | None = None,
    truncate_dim: int | None = None,
) -> NanoBEIREvaluator:
    kwargs: dict[str, Any] = {
        "dataset_id": "jangedoo/NanoBEIR-ne",
        "show_progress_bar": show_progress_bar,
    }
    if batch_size is not None:
        kwargs["batch_size"] = batch_size
    if truncate_dim is not None:
        kwargs["truncate_dim"] = truncate_dim
    return NanoBEIREvaluator(**kwargs)


def _build_evaluators(
    dataset_names: list[str],
    *,
    show_progress_bar: bool = True,
    batch_size: int | None = None,
    precision: Precision | None = None,
    truncate_dim: int | None = None,
) -> dict[str, Any]:
    evaluators = {}
    for name in dataset_names:
        if name == "stsb":
            evaluators["stsb"] = _build_stsb_evaluator(
                show_progress_bar=show_progress_bar,
                batch_size=batch_size,
                precision=precision,
                truncate_dim=truncate_dim,
            )
        elif name == "nanobeir":
            evaluators["nanobeir"] = _build_nanobeir_evaluator(
                show_progress_bar=show_progress_bar,
                batch_size=batch_size,
                truncate_dim=truncate_dim,
            )
        elif name == "nanobeir_ne":
            evaluators["nanobeir_ne"] = _build_nanobeir_ne_evaluator(
                show_progress_bar=show_progress_bar,
                batch_size=batch_size,
                truncate_dim=truncate_dim,
            )
        else:
            raise ValueError(
                f"Unknown dataset '{name}'. Supported: {SUPPORTED_DATASETS}"
            )
    return evaluators


def _resolve_torch_dtype(precision: Precision | None) -> torch.dtype:
    """Resolve a precision string to a torch dtype, auto-detecting bf16 support.

    For ``"bf16"`` we check whether the target device (GPU if available, else
    CPU) actually supports bfloat16.  If it does not, we silently fall back to
    float32 so evaluation never crashes on older hardware.
    """
    if precision is None or precision == "float32":
        return torch.float32

    if precision == "bf16":
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            logger.warning(
                "CUDA device does not support bf16; falling back to float32."
            )
            return torch.float32
        # CPU bf16 support
        try:
            # torch.backends.cpu.is_bf16_supported() may not exist on older builds
            supported = bool(torch.backends.cpu.is_bf16_supported())  # type: ignore[attr-defined]
        except AttributeError:
            supported = False
        if supported:
            return torch.bfloat16
        logger.warning("CPU does not support bf16; falling back to float32.")
        return torch.float32

    if precision == "fp16":
        if not torch.cuda.is_available():
            logger.warning(
                "fp16 requested but no CUDA device available; falling back to float32."
            )
            return torch.float32
        return torch.float16

    # Should not reach here
    return torch.float32


def _load_model(
    model_or_path: str,
    *,
    trust_remote_code: bool = True,
    precision: Precision | None = None,
    device: str | None = None,
) -> SentenceTransformer:
    """Load a SentenceTransformer with optional half-precision and device."""
    dtype = _resolve_torch_dtype(precision)
    model_kwargs: dict[str, Any] = {}
    if dtype != torch.float32:
        model_kwargs["torch_dtype"] = dtype
    return SentenceTransformer(
        model_or_path,
        trust_remote_code=trust_remote_code,
        device=device,
        model_kwargs=model_kwargs or None,
    )


def _free_model(model: SentenceTransformer | None) -> None:
    """Release a model from memory and clear caches."""
    if model is None:
        return
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


def _run_evaluators(
    model: SentenceTransformer,
    evaluators: dict[str, Any],
    *,
    show_progress_bar: bool = True,
) -> dict[str, dict[str, float]]:
    results = {}
    iterator = (
        tqdm(
            evaluators.items(),
            desc="Evaluating datasets",
            disable=not show_progress_bar,
        )
        if show_progress_bar
        else evaluators.items()
    )
    for name, evaluator in iterator:
        if show_progress_bar and isinstance(iterator, tqdm):
            iterator.set_postfix_str(name)
        raw = model.evaluate(evaluator)
        results[name] = raw
    return results


def _relative(base_val: float, pruned_val: float) -> float:
    """Pruned score as a fraction of base (base = 1.0)."""
    if base_val == 0:
        return float("nan")
    return round(pruned_val / base_val, 4)


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
        return round(
            (1 - self.pruned_stats.vocab_size / self.base_stats.vocab_size) * 100, 2
        )

    @property
    def param_reduction_pct(self) -> float:
        return round(
            (1 - self.pruned_stats.num_parameters / self.base_stats.num_parameters)
            * 100,
            2,
        )

    @property
    def embedding_reduction_pct(self) -> float:
        return round(
            (
                1
                - self.pruned_stats.embedding_parameters
                / self.base_stats.embedding_parameters
            )
            * 100,
            2,
        )

    @property
    def embedding_saving_mb(self) -> float:
        return round(
            self.base_stats.embedding_size_mb - self.pruned_stats.embedding_size_mb, 2
        )

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
        from rich.console import Console
        from rich.style import Style
        from rich.table import Table

        console = Console()
        s = self.to_dict()["stats"]

        console.print()
        console.print("[bold underline]PRUNING EVALUATION REPORT[/]")
        console.print(f"  [dim]Base  :[/] {self.base_model}")
        console.print(f"  [dim]Pruned:[/] {self.pruned_model}")

        size_table = Table(
            title="Model Size",
            title_style=Style(bold=True),
            box=None,
            header_style="bold cyan",
            collapse_padding=True,
        )
        size_table.add_column("Attribute", style="cyan", no_wrap=True)
        size_table.add_column("Base", justify="right")
        size_table.add_column("Pruned", justify="right")
        size_table.add_column("Reduction", justify="right")

        size_table.add_row(
            "Vocab size",
            f"{s['vocab_size']['base']:>10,}",
            f"{s['vocab_size']['pruned']:>10,}",
            f"[red]-{s['vocab_size']['reduction_pct']}%[/]",
        )
        size_table.add_row(
            "Parameters",
            f"{s['num_parameters']['base']:>10,}",
            f"{s['num_parameters']['pruned']:>10,}",
            f"[red]-{s['num_parameters']['reduction_pct']}%[/]",
        )
        size_table.add_row(
            "Embedding params",
            f"{s['embedding_parameters']['base']:>10,}",
            f"{s['embedding_parameters']['pruned']:>10,}",
            f"[red]-{s['embedding_parameters']['reduction_pct']}%[/]",
        )
        size_table.add_row(
            "Embedding size",
            f"{self.base_stats.embedding_size_mb:>8.1f} MB",
            f"{self.pruned_stats.embedding_size_mb:>8.1f} MB",
            f"[red]-{self.embedding_saving_mb} MB[/]",
        )
        console.print()
        console.print(size_table)

        rel = self.relative_metrics()
        metric_table = Table(
            title="Metrics  (base → pruned  [relative])",
            title_style=Style(bold=True),
            box=None,
            header_style="bold cyan",
            collapse_padding=True,
        )
        metric_table.add_column("Dataset", style="yellow", no_wrap=True)
        metric_table.add_column("Metric")
        metric_table.add_column("Base", justify="right")
        metric_table.add_column("Pruned", justify="right")
        metric_table.add_column("Relative", justify="right")

        for dataset in sorted(self.base_metrics):
            for metric, base_val in self.base_metrics[dataset].items():
                pruned_val = self.pruned_metrics.get(dataset, {}).get(
                    metric, float("nan")
                )
                ratio = rel[dataset].get(metric, float("nan"))
                ratio_str = (
                    f"[green]{ratio:.4f}[/]" if ratio >= 1.0 else f"[red]{ratio:.4f}[/]"
                )
                metric_table.add_row(
                    dataset,
                    metric,
                    f"{base_val:.4f}",
                    f"{pruned_val:.4f}",
                    ratio_str,
                )
        console.print()
        console.print(metric_table)
        console.print()


def compare(
    base_model: str | SentenceTransformer,
    pruned_model: str | SentenceTransformer,
    dataset_names: list[str] | None = None,
    trust_remote_code: bool = True,
    *,
    batch_size: int | None = None,
    show_progress_bar: bool = True,
    precision: Precision | None = None,
    device: str | None = None,
) -> PruningReport:
    """Compare a base model against its pruned counterpart.

    Args:
        base_model: HF model id or loaded SentenceTransformer.
        pruned_model: Local path, HF repo id, or loaded SentenceTransformer.
        dataset_names: Which benchmark datasets to run. Defaults to ["stsb"].
        trust_remote_code: Passed to SentenceTransformer when loading from a path/repo.
        batch_size: Override the evaluator default batch size.
        show_progress_bar: Show tqdm progress bars for encoding and datasets.
        precision: Inference precision - ``"float32"``, ``"bf16"``, or ``"fp16"``.
            ``"bf16"`` auto-detects support and falls back to float32 if unavailable.
        device: Device to place the model on (e.g. ``"cuda"``, ``"cpu"``).

    Returns:
        PruningReport with raw metrics, relative scores, and model size stats.
    """
    if dataset_names is None:
        dataset_names = ["stsb"]

    if isinstance(base_model, str):
        base_model_id = base_model
        base_st = _load_model(
            base_model,
            trust_remote_code=trust_remote_code,
            precision=precision,
            device=device,
        )
    else:
        base_model_id = getattr(base_model, "model_name_or_path", "base")
        base_st = base_model

    base_evaluators = _build_evaluators(
        dataset_names,
        show_progress_bar=show_progress_bar,
        batch_size=batch_size,
        precision=precision,
    )
    base_stats = _model_stats(base_st)
    base_metrics = _run_evaluators(
        base_st, base_evaluators, show_progress_bar=show_progress_bar
    )

    if isinstance(base_model, str):
        _free_model(base_st)

    if isinstance(pruned_model, str):
        pruned_model_id = pruned_model
        pruned_st = _load_model(
            pruned_model,
            trust_remote_code=trust_remote_code,
            precision=precision,
            device=device,
        )
    else:
        pruned_model_id = getattr(pruned_model, "model_name_or_path", "pruned")
        pruned_st = pruned_model

    pruned_evaluators = _build_evaluators(
        dataset_names,
        show_progress_bar=show_progress_bar,
        batch_size=batch_size,
        precision=precision,
    )
    pruned_stats = _model_stats(pruned_st)
    pruned_metrics = _run_evaluators(
        pruned_st, pruned_evaluators, show_progress_bar=show_progress_bar
    )

    if isinstance(pruned_model, str):
        _free_model(pruned_st)

    return PruningReport(
        base_model=base_model_id,
        pruned_model=pruned_model_id,
        base_stats=base_stats,
        pruned_stats=pruned_stats,
        base_metrics=base_metrics,
        pruned_metrics=pruned_metrics,
    )


def evaluate(
    model: str | SentenceTransformer,
    dataset_names: list[str] | None = None,
    trust_remote_code: bool = True,
    *,
    batch_size: int | None = None,
    show_progress_bar: bool = True,
    precision: Precision | None = None,
    device: str | None = None,
):
    """Evaluate a single model on the given datasets.

    Args:
        model: HF model id or loaded SentenceTransformer.
        dataset_names: Which benchmark datasets to run. Defaults to ["stsb"].
        trust_remote_code: Passed to SentenceTransformer when loading from a path/repo.
        batch_size: Override the evaluator default batch size.
        show_progress_bar: Show tqdm progress bars for encoding and datasets.
        precision: Inference precision - ``"float32"``, ``"bf16"``, or ``"fp16"``.
        device: Device to place the model on (e.g. ``"cuda"``, ``"cpu"``).

    Returns:
        Dict of ``{dataset_name: {metric: score}}``.
    """
    dataset_names = dataset_names or ["stsb"]
    evaluators = _build_evaluators(
        dataset_names=dataset_names,
        show_progress_bar=show_progress_bar,
        batch_size=batch_size,
        precision=precision,
    )
    if isinstance(model, str):
        model = _load_model(
            model,
            trust_remote_code=trust_remote_code,
            precision=precision,
            device=device,
        )

    metrics = _run_evaluators(model, evaluators, show_progress_bar=show_progress_bar)
    return metrics


class PrunedQueryBaseDocumentModel:
    """this is a simple wrapper with uses base_model to encode document and pruned_model
    to encode query. this is supposed to be used with SentenceTransformer's evaluators.
    This class replicates typical scenario where a batch job with original model computes
    the document embeddings where latency is not that important and the pruned model is
    used to encode user's query on the fly."""

    def __init__(
        self, base_model: SentenceTransformer, pruned_model: SentenceTransformer
    ):
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
        logger.debug("encoding inputs using pruned model")
        return self.pruned_model.encode(inputs, *args, **kwargs)

    def encode_document(self, inputs, *args, **kwargs):
        logger.debug("encoding documents using base model")
        return self.base_model.encode_document(inputs, *args, **kwargs)

    def encode_query(self, inputs, *args, **kwargs):
        logger.debug("encoding queries using pruned model")
        return self.pruned_model.encode_query(inputs, *args, **kwargs)

    def evaluate(self, *args, **kwargs):
        return self.pruned_model.evaluate(*args, **kwargs)
