from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from typing import Union, get_args, get_origin

import click
import dotenv
from sentence_transformers import SentenceTransformer

from model_pruner.evaluation import SUPPORTED_DATASETS, PruningReport
from model_pruner.evaluation import compare as compare_fn
from model_pruner.prune.groupers import DBSCANGrouper
from model_pruner.prune.pruner import prune
from model_pruner.prune.reducers import ReduceToFirst, ReduceToMean

dotenv.load_dotenv()
GROUPER_REGISTRY: dict[str, type] = {
    "dbscan": DBSCANGrouper,
}

REDUCER_REGISTRY: dict[str, type] = {
    "first": ReduceToFirst,
    "mean": ReduceToMean,
}


def _extract_value(args: list[str], flag: str) -> str | None:
    """Find the value of --flag in a raw args list, handling both --flag val and --flag=val forms."""
    for i, arg in enumerate(args):
        if arg == flag and i + 1 < len(args):
            return args[i + 1]
        if arg.startswith(f"{flag}="):
            return arg.split("=", 1)[1]
    return None


def _annotation_to_click_type(
    annotation, default=inspect.Parameter.empty
) -> click.ParamType:
    """Map a Python type annotation (or infer from default) to a Click type."""
    if annotation is not inspect.Parameter.empty:
        origin = get_origin(annotation)
        if origin is Union:
            inner = [a for a in get_args(annotation) if a is not type(None)]
            if inner:
                return _annotation_to_click_type(inner[0])
        type_map = {
            int: click.INT,
            float: click.FLOAT,
            bool: click.BOOL,
            str: click.STRING,
        }
        if annotation in type_map:
            return type_map[annotation]
    if default is not inspect.Parameter.empty and default is not None:
        type_map = {
            int: click.INT,
            float: click.FLOAT,
            bool: click.BOOL,
            str: click.STRING,
        }
        if type(default) in type_map:
            return type_map[type(default)]
    return click.STRING


def _options_from_class(cls: type, prefix: str) -> list[click.Option]:
    """Introspect cls.__init__ and produce --{prefix}-{name} Click options."""
    options = []
    sig = inspect.signature(cls.__init__)
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        flag = f"--{prefix}-{name.replace('_', '-')}"
        default = (
            param.default if param.default is not inspect.Parameter.empty else None
        )
        click_type = _annotation_to_click_type(param.annotation, param.default)
        options.append(
            click.Option(
                [flag],
                default=default,
                show_default=True,
                type=click_type,
                help=f"[{cls.__name__}] {name}",
            )
        )
    return options


def _resolve(registry: dict, name: str, kind: str) -> type:
    if name not in registry:
        valid = ", ".join(sorted(registry.keys()))
        raise click.BadParameter(
            f"Unknown {kind} '{name}'. Valid choices: {valid}", param_hint=f"--{kind}"
        )
    return registry[name]


def _extract_prefixed(kwargs: dict, prefix: str) -> dict:
    """Return kwargs whose keys start with prefix, stripped of the prefix. Skips Nones."""
    return {
        k[len(prefix) :]: v
        for k, v in kwargs.items()
        if k.startswith(prefix) and v is not None
    }


class PruneCommand(click.Command):
    """Click Command that injects grouper- and reducer-specific options at parse time.

    This allows --help to show the concrete parameters for whichever grouper/reducer
    is selected, with their types and defaults derived from the class __init__ signature.
    Adding a new class to GROUPER_REGISTRY or REDUCER_REGISTRY is all that's needed to
    surface its parameters here.
    """

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        grouper_name = _extract_value(args, "--grouper") or "dbscan"
        reducer_name = _extract_value(args, "--reducer") or "mean"

        grouper_cls = GROUPER_REGISTRY.get(grouper_name)
        reducer_cls = REDUCER_REGISTRY.get(reducer_name)

        if grouper_cls:
            self.params.extend(_options_from_class(grouper_cls, "grouper"))
        if reducer_cls:
            self.params.extend(_options_from_class(reducer_cls, "reducer"))

        return super().parse_args(ctx, args)


def _build_stats_section(report: PruningReport) -> str:
    d = report.to_dict()
    s = d["stats"]
    m = d["metrics"]
    rel = m.get("relative", {})

    rows = [
        f"| {dataset} / {metric} | {base_val:.4f} | {m['pruned'].get(dataset, {}).get(metric, float('nan')):.4f} | {rel.get(dataset, {}).get(metric, float('nan')):.4f} |"
        for dataset, scores in m["base"].items()
        for metric, base_val in scores.items()
    ]
    metric_rows = "\n".join(rows) if rows else "| — | — | — | — |"

    base_mb = report.base_stats.embedding_size_mb
    pruned_mb = report.pruned_stats.embedding_size_mb
    saving_mb = report.embedding_saving_mb

    return f"""\
## Pruning statistics

| | Base | Pruned | Reduction |
|---|---:|---:|---:|
| Vocab size | {s["vocab_size"]["base"]:,} | {s["vocab_size"]["pruned"]:,} | {s["vocab_size"]["reduction_pct"]}% |
| Total parameters | {s["num_parameters"]["base"]:,} | {s["num_parameters"]["pruned"]:,} | {s["num_parameters"]["reduction_pct"]}% |
| Embedding parameters | {s["embedding_parameters"]["base"]:,} | {s["embedding_parameters"]["pruned"]:,} | {s["embedding_parameters"]["reduction_pct"]}% |
| Embedding size (MB) | {base_mb:.1f} | {pruned_mb:.1f} | {saving_mb:.1f} MB saved |

## Evaluation

| Dataset / Metric | Base | Pruned | Relative (base = 1.0) |
|---|---:|---:|---:|
{metric_rows}
"""


def _build_model_card(
    base_model_id: str,
    pruned_model_id: str,
    report: PruningReport | None,
) -> str:
    display_name = (
        pruned_model_id.split("/")[-1] if "/" in pruned_model_id else pruned_model_id
    )
    stats_section = _build_stats_section(report) if report else ""

    return f"""\
---
base_model: {base_model_id}
pipeline_tag: sentence-similarity
library_name: sentence-transformers
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- token-pruning
---

# {display_name}

This model is a **token-embedding pruned** version of
[{base_model_id}](https://huggingface.co/{base_model_id}).

Token-embedding pruning clusters semantically similar tokens in the embedding
space (using DBSCAN) and merges each cluster into a single shared embedding,
shrinking the vocabulary and reducing memory without retraining the transformer
layers.

## How to use

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("{pruned_model_id}", trust_remote_code=True)
embeddings = model.encode(["Hello world", "How are you?"])
```

> **Note:** `trust_remote_code=True` is required because the model ships a small
> custom tokenizer class (`pruned_tokenizer.py`) that applies the id remapping
> after tokenization. No additional package installation is needed.

{stats_section}
## Citation

If you use this model or the pruning approach, please cite:

```bibtex
@misc{{subedi2025tokenpruning,
  author = {{Sanjaya Subedi}},
  title  = {{Token Embedding Pruning for Sentence Transformers}},
  year   = {{2026}},
  note   = {{Available at: [link to be added upon publication]}}
}}
```
"""


@click.group()
def cli() -> None:
    """Prune and evaluate SentenceTransformer embedding vocabularies."""


@cli.command("prune", cls=PruneCommand)
@click.argument("model")
@click.argument("output_dir")
@click.option(
    "--grouper",
    default="dbscan",
    show_default=True,
    type=click.Choice(list(GROUPER_REGISTRY.keys())),
    help="Token grouping algorithm.",
)
@click.option(
    "--reducer",
    default="mean",
    show_default=True,
    type=click.Choice(list(REDUCER_REGISTRY.keys())),
    help="Cluster reduction strategy.",
)
@click.option(
    "--preserve",
    multiple=True,
    metavar="TOKEN",
    help="Token string to never merge. Repeatable.",
)
@click.option(
    "--evaluate/--no-evaluate",
    "run_eval",
    default=False,
    show_default=True,
    help="Run evaluation immediately after pruning.",
)
@click.option(
    "--dataset",
    multiple=True,
    default=("stsb",),
    show_default=True,
    type=click.Choice(SUPPORTED_DATASETS),
    help="Benchmark dataset (used with --evaluate). Repeatable.",
)
@click.option(
    "--report-output",
    default=None,
    type=click.Path(dir_okay=False, writable=True),
    help="Path to save evaluation JSON report.",
)
@click.option(
    "--push-to-hub",
    is_flag=True,
    default=False,
    help="Push the pruned model to Hugging Face Hub after saving.",
)
@click.option(
    "--hub-model-id",
    default=None,
    type=click.STRING,
    help="Huggingface model ID to use when pushing to hub e.g. jangedoo/my-pruned-model. Requires HF_TOKEN environemnt variable to be set",
)
@click.option(
    "--batch-size",
    default=None,
    type=click.INT,
    help="Batch size for evaluation encoding. Uses evaluator default if omitted.",
)
@click.option(
    "--precision",
    default="float32",
    show_default=True,
    type=click.Choice(["float32", "bf16", "fp16"]),
    help="Inference precision for evaluation. 'bf16' auto-detects support.",
)
@click.option(
    "--device",
    default=None,
    type=click.STRING,
    help="Device for evaluation (e.g. 'cuda', 'cpu'). Auto-detected if omitted.",
)
@click.option(
    "--no-progress-bar",
    is_flag=True,
    default=False,
    help="Disable tqdm progress bars during evaluation.",
)
def prune_cmd(
    model,
    output_dir,
    grouper,
    reducer,
    preserve,
    run_eval,
    dataset,
    report_output,
    push_to_hub,
    hub_model_id,
    batch_size,
    precision,
    device,
    no_progress_bar,
    **kwargs,
):
    """Prune the vocabulary of a SentenceTransformer model.

    MODEL is a Hugging Face model ID or local path.
    OUTPUT_DIR is where the pruned model will be saved.
    """
    grouper_cls = _resolve(GROUPER_REGISTRY, grouper, "grouper")
    reducer_cls = _resolve(REDUCER_REGISTRY, reducer, "reducer")

    grouper_instance = grouper_cls(**_extract_prefixed(kwargs, "grouper_"))
    reducer_instance = reducer_cls(**_extract_prefixed(kwargs, "reducer_"))

    if push_to_hub and "HF_TOKEN" not in os.environ:
        raise ValueError(
            f"push to hub is enabled but no huggingface token (HF_TOKEN) found in environment variables."
        )
    if push_to_hub and not hub_model_id:
        raise ValueError(
            f"push to hub is enabled but hub model id was not passed. Pass it using --hub-model-id 'username/my-pruned-model'"
        )

    click.echo(f"Loading {model} ...")
    loaded_model = SentenceTransformer(model)

    prune(
        model=loaded_model,
        grouper=grouper_instance,
        reducer=reducer_instance,
        tokens_to_preserve=list(preserve) or None,
    )

    click.echo(f"Saving pruned model to {output_dir} ...")
    loaded_model.save_pretrained(output_dir)

    report = None
    if run_eval:
        click.echo("Evaluating model")
        report = compare_fn(
            base_model=model,
            pruned_model=output_dir,
            dataset_names=list(dataset) or None,
            batch_size=batch_size,
            show_progress_bar=not no_progress_bar,
            precision=precision,
            device=device,
        )
        report.print_summary()
        if report_output:
            with open(report_output, "w") as f:
                json.dump(report.to_dict(), f, indent=2)
            click.echo(f"Report saved to {report_output}")

    (Path(output_dir) / "README.md").write_text(
        _build_model_card(
            model,
            pruned_model_id=hub_model_id if push_to_hub else output_dir,
            report=report,
        )
    )

    if push_to_hub:
        click.echo(f"Pushing pruned model to Huggingface Hub {hub_model_id}")
        SentenceTransformer(output_dir, trust_remote_code=True).push_to_hub(
            hub_model_id
        )
        click.echo("Uploaded to Huggingface Hub")

    click.echo("Done.")


@cli.command("evaluate")
@click.argument("base_model")
@click.argument("pruned_model")
@click.option(
    "--dataset",
    multiple=True,
    default=("stsb",),
    show_default=True,
    type=click.Choice(SUPPORTED_DATASETS),
    help="Benchmark dataset to run. Repeatable.",
)
@click.option(
    "--no-trust-remote-code",
    is_flag=True,
    default=False,
    help="Disable trust_remote_code when loading the pruned model.",
)
@click.option(
    "--output",
    default=None,
    type=click.Path(dir_okay=False, writable=True),
    help="Path to save JSON report.",
)
@click.option(
    "--batch-size",
    default=None,
    type=click.INT,
    help="Batch size for evaluation encoding. Uses evaluator default if omitted.",
)
@click.option(
    "--precision",
    default="float32",
    show_default=True,
    type=click.Choice(["float32", "bf16", "fp16"]),
    help="Inference precision for evaluation. 'bf16' auto-detects support.",
)
@click.option(
    "--device",
    default=None,
    type=click.STRING,
    help="Device for evaluation (e.g. 'cuda', 'cpu'). Auto-detected if omitted.",
)
@click.option(
    "--no-progress-bar",
    is_flag=True,
    default=False,
    help="Disable tqdm progress bars during evaluation.",
)
def evaluate_cmd(
    base_model,
    pruned_model,
    dataset,
    no_trust_remote_code,
    output,
    batch_size,
    precision,
    device,
    no_progress_bar,
):
    """Compare a base model against its pruned counterpart.

    BASE_MODEL is a Hugging Face model ID or local path.
    PRUNED_MODEL is a local path or Hugging Face model ID.
    """
    report = compare_fn(
        base_model=base_model,
        pruned_model=pruned_model,
        dataset_names=list(dataset) or None,
        trust_remote_code=not no_trust_remote_code,
        batch_size=batch_size,
        show_progress_bar=not no_progress_bar,
        precision=precision,
        device=device,
    )
    report.print_summary()
    if output:
        with open(output, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        click.echo(f"Report saved to {output}")
