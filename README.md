# Token Embedding Pruner

## Quick Start
Run the following command to prune the embeddings layer

```
model-pruner prune \
    --grouper dbscan \
    --grouper-eps 0.4 \
    --grouper-min-samples 2 \
    --grouper-metric cosine \
    --reducer mean \
    --evaluate \
    --dataset stsb \
    --dataset nanobeir \
    --push-to-hub \
    --hub-model-id "jangedoo/all-MiniLM-L6-v2-pruned" \
    "sentence-transformers/all-MiniLM-L6-v2" \
    "./pruned/all-MiniLM-L6-v2-pruned"
```

## Evaluate
There are two datasets currently available to evaluate: stsb and nanobeir. You can compare the base model and pruned model using the following command.
```
model-pruner evaluate --dataset stsb --dataset nanobeir "sentence-transformers/all-MiniLM-L6-v2" "./pruned/all-MiniLM-L6-v2-pruned"
```