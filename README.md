# Token Embedding Pruner

## Quick Start
Run the following command to prune the embeddings layer

```bash
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


model-pruner prune \
    --grouper dbscan \
    --grouper-eps 0.4 \
    --grouper-min-samples 2 \
    --grouper-metric cosine \
    --reducer mean \
    --evaluate \
    --dataset stsb \
    --dataset nanobeir \
    --dataset nanobeir_ne \
    --report-output ./multilingual_e5_small_eval.json \
    --push-to-hub \
    --hub-model-id "jangedoo/multilingual-e5-small-pruned" \
    "intfloat/multilingual-e5-small" \
    "./pruned/multilingual-e5-small-pruned"


# domain adaptation

model-pruner prune \
    --grouper domain \
    --grouper-dataset-name "HuggingFaceFW/fineweb-edu" \
    --grouper-subset "sample-10BT" \
    --grouper-split train \
    --grouper-max-rows 100000 \
    --reducer mean \
    --evaluate \
    --dataset stsb \
    --dataset nanobeir \
    --report-output ./multilingual_e5_small_en_eval.json \
    --push-to-hub \
    --hub-model-id "jangedoo/multilingual-e5-small-en-pruned" \
    "intfloat/multilingual-e5-small" \
    "./pruned/multilingual-e5-small-en-pruned"    



model-pruner prune \
    --grouper domain \
    --grouper-dataset-name "jangedoo/nepali-corpus" \
    --grouper-subset "" \
    --grouper-split train \
    --grouper-max-rows 300000 \
    --reducer mean \
    --evaluate \
    --dataset stsb_ne \
    --dataset nanobeir_ne \
    --report-output ./multilingual_e5_small_ne_eval.json \
    --push-to-hub \
    --hub-model-id "jangedoo/multilingual-e5-small-ne-pruned" \
    "intfloat/multilingual-e5-small" \
    "./pruned/multilingual-e5-small-ne-pruned"


model-pruner prune \
    --grouper domain \
    --grouper-dataset-name "jangedoo/nepali-corpus" \
    --grouper-subset "" \
    --grouper-split train \
    --grouper-max-rows 300000 \
    --reducer mean \
    --evaluate \
    --dataset stsb_ne \
    --dataset nanobeir_ne \
    --report-output ./eval_results/embeddinggemma_ne_eval.json \
    --push-to-hub \
    --hub-model-id "jangedoo/embeddinggemma-300m-ne-pruned" \
    "google/embeddinggemma-300m" \
    "./pruned/embeddinggemma-300m-ne-pruned"       
```

## Evaluate
There are two datasets currently available to evaluate: stsb and nanobeir. You can compare the base model and pruned model using the following command.


```bash
model-pruner evaluate --dataset stsb --dataset nanobeir "sentence-transformers/all-MiniLM-L6-v2" "./pruned/all-MiniLM-L6-v2-pruned"

model-pruner evaluate --dataset stsb --dataset nanobeir --dataset nanobeir_ne "sentence-transformers/all-MiniLM-L6-v2" "jangedoo/all-MiniLM-L6-v2-pruned" --output ./minilm_eval.json

model-pruner evaluate --dataset stsb --dataset nanobeir --dataset nanobeir_ne "google/embeddinggemma-300m" "jangedoo/embeddinggemma-300m-pruned" --output ./embeddinggemma_eval.json

model-pruner evaluate --dataset stsb --dataset nanobeir "intfloat/e5-small-v2" "jangedoo/e5-small-v2-pruned" --output ./e5_small_v2_eval.json
```