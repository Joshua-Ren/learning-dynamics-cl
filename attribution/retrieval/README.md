# Translation Retrieval experiment

This experiment directly measures whether an English query retrieves the four non-English translations belonging to the same source example. It evaluates representation quality and ranking only; it does not run downstream answer generation. Agent Memory is documented separately in `../agent_memory/`.

## Experiment protocol

For every source example, the candidate pool contains Chinese, French, Korean, and Spanish versions. Each English example is a query. The scorer ranks all translated candidates and saves the top four.

The primary metrics are:

- `top4_item_accuracy_mean`: fraction of the four retrieved items that share the query's source;
- `top4_all_same_source_rate`: fraction of queries whose entire top four contains the four correct translations;
- `queries_with_all_4_same_source`: count form of the second metric.

## Contents

```text
retrieval/
├── data/
│   ├── gsm8k_train_first50_translated_manual.json
│   └── medical_translated_test_top10.json
├── scores/             # 15 JSON score outputs plus the Llama GH layer-sweep log
├── reference/
│   └── translation_top4.tsv
├── scripts/
│   └── run_model_suite.py
└── src/
    └── run_forvalue_translation_top4.py
```

The GSM8K dataset has 50 English queries and 200 translated candidates. The medical MMMLU dataset combines four subsets with 10 examples each, giving 40 queries and 160 translated candidates. The exact translated datasets are included because translation text is not expected to regenerate bit for bit.

## Run the complete suite for a new model

The following is the complete command. **Only change the value of `--model-name`** to evaluate another Hugging Face model or local checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python attribution/retrieval/scripts/run_model_suite.py \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --allow-download
```

This command runs four fixed configurations sequentially:

| dataset | score | query | maximum length | GH configuration |
| --- | --- | --- | ---: | --- |
| Medical MMMLU | RH | question + answer | 192 | none |
| Medical MMMLU | Both | question + answer | 192 | all non-final layers, bottom-relative, input-layernorm |
| GSM8K first-50 | RH | question + answer | 768 | none |
| GSM8K first-50 | Both | question + answer | 768 | all non-final layers, bottom-relative, input-layernorm |

The launcher automatically discovers the model layer count, resolves `all` to layers 1 through L-1, creates a filename-safe model directory, and writes all four JSON files under:

```text
outputs/attribution/retrieval/model_suites/<model-slug>/
```

Downloads are disabled when `--allow-download` is omitted. Existing results are protected; add `--overwrite` to intentionally rerun the same model. Use `--dry-run` to print all four resolved commands without loading the model.

## Reproduce a released retrieval suite

Run from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 python attribution/scripts/reproduce.py \
  retrieval-suite qwen3_gsm8k --allow-download
```

Available suites are `qwen3_gsm8k`, `qwen3_medical`, and `llama32_medical`. Each suite recomputes RH, GH, and RH+GH configurations. Outputs go to `outputs/attribution/retrieval/scores/`.

Recompute one saved configuration by filename or stem:

```bash
CUDA_VISIBLE_DEVICES=0 python attribution/scripts/reproduce.py score \
  forvalue_translation_top4_all_english_gh_all.json \
  --allow-download
```

Compare aggregate metrics and exact ranking against the released score:

```bash
python attribution/scripts/validate_release.py \
  --compare-score \
  attribution/retrieval/scores/forvalue_translation_top4_all_english_gh_all.json \
  outputs/attribution/retrieval/scores/forvalue_translation_top4_all_english_gh_all.json \
  --strict-ranking
```

For custom configurations, inspect the lower-level entry point:

```bash
python attribution/retrieval/src/run_forvalue_translation_top4.py --help
```

## Saved configurations

Medical retrieval and Qwen3 GSM8K retrieval use the English question plus answer as the query, a maximum length of 192, and bottom-relative GH layers. The Llama GSM8K sweep uses maximum length 768, GH input layer normalization, and top-relative prefixes beginning at layer 2. Top-relative index 1 is excluded because it corresponds to the final RH/post-norm state.

The full Llama sweep output is retained as `scores/forvalue_gsm8k_train50_top4_llama32_3b_gh_layers_sweep.log`. Absolute paths inside that file are historical log text only and are not used by any launcher.

## Qwen2.5-1.5B and Llama results

| dataset/model | best released score | top-4 item accuracy | all-four same-source rate |
| --- | --- | ---: | ---: |
| Medical MMMLU / Qwen2.5-1.5B-Instruct | Both, all GH layers | 62.50% | 25.00% |
| Medical MMMLU / Llama-3.2-3B-Instruct | Both, all GH layers | 62.50% | 25.00% |
| GSM8K first-50 / Qwen2.5-1.5B-Instruct | Both, all GH layers | 60.00% | 14.00% |
| GSM8K first-50 / Llama-3.2-3B-Instruct | Both, all GH layers (1-27) | 93.00% | 76.00% |

Every released configuration and its source artifact are listed in `reference/translation_top4.tsv`.
