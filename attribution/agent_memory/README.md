# Agent Memory experiment

This experiment tests whether a translated worked example retrieved from memory helps a language model solve a GSM8K question that the same model answered incorrectly without memory. It is a downstream generation experiment; it is distinct from the direct translation-retrieval benchmark in `../retrieval/`.

## Experiment pipeline

1. Run greedy GSM8K inference without memory on a fixed first-200 or first-400 source pool.
2. Keep that model's incorrect examples and provide Chinese, French, Korean, and Spanish versions of each question and answer.
3. Use the English question as the query and every translated question-plus-answer pair as a memory candidate.
4. Rank candidates with RH, GH, RH+GH, or the native last-hidden-state baseline and select top-1.
5. Add the selected worked example to the prompt, generate a new answer, and report both source-selection accuracy and answer accuracy.

Candidate order is source-row order followed by `zh`, `fr`, `ko`, and `es`. Saved score JSONs retain the ranked top four candidates for every query, which is sufficient to reconstruct every released top-1 memory prompt.

## Contents

```text
agent_memory/
├── data/               # exact source pools and translated self-wrong subsets
├── scores/             # 24 saved ForValue/native top-4 score files
├── reference/
│   ├── baselines/      # five complete no-memory filtering outputs
│   ├── memory_baselines.tsv
│   └── memory_top1.tsv
└── src/
    ├── run_gsm8k_memory_agent_top1.py
    ├── run_gsm8k_no_memory_baseline.py
    ├── run_top1_agent_from_saved_selection.py
    ├── run_top1_agent_random_selection.py
    └── translate_gsm8k_wrong_examples.py
```

## Released runs and data lineage

| run ID | source pool | no-memory model | self-wrong queries | translated candidates |
| --- | ---: | --- | ---: | ---: |
| `qwen25_1p5b` | GSM8K train 0-199 | Qwen2.5-1.5B-Instruct | 62 | 248 |
| `llama32_3b` | GSM8K train 0-199 | Llama-3.2-3B-Instruct | 36 | 144 |
| `qwen3_1p7b` | GSM8K train 0-199 | Qwen3-1.7B | 51 | 204 |
| `qwen3_4b` | GSM8K train 0-399 | Qwen3-4B-Instruct-2507 | 34 | 136 |

The exact translated JSONs are released because natural-language translation is not bitwise deterministic. The Qwen2.5 and Llama subsets contain manually checked compact translations; the Qwen3 subsets were produced with Qwen3-4B-Instruct-2507. Full no-memory outputs and their wrong indices are under `reference/baselines/`.

## Reproduce downstream results from saved scores

The recommended path reuses the released deterministic selections and only reruns answer generation:

```bash
CUDA_VISIBLE_DEVICES=0 python attribution/scripts/reproduce.py \
  memory qwen25_1p5b --allow-download
```

The default output is `outputs/attribution/agent_memory/qwen25_1p5b_top1.json`. Compare it with the released reference table:

```bash
python attribution/scripts/validate_release.py \
  --memory-result qwen25_1p5b \
  outputs/attribution/agent_memory/qwen25_1p5b_top1.json
```

Available run IDs are `qwen25_1p5b`, `llama32_3b`, `qwen3_1p7b`, and `qwen3_4b`.

Run the NumPy seed-42 random-memory control with:

```bash
CUDA_VISIBLE_DEVICES=0 python attribution/scripts/reproduce.py \
  random qwen25_1p5b --allow-download
```

## Recompute a memory score

The unified launcher reads the saved JSON metadata and restores the model, dataset, query field, channels, GH layers, layer-index convention, input layer normalization, maximum length, and top-k settings:

```bash
CUDA_VISIBLE_DEVICES=0 python attribution/scripts/reproduce.py score \
  forvalue_gsm8k_train_first200_qwen25_1p5b_instruct_wrong_question_query_rh.json \
  --allow-download
```

The output is written under `outputs/attribution/agent_memory/scores/`. Compare metrics and exact top-4 order with:

```bash
python attribution/scripts/validate_release.py \
  --compare-score \
  attribution/agent_memory/scores/forvalue_gsm8k_train_first200_qwen25_1p5b_instruct_wrong_question_query_rh.json \
  outputs/attribution/agent_memory/scores/forvalue_gsm8k_train_first200_qwen25_1p5b_instruct_wrong_question_query_rh.json \
  --strict-ranking
```

For an end-to-end score-and-generation run, use `python attribution/agent_memory/src/run_gsm8k_memory_agent_top1.py --help`.

## Main saved results

| model/run | best released selector | answer accuracy | random seed-42 |
| --- | --- | ---: | ---: |
| Qwen2.5-1.5B-Instruct, first-200 self-wrong | RH / Both | 37/62 = 59.68% | 31/62 = 50.00% |
| Llama-3.2-3B-Instruct, first-200 self-wrong | Both | 34/36 = 94.44% | 14/36 = 38.89% |
| Qwen3-1.7B, first-200 self-wrong | RH | 29/51 = 56.86% | 19/51 = 37.25% |

All selectors, selection counts, answer counts, and missing-final-marker counts are in `reference/memory_top1.tsv`.

## Generation protocol and reproducibility

No-memory filtering and memory evaluation use chat templates and greedy decoding. Qwen3 thinking is disabled.

- filtering: maximum prompt length 2,048 and 256 new tokens;
- memory evaluation: maximum prompt length 2,048 and 512 new tokens;
- random memory: NumPy `default_rng(42)` over all translated candidates.

The selected memory and prompt are deterministic once a saved score is used. Greedy CUDA generation can still change at numerical boundaries across GPU, PyTorch, Transformers, or kernel versions. One one-answer change was observed in a Qwen3-1.7B GH rerun while all 51 top-1 memories stayed identical. The result validator therefore permits one answer of tolerance by default; pass `--answer-tolerance 0` for strict comparison.
