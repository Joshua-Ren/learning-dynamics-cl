# Forgetting and Behavioral Erosion Experiments

This directory contains the code used for the GSM8K-to-MMLU experiments in
Section 5. It is kept standalone, but its training and model-loading paths require
the accompanying LLaMA-Factory fork to be installed in the same environment.

The release covers two experimental blocks:

1. **Behavioral erosion (Section 5.2).** Full-parameter SFT or EAFT on GSM8K,
   followed by generative MMLU evaluation. We measure exact instruction
   following, use of the GSM8K-specific `####` marker, multiple-choice
   accuracy, and first-token probability mass on `A/B/C/D`.
2. **Pre-finetuning prediction (Section 5.3).** CH1 and CH2 are computed from
   the base model with forward-pass quantities, aggregated by MMLU subject,
   and correlated with the subject-wise erosion observed after fine-tuning.

The directory intentionally excludes model checkpoints, generated responses,
paper tables, job logs, scheduler configuration, and machine-specific paths.

## Requirements

Install the top-level `requirements.txt`, then install the accompanying
LLaMA-Factory fork in the same environment using that project's normal
installation instructions. The fork must provide the EAFT additions documented
in `EAFT.md` when reproducing EAFT runs.

Run all commands from this repository root. Hugging Face credentials, if
needed for a gated model, must be supplied through the normal Hugging Face
login mechanism or an environment variable; this release contains no tokens.

## Models and templates

The Section 5 experiments use the following LLaMA-Factory templates:

| Model | `TEMPLATE` |
|---|---|
| `Qwen/Qwen2.5-1.5B-Instruct` | `qwen` |
| `Qwen/Qwen3-4B-Instruct-2507` | `qwen3_nothink` |
| `Qwen/Qwen2.5-7B-Instruct` | `qwen` |
| `meta-llama/Llama-3.2-3B-Instruct` | `llama3` |

## 1. Fine-tune on GSM8K

The wrapper performs full-parameter fine-tuning for one epoch by default. The
effective global batch size is checked from `NUM_GPUS`,
`PER_DEVICE_BATCH_SIZE`, and `GLOBAL_BATCH_SIZE`.

Standard SFT:

```bash
BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
TEMPLATE=qwen \
METHOD=sft \
OUTPUT_DIR=outputs/qwen25-1.5b-sft \
bash forgetting/scripts/train_gsm8k.sh
```

EAFT:

```bash
BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
TEMPLATE=qwen \
METHOD=eaft \
EAFT_ALPHA=1.0 \
OUTPUT_DIR=outputs/qwen25-1.5b-eaft \
bash forgetting/scripts/train_gsm8k.sh
```

For multi-GPU training, set `NUM_GPUS`. For large checkpoints, pass a
DeepSpeed ZeRO-3 configuration with `DEEPSPEED_CONFIG`.

## 2. Evaluate behavioral erosion

Prepare the full MMLU test split using the exact prompt in
`evaluation/prompts.py`:

```bash
python -m forgetting.evaluation.prepare_mmlu \
  --output_file outputs/mmlu.jsonl
```

Evaluate base, SFT, and EAFT independently. Decoding is greedy by default and
the evaluator also records the first-token probability mass assigned to the
four requested choices.

```bash
python -m forgetting.evaluation.evaluate_checkpoint \
  --input_file outputs/mmlu.jsonl \
  --output_file outputs/base.jsonl \
  --model_kind base \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --template qwen

python -m forgetting.evaluation.evaluate_checkpoint \
  --input_file outputs/mmlu.jsonl \
  --output_file outputs/sft.jsonl \
  --model_kind sft \
  --model_name_or_path outputs/qwen25-1.5b-sft \
  --template qwen

python -m forgetting.evaluation.evaluate_checkpoint \
  --input_file outputs/mmlu.jsonl \
  --output_file outputs/eaft.jsonl \
  --model_kind eaft \
  --model_name_or_path outputs/qwen25-1.5b-eaft \
  --template qwen
```

Build whole-dataset and subject-level behavior tables:

```bash
python -m forgetting.evaluation.summarize_mmlu \
  --base_file outputs/base.jsonl \
  --sft_file outputs/sft.jsonl \
  --eaft_file outputs/eaft.jsonl \
  --output_dir outputs/mmlu_behavior
```

`non_if_rate` is the fraction of generations that, after stripping common
visible EOS markers and whitespace, are not exactly `A`, `B`, `C`, or `D`.
`hash_rate` is the fraction containing `####`. `choice_mass` is
`P(A)+P(B)+P(C)+P(D)` at the first assistant-token position.

## 3. Compute pre-finetuning CH1/CH2 scores

First run the lightweight closed-form checks:

```bash
python forgetting/scripts/validate_sequence_score.py
```

The paper setting samples 25 MMLU examples per subject and 100 shared GSM8K
training examples. All supervised GSM8K response tokens are used.

The command below reproduces the historical candidate pool; see `REPRODUCIBILITY.md`.

```bash
python forgetting/scripts/run_sequence_score.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --template qwen \
  --num_mmlu_per_subject 25 \
  --num_gsm8k 100 \
  --gsm8k_candidate_pool_size 300 \
  --update_token_mode all_supervised \
  --output_dir outputs/qwen25-1.5b-sequence-score
```

The run writes sampled indices, tokenization diagnostics, pair-level scores,
MMLU-example scores, and `subject_scores.csv`. No parameter gradients or model
updates are required for this calculation.

Finally correlate the pre-finetuning subject scores with post-SFT behavior:

```bash
python -m forgetting.evaluation.correlate_subjects \
  --behavior_file outputs/mmlu_behavior/subject_behavior.csv \
  --score_file outputs/qwen25-1.5b-sequence-score/subject_scores.csv \
  --condition sft \
  --output_file outputs/qwen25-1.5b-correlations.csv
```

See [METHOD.md](METHOD.md) for the score definitions and aggregation order.
