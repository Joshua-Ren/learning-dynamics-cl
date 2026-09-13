#!/usr/bin/env bash
# Reproduce the paired full-parameter GSM8K -> MMLU format-forgetting protocol.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
MODEL="${MODEL:?Set MODEL, e.g. Qwen/Qwen2.5-1.5B}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL}}"
BRIDGE_VARIANT="${BRIDGE_VARIANT:-question_answer}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-0}"
SKIP_BRIDGE="${SKIP_BRIDGE:-0}"
PREPARE="${PREPARE:-1}"
SEED="${SEED:-42}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MMLU_SFT_SAMPLES="${MMLU_SFT_SAMPLES:-5000}"
DATA="${DATA:-${ROOT}/artifacts/data}"

case "$BRIDGE_VARIANT" in
  question_answer|problem_result) ;;
  *) echo "BRIDGE_VARIANT must be question_answer or problem_result" >&2; exit 2 ;;
esac
case "$CHAT_TEMPLATE" in
  0|1) ;;
  *) echo "CHAT_TEMPLATE must be 0 or 1" >&2; exit 2 ;;
esac
case "$SKIP_BRIDGE" in
  0|1) ;;
  *) echo "SKIP_BRIDGE must be 0 or 1" >&2; exit 2 ;;
esac

MODEL_SLUG="${MODEL//\//_}"
MODEL_SLUG="${MODEL_SLUG//:/_}"
RUN="${RUN:-${ROOT}/artifacts/reproduced_runs/${MODEL_SLUG}_mmlu${MMLU_SFT_SAMPLES}_${BRIDGE_VARIANT}_seed${SEED}}"

if [[ -e "$RUN" ]]; then
  echo "Refusing to overwrite existing run directory: $RUN" >&2
  exit 2
fi

if [[ "$PREPARE" == "1" ]]; then
  "$PYTHON_BIN" prepare_datasets.py \
    --datasets gsm8k,mmlu,mmlu_sft \
    --output_dir "$DATA" \
    --gsm8k_split train \
    --gsm8k_variants question_answer,problem_result \
    --mmlu_variants question_answer,problem_result \
    --mmlu_sft_samples "$MMLU_SFT_SAMPLES" \
    --mmlu_sft_seed "$SEED" \
    --mmlu_sft_variant "$BRIDGE_VARIANT"
fi

for file in \
  "$DATA/gsm8k_question_answer.jsonl" \
  "$DATA/gsm8k_problem_result.jsonl" \
  "$DATA/mmlu.jsonl" \
  "$DATA/mmlu_problem_result.jsonl"; do
  [[ -f "$file" ]] || { echo "Missing input: $file. Set PREPARE=1 to create it." >&2; exit 2; }
done

if [[ "$SKIP_BRIDGE" == "0" ]]; then
  BRIDGE_FILE="$DATA/mmlu_auxiliary_train_$BRIDGE_VARIANT"_n"$MMLU_SFT_SAMPLES"_seed"$SEED".jsonl
  [[ -f "$BRIDGE_FILE" ]] || { echo "Missing bridge data: $BRIDGE_FILE" >&2; exit 2; }
fi

if [[ "$CHAT_TEMPLATE" == "1" ]]; then
  PROMPT_ARGS=(--chat_template)
else
  PROMPT_ARGS=(--no-chat_template)
fi

TRAIN_ARGS=(
  --finetune_mode full
  --learning_rate "$LEARNING_RATE"
  --num_train_epochs "$EPOCHS"
  --per_device_train_batch_size "$BATCH_SIZE"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --max_length "$MAX_LENGTH"
  --seed "$SEED"
  --save_strategy no
  --no-bf16 --no-fp16
  --gradient_checkpointing
  "${PROMPT_ARGS[@]}"
)

train_full () {
  local train_file="$1"
  local initial_model="$2"
  local output_dir="$3"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" train_gsm8k.py \
    --train_file "$train_file" \
    --model_name "$initial_model" \
    --output_dir "$output_dir" \
    "${TRAIN_ARGS[@]}"
}

mkdir -p "$RUN"
if [[ "$SKIP_BRIDGE" == "1" ]]; then
  MMLU_BASE="$MODEL"
else
  MMLU_BASE="${RUN}/mmlu_aux${MMLU_SFT_SAMPLES}_${BRIDGE_VARIANT}"
  train_full "$BRIDGE_FILE" "$MODEL" "$MMLU_BASE"
fi

QA_FT="$RUN/gsm8k_question_answer"
PR_FT="$RUN/gsm8k_problem_result"
train_full "$DATA/gsm8k_question_answer.jsonl" "$MMLU_BASE" "$QA_FT"
train_full "$DATA/gsm8k_problem_result.jsonl" "$MMLU_BASE" "$PR_FT"

MMLU_QA="$RUN/mmlu_question_answer_predictions"
MMLU_PR="$RUN/mmlu_problem_result_predictions"
GSM_OUT="$RUN/gsm8k_test_predictions"
mkdir -p "$MMLU_QA" "$MMLU_PR" "$GSM_OUT"

eval_mmlu () {
  local eval_file="$1"
  local checkpoint="$2"
  local run_name="$3"
  local output_path="$4"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" evaluate_mmlu.py \
    --eval_file "$eval_file" \
    --model_name "$checkpoint" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --run_name "$run_name" \
    --output_path "$output_path" \
    --batch_size 32 \
    --max_new_tokens 32 \
    "${PROMPT_ARGS[@]}"
}

eval_mmlu "$DATA/mmlu.jsonl" "$MMLU_BASE" baseline "$MMLU_QA/baseline.jsonl"
eval_mmlu "$DATA/mmlu.jsonl" "$QA_FT" gsm8k_question_answer "$MMLU_QA/question_answer.jsonl"
eval_mmlu "$DATA/mmlu.jsonl" "$PR_FT" gsm8k_problem_result "$MMLU_QA/problem_result.jsonl"
eval_mmlu "$DATA/mmlu_problem_result.jsonl" "$MMLU_BASE" baseline "$MMLU_PR/baseline.jsonl"
eval_mmlu "$DATA/mmlu_problem_result.jsonl" "$QA_FT" gsm8k_question_answer "$MMLU_PR/question_answer.jsonl"
eval_mmlu "$DATA/mmlu_problem_result.jsonl" "$PR_FT" gsm8k_problem_result "$MMLU_PR/problem_result.jsonl"

"$PYTHON_BIN" analyze_format_changes.py \
  --baseline "$MMLU_QA/baseline.jsonl" \
  --comparisons "gsm8k_question_answer=$MMLU_QA/question_answer.jsonl" "gsm8k_problem_result=$MMLU_QA/problem_result.jsonl" \
  --output_json "$MMLU_QA/format_comparison.json" \
  --output_markdown "$MMLU_QA/format_comparison.md"

"$PYTHON_BIN" analyze_format_changes.py \
  --baseline "$MMLU_PR/baseline.jsonl" \
  --comparisons "gsm8k_question_answer=$MMLU_PR/question_answer.jsonl" "gsm8k_problem_result=$MMLU_PR/problem_result.jsonl" \
  --output_json "$MMLU_PR/format_comparison.json" \
  --output_markdown "$MMLU_PR/format_comparison.md"

eval_gsm () {
  local checkpoint="$1"
  local variant="$2"
  local output_path="$3"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" evaluate_gsm8k.py \
    --model_name "$checkpoint" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --format_variant "$variant" \
    --output_path "$output_path" \
    --batch_size 32 \
    --max_new_tokens 256 \
    "${PROMPT_ARGS[@]}"
}

eval_gsm "$MMLU_BASE" question_answer "$GSM_OUT/base_question_answer.jsonl"
eval_gsm "$MMLU_BASE" problem_result "$GSM_OUT/base_problem_result.jsonl"
eval_gsm "$QA_FT" question_answer "$GSM_OUT/question_answer_matching.jsonl"
eval_gsm "$PR_FT" problem_result "$GSM_OUT/problem_result_matching.jsonl"
eval_gsm "$PR_FT" question_answer "$GSM_OUT/problem_result_to_question_answer.jsonl"

printf 'Completed reproducible run: %s\n' "$RUN"
