#!/usr/bin/env bash
# Run the exact paired GSM8K -> MMLU format-forgetting experiment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/artifacts/data}"
RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/artifacts/run}"
PREPARE="${PREPARE:-1}"
TRAIN="${TRAIN:-1}"
EVALUATE="${EVALUATE:-1}"
ANALYZE="${ANALYZE:-1}"

# Keep these settings identical between question_answer and problem_result.
SEED="${SEED:-42}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MMLU_BATCH_SIZE="${MMLU_BATCH_SIZE:-8}"
MMLU_MAX_NEW_TOKENS="${MMLU_MAX_NEW_TOKENS:-32}"

QA_ADAPTER="${QA_ADAPTER:-${RUN_DIR}/question_answer}"
RESULT_ADAPTER="${RESULT_ADAPTER:-${RUN_DIR}/problem_result}"
PREDICTION_DIR="${PREDICTION_DIR:-${RUN_DIR}/mmlu_predictions}"

if [[ "${PREPARE}" == "1" ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_datasets.py" \
    --datasets gsm8k,mmlu \
    --output_dir "${DATA_DIR}"
fi

if [[ "${TRAIN}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" "${SCRIPT_DIR}/train_gsm8k.py" \
    --train_file "${DATA_DIR}/gsm8k_question_answer.jsonl" \
    --model_name "${MODEL_NAME}" \
    --output_dir "${QA_ADAPTER}" \
    --seed "${SEED}" \
    --learning_rate "${LEARNING_RATE}" \
    --num_train_epochs "${EPOCHS}" \
    --per_device_train_batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --max_length "${MAX_LENGTH}"

  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" "${SCRIPT_DIR}/train_gsm8k.py" \
    --train_file "${DATA_DIR}/gsm8k_problem_result.jsonl" \
    --model_name "${MODEL_NAME}" \
    --output_dir "${RESULT_ADAPTER}" \
    --seed "${SEED}" \
    --learning_rate "${LEARNING_RATE}" \
    --num_train_epochs "${EPOCHS}" \
    --per_device_train_batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --max_length "${MAX_LENGTH}"
fi

if [[ "${EVALUATE}" == "1" ]]; then
  mkdir -p "${PREDICTION_DIR}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_mmlu.py" \
    --eval_file "${DATA_DIR}/mmlu.jsonl" \
    --model_name "${MODEL_NAME}" \
    --run_name baseline \
    --output_path "${PREDICTION_DIR}/baseline.jsonl" \
    --batch_size "${MMLU_BATCH_SIZE}" \
    --max_new_tokens "${MMLU_MAX_NEW_TOKENS}"

  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_mmlu.py" \
    --eval_file "${DATA_DIR}/mmlu.jsonl" \
    --model_name "${MODEL_NAME}" \
    --adapter_path "${QA_ADAPTER}" \
    --run_name question_answer \
    --output_path "${PREDICTION_DIR}/question_answer.jsonl" \
    --batch_size "${MMLU_BATCH_SIZE}" \
    --max_new_tokens "${MMLU_MAX_NEW_TOKENS}"

  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_mmlu.py" \
    --eval_file "${DATA_DIR}/mmlu.jsonl" \
    --model_name "${MODEL_NAME}" \
    --adapter_path "${RESULT_ADAPTER}" \
    --run_name problem_result \
    --output_path "${PREDICTION_DIR}/problem_result.jsonl" \
    --batch_size "${MMLU_BATCH_SIZE}" \
    --max_new_tokens "${MMLU_MAX_NEW_TOKENS}"
fi

if [[ "${ANALYZE}" == "1" ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_format_changes.py" \
    --baseline "${PREDICTION_DIR}/baseline.jsonl" \
    --comparisons \
      "question_answer=${PREDICTION_DIR}/question_answer.jsonl" \
      "problem_result=${PREDICTION_DIR}/problem_result.jsonl" \
    --output_json "${RUN_DIR}/format_comparison.json" \
    --output_markdown "${RUN_DIR}/format_comparison.md"
fi
