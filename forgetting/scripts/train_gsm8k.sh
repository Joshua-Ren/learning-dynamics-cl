#!/usr/bin/env bash

set -euo pipefail

# Run from the root of the accompanying LLaMA-Factory repository. This wrapper
# deliberately contains no environment installation or scheduler assumptions.
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
cd "$REPO_ROOT"

BASE_MODEL="${BASE_MODEL:?Set BASE_MODEL to a Hugging Face model or local checkpoint}"
TEMPLATE="${TEMPLATE:?Set TEMPLATE to the matching LLaMA-Factory chat template}"
METHOD="${METHOD:-sft}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR for the trained checkpoint}"

NUM_GPUS="${NUM_GPUS:-1}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1.0}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
CUTOFF_LEN="${CUTOFF_LEN:-2048}"
EAFT_ALPHA="${EAFT_ALPHA:-1.0}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-}"
DATASET_DIR="${DATASET_DIR:-$REPO_ROOT/forgetting/configs}"

if [[ "$METHOD" != "sft" && "$METHOD" != "eaft" ]]; then
  echo "METHOD must be sft or eaft, got: $METHOD" >&2
  exit 2
fi

denominator=$((NUM_GPUS * PER_DEVICE_BATCH_SIZE))
if (( denominator <= 0 || GLOBAL_BATCH_SIZE % denominator != 0 )); then
  echo "GLOBAL_BATCH_SIZE must be divisible by NUM_GPUS * PER_DEVICE_BATCH_SIZE" >&2
  exit 2
fi
GRADIENT_ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / denominator))

extra_args=()
if [[ "$METHOD" == "eaft" ]]; then
  extra_args+=(--use_eaft_loss true --eaft_alpha "$EAFT_ALPHA")
fi
if [[ -n "$DEEPSPEED_CONFIG" ]]; then
  extra_args+=(--deepspeed "$DEEPSPEED_CONFIG")
  export FORCE_TORCHRUN=1
fi
if (( NUM_GPUS > 1 )); then
  export FORCE_TORCHRUN=1
  export NPROC_PER_NODE="$NUM_GPUS"
fi

llamafactory-cli train \
  --model_name_or_path "$BASE_MODEL" \
  --trust_remote_code \
  --stage sft \
  --do_train \
  --finetuning_type full \
  --dataset_dir "$DATASET_DIR" \
  --dataset gsm8k_sft \
  --template "$TEMPLATE" \
  --cutoff_len "$CUTOFF_LEN" \
  --preprocessing_num_workers 8 \
  --dataloader_num_workers 4 \
  --output_dir "$OUTPUT_DIR" \
  --overwrite_output_dir true \
  --logging_steps 10 \
  --save_strategy epoch \
  --save_total_limit 1 \
  --save_only_model false \
  --report_to none \
  --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --learning_rate "$LEARNING_RATE" \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --bf16 \
  --gradient_checkpointing true \
  --ddp_timeout 180000000 \
  "${extra_args[@]}"
