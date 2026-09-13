# GSM8K → MMLU answer-format forgetting

> Lean reproducible package: code and the exact training/evaluation protocol are retained here.
> By default, the reproduction wrapper materializes GSM8K and MMLU inputs before training.
> The 379-GB model archive and historical raw outputs are not duplicated; see EXTERNAL_ARTIFACTS.md for the local checkpoint inventory.
> Completed numerical results remain documented in this README.

Use scripts/reproduce_full_parameter.sh with MODEL and GPU set to reproduce the MMLU bridge, both GSM8K continuations,
both MMLU prompts, format analysis, and GSM8K evaluation.

This experiment tests whether the GSM8K prompt marker that is closest to MMLU's `Answer:` marker causes more MMLU answer-format migration.

It materializes two **paired** GSM8K training files:

| Condition | Prompt |
| --- | --- |
| `question_answer` | `\nQuestion: {question}\n\nAnswer:` |
| `problem_result` | `\nProblem: {question}\n\nresult:` |

The GSM8K completion is unchanged in both conditions, including its `####` final-answer convention. The data manifest stores a SHA-256 hash over `example_id + completion`; the two hashes must be identical. Both files have the same `source_row` order, so use the same seed and training hyperparameters.

MMLU has paired `Question`/`Answer` and `Problem`/`result` evaluation prompts; both ask for only `A`, `B`, `C`, or `D`.

## Completed full-parameter study

### Study design

The completed results below intentionally separate two model families. Do not compare their absolute levels as if they had the same starting checkpoint or rendering path.

| Family | Starting checkpoints | Intermediate MMLU SFT | Prompt rendering |
| --- | --- | --- | --- |
| Direct Instruct | Qwen2.5-1.5B-Instruct, Llama-3.2-3B-Instruct, Qwen3-4B-Instruct-2507 | None; GSM8K is applied directly | Native chat template |
| Base with MMLU bridge | Qwen2.5-1.5B, Llama-3.2-3B, Qwen3-4B-Base | 5,000 random MMLU auxiliary_train examples in Question/Answer (original suite) or Problem/result (reverse suite), then GSM8K | Raw prompts; no chat template |

All completed training runs are full-parameter, one epoch, FP32, learning rate 1e-5, batch size 4, gradient accumulation 8, maximum length 1024, and seed 42. GSM8K has 7,473 training examples; MMLU test has 14,042 examples; GSM8K test has 1,319 examples.


All Base-with-MMLU-bridge runs additionally use gradient checkpointing (`--gradient_checkpointing`); it is required for the FP32 3B/4B runs on the available 96-GB GPUs. It changes the activation-memory tradeoff only, not the full-parameter optimization or effective batch size.
For GSM8K, only the labels in the paired prompt change:

| Condition | Exact prompt |
| --- | --- |
| question_answer | leading newline, then Question: question, blank line, Answer: |
| problem_result | leading newline, then Problem: question, blank line, result: |

The completion, including GSM8K's final #### number convention, is byte-identical across the two conditions.

### Results

#### Direct Instruct to GSM8K: MMLU answer-format retention

The following is canonical MMLU format ratio: the generated answer is exactly a single A/B/C/D letter. Each cell is Question/Answer MMLU prompt / Problem/result MMLU prompt. It is a format metric, not MMLU content accuracy.

| Model | Instruct base | GSM8K Q/A FT | GSM8K P/R FT |
| --- | ---: | ---: | ---: |
| Qwen2.5-1.5B-Instruct | 75.27% / 96.14% | 73.20% / 72.52% | 52.63% / 31.06% |
| Llama-3.2-3B-Instruct | 92.34% / 97.02% | 47.59% / 44.85% | 44.10% / 39.87% |
| Qwen3-4B-Instruct | 85.89% / 76.93% | 0.00% / 0.05% | 32.12% / 0.00% |

Direct full GSM8K SFT causes substantial MMLU format migration for all three Instruct families. The intervention is not uniformly mitigated by Problem/result: Qwen3-Instruct remains highly sensitive, while Qwen2.5 and Llama degrade under both GSM8K templates.

The completed direct-Instruct record above is the MMLU format study; GSM8K numeric test accuracy was collected only for the matched Base-with-MMLU-bridge suite below, so unreported Instruct GSM8K content values are not zeros.

#### Base with MMLU bridge: MMLU canonical format ratio

These models first receive the same 5,000-example MMLU Question/Answer bridge. Therefore their base MMLU responses are nearly all canonical letters under both MMLU prompts. Cells again mean Q/A MMLU prompt / P/R MMLU prompt.

| Model | MMLU-bridge base | GSM8K Q/A FT | GSM8K P/R FT |
| --- | ---: | ---: | ---: |
| Qwen2.5-1.5B-Base | 100.00% / 99.85% | 97.35% / 92.47% | 100.00% / 93.13% |
| Llama-3.2-3B-Base | 99.98% / 99.94% | 44.77% / 87.91% | 99.53% / 46.51% |
| Qwen3-4B-Base | 100.00% / 99.97% | 68.64% / 95.01% | 99.99% / 13.30% |

The off-diagonal drops are informative: Llama Q/A FT loses format mainly under Q/A MMLU evaluation, whereas P/R FT loses format mainly under P/R evaluation. Qwen3 P/R FT is especially sensitive to the P/R MMLU cue.

#### Base with MMLU bridge: MMLU content test accuracy

MMLU content is relaxed A/B/C/D accuracy; an answer can receive content credit even if it is noncanonical. Each cell is Q/A MMLU prompt / P/R MMLU prompt.

| Model | MMLU-bridge base | GSM8K Q/A FT | GSM8K P/R FT |
| --- | ---: | ---: | ---: |
| Qwen2.5-1.5B-Base | 59.31% / 58.41% | 58.08% / 54.89% | 58.89% / 55.73% |
| Llama-3.2-3B-Base | 55.28% / 53.95% | 40.03% / 49.58% | 53.81% / 42.44% |
| Qwen3-4B-Base | 71.04% / 70.74% | 53.67% / 68.15% | 70.65% / 23.81% |

Thus answer-format migration is not the sole effect: content accuracy can also fall sharply. The strongest instance is Qwen3 P/R FT evaluated with the P/R MMLU prompt, 70.74% to 23.81%.

#### Base with MMLU bridge: GSM8K test accuracy

Each cell is numeric GSM8K accuracy; the value in brackets is strict accuracy requiring a correct #### final answer. Base models are evaluated under both prompts. Q/A FT uses its matching Q/A prompt; P/R FT is reported both with its matching P/R prompt and with the Q/A cross-template prompt.

| Model | Bridge base Q/A / P/R | Q/A FT on Q/A | P/R FT on P/R | P/R FT on Q/A |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-1.5B-Base | 8.42% [0.08%] / 9.25% [0.00%] | 41.24% [41.24%] | 41.77% [41.62%] | 39.73% [39.27%] |
| Llama-3.2-3B-Base | 5.69% [0.00%] / 5.53% [0.00%] | 34.95% [34.72%] | 35.78% [35.78%] | 33.43% [33.28%] |
| Qwen3-4B-Base | 22.37% [0.00%] / 22.29% [0.00%] | 68.16% [67.85%] | 67.10% [66.87%] | 68.01% [67.63%] |

The MMLU-bridge bases almost never emit ####. After GSM8K SFT, #### appears in 89.6% to 93.4% of generations. P/R to Q/A prompt transfer lowers numeric GSM8K accuracy by 2.50 points for Qwen2.5 and 2.35 points for Llama, but changes Qwen3 by +0.90 points. This shows a template effect, but not a universal collapse of GSM8K answer formatting.

#### Reverse P/R MMLU bridge: complete full-parameter study

This is the requested reverse control: each Base model first receives the same deterministic 5,000-example MMLU sample, rendered as `Problem: ...` followed by `result:`, before both paired GSM8K continuations. The sample indices, seed, one epoch, FP32 full-parameter optimizer settings, and raw-prompt rendering are otherwise unchanged. Cells in the first two tables are Q/A MMLU prompt / P/R MMLU prompt.

| Model | P/R-bridge base format | GSM8K Q/A FT | GSM8K P/R FT |
| --- | ---: | ---: | ---: |
| Qwen2.5-1.5B-Base | 100.00% / 99.99% | 90.14% / 99.01% | 90.41% / 85.98% |
| Llama-3.2-3B-Base | 99.99% / 99.99% | 70.89% / 99.34% | 89.31% / 82.35% |
| Qwen3-4B-Base | 100.00% / 100.00% | 71.72% / 96.08% | 91.23% / 88.35% |

| Model | P/R-bridge base content | GSM8K Q/A FT | GSM8K P/R FT |
| --- | ---: | ---: | ---: |
| Qwen2.5-1.5B-Base | 59.76% / 59.76% | 56.97% / 59.04% | 56.87% / 55.91% |
| Llama-3.2-3B-Base | 55.52% / 55.32% | 44.97% / 52.79% | 50.19% / 48.19% |
| Qwen3-4B-Base | 71.36% / 71.15% | 58.67% / 68.93% | 66.78% / 65.81% |

The reverse control migrates the strongest format-forgetting effect with the MMLU marker. The table below is canonical-to-noncanonical (equivalently here, format-changed) ratio relative to the P/R-bridge base:

| Model | Q/A MMLU: Q/A FT / P/R FT | P/R MMLU: Q/A FT / P/R FT |
| --- | ---: | ---: |
| Qwen2.5-1.5B-Base | 9.86% / 9.59% | 0.98% / 14.03% |
| Llama-3.2-3B-Base | 29.10% / 10.68% | 0.65% / 17.65% |
| Qwen3-4B-Base | 28.28% / 8.77% | 3.92% / 11.65% |

For all three Base models, Q/A GSM8K tuning produces its largest format migration on Q/A MMLU, while P/R GSM8K tuning produces its largest migration on P/R MMLU. This demonstrates cue-specific format forgetting rather than a fixed preference for one template.

| Model | P/R-bridge GSM base Q/A / P/R | Q/A FT on Q/A | P/R FT on P/R | P/R FT on Q/A |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-1.5B-Base | 12.66% [0.00%] / 11.52% [0.00%] | 58.38% [58.15%] | 58.61% [58.45%] | 56.86% [56.71%] |
| Llama-3.2-3B-Base | 5.46% [0.00%] / 6.14% [0.00%] | 46.25% [46.10%] | 45.87% [45.72%] | 45.79% [45.56%] |
| Qwen3-4B-Base | 31.08% [0.00%] / 32.15% [0.00%] | 79.30% [79.00%] | 79.98% [79.83%] | 79.61% [79.38%] |


### Exact CH1 attribution at the MMLU answer token

To test whether the template effect comes from the hidden state or from the prediction-error term, we evaluate the **original non-approximate CH1/RH score** on the Qwen2.5-1.5B MMLU-bridge checkpoint. We use 50 paired MMLU test examples and 50 paired GSM8K training examples (seed 42). On the MMLU side, only the hidden state and error that predict the gold option immediately after `Answer:` or `result:` are retained. GSM8K retains all teacher-forced positions.

The score is the original vocabulary-aligned representation score, not the `h_dot × error_dot` approximation:

\[
R(v)=\sum_t\left(\mathbb{1}[y_t=v]-p_t(v)\right)h_t,
\qquad
S_{\mathrm{CH1}}(i,j)=\sum_{v\in V_i\cap V_j}R_i(v)^\top R_j(v).
\]

For each GSM8K source example, the Q/A and P/R token sequences have the same retained length. This permits positional counterfactual recombination of GSM hidden states `H` and prediction errors `E`: `S(H_QA, E_QA)`, `S(H_QA, E_PR)`, `S(H_PR, E_QA)`, and `S(H_PR, E_PR)`. We report the symmetric Shapley attribution, which exactly satisfies `ΔS = ΔS_H + ΔS_E` and does not depend on whether `H` or `E` is switched first.

![Exact CH1 hidden-state and prediction-error attribution](artifacts/score_analysis_qwen25_base_mmlu5000_exact_ch1_component_attribution_answer_first_n50_seed42/exact_ch1_h_error_attribution.png)

The four exact-score counterfactuals make the same result explicit: replacing only the GSM prediction-error term nearly reaches the P/R score, while replacing only the GSM hidden states remains near the Q/A baseline.

![Exact CH1 counterfactual path](artifacts/score_analysis_qwen25_base_mmlu5000_exact_ch1_component_attribution_answer_first_n50_seed42/exact_ch1_counterfactual_path.png)

The table reports GSM P/R minus GSM Q/A. A positive value is less negative CH1 and therefore predicts less first-order reduction of the MMLU gold-answer likelihood (a more protective update).

| MMLU prompt | Exact CH1 change | Hidden-state contribution | Prediction-error contribution |
| --- | ---: | ---: | ---: |
| Q/A (primary) | +14.87 [-56.39, 82.56] | -0.43 [-29.55, 24.40] | +15.30 [-49.07, 80.05] |
| P/R (control) | +69.29 [-24.27, 169.65] | -0.17 [-20.64, 19.91] | +69.46 [-18.55, 169.02] |

For the primary Q/A condition, nearly all of the point-estimate protection is due to the prediction-error contribution; the net hidden-state contribution is close to zero and slightly negative. Thus this exact CH1 diagnostic does **not** support the claim that P/R protects MMLU answer likelihood by lowering hidden-state similarity. However, every 95% cluster-bootstrap interval crosses zero at 50×50 samples, so this is directional evidence rather than a statistically resolved effect. It is also a local, teacher-forced, first-order likelihood proxy; it does not by itself establish a change in generated MMLU format accuracy.

Reproduce the exact score attribution and figure with:

~~~bash
CUDA_VISIBLE_DEVICES=0 python score_analysis/analyze_exact_ch1_component_attribution.py \
  --model_name /path/to/mmlu_bridge_checkpoint \
  --tokenizer_name Qwen/Qwen2.5-1.5B \
  --data_dir artifacts/data \
  --output_dir artifacts/exact_ch1_answer_first \
  --num_mmlu 50 --num_gsm 50 --seed 42 \
  --max_length 1024 --prediction_topk 32 \
  --lowest_likelihood_ratio 1.0 --bootstrap_samples 2000

python score_analysis/plot_exact_ch1_component_attribution.py \
  --input_json artifacts/exact_ch1_answer_first/summary.json
~~~

### Reproduce the completed full-parameter protocol

Run from the repository directory. PYTHON_BIN must provide torch, transformers, and datasets. For local checkpoints whose saved tokenizer metadata is newer than the active transformers version, always pass tokenizer_path as the original Hugging Face model name during evaluation.

~~~bash
cd forgetting/forgetting_format  # from the learning-dynamics-cl repository root
PYTHON_BIN=python
GPU=0
DATA=$PWD/artifacts/data
RUN=$PWD/artifacts/reproduce_full

# Choose one family.
# Base family: raw prompts.
MODEL=Qwen/Qwen2.5-1.5B
TRAIN_MMLU_TEMPLATE_ARGS=--no-chat_template
MMLU_BRIDGE_VARIANT=question_answer  # use problem_result for the reverse bridge
GSM_TEMPLATE_ARGS=--no-chat_template

# Instruct family example: native chat template.
# MODEL=meta-llama/Llama-3.2-3B-Instruct
# TRAIN_MMLU_TEMPLATE_ARGS=
# GSM_TEMPLATE_ARGS=--chat_template
~~~

Prepare the paired GSM8K files, both MMLU test prompts, and the deterministic 5,000-example bridge dataset:

~~~bash
$PYTHON_BIN prepare_datasets.py \
  --datasets gsm8k,mmlu,mmlu_sft \
  --output_dir $DATA \
  --gsm8k_split train \
  --gsm8k_variants question_answer,problem_result \
  --mmlu_variants question_answer,problem_result \
  --mmlu_sft_samples 5000 \
  --mmlu_sft_seed 42 \
  --mmlu_sft_variant $MMLU_BRIDGE_VARIANT
~~~

For the Base-with-bridge family, train the MMLU bridge first. For the direct-Instruct family, skip this step and set MMLU_BASE to MODEL.

~~~bash
MMLU_BASE=$RUN/mmlu_aux5000_$MMLU_BRIDGE_VARIANT
CUDA_VISIBLE_DEVICES=$GPU $PYTHON_BIN train_gsm8k.py \
  --train_file $DATA/mmlu_auxiliary_train_${MMLU_BRIDGE_VARIANT}_n5000_seed42.jsonl \
  --model_name $MODEL \
  --output_dir $MMLU_BASE \
  --finetune_mode full \
  --learning_rate 1e-5 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --max_length 1024 \
  --seed 42 \
  --save_strategy no \
  --no-bf16 --no-fp16 \
  --gradient_checkpointing \
  $TRAIN_MMLU_TEMPLATE_ARGS

# Direct-Instruct variant:
# MMLU_BASE=$MODEL
~~~

Train the paired GSM8K continuations from the exact same MMLU_BASE checkpoint:

~~~bash
QA_FT=$RUN/gsm8k_question_answer
PR_FT=$RUN/gsm8k_problem_result

CUDA_VISIBLE_DEVICES=$GPU $PYTHON_BIN train_gsm8k.py \
  --train_file $DATA/gsm8k_question_answer.jsonl \
  --model_name $MMLU_BASE \
  --output_dir $QA_FT \
  --finetune_mode full \
  --learning_rate 1e-5 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --max_length 1024 \
  --seed 42 \
  --save_strategy no \
  --no-bf16 --no-fp16 \
  --gradient_checkpointing \
  $TRAIN_MMLU_TEMPLATE_ARGS

CUDA_VISIBLE_DEVICES=$GPU $PYTHON_BIN train_gsm8k.py \
  --train_file $DATA/gsm8k_problem_result.jsonl \
  --model_name $MMLU_BASE \
  --output_dir $PR_FT \
  --finetune_mode full \
  --learning_rate 1e-5 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --max_length 1024 \
  --seed 42 \
  --save_strategy no \
  --no-bf16 --no-fp16 \
  --gradient_checkpointing \
  $TRAIN_MMLU_TEMPLATE_ARGS
~~~

Evaluate all three checkpoints on both MMLU prompt templates. The helper retains raw predictions, which are required for format analysis.

~~~bash
MMLU_QA=$RUN/mmlu_question_answer_predictions
MMLU_PR=$RUN/mmlu_problem_result_predictions
mkdir -p $MMLU_QA $MMLU_PR

eval_mmlu () {
  EVAL_FILE=$1
  CHECKPOINT=$2
  RUN_NAME=$3
  OUT_FILE=$4
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON_BIN evaluate_mmlu.py \
    --eval_file $EVAL_FILE \
    --model_name $CHECKPOINT \
    --tokenizer_path $MODEL \
    --run_name $RUN_NAME \
    --output_path $OUT_FILE \
    --batch_size 32 \
    --max_new_tokens 32 \
    $TRAIN_MMLU_TEMPLATE_ARGS
}

eval_mmlu $DATA/mmlu.jsonl $MMLU_BASE baseline $MMLU_QA/baseline.jsonl
eval_mmlu $DATA/mmlu.jsonl $QA_FT gsm8k_question_answer $MMLU_QA/question_answer.jsonl
eval_mmlu $DATA/mmlu.jsonl $PR_FT gsm8k_problem_result $MMLU_QA/problem_result.jsonl
eval_mmlu $DATA/mmlu_problem_result.jsonl $MMLU_BASE baseline $MMLU_PR/baseline.jsonl
eval_mmlu $DATA/mmlu_problem_result.jsonl $QA_FT gsm8k_question_answer $MMLU_PR/question_answer.jsonl
eval_mmlu $DATA/mmlu_problem_result.jsonl $PR_FT gsm8k_problem_result $MMLU_PR/problem_result.jsonl

$PYTHON_BIN analyze_format_changes.py \
  --baseline $MMLU_QA/baseline.jsonl \
  --comparisons gsm8k_question_answer=$MMLU_QA/question_answer.jsonl gsm8k_problem_result=$MMLU_QA/problem_result.jsonl \
  --output_json $MMLU_QA/format_comparison.json \
  --output_markdown $MMLU_QA/format_comparison.md
~~~

Evaluate GSM8K test. The final call is the requested cross-template test: P/R-trained weights with the Q/A prompt.

~~~bash
eval_gsm () {
  CHECKPOINT=$1
  VARIANT=$2
  OUT_FILE=$3
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON_BIN evaluate_gsm8k.py \
    --model_name $CHECKPOINT \
    --tokenizer_path $MODEL \
    --format_variant $VARIANT \
    --output_path $OUT_FILE \
    --batch_size 32 \
    --max_new_tokens 256 \
    $GSM_TEMPLATE_ARGS
}

mkdir -p $RUN/gsm8k_test_predictions
eval_gsm $MMLU_BASE question_answer $RUN/gsm8k_test_predictions/base_question_answer.jsonl
eval_gsm $MMLU_BASE problem_result $RUN/gsm8k_test_predictions/base_problem_result.jsonl
eval_gsm $QA_FT question_answer $RUN/gsm8k_test_predictions/question_answer_matching.jsonl
eval_gsm $PR_FT problem_result $RUN/gsm8k_test_predictions/problem_result_matching.jsonl
eval_gsm $PR_FT question_answer $RUN/gsm8k_test_predictions/problem_result_to_question_answer.jsonl
~~~

evaluate_gsm8k.py writes one JSONL with every response and an adjacent summary JSON. Its numeric metric gives priority to a #### final answer when present; otherwise it uses the final generated number. Strict hash accuracy always requires a correct #### final answer.


## Legacy LoRA quick run

From `learning-dynamics-cl/forgetting/forgetting_format`:

```bash
GPU=0 MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct bash run_format_comparison.sh
```

The script runs, sequentially:

1. `prepare_datasets.py`: writes `artifacts/data/gsm8k_question_answer.jsonl`, `gsm8k_problem_result.jsonl`, `mmlu.jsonl`, and `gsm8k_format_manifest.json`.
2. `train_gsm8k.py`: starts each condition from the same base model with the same LoRA/SFT settings.
3. `evaluate_mmlu.py`: greedily generates MMLU answers for base, `question_answer`, and `problem_result`.
4. `analyze_format_changes.py`: aligns all runs by `example_id` and writes `artifacts/run/format_comparison.json` and `format_comparison.md`.

To reuse already prepared data and adapters:

```bash
PREPARE=0 TRAIN=0 EVALUATE=1 ANALYZE=1 \
  GPU=0 MODEL_NAME=/path/to/base_model \
  QA_ADAPTER=/path/to/question_answer_adapter \
  RESULT_ADAPTER=/path/to/problem_result_adapter \
  bash run_format_comparison.sh
```

The default training settings can be overridden with `SEED`, `LEARNING_RATE`, `EPOCHS`, `BATCH_SIZE`, `GRADIENT_ACCUMULATION_STEPS`, and `MAX_LENGTH`. Keep them exactly equal between the two conditions.

## Metrics

For every MMLU generation, the evaluator saves the raw response and classifies it as `letter_only`, `answer_label`, `result_label`, `gsm8k_hash` (contains `###`), `verbal_answer`, `other`, or `empty`.

- **Relaxed accuracy**: extracts an A/B/C/D choice from the response. This is closer to content/knowledge accuracy.
- **Strict accuracy**: correct only if the response is also exactly in the required letter-only MMLU format.
- **Format error ratio**: fraction that is not letter-only.
- **Correct-but-noncanonical ratio**: the model selected the correct letter but violated the requested output format.
- **Answer-format changed ratio**: fraction of MMLU examples whose format category changed versus the base model.
- **Canonical → noncanonical ratio**: examples that were letter-only in the base run but become noncanonical after GSM8K tuning.
- **Format-mediated share of baseline strict losses**: among base strict-correct examples that stop being strict-correct, the fraction whose extracted answer remains correct but whose format is no longer canonical.

A format-forgetting explanation is supported when strict MMLU accuracy drops much more than relaxed accuracy, while the correct-but-noncanonical and canonical-to-noncanonical ratios increase. If `problem_result` reduces these values relative to `question_answer`, that is evidence for the proposed context-marker similarity mechanism rather than purely a loss of multiple-choice knowledge.
