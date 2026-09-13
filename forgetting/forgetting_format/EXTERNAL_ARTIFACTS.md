# Existing checkpoint archive

This experiment package is self-contained for a fresh reproduction, but it
does not copy the previously trained tensors. The original local archive is:

    /data/dwenlong/Forgetting_dynamic/forgetting_format/artifacts

At migration time it contained 57 model/optimizer files totalling
378,934,518,858 bytes. Those files are deliberately excluded so that this
repository does not acquire a second 379-GB copy. No symlink is used: deleting
or moving the old archive will not affect a fresh run of this package.

Key completed full-parameter Base bridge checkpoints in that archive:

| Protocol | Qwen2.5-1.5B | Llama-3.2-3B | Qwen3-4B |
| --- | --- | --- | --- |
| Q/A MMLU bridge | run_full_e1_mmlu_aux5000_question_answer_5000effective_qwen25_1p5b_base_fp32_len1024_lr1e5_bs4ga8 | run_full_e1_mmlu_aux5000_question_answer_5000effective_llama32_3b_base_fp32_len1024_lr1e5_bs4ga8_retry1 | run_full_e1_mmlu_aux5000_question_answer_5000effective_qwen3_4b_base_fp32_len1024_lr1e5_bs4ga8 |
| P/R MMLU bridge | run_full_e1_mmlu_aux5000_problem_result_5000effective_qwen25_1p5b_base_fp32_len1024_lr1e5_bs4ga8 | run_full_e1_mmlu_aux5000_problem_result_5000effective_llama32_3b_base_fp32_len1024_lr1e5_bs4ga8 | run_full_e1_mmlu_aux5000_problem_result_5000effective_qwen3_4b_base_fp32_len1024_lr1e5_bs4ga8 |

Each bridge has two continuations named
run_full_e1_from_mmlu_aux5000[_problem_result]_<model>_base_gsm8k_question_answer_...
and ..._gsm8k_problem_result_.... The completed table values are retained in
this package's README; generated summaries are intentionally not copied.

To evaluate an existing archived checkpoint without re-training, pass its
absolute path as --model_name to evaluate_mmlu.py or evaluate_gsm8k.py, and
pass its original Hugging Face model ID through --tokenizer_path.
