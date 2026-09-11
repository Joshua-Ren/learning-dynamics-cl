# Experiments represented by this release

## Main Chapter 5 experiments

The released workflow supports the experiments used for behavioral erosion and
its pre-finetuning prediction:

1. Full-parameter GSM8K SFT for one epoch.
2. Full-parameter GSM8K EAFT for one epoch with `alpha = 1.0`.
3. Greedy generation on the complete MMLU test split for base, SFT, and EAFT
   checkpoints.
4. MMLU likelihood accuracy, extraction accuracy, exact response-format
   compliance, `####` transfer, and first-token `P(A/B/C/D)`.
5. Subject-wise instruction-following and hash-tag rates over all MMLU test
   examples.
6. Pre-finetuning GSM8K-to-MMLU CH1, CH2, and CH1+2 scores using 100 shared
   GSM8K examples and 25 MMLU examples per subject.
7. Final-layer hidden cosine and dot-product representation baselines.
8. Pearson and Spearman correlations, including p-values, across the 57 MMLU
   subjects.

The model/template combinations used in the study include Qwen2.5-1.5B,
Qwen3-4B, Qwen2.5-7B, and Llama3.2-3B instruction-tuned checkpoints.

## Retained diagnostic variants

The sequence-score runner also retains variants used during analysis:

- all supervised GSM8K response tokens (the main setting);
- only the first response token;
- only the first token after a final-answer marker such as `####`;
- sampled MMLU observations;
- a subject-mean observation formed by averaging forward-pass features over
  every MMLU example in a subject;
- sum, mean, and mean-absolute pair-score aggregations.

These options are useful for ablations, but they should not be presented as the
default Chapter 5 protocol unless explicitly reported.

## Deliberately excluded

The working repository also contains exploratory IFEval and Dolly evaluation,
spreadsheet-generation utilities, cluster submission scripts, intermediate
snapshots, and model-specific debugging code. They are not required to
reproduce the Chapter 5 MMLU results and are therefore omitted from this
minimal release. No generated model outputs or checkpoints are included.
