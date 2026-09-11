# Reproducibility note

The completed sequence-score runs underlying the reported subject-level
analysis sampled 100 GSM8K examples, with seed 42, from a deterministic
candidate pool consisting of the first 300 rows of the GSM8K training split.
Their saved indices range from 3 to 294 and are shared across the reported
model comparisons.

To reproduce that historical sampling exactly, use:

```text
--num_gsm8k 100 --gsm8k_candidate_pool_size 300 --seed 42
```

Omitting `--gsm8k_candidate_pool_size` instead samples from the complete GSM8K
training split, which is the cleaner interpretation of "100 randomly sampled
GSM8K training examples" but will produce a different sample and may change
the correlations. This choice should be kept consistent between the manuscript
description, released command, and reported table.

The MMLU side samples 25 examples independently within every subject with the
same seed. All models must reuse the saved GSM8K and MMLU index files for a
strict cross-model comparison.
