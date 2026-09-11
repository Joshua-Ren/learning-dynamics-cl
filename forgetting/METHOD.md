# Method summary

## Behavioral erosion

An MMLU observation ends immediately before the first assistant response token
and asks for only one of `A`, `B`, `C`, or `D`. After GSM8K fine-tuning, task
accuracy and response behavior are measured separately:

- likelihood accuracy is the argmax among the four choice-token probabilities;
- extraction accuracy parses a choice from the generated response;
- non-IF records whether the complete stripped response is not exactly one
  choice letter;
- hash-tag behavior records whether the response contains `####`;
- choice mass is the first-token probability
  `P(A) + P(B) + P(C) + P(D)`.

The generative metrics use every example in the MMLU test split. Subject-level
rates are computed before correlations are evaluated across the 57 subjects.

## Sequence interaction score

For one supervised GSM8K update token `u` and one MMLU observation objective
`o`, let

```text
g = d log p(y | z) / dz = one_hot(y) - softmax(z)
```

be the output-space force, `W` the language-model readout matrix, and
`h_l` the RMS-normalized residual stream entering transformer block `l`.
The implementation computes

```text
gg       = <g_u, g_o>
gwwg     = <W^T g_u, W^T g_o>
hh_l     = <h_L,u, h_L,o>
hh_other = sum_{l < L} <h_l,u, h_l,o>
kembd    = token-multiset overlap between the update prefix and MMLU prompt

CH1 = gg * hh_l
CH2 = gwwg * (hh_other + kembd)
CH1+2 = CH1 + CH2
```

The main observation objective is the instruction-following mass

```text
log sum_{c in {A,B,C,D}} p(c | s_o).
```

Its closed-form gradient is used directly. Four individual choice objectives
are also retained as diagnostics. The exact one-token continuation IDs for
`A/B/C/D` are resolved after applying each model's chat template and are saved
with every run.

For the paper setting, each pair score is averaged over all supervised GSM8K
response tokens. Pair scores are then averaged over 100 shared GSM8K examples
and 25 sampled observations per MMLU subject. Hidden cosine and dot-product
similarities at the final layer are reported as representation baselines.

All quantities are obtained from forward passes. The implementation does not
backpropagate through model parameters and does not materialize full-parameter
Jacobians.
