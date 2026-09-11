# EAFT baseline in the parent repository

EAFT is already integrated into the accompanying LLaMA-Factory fork. The
release wrapper enables it with:

```text
--use_eaft_loss true --eaft_alpha 1.0
```

The relevant parent-repository locations are:

- `src/llamafactory/hparams/finetuning_args.py`: defines `use_eaft_loss` and
  `eaft_alpha`;
- `src/llamafactory/train/sft/trainer.py`: selects the EAFT loss for SFT;
- `src/llamafactory/train/trainer_utils.py`: implements the token weighting.

For each supervised token, the implementation computes ordinary cross entropy
and estimates predictive entropy from the model's top-20 logits. The top-20
logits are renormalized, their entropy is divided by `3.0`, and the resulting
quantity is raised to `alpha`. The detached weight multiplies the token's
cross-entropy loss:

```text
w_t = (H(top20(softmax(z_t))) / 3.0)^alpha
L_EAFT = mean_t w_t * CE(z_t, y_t)
```

Only supervised response tokens participate in the reduction. With distributed
training, LLaMA-Factory's normal token-count normalization is retained. The
Chapter 5 runs use `alpha = 1.0` and full-parameter fine-tuning.
