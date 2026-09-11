# CH1 and CH2 Approximation

For an update token `u` and observation token `o`, Section 3 predicts the observation token's one-step log-probability change as

```text
approx(o, u) = lr * [ch1(o, u) + ch2(o, u)].
```

## Causal-LM alignment

For a supervised target at token position `pos`, all quantities are obtained from the context position `pos - 1`:

```text
logit_pos = hidden_pos = label_pos - 1.
```

The target is `y_t = labels[label_pos]`; masked labels (`-100`) are excluded.

## Per-token quantities

```text
g_t = one_hot(y_t) - softmax(logits_t)
h_t = hidden/residual representation at the selected context position
```

`g_t` is the gradient of `log p(y_t | s_t)` with respect to its logits.

## CH1

```text
ch1(o, u) = <g_o, g_u> * <h_o, h_u>.
```

This is the pure readout contribution when `h` is the representation directly entering the LM head. For models with tied input/output embeddings, a literal optimizer update of the shared tensor also has an input-embedding contribution; the functional untied-readout diagnostic separates that issue from the CH1 identity.

## Estimated CH2

Let `W` be the LM-head weight, and let `h_tilde_{s,t}` be the normalized input to selected attention/MLP projection streams at block/input stream `s`.

```text
readout_align(o, u) = <W^T g_o, W^T g_u>
layer_overlap(o, u) = sum_s <h_tilde_{s,o}, h_tilde_{s,u}>
ch2(o, u) = readout_align(o, u) * layer_overlap(o, u).
```

The implementation saves this forward-computable estimate as `ch2`; it is distinct from `ch2_exact_backbone`, which is an exact gradient diagnostic.

## Sign convention and saved fields

The validation update uses SGD on `loss = -logp_update`. Therefore

```text
first_order_exact = lr * <grad logp_observe, grad logp_update>
delta_logp = logp_observe_after - logp_observe_before.
```

Each pair CSV includes:

```text
ch1, ch2, approx_raw = ch1 + ch2, approx = lr * approx_raw,
ch1_scaled, ch2_scaled, first_order_exact, delta_logp.
```
