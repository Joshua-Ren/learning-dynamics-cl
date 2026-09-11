# Interaction decomposition and validation

This module contains the code for the structured token-level update--behavior interaction developed in Sections 2--3, the empirical validation in Figure 4, the update-energy diagnostics in Section 5.1 (Figure 6), and the additional diagnostics in Appendix B.2. It is the shared analytical foundation used by the paper's later selection, forgetting, and plasticity arguments; it is not the complete Section 4 data-selection pipeline.

## Main entry points

| Entry point | Purpose |
| --- | --- |
| `src/run_section3_validation.py` | Compare CH1+CH2 with exact first-order and actual one-step changes. |
| `src/run_hh_analysis.py` | Measure hidden-state pair geometry across layers. |
| `src/run_one_step_forgetting.py` | Measure log-probability changes after one supervised-token update. |
| `src/get_logits_sft.py` | Extract teacher-forced SFT token logits. |
| `src/get_logits_grpo.py` | Extract greedy or sampled generation-time logits. |
| `src/diagnose_per_layer_ch2.py` | Inspect per-layer and cumulative CH2 terms. |
| `src/diagnose_last_block_ch2.py` | Run the final-block factorization ladder. |
| `src/run_single_block_update_sweep.py` | Restrict actual one-step updates to individual transformer blocks. |

See `docs/ch1_ch2.md` for definitions and sign conventions and `docs/appendix_b2.md` for the diagnostic scope.

## Run

Install the top-level requirements and run from the repository root:

```bash
python interaction/src/run_section3_validation.py \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --output_dir outputs/interaction/section3 \
  --allow_download \
  --save_csv
```

Without `--allow_download`, runners use only locally cached Hugging Face assets. Defaults intentionally use a very small number of samples and token pairs, but the exact cost still depends on model size and `--param_scope`.

A smaller hidden-state workflow is:

```bash
python interaction/src/run_hh_analysis.py \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --output_dir outputs/interaction/hidden-geometry
```

Pass `--allow_download` if the selected runner exposes it and local assets are unavailable. Use `python interaction/src/<entry>.py --help` for all options.

## Outputs and notebooks

Runners write resolved configuration, diagnostics, pair-level rows, and summaries beneath the requested output directory. Raw artifacts are intentionally not versioned. The notebooks under `notebooks/` expect runner-produced artifacts and write figures locally.

For a target at sequence position `pos`, causal-LM scoring uses logits at `pos - 1`. Section 3 defaults to FP32 and plain SGD for its one-step validation update.
