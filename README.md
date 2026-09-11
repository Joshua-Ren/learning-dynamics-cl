# Learning Dynamics of Continual Learning

> A unified view of data attribution, forgetting, and plasticity loss.

Paper coming soon | [Project page](https://joshua-ren.github.io/learning-dynamics-cl/) | [Code](.)

## TL;DR

Every time a model learns something new, it changes something else. This repository provides minimal implementations for the paper's token-level interaction analysis, forgetting experiments, and plasticity-loss experiments. The code exposes how an update helps or harms another behavior through two channels, then follows that interaction from one-step validation to accumulated erosion and long-horizon loss of learnability.

## One interaction, two channels, three tasks

For an updating token `u` and an observed token `o`, the paper approximates the change in observed log-probability as

```text
Delta_t(o, u) ~= eta (g_o^T g_u)(h_L,o^T h_L,u)
                + eta (g_o^T W W^T g_u) sum_l (h_l,o^T h_l,u) + c
```

- **CH1** is the direct, token-aligned interaction.
- **CH2** is the diffuse interaction mediated by the shared readout geometry.

| Paper theme | Question | Release coverage |
| --- | --- | --- |
| Selection / attribution (Sec. 4) | What should the model learn from? | Uses the interaction developed and validated in `interaction/`; the supplied release does not include the full Sec. 4 selection pipeline. |
| Interference / forgetting (Sec. 5) | What does an update break? | `interaction/` for collision/update energy; `forgetting/` for erosion. |
| Plasticity loss (Sec. 6) | Will future learning still work? | `plasticity/` |

## Repository structure

```text
interaction/   # Secs. 2-3, 5.1, App. B.2: CH1/CH2 validation and collision diagnostics
forgetting/    # Secs. 5.2-5.3: behavioral erosion and pre-finetuning prediction
plasticity/    # Sec. 6: readout transmission, long-horizon training, reset interventions
outputs/       # generated locally; ignored by Git
```

Each module has a focused README with its data assumptions, entry points, and outputs.

## Installation

Python 3.10 is the common version used by the supplied releases.

```bash
conda create -n learning-dynamics python=3.10
conda activate learning-dynamics
pip install -r requirements.txt
```

The forgetting training and model-loading paths additionally require the companion LLaMA-Factory fork described in [forgetting/README.md](forgetting/README.md). Install that fork in the same environment before running those paths. Hugging Face authentication, when required for a gated model, must be configured externally.

## Experiments

Run commands from the repository root.

### 1. Interaction validation

This covers the empirical validation in Sec. 3 (Figure 4); the same module also contains the SFT-vs-generation update-energy diagnostics for Sec. 5.1 (Figure 6) and layer-wise diagnostics from Appendix B.2.

```bash
python interaction/src/run_section3_validation.py \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --output_dir outputs/interaction/section3 \
  --allow_download \
  --save_csv
```

The runner writes its resolved configuration, pair-level measurements, summary metrics, and optional CSV artifacts. Downloads are disabled unless `--allow_download` is supplied. See [interaction/README.md](interaction/README.md) and use `--help` for the smaller hidden-state and one-step workflows.

### 2. Interference and forgetting

The cheapest correctness check validates the closed-form token-force and sequence-score calculations without loading a language model:

```bash
python forgetting/scripts/validate_sequence_score.py
```

The paper-scale workflow fine-tunes on GSM8K, evaluates generative MMLU behavior, computes pre-finetuning CH1/CH2 scores, and correlates those scores with subject-wise erosion:

```bash
python forgetting/scripts/run_sequence_score.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --template qwen \
  --num_mmlu_per_subject 25 \
  --num_gsm8k 100 \
  --gsm8k_candidate_pool_size 300 \
  --update_token_mode all_supervised \
  --output_dir outputs/forgetting/sequence-score
```

See [forgetting/README.md](forgetting/README.md) for training, evaluation, aggregation, and the required LLaMA-Factory integration.

### 3. Plasticity loss

A small SFT invocation checks the public Python entry point; it is not a reproduction of the long-horizon experiment:

```bash
PYTHONPATH=plasticity/src python -m plasticity_loss_sft.train_sft \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --dataset_limit 32 \
  --max_steps 1 \
  --output_dir outputs/plasticity/smoke \
  --no_bf16
```

The full Sec. 6 pipeline prepares probe sets, tracks the normalized readout-transmission score during sequential SFT or continued pretraining, and evaluates readout/reset interventions. See [plasticity/README.md](plasticity/README.md); every retained workflow has a direct Python entry point.

## Reproducing paper results

This is a minimal code release, not a one-command artifact bundle. Exact paper-scale runs require external Hugging Face datasets and checkpoints, gated-model access where applicable, substantial GPU resources, and the companion LLaMA-Factory fork for the forgetting experiments. Raw outputs, checkpoints, model caches, W&B histories, and paper figures are intentionally not committed.

The default commands write beneath `outputs/`. Use explicit path arguments for external datasets and checkpoints, and inspect each command with `--help` before launching a full run.

## Acknowledgements

Repository release engineering, integration, and documentation cleanup were completed with assistance from OpenAI Codex.

## Citation

The manuscript is currently a preprint and does not provide an arXiv identifier.

```bibtex
@misc{ren2026learningdynamicscl,
  title  = {Learning Dynamics of Continual Learning: A Unified View of Data Attribution, Forgetting, and Plasticity Loss},
  author = {Ren, Yi and Deng, Wenlong and Hong, Guanzhe and Lyle, Clare and Gal, Yarin},
  year   = {2026},
  note   = {Preprint}
}
```
