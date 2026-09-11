# Plasticity-loss experiments

This module contains the Section 6 implementation for task-conditioned readout transmission, sequential SFT, long-horizon continued pretraining (CPT), and readout/reset interventions. Datasets, model weights, checkpoints, W&B histories, outputs, and figures are intentionally excluded.

## Setup

Install the top-level requirements, then expose the source package:

```bash
export PYTHONPATH=plasticity/src
```

All examples below run from the repository root. Hugging Face credentials and optional W&B configuration must be supplied externally.

## Main workflows

### Prepare downstream tasks and probes

```bash
python -m plasticity_loss_sft.prepare_subsets \
  --output_dir data/prepared_subsets \
  --train_size 1000 \
  --probe_size 100
```

This writes deterministic GSM8K, MBPP, and Dolly QA train/probe subsets plus selection manifests.

### Small entry-point check

```bash
python -m plasticity_loss_sft.train_sft \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --dataset_limit 32 \
  --max_steps 1 \
  --output_dir outputs/plasticity/smoke \
  --no_bf16
```

This is a smoke-scale baseline, not the paper experiment.

### Sequential SFT and readout transmission

```bash
python -m plasticity_loss_sft.train_sequential_sft \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --task_sequence \
    gsm8k=data/prepared_subsets/gsm8k/train.jsonl \
    mbpp=data/prepared_subsets/mbpp/train.jsonl \
    dolly_qa=data/prepared_subsets/dolly_qa/train.jsonl \
  --probe_eval_specs \
    gsm8k=data/prepared_subsets/gsm8k/probe.jsonl \
    mbpp=data/prepared_subsets/mbpp/probe.jsonl \
    dolly_qa=data/prepared_subsets/dolly_qa/probe.jsonl \
  --output_dir outputs/plasticity/sequential-sft
```

Paper-scale defaults are intentionally long-running. Inspect and override epochs, task rounds, checkpointing, and probe frequency before launching.

### Continued pretraining

```bash
python -m plasticity_loss_sft.train_cpt_with_plasticity \
  --cpt_data_path /path/to/tokenized_cpt_jsonl_or_directory \
  --eval_specs \
    gsm8k=data/prepared_subsets/gsm8k/probe.jsonl \
    mbpp=data/prepared_subsets/mbpp/probe.jsonl \
    dolly_qa=data/prepared_subsets/dolly_qa/probe.jsonl \
  --output_dir outputs/plasticity/cpt
```

Checkpoint syncing is optional. If used, provide both `--checkpoint_sync_dir` and `--checkpoint_sync_host`; no cluster hostname or storage path is built in.

### Geometry and interventions

- `run_readout_geometry.py` and `compare_readout_geometry*.py`: compute and compare readout spectra.
- `sft_from_cpt_checkpoints.py`: discover CPT trajectories and run downstream checkpoint sweeps.
- `readout_transmission.py`: shared transmission-score calculations.
- `prepare_cpt_data.py`, `prepare_hybrid_cpt_data.py`, and `retokenize_hybrid_cpt_for_model.py`: data preparation.
- `verify_checkpoint.py` and `validate_sft_setup.py`: lightweight preflight checks.

Use `python -m plasticity_loss_sft.<module> --help` for exact arguments. Full runs require substantial GPU compute and external datasets/checkpoints; no full experiment is started by this release.
