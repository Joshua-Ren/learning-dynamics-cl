# Generated artifacts

This directory intentionally starts empty except for this note.

scripts/reproduce_full_parameter.sh prepares GSM8K and MMLU data/ by default,
then writes checkpoints, predictions, format reports, and GSM8K summaries to
reproduced_runs/. Set PREPARE=0 only when reusing an already materialized data
directory.

All generated artifacts are ignored by Git. The historical local checkpoint
archive is described in ../EXTERNAL_ARTIFACTS.md.
