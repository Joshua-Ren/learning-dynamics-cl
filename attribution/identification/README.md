# Data identification benchmark

This task asks whether a score can identify which of 10 balanced classes produced a held-out example. Each dataset contains 900 training examples (90 per class) and 100 test examples (10 per class). Performance is reported as one-vs-rest AUC and Recall@90, averaged over the 10 classes.

## Contents

```text
identification/
├── data/       # six self-contained Hugging Face datasets (train/test)
├── scores/     # measured Qwen2.5-1.5B metrics and timing JSONs
├── reference/  # machine-readable measured results and paper values
├── src/        # task-specific launcher
└── README.md
```

The launcher reuses [`../common/forvalue_streaming_ghrh.py`](../common/forvalue_streaming_ghrh.py), so the representation and scoring logic are not duplicated. In that implementation, `rh` is CH1 and `gh` is CH2; the combined method uses both channels.

## Reproduction protocol

The saved runs use:

- base model `Qwen/Qwen2.5-1.5B`;
- `total_unique` vocabulary, all token positions, and maximum sequence length 300;
- all 27 intermediate GH layers, indexed from the bottom;
- embedding batch size 15;
- 10 classes, 90 train examples per class, and 10 held-out examples per class.

Three presets are released: `ch1`, raw `ch1-ch2`, and `ch1-ch2-inputln`, which applies the model's input RMSNorm before constructing CH2.

## Saved Qwen2.5-1.5B results

Values are mean ± standard deviation over the 10 classes.

| dataset | preset | AUC | Recall@90 |
| --- | --- | ---: | ---: |
| Sentence transformations | CH1 | 1.000 ± 0.001 | 0.990 ± 0.023 |
| Sentence transformations | CH1+CH2 | 1.000 ± 0.000 | 0.997 ± 0.009 |
| Math problems (w/o reasoning) | CH1 | 1.000 ± 0.000 | 0.998 ± 0.011 |
| Math problems (w/o reasoning) | CH1+CH2 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Math problems (w/ reasoning) | CH1 | 1.000 ± 0.000 | 0.998 ± 0.007 |
| Math problems (w/ reasoning) | CH1+CH2 | 1.000 ± 0.000 | 1.000 ± 0.000 |

Exact unrounded values, configuration metadata, run times, and source artifacts are indexed in [`reference/qwen25_1p5b_reproduction.tsv`](reference/qwen25_1p5b_reproduction.tsv). The values transcribed from the supplied paper table are kept separately in [`reference/paper_table_excerpt.tsv`](reference/paper_table_excerpt.tsv).

CH1 closely reproduces the displayed paper values. With input RMSNorm, CH1+CH2 matches the reported math-with-reason result, while the other two datasets are higher than the displayed values. Since CH1 already agrees, the remaining discrepancy is isolated to the CH2 construction, scaling, or layer aggregation rather than to the dataset split or evaluation metric.

## Run

From the repository root, run all three datasets with the input-RMSNorm preset:

```bash
CUDA_VISIBLE_DEVICES=0 python attribution/identification/src/run_identification.py all \
  --preset ch1-ch2-inputln
```

Or use the unified launcher:

```bash
python attribution/scripts/reproduce.py identification all \
  --preset ch1-ch2-inputln --dry-run
```

Replace the preset with `ch1` or `ch1-ch2` for the other configurations. Model downloads are disabled by default; add `--allow-download` only when needed. Outputs go to `outputs/attribution/identification/`, and existing outputs are protected unless `--overwrite` is given.

The full CH1+CH2 run materializes roughly 114 GB of CPU-side representations per dataset, so check available RAM before launching it.
