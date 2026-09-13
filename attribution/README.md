# Selection and attribution experiments

This directory contains three separate Section 4 experiments. They share the ForValue representation and streaming-score implementation, but have different datasets, evaluation targets, and outputs.

| experiment | question | primary metric | documentation |
| --- | --- | --- | --- |
| Data Identification | Can the score identify which balanced data class produced a held-out example? | one-vs-rest AUC and Recall@90 | [`identification/README.md`](identification/README.md) |
| Agent Memory | Does a retrieved multilingual worked example help the model answer a GSM8K problem it previously missed? | downstream answer accuracy and top-1 source selection | [`agent_memory/README.md`](agent_memory/README.md) |
| Translation Retrieval | Can an English query recover its translations from a multilingual candidate pool? | top-4 same-source retrieval accuracy | [`retrieval/README.md`](retrieval/README.md) |

## Directory layout

```text
attribution/
├── identification/     # balanced data-identification benchmark
│   ├── data/
│   ├── scores/
│   ├── reference/
│   ├── src/
│   └── README.md
├── agent_memory/       # downstream GSM8K memory experiment
│   ├── data/
│   ├── scores/
│   ├── reference/
│   ├── src/
│   └── README.md
├── retrieval/          # direct translation-retrieval experiment
│   ├── data/
│   ├── scores/
│   ├── reference/
│   ├── src/
│   └── README.md
├── common/             # shared ForValue representation and scoring implementation
├── scripts/            # unified launch and integrity-check entry points
└── MANIFEST.json       # bytes and SHA-256 for every released artifact
```

## Quick integrity check

Run from the repository root. This is offline and does not load a model:

```bash
python attribution/scripts/validate_release.py
```

List all released runs and saved scores:

```bash
python attribution/scripts/reproduce.py list
```

See the experiment-specific README before launching a GPU run. Model weights are loaded from Hugging Face and are not included in this repository. Downloads remain disabled unless `--allow-download` is supplied.
