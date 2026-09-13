# Attribution reproducibility contract

The attribution release contains three independent experiments:

- [`identification/README.md`](identification/README.md) documents balanced data identification with CH1, CH1+CH2, and input-RMSNorm CH1+CH2.
- [`agent_memory/README.md`](agent_memory/README.md) documents downstream GSM8K generation with retrieved multilingual memory.
- [`retrieval/README.md`](retrieval/README.md) documents direct cross-language top-4 retrieval.

They share only the ForValue representation and streaming scoring implementation in `common/`. Experiment data, saved scores, reference tables, source entry points, output paths, and documentation are separated under their respective folders.

`MANIFEST.json` records file size and SHA-256 for every released code, data, score, reference, and documentation artifact. Verify the whole release offline with:

```bash
python attribution/scripts/validate_release.py
```

The validator checks identification dataset shards and saved metrics, translated-data schemas, exact query order, candidate source/language/index alignment, saved top-4 metrics, Agent Memory top-1 selections, baseline counts, reference tables, and every manifest hash. The supplied paper-table transcription and the newly measured Qwen2.5-1.5B results are preserved separately. Use `--compare-score` for a recomputed retrieval/selection score and `--memory-result` for a downstream Agent Memory output.

Model checkpoints are not included. The unified launcher defaults to locally cached Hugging Face files; pass `--allow-download` explicitly when network access and model permissions are available.
