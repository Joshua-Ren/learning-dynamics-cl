from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from collections.abc import Mapping
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from plasticity_loss_sft.modeling import get_lm_head_weight
from plasticity_loss_sft.compute_gu_norms import (
    RenderedTarget,
    iter_jsonl,
    percentile,
    render_target_tokens,
    resolve_device,
    resolve_dtype,
    write_json,
)


TASKS = ("gsm8k", "mbpp", "dolly_qa")
SUMMARY_FIELDS = (
    "transmission_mean",
    "transmission_median",
    "transmission_p90",
    "transmission_p95",
    "transmission_p99",
    "gu_norm_sq_mean",
    "target_token_probability_mean",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute task-conditioned readout transmission ||W^T(e_y - p)||_2^2 on probe sets."
    )
    parser.add_argument(
        "--model_specs",
        nargs="+",
        required=True,
        help="Model specs as label=path_or_hf_name. The first spec is used as the comparison baseline.",
    )
    parser.add_argument("--subsets_dir", default="data/prepared_subsets")
    parser.add_argument("--output_dir", default="analysis/readout_transmission/gsm8k_sft")
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--split", default="probe")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--dtype", default="auto", choices=("auto", "bf16", "fp16", "fp32"))
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--softmax_block_size", type=int, default=128)
    parser.add_argument("--flush_every", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_specs = parse_model_specs(args.model_specs)
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    report: dict[str, Any] = {
        "subsets_dir": args.subsets_dir,
        "split": args.split,
        "device": str(device),
        "dtype": str(dtype),
        "softmax_block_size": args.softmax_block_size,
        "max_examples": args.max_examples,
        "max_length": args.max_length,
        "baseline_label": model_specs[0]["label"],
        "models": {},
    }

    for spec in model_specs:
        model_summary = process_model(
            model_label=spec["label"],
            model_name_or_path=spec["model_name_or_path"],
            tasks=tasks,
            split=args.split,
            subsets_dir=Path(args.subsets_dir),
            output_dir=output_dir,
            device=device,
            dtype=dtype,
            max_examples=args.max_examples,
            max_length=args.max_length,
            softmax_block_size=args.softmax_block_size,
            flush_every=args.flush_every,
        )
        report["models"][spec["label"]] = model_summary
        write_json(output_dir / f"{spec['label']}_summary.json", model_summary)

    comparison_rows = build_comparison_rows(report)
    report["comparison"] = comparison_rows
    write_json(output_dir / "readout_transmission_report.json", report)
    write_comparison_csv(output_dir / "readout_transmission_comparison.csv", comparison_rows)
    write_comparison_markdown(output_dir / "readout_transmission_comparison.md", comparison_rows)
    print_comparison(comparison_rows)
    print(f"Wrote report: {output_dir / 'readout_transmission_report.json'}")


def parse_model_specs(values: list[str]) -> list[dict[str, str]]:
    specs = []
    seen = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected model spec label=path_or_hf_name, got {value!r}")
        label, model_name_or_path = value.split("=", 1)
        label = label.strip()
        model_name_or_path = model_name_or_path.strip()
        if not label or not model_name_or_path:
            raise ValueError(f"Invalid empty model spec component in {value!r}")
        if label in seen:
            raise ValueError(f"Duplicate model label: {label}")
        seen.add(label)
        specs.append({"label": label, "model_name_or_path": model_name_or_path})
    if not specs:
        raise ValueError("At least one model spec is required.")
    return specs


def process_model(
    model_label: str,
    model_name_or_path: str,
    tasks: list[str],
    split: str,
    subsets_dir: Path,
    output_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    max_examples: int | None,
    max_length: int | None,
    softmax_block_size: int,
    flush_every: int,
) -> dict[str, Any]:
    print(f"Loading {model_label}: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()
    model.config.use_cache = False
    lm_head_weight = get_lm_head_weight(model).detach()

    model_dir = output_dir / model_label
    model_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "model_label": model_label,
        "model_name_or_path": model_name_or_path,
        "lm_head_shape": list(lm_head_weight.shape),
        "tasks": {},
    }
    for task in tasks:
        task_summary = process_task(
            task=task,
            split=split,
            subsets_dir=subsets_dir,
            output_dir=model_dir,
            model_label=model_label,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_examples=max_examples,
            max_length=max_length,
            softmax_block_size=softmax_block_size,
            flush_every=flush_every,
        )
        summary["tasks"][task] = task_summary
        write_json(model_dir / f"{task}_{split}_summary.json", task_summary)
        print(
            f"{model_label}/{task}: examples={task_summary['examples']} "
            f"tokens={task_summary['tokens']} T_mean={task_summary['transmission_mean']:.6f} "
            f"T_p95={task_summary['transmission_p95']:.6f}"
        )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return summary


def process_task(
    task: str,
    split: str,
    subsets_dir: Path,
    output_dir: Path,
    model_label: str,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    device: torch.device,
    max_examples: int | None,
    max_length: int | None,
    softmax_block_size: int,
    flush_every: int,
) -> dict[str, Any]:
    input_path = subsets_dir / task / f"{split}.jsonl"
    output_path = output_dir / f"{task}_{split}_readout_transmission_tokens.jsonl"
    values: list[float] = []
    gu_norm_values: list[float] = []
    probability_values: list[float] = []
    examples = 0
    skipped_for_length = 0

    with input_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as sink:
        buffer: list[str] = []
        for record in iter_jsonl(source):
            if max_examples is not None and examples >= max_examples:
                break
            rendered = render_target_tokens(record, tokenizer)
            if max_length is not None and len(rendered.input_ids) > max_length:
                skipped_for_length += 1
                continue
            rows = compute_example_rows(
                task=task,
                record=record,
                rendered=rendered,
                tokenizer=tokenizer,
                model=model,
                device=device,
                softmax_block_size=softmax_block_size,
            )
            for row in rows:
                values.append(float(row["readout_transmission"]))
                gu_norm_values.append(float(row["gu_norm_sq"]))
                probability_values.append(float(row["target_token_probability"]))
                buffer.append(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
                if len(buffer) >= flush_every:
                    sink.writelines(buffer)
                    buffer.clear()
            examples += 1
            if examples % 25 == 0:
                print(f"{model_label}/{task}: processed {examples} examples, {len(values)} target tokens")
        if buffer:
            sink.writelines(buffer)

    summary = summarize_task(values, gu_norm_values, probability_values)
    summary.update(
        {
            "task": task,
            "split": split,
            "examples": examples,
            "skipped_for_length": skipped_for_length,
            "tokens": len(values),
            "tokens_path": str(output_path),
        }
    )
    validate_summary(summary)
    return summary


def compute_example_rows(
    task: str,
    record: Mapping[str, Any],
    rendered: RenderedTarget,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    device: torch.device,
    softmax_block_size: int,
) -> list[dict[str, Any]]:
    input_ids = torch.tensor([rendered.input_ids], dtype=torch.long, device=device)
    target_positions = torch.tensor(rendered.target_positions, dtype=torch.long, device=device)
    target_ids = input_ids[0, target_positions]
    logit_positions = target_positions - 1
    lm_head_weight = get_lm_head_weight(model).detach()

    rows = []
    with torch.inference_mode():
        logits = model(input_ids=input_ids).logits[0, logit_positions]
        for start in range(0, logits.shape[0], softmax_block_size):
            end = start + softmax_block_size
            block_logits = logits[start:end].float()
            block_targets = target_ids[start:end]
            probs = torch.softmax(block_logits, dim=-1)
            p_y = probs.gather(1, block_targets[:, None]).squeeze(1)
            gu_norm_sq = 1.0 - (2.0 * p_y) + torch.sum(probs * probs, dim=-1)

            expected_readout = probs.to(lm_head_weight.dtype) @ lm_head_weight
            target_readout = lm_head_weight.index_select(0, block_targets)
            z = target_readout - expected_readout
            transmission = torch.sum(z.float() * z.float(), dim=-1)

            for local_index, token_id, probability, norm_sq, transmission_value in zip(
                range(start, min(end, logits.shape[0])),
                block_targets.detach().cpu().tolist(),
                p_y.detach().cpu().tolist(),
                gu_norm_sq.detach().cpu().tolist(),
                transmission.detach().cpu().tolist(),
                strict=True,
            ):
                rows.append(
                    {
                        "task": task,
                        "example_id": record.get("example_id"),
                        "token_position": int(local_index),
                        "sequence_position": int(rendered.target_positions[local_index]),
                        "token_id": int(token_id),
                        "token": tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False),
                        "target_token_probability": float(probability),
                        "gu_norm_sq": float(norm_sq),
                        "readout_transmission": float(transmission_value),
                    }
                )
    return rows


def summarize_task(
    transmission_values: list[float],
    gu_norm_values: list[float],
    probability_values: list[float],
) -> dict[str, float]:
    transmission_summary = summarize_values(transmission_values)
    return {
        "transmission_mean": transmission_summary["mean"],
        "transmission_median": transmission_summary["median"],
        "transmission_p90": transmission_summary["p90"],
        "transmission_p95": transmission_summary["p95"],
        "transmission_p99": transmission_summary["p99"],
        "transmission_max": transmission_summary["max"],
        "gu_norm_sq_mean": mean(gu_norm_values) if gu_norm_values else math.nan,
        "target_token_probability_mean": mean(probability_values) if probability_values else math.nan,
    }


def summarize_values(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "mean": math.nan,
            "median": math.nan,
            "p90": math.nan,
            "p95": math.nan,
            "p99": math.nan,
            "max": math.nan,
        }
    sorted_values = sorted(values)
    return {
        "mean": mean(sorted_values),
        "median": median(sorted_values),
        "p90": percentile(sorted_values, 0.90),
        "p95": percentile(sorted_values, 0.95),
        "p99": percentile(sorted_values, 0.99),
        "max": sorted_values[-1],
    }


def build_comparison_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    baseline_label = str(report["baseline_label"])
    baseline = report["models"][baseline_label]
    rows = []
    for model_label, model_summary in report["models"].items():
        for task, summary in model_summary["tasks"].items():
            baseline_summary = baseline["tasks"][task]
            row: dict[str, Any] = {
                "model_label": model_label,
                "baseline_label": baseline_label,
                "task": task,
                "tokens": summary["tokens"],
                "examples": summary["examples"],
            }
            for field in SUMMARY_FIELDS:
                value = float(summary[field])
                base_value = float(baseline_summary[field])
                row[field] = value
                row[f"base_{field}"] = base_value
                row[f"relative_change_{field}"] = relative_change(value, base_value)
            rows.append(row)
    return rows


def relative_change(value: float, base_value: float) -> float:
    if base_value == 0.0:
        return math.nan
    return (value - base_value) / base_value


def validate_summary(summary: Mapping[str, Any]) -> None:
    if int(summary["tokens"]) <= 0:
        raise RuntimeError(f"No target tokens evaluated for {summary['task']}")
    for field in SUMMARY_FIELDS:
        if not math.isfinite(float(summary[field])):
            raise RuntimeError(f"Non-finite {field} for {summary['task']}: {summary[field]}")


def write_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["model_label", "baseline_label", "task", "examples", "tokens"]
    for field in SUMMARY_FIELDS:
        fieldnames.extend([field, f"base_{field}", f"relative_change_{field}"])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def write_comparison_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Readout Transmission Comparison",
        "",
        "| Model | Task | Tokens | T mean | T median | T p95 | mean ||g||^2 | mean p(y) | rel T mean | rel T p95 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['model_label']}` | `{row['task']}` | {row['tokens']} | "
            f"{row['transmission_mean']:.6f} | {row['transmission_median']:.6f} | "
            f"{row['transmission_p95']:.6f} | {row['gu_norm_sq_mean']:.6f} | "
            f"{row['target_token_probability_mean']:.6f} | "
            f"{row['relative_change_transmission_mean']:.8f} | "
            f"{row['relative_change_transmission_p95']:.8f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_comparison(rows: list[dict[str, Any]]) -> None:
    print("model,task,tokens,T_mean,T_median,T_p95,gu_norm_sq_mean,target_prob_mean,rel_T_mean")
    for row in rows:
        print(
            f"{row['model_label']},{row['task']},{row['tokens']},"
            f"{row['transmission_mean']:.6f},{row['transmission_median']:.6f},"
            f"{row['transmission_p95']:.6f},{row['gu_norm_sq_mean']:.6f},"
            f"{row['target_token_probability_mean']:.6f},"
            f"{row['relative_change_transmission_mean']:.8f}"
        )


if __name__ == "__main__":
    main()
