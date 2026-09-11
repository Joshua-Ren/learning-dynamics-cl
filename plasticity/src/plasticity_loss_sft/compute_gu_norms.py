from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


TASKS = ("gsm8k", "mbpp", "dolly_qa")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute token-level ||e_y - softmax(logits)||_2^2 on prepared train subsets."
    )
    parser.add_argument("--subsets_dir", default="data/prepared_subsets")
    parser.add_argument("--output_dir", default="analysis/gu_norms_train")
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--split", default="train", choices=("train", "probe"))
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--dtype", default="auto", choices=("auto", "bf16", "fp16", "fp32"))
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--softmax_block_size", type=int, default=256)
    parser.add_argument("--histogram_bins", type=int, default=80)
    parser.add_argument("--flush_every", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()
    model.config.use_cache = False

    report = {
        "model_name": args.model_name,
        "split": args.split,
        "subsets_dir": args.subsets_dir,
        "device": str(device),
        "dtype": str(dtype),
        "softmax_block_size": args.softmax_block_size,
        "tasks": {},
    }

    for task in tasks:
        summary = process_task(
            task=task,
            split=args.split,
            subsets_dir=Path(args.subsets_dir),
            output_dir=output_dir,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_examples=args.max_examples,
            max_length=args.max_length,
            softmax_block_size=args.softmax_block_size,
            histogram_bins=args.histogram_bins,
            flush_every=args.flush_every,
        )
        report["tasks"][task] = summary
        write_json(output_dir / f"{task}_gu_norm_summary.json", summary)
        print(
            f"{task}: examples={summary['examples']} tokens={summary['tokens']} "
            f"mean={summary['mean']:.6f} p95={summary['p95']:.6f} max={summary['max']:.6f}"
        )

    write_json(output_dir / "gu_norm_report.json", report)
    print(f"Wrote report: {output_dir / 'gu_norm_report.json'}")


def process_task(
    task: str,
    split: str,
    subsets_dir: Path,
    output_dir: Path,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    device: torch.device,
    max_examples: int | None,
    max_length: int | None,
    softmax_block_size: int,
    histogram_bins: int,
    flush_every: int,
) -> dict[str, Any]:
    input_path = subsets_dir / task / f"{split}.jsonl"
    output_path = output_dir / f"{task}_{split}_gu_norm_tokens.jsonl"
    values: list[float] = []
    examples = 0
    skipped_for_length = 0

    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as sink:
        buffer: list[str] = []
        for record in iter_jsonl(source):
            if max_examples is not None and examples >= max_examples:
                break
            rendered = render_target_tokens(record, tokenizer)
            if max_length is not None and len(rendered.input_ids) > max_length:
                skipped_for_length += 1
                continue
            rows, norm_values = compute_example_rows(
                task=task,
                record=record,
                rendered=rendered,
                tokenizer=tokenizer,
                model=model,
                device=device,
                softmax_block_size=softmax_block_size,
            )
            values.extend(norm_values)
            for row in rows:
                buffer.append(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
                if len(buffer) >= flush_every:
                    sink.writelines(buffer)
                    buffer.clear()
            examples += 1
            if examples % 50 == 0:
                print(f"{task}: processed {examples} examples, {len(values)} target tokens")
        if buffer:
            sink.writelines(buffer)

    summary = summarize_values(values)
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
    histogram = build_histogram(values, histogram_bins)
    histogram_path = output_dir / f"{task}_{split}_gu_norm_histogram.json"
    histogram_svg_path = output_dir / f"{task}_{split}_gu_norm_histogram.svg"
    write_json(histogram_path, histogram)
    write_histogram_svg(histogram_svg_path, task, split, histogram)
    summary["histogram_path"] = str(histogram_path)
    summary["histogram_svg_path"] = str(histogram_svg_path)
    return summary


class RenderedTarget:
    def __init__(
        self,
        input_ids: list[int],
        target_positions: list[int],
        completion_start_char: int,
    ) -> None:
        self.input_ids = input_ids
        self.target_positions = target_positions
        self.completion_start_char = completion_start_char


def render_target_tokens(
    record: Mapping[str, Any],
    tokenizer: PreTrainedTokenizerBase,
) -> RenderedTarget:
    messages = record["messages"]
    assistant_index = last_assistant_index(messages)
    if tokenizer.chat_template is None:
        prompt, full_text = render_plain_prompt_completion(messages, assistant_index)
    else:
        prompt_messages = messages[:assistant_index]
        full_messages = messages[: assistant_index + 1]
        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    if not full_text.startswith(prompt):
        raise RuntimeError(f"Prompt is not a prefix for example {record.get('example_id')}")

    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = [int(token_id) for token_id in encoded["input_ids"]]
    completion_start_char = len(prompt)
    target_positions = [
        position
        for position, (_start, end) in enumerate(encoded["offset_mapping"])
        if end > completion_start_char and position > 0
    ]
    if not target_positions:
        raise RuntimeError(f"No assistant target tokens for example {record.get('example_id')}")
    return RenderedTarget(input_ids, target_positions, completion_start_char)


def render_plain_prompt_completion(
    messages: list[Mapping[str, Any]],
    assistant_index: int,
) -> tuple[str, str]:
    prompt_parts = [plain_message_block(message) for message in messages[:assistant_index]]
    assistant_content = str(messages[assistant_index]["content"]).strip()
    if not assistant_content:
        raise RuntimeError("Rendered assistant completion is empty.")
    prompt = "\n\n".join(prompt_parts)
    if prompt:
        prompt += "\n\n"
    prompt += "Assistant:\n"
    return prompt, prompt + assistant_content


def plain_message_block(message: Mapping[str, Any]) -> str:
    role = str(message["role"]).strip().lower()
    content = str(message["content"]).strip()
    if role == "user":
        label = "User"
    elif role == "assistant":
        label = "Assistant"
    elif role == "system":
        label = "System"
    else:
        label = role.capitalize() or "Message"
    return f"{label}:\n{content}"


def compute_example_rows(
    task: str,
    record: Mapping[str, Any],
    rendered: RenderedTarget,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    device: torch.device,
    softmax_block_size: int,
) -> tuple[list[dict[str, Any]], list[float]]:
    input_ids = torch.tensor([rendered.input_ids], dtype=torch.long, device=device)
    target_positions = torch.tensor(rendered.target_positions, dtype=torch.long, device=device)
    target_ids = input_ids[0, target_positions]
    logit_positions = target_positions - 1

    with torch.inference_mode():
        logits = model(input_ids=input_ids).logits[0, logit_positions]
        p_y_parts = []
        norm_parts = []
        for start in range(0, logits.shape[0], softmax_block_size):
            block_logits = logits[start : start + softmax_block_size].float()
            block_targets = target_ids[start : start + softmax_block_size]
            probs = torch.softmax(block_logits, dim=-1)
            p_y = probs.gather(1, block_targets[:, None]).squeeze(1)
            norm_sq = 1.0 - (2.0 * p_y) + torch.sum(probs * probs, dim=-1)
            p_y_parts.append(p_y.cpu())
            norm_parts.append(norm_sq.cpu())

    p_y_values = torch.cat(p_y_parts).tolist()
    norm_values = [float(value) for value in torch.cat(norm_parts).tolist()]
    target_id_values = [int(value) for value in target_ids.cpu().tolist()]
    target_position_values = [int(value) for value in target_positions.cpu().tolist()]

    rows = []
    example_id = record.get("example_id")
    for token_index, (sequence_position, token_id, probability, norm_sq) in enumerate(
        zip(target_position_values, target_id_values, p_y_values, norm_values, strict=True)
    ):
        rows.append(
            {
                "task": task,
                "example_id": example_id,
                "token_position": token_index,
                "sequence_position": sequence_position,
                "token_id": token_id,
                "token": tokenizer.decode([token_id], clean_up_tokenization_spaces=False),
                "target_token_probability": float(probability),
                "gu_norm_sq": norm_sq,
            }
        )
    return rows, norm_values


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


def percentile(sorted_values: list[float], q: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def build_histogram(values: list[float], bins: int) -> dict[str, Any]:
    if not values:
        return {"bins": bins, "counts": [], "edges": []}
    minimum = min(values)
    maximum = max(values)
    if minimum == maximum:
        return {"bins": 1, "counts": [len(values)], "edges": [minimum, maximum]}
    width = (maximum - minimum) / bins
    counts = [0 for _ in range(bins)]
    for value in values:
        index = min(int((value - minimum) / width), bins - 1)
        counts[index] += 1
    edges = [minimum + i * width for i in range(bins + 1)]
    return {"bins": bins, "counts": counts, "edges": edges}


def write_histogram_svg(path: Path, task: str, split: str, histogram: Mapping[str, Any]) -> None:
    counts = list(histogram["counts"])
    width = 900
    height = 420
    left = 70
    right = 20
    top = 40
    bottom = 70
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_count = max(counts) if counts else 1
    bar_width = plot_width / max(len(counts), 1)
    bars = []
    for index, count in enumerate(counts):
        bar_height = 0 if max_count == 0 else plot_height * count / max_count
        x = left + index * bar_width
        y = top + plot_height - bar_height
        bars.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(bar_width - 1, 1):.2f}" '
            f'height="{bar_height:.2f}" fill="#4c78a8" />'
        )
    edges = histogram.get("edges") or [0, 0]
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white" />
  <text x="{width / 2:.0f}" y="24" text-anchor="middle" font-family="sans-serif" font-size="18">||g_u||_2^2 histogram: {task}/{split}</text>
  <line x1="{left}" y1="{top + plot_height}" x2="{width - right}" y2="{top + plot_height}" stroke="#333" />
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#333" />
  {''.join(bars)}
  <text x="{left}" y="{height - 28}" text-anchor="middle" font-family="sans-serif" font-size="12">{edges[0]:.4g}</text>
  <text x="{width - right}" y="{height - 28}" text-anchor="end" font-family="sans-serif" font-size="12">{edges[-1]:.4g}</text>
  <text x="{width / 2:.0f}" y="{height - 8}" text-anchor="middle" font-family="sans-serif" font-size="13">||g_u||_2^2</text>
  <text x="18" y="{top + 12}" text-anchor="start" font-family="sans-serif" font-size="12">max bin {max_count}</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def iter_jsonl(lines: Iterable[str]) -> Iterable[dict[str, Any]]:
    for line in lines:
        if line.strip():
            yield json.loads(line)


def last_assistant_index(messages: list[Mapping[str, Any]]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "assistant":
            return index
    raise ValueError("Expected at least one assistant message.")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
