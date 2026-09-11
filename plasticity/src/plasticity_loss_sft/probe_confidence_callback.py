from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase, TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from plasticity_loss_sft.data import load_instruction_dataset, to_prompt_completion_dataset
from plasticity_loss_sft.modeling import get_lm_head_weight, supports_assistant_token_mask


@dataclass(frozen=True)
class ProbeTask:
    name: str
    dataset_path: str
    split: str = "probe"


@dataclass(frozen=True)
class EncodedProbeExample:
    input_ids: list[int]
    response_mask: list[int]


@dataclass(frozen=True)
class FixedForceBlock:
    target_ids: torch.Tensor
    probs: torch.Tensor
    g2: torch.Tensor


class ProbeConfidenceWandbCallback(TrainerCallback):
    def __init__(
        self,
        probe_tasks: list[ProbeTask],
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int,
        eval_steps: int,
        batch_size: int,
        limit: int | None,
        seed: int,
        output_dir: str | None = None,
        plasticity_block_size: int = 8,
        continuous_epoch_offset: float = 0.0,
        sequential_global_step_offset: int = 0,
        wandb_step_offset: int = 0,
        task_name: str | None = None,
        task_index: int | None = None,
        task_round: int | None = None,
        task_segment_index: int | None = None,
    ) -> None:
        if eval_steps < 0:
            raise ValueError("probe eval_steps must be non-negative")
        if batch_size <= 0:
            raise ValueError("probe batch_size must be positive")
        if plasticity_block_size <= 0:
            raise ValueError("plasticity_block_size must be positive")
        self.probe_tasks = probe_tasks
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.eval_steps = eval_steps
        self.batch_size = batch_size
        self.limit = limit
        self.seed = seed
        self.output_dir = Path(output_dir) if output_dir else None
        self.plasticity_block_size = plasticity_block_size
        self.continuous_epoch_offset = continuous_epoch_offset
        self.sequential_global_step_offset = sequential_global_step_offset
        self.wandb_step_offset = wandb_step_offset
        self.task_name = task_name
        self.task_index = task_index
        self.task_round = task_round
        self.task_segment_index = task_segment_index
        self.use_assistant_mask = supports_assistant_token_mask(tokenizer)
        self.encoded = self._load_and_encode_tasks()
        self.last_logged_step: int | None = None
        self.csv_path = self.output_dir / "probe_plasticity_metrics.csv" if self.output_dir else None
        self.jsonl_path = self.output_dir / "probe_plasticity_metrics.jsonl" if self.output_dir else None
        self.fixed_force_cache: dict[tuple[str, str], list[FixedForceBlock]] = {}

    def set_context(
        self,
        continuous_epoch_offset: float,
        sequential_global_step_offset: int,
        wandb_step_offset: int = 0,
        task_name: str | None = None,
        task_index: int | None = None,
        task_round: int | None = None,
        task_segment_index: int | None = None,
    ) -> None:
        self.continuous_epoch_offset = continuous_epoch_offset
        self.sequential_global_step_offset = sequential_global_step_offset
        self.wandb_step_offset = wandb_step_offset
        self.task_name = task_name
        self.task_index = task_index
        self.task_round = task_round
        self.task_segment_index = task_segment_index

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        self._log(state=state, model=kwargs.get("model"), event="train_begin", force=True)

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        if self.eval_steps == 0:
            return
        if state.global_step <= 0 or state.global_step % self.eval_steps != 0:
            return
        self._log(state=state, model=kwargs.get("model"), event="step", force=False)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        self._log(state=state, model=kwargs.get("model"), event="train_end", force=True)

    def _load_and_encode_tasks(self) -> dict[tuple[str, str], list[EncodedProbeExample]]:
        encoded: dict[tuple[str, str], list[EncodedProbeExample]] = {}
        for task in self.probe_tasks:
            dataset = load_instruction_dataset(task.dataset_path, "train", self.limit, self.seed)
            examples: list[EncodedProbeExample]
            if not self.use_assistant_mask:
                prompt_completion = to_prompt_completion_dataset(dataset, self.tokenizer)
                examples = [self._encode_prompt_completion(row) for row in prompt_completion]
            else:
                examples = [self._encode_messages(row["messages"]) for row in dataset]
            examples = [example for example in examples if sum(example.response_mask) > 0]
            if not examples:
                raise RuntimeError(
                    f"No probe examples with response tokens for task {task.name}/{task.split}"
                )
            encoded[(task.name, task.split)] = examples
        return encoded

    def _encode_messages(self, messages: list[dict[str, str]]) -> EncodedProbeExample:
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            add_generation_prompt=False,
        )
        input_ids = list(encoded["input_ids"])
        mask = encoded.get("assistant_masks")
        if mask is None:
            mask = encoded.get("assistant_tokens_mask")
        if mask is None:
            raise RuntimeError("Tokenizer did not return assistant token masks for probe scoring.")
        response_mask = [int(value) for value in mask]
        return self._truncate(EncodedProbeExample(input_ids=input_ids, response_mask=response_mask))

    def _encode_prompt_completion(self, row: dict[str, str]) -> EncodedProbeExample:
        prompt_ids = self.tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
        completion_ids = self.tokenizer(row["completion"], add_special_tokens=False)["input_ids"]
        input_ids = list(prompt_ids) + list(completion_ids)
        response_mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
        return self._truncate(EncodedProbeExample(input_ids=input_ids, response_mask=response_mask))

    def _truncate(self, example: EncodedProbeExample) -> EncodedProbeExample:
        input_ids = example.input_ids[: self.max_seq_length]
        response_mask = example.response_mask[: self.max_seq_length]
        return EncodedProbeExample(input_ids=input_ids, response_mask=response_mask)

    def _log(self, state: TrainerState, model: Any, event: str, force: bool) -> None:
        if model is None:
            return
        if not force and self.last_logged_step == state.global_step:
            return
        if force and self.last_logged_step == state.global_step:
            return

        sequential_step = self.sequential_global_step_offset + int(state.global_step)
        continuous_epoch = self.continuous_epoch_offset + float(state.epoch or 0.0)
        payload: dict[str, int | float | str] = {
            "sequential_global_step": sequential_step,
            "continuous_epoch": continuous_epoch,
            "probe/event": event,
            "plasticity/event": event,
        }
        task_context = self._task_context()
        payload.update({key: value for key, value in task_context.items() if value is not None})

        output_rows: list[dict[str, int | float | str | None]] = []
        was_training = bool(model.training)
        model.eval()
        try:
            lm_head_weight = get_lm_head_weight(model).detach()
            for (task_name, split), examples in self.encoded.items():
                metrics = self._evaluate_task(model, lm_head_weight, examples, (task_name, split))
                self._add_metrics_to_payload(payload, task_name, split, metrics)
                output_rows.append(
                    self._build_output_row(
                        metrics=metrics,
                        event=event,
                        sequential_step=sequential_step,
                        continuous_epoch=continuous_epoch,
                        eval_task=task_name,
                        split=split,
                    )
                )
        finally:
            if was_training:
                model.train()

        self._append_local_outputs(output_rows)
        self._wandb_log(payload, self.wandb_step_offset + sequential_step)
        self.last_logged_step = int(state.global_step)

    def _task_context(self) -> dict[str, int | float | str | None]:
        return {
            "task/name": self.task_name,
            "task/index": self.task_index,
            "task/round": self.task_round,
            "task/segment_index": self.task_segment_index,
        }

    def _add_metrics_to_payload(
        self,
        payload: dict[str, int | float | str],
        task_name: str,
        split: str,
        metrics: dict[str, float],
    ) -> None:
        split_probe_prefix = f"probe/{task_name}/{split}"
        for field in (
            "nll",
            "confidence",
            "perplexity",
            "token_accuracy",
            "num_tokens",
            "num_examples",
        ):
            payload[f"{split_probe_prefix}/{field}"] = metrics[field]
        if split == "probe":
            legacy_prefix = f"probe/{task_name}"
            for field in (
                "nll",
                "confidence",
                "perplexity",
                "token_accuracy",
                "num_tokens",
                "num_examples",
            ):
                payload[f"{legacy_prefix}/{field}"] = metrics[field]

        plasticity_prefix = f"plasticity/{task_name}/{split}"
        for field in (
            "g2_mean",
            "wtg2_mean",
            "R_weighted",
            "R_mean",
            "R_median",
            "R_p95",
            "R_fixed_mean",
            "R_fixed_weighted",
        ):
            payload[f"{plasticity_prefix}/{field}"] = metrics[field]

    def _build_output_row(
        self,
        metrics: dict[str, float],
        event: str,
        sequential_step: int,
        continuous_epoch: float,
        eval_task: str,
        split: str,
    ) -> dict[str, int | float | str | None]:
        return {
            "event": event,
            "global_step": sequential_step,
            "continuous_epoch": continuous_epoch,
            "current_training_task": self.task_name,
            "current_task_index": self.task_index,
            "current_round_index": self.task_round,
            "current_segment_index": self.task_segment_index,
            "eval_task": eval_task,
            "split": split,
            "num_examples": metrics["num_examples"],
            "num_tokens": metrics["num_tokens"],
            "nll": metrics["nll"],
            "confidence": metrics["confidence"],
            "perplexity": metrics["perplexity"],
            "token_accuracy": metrics["token_accuracy"],
            "g2_mean": metrics["g2_mean"],
            "wtg2_mean": metrics["wtg2_mean"],
            "R_weighted": metrics["R_weighted"],
            "R_mean": metrics["R_mean"],
            "R_median": metrics["R_median"],
            "R_p95": metrics["R_p95"],
            "R_fixed_mean": metrics["R_fixed_mean"],
            "R_fixed_weighted": metrics["R_fixed_weighted"],
        }

    @torch.no_grad()
    def _evaluate_task(
        self,
        model: Any,
        lm_head_weight: torch.Tensor,
        examples: list[EncodedProbeExample],
        cache_key: tuple[str, str],
    ) -> dict[str, float]:
        total_nll = 0.0
        total_correct = 0
        total_tokens = 0
        total_examples = 0
        g2_sum = 0.0
        wtg2_sum = 0.0
        r_values: list[float] = []
        r_fixed_values: list[float] = []
        fixed_wtg2_sum = 0.0
        fixed_g2_sum = 0.0
        build_fixed_cache = cache_key not in self.fixed_force_cache
        fixed_blocks: list[FixedForceBlock] = [] if build_fixed_cache else self.fixed_force_cache[cache_key]
        fixed_block_cursor = 0
        device = next(model.parameters()).device
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0

        for start in range(0, len(examples), self.batch_size):
            batch = examples[start : start + self.batch_size]
            max_len = max(len(example.input_ids) for example in batch)
            input_ids = torch.full((len(batch), max_len), int(pad_id), dtype=torch.long, device=device)
            attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long, device=device)
            response_mask = torch.zeros((len(batch), max_len), dtype=torch.bool, device=device)
            for row_index, example in enumerate(batch):
                length = len(example.input_ids)
                input_ids[row_index, :length] = torch.tensor(example.input_ids, dtype=torch.long, device=device)
                attention_mask[row_index, :length] = 1
                response_mask[row_index, :length] = torch.tensor(
                    example.response_mask, dtype=torch.bool, device=device
                )

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1, :].float()
            targets = input_ids[:, 1:]
            target_mask = response_mask[:, 1:] & attention_mask[:, 1:].bool()
            if not target_mask.any():
                continue

            selected_logits = logits[target_mask]
            selected_targets = targets[target_mask]
            predictions = selected_logits.argmax(dim=-1)
            total_correct += int((predictions == selected_targets).sum().item())
            token_count = int(selected_targets.numel())
            total_tokens += token_count
            total_examples += len(batch)

            block_metrics = self._score_selected_tokens(
                selected_logits=selected_logits,
                selected_targets=selected_targets,
                lm_head_weight=lm_head_weight,
                build_fixed_cache=build_fixed_cache,
                fixed_blocks=fixed_blocks,
                fixed_block_cursor=fixed_block_cursor,
            )
            fixed_block_cursor = int(block_metrics["fixed_block_cursor"])
            total_nll += block_metrics["nll_sum"]
            g2_sum += block_metrics["g2_sum"]
            wtg2_sum += block_metrics["wtg2_sum"]
            fixed_wtg2_sum += block_metrics["fixed_wtg2_sum"]
            fixed_g2_sum += block_metrics["fixed_g2_sum"]
            r_values.extend(block_metrics["r_values"])
            r_fixed_values.extend(block_metrics["r_fixed_values"])

        if build_fixed_cache:
            self.fixed_force_cache[cache_key] = fixed_blocks
        elif fixed_block_cursor != len(fixed_blocks):
            raise RuntimeError(
                f"Fixed-force cache length mismatch for {cache_key}: "
                f"used {fixed_block_cursor} of {len(fixed_blocks)} blocks"
            )

        if total_tokens == 0:
            raise RuntimeError("Probe evaluation produced zero scored response tokens.")
        nll = total_nll / total_tokens
        r_sorted = sorted(r_values)
        r_fixed_sorted = sorted(r_fixed_values)
        return {
            "nll": nll,
            "confidence": math.exp(-nll),
            "perplexity": math.exp(nll) if nll < 50.0 else math.inf,
            "token_accuracy": total_correct / total_tokens,
            "num_tokens": float(total_tokens),
            "num_examples": float(total_examples),
            "g2_mean": g2_sum / total_tokens,
            "wtg2_mean": wtg2_sum / total_tokens,
            "R_weighted": wtg2_sum / max(g2_sum, 1e-12),
            "R_mean": sum(r_values) / len(r_values) if r_values else math.nan,
            "R_median": percentile(r_sorted, 0.50),
            "R_p95": percentile(r_sorted, 0.95),
            "R_fixed_mean": sum(r_fixed_values) / len(r_fixed_values) if r_fixed_values else math.nan,
            "R_fixed_weighted": fixed_wtg2_sum / max(fixed_g2_sum, 1e-12),
        }

    def _score_selected_tokens(
        self,
        selected_logits: torch.Tensor,
        selected_targets: torch.Tensor,
        lm_head_weight: torch.Tensor,
        build_fixed_cache: bool,
        fixed_blocks: list[FixedForceBlock],
        fixed_block_cursor: int,
    ) -> dict[str, Any]:
        nll_sum = 0.0
        g2_sum = 0.0
        wtg2_sum = 0.0
        fixed_g2_sum = 0.0
        fixed_wtg2_sum = 0.0
        r_values: list[float] = []
        r_fixed_values: list[float] = []
        for start in range(0, selected_logits.shape[0], self.plasticity_block_size):
            end = start + self.plasticity_block_size
            block_logits = selected_logits[start:end]
            block_targets = selected_targets[start:end]
            log_probs = torch.log_softmax(block_logits, dim=-1)
            probs = log_probs.exp()
            p_y = probs.gather(1, block_targets[:, None]).squeeze(1)
            token_nll = -log_probs.gather(1, block_targets[:, None]).squeeze(1)
            g2 = 1.0 - (2.0 * p_y) + torch.sum(probs * probs, dim=-1)

            expected_readout = probs.to(lm_head_weight.dtype) @ lm_head_weight
            target_readout = lm_head_weight.index_select(0, block_targets)
            wtg = target_readout - expected_readout
            wtg2 = torch.sum(wtg.float() * wtg.float(), dim=-1)
            r = wtg2 / g2.clamp_min(1e-12)

            if build_fixed_cache:
                fixed_block = FixedForceBlock(
                    target_ids=block_targets.detach().cpu().long(),
                    probs=probs.detach().cpu().to(torch.float16),
                    g2=g2.detach().cpu().float(),
                )
                fixed_blocks.append(fixed_block)
            else:
                if fixed_block_cursor >= len(fixed_blocks):
                    raise RuntimeError("Fixed-force cache is shorter than the current evaluation stream.")
                fixed_block = fixed_blocks[fixed_block_cursor]
                fixed_block_cursor += 1
                cached_targets = fixed_block.target_ids.to(device=block_targets.device, dtype=torch.long)
                if cached_targets.shape != block_targets.shape or not torch.equal(cached_targets, block_targets):
                    raise RuntimeError("Fixed-force cache target tokens do not match current evaluation tokens.")

            base_probs = fixed_block.probs.to(device=lm_head_weight.device, dtype=lm_head_weight.dtype)
            base_g2 = fixed_block.g2.to(device=lm_head_weight.device, dtype=torch.float32)
            fixed_expected_readout = base_probs @ lm_head_weight
            fixed_wtg = target_readout - fixed_expected_readout
            fixed_wtg2 = torch.sum(fixed_wtg.float() * fixed_wtg.float(), dim=-1)
            fixed_r = fixed_wtg2 / base_g2.clamp_min(1e-12)

            nll_sum += float(token_nll.sum().item())
            g2_sum += float(g2.sum().item())
            wtg2_sum += float(wtg2.sum().item())
            fixed_g2_sum += float(base_g2.sum().item())
            fixed_wtg2_sum += float(fixed_wtg2.sum().item())
            r_values.extend(float(value) for value in r.detach().cpu().tolist())
            r_fixed_values.extend(float(value) for value in fixed_r.detach().cpu().tolist())
        return {
            "nll_sum": nll_sum,
            "g2_sum": g2_sum,
            "wtg2_sum": wtg2_sum,
            "fixed_g2_sum": fixed_g2_sum,
            "fixed_wtg2_sum": fixed_wtg2_sum,
            "r_values": r_values,
            "r_fixed_values": r_fixed_values,
            "fixed_block_cursor": fixed_block_cursor,
        }

    def _append_local_outputs(self, rows: list[dict[str, int | float | str | None]]) -> None:
        if not rows or self.output_dir is None or self.csv_path is None or self.jsonl_path is None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys())
        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for row in rows:
                writer.writerow(row)
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")

    def _wandb_log(self, values: dict[str, int | float | str], step: int) -> None:
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is not None:
            wandb.log(values, step=step)


def parse_probe_specs(values: list[str]) -> list[ProbeTask]:
    tasks: list[ProbeTask] = []
    seen = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected probe spec task_name[:split]=jsonl, got {value!r}")
        name_spec, dataset_path = value.split("=", 1)
        name_spec = name_spec.strip()
        dataset_path = dataset_path.strip()
        if ":" in name_spec:
            name, split = name_spec.split(":", 1)
            name = name.strip()
            split = split.strip()
        else:
            name = name_spec
            split = "probe"
        if not name or not split or not dataset_path:
            raise ValueError(f"Invalid empty probe spec component in {value!r}")
        key = (name, split)
        if key in seen:
            raise ValueError(f"Duplicate probe task/split: {name}:{split}")
        if not Path(dataset_path).is_file():
            raise FileNotFoundError(f"Missing probe dataset: {dataset_path}")
        seen.add(key)
        tasks.append(ProbeTask(name=name, split=split, dataset_path=dataset_path))
    return tasks


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
