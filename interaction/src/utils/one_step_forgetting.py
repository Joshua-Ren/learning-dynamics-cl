from collections import defaultdict
from pathlib import Path
import json

import pandas as pd
import torch
import torch.nn.functional as F


def clone_model_state(model):
    # Keep the restore checkpoint on CPU so exact-gradient diagnostics do not
    # reserve a second full model copy on GPU memory.
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def restore_model_state(model, state_dict):
    model.load_state_dict(state_dict)


def iter_supervised_positions(sample, max_positions=None):
    """
    Yield supervised token positions from a processed causal-LM sample.

    The token at position pos is scored by logits at pos - 1, so pos == 0 is
    skipped even if it has a label.
    """
    labels = sample["labels"]
    attention_mask = sample.get("attention_mask")

    if labels.dim() != 1:
        raise ValueError("sample['labels'] must have shape [seq_len].")
    if attention_mask is not None and attention_mask.dim() != 1:
        raise ValueError("sample['attention_mask'] must have shape [seq_len].")
    if max_positions is not None and max_positions <= 0:
        return

    count = 0
    for pos in range(1, labels.shape[0]):
        if labels[pos].item() == -100:
            continue
        if attention_mask is not None and attention_mask[pos].item() == 0:
            continue
        yield pos
        count += 1
        if max_positions is not None and count >= max_positions:
            break


def apply_single_token_update(model, optimizer, input_ids, attention_mask, labels, pos):
    """
    Apply one optimizer step using exactly one supervised label position.

    Args:
        input_ids: [1, seq_len]
        attention_mask: [1, seq_len]
        labels: [1, seq_len]
        pos: supervised target token position. The model's causal LM loss handles
            the internal shift.
    """
    model.train()

    single_labels = torch.full_like(labels, -100)
    single_labels[0, pos] = labels[0, pos]

    model.zero_grad(set_to_none=True)
    optimizer.zero_grad()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=single_labels,
        output_hidden_states=False,
        output_attentions=False,
    )
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    return loss.item()


def _model_device(model):
    return next(model.parameters()).device


def _as_batch_tensor(value, device):
    if torch.is_tensor(value):
        tensor = value
    else:
        tensor = torch.tensor(value, dtype=torch.long)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    return tensor.to(device)


def _sample_id(sample, default=None):
    for key in ("probe_id", "example_id", "sample_id", "raw_index"):
        if key in sample:
            return sample[key]
    return default


@torch.no_grad()
def eval_supervised_logprob(model, sample):
    """
    Compute teacher-forced log p(supervised labels | prefix) for one sample.

    Expects standard supervised fields:
        input_ids: [seq_len]
        attention_mask: [seq_len]
        labels: [seq_len], with ignored positions set to -100

    For causal LMs, labels[pos] is scored by logits[pos - 1].
    """
    model.eval()
    device = _model_device(model)
    input_ids = _as_batch_tensor(sample["input_ids"], device)
    attention_mask = _as_batch_tensor(sample["attention_mask"], device)
    labels = _as_batch_tensor(sample["labels"], device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=False,
        output_attentions=False,
    )
    log_probs = F.log_softmax(outputs.logits, dim=-1).float()

    token_logps = []
    for pos in iter_supervised_positions(
        {"labels": labels[0].detach().cpu(), "attention_mask": attention_mask[0].detach().cpu()}
    ):
        target_token_id = int(labels[0, pos].item())
        token_logps.append(log_probs[0, pos - 1, target_token_id])

    if not token_logps:
        total_logp = 0.0
        avg_logp = None
    else:
        stacked = torch.stack(token_logps)
        total_logp = stacked.sum().item()
        avg_logp = stacked.mean().item()

    return {
        "probe_id": _sample_id(sample),
        "domain": sample.get("domain"),
        "dataset": sample.get("dataset"),
        "example_id": sample.get("example_id"),
        "raw_index": sample.get("raw_index"),
        "total_logp": total_logp,
        "avg_logp": avg_logp,
        "num_scored_tokens": len(token_logps),
    }


@torch.no_grad()
def eval_probe_samples(model, samples):
    results = []
    for sample_index, sample in enumerate(samples):
        row = eval_supervised_logprob(model, sample)
        if row["probe_id"] is None:
            row["probe_id"] = sample_index
        results.append(row)
    result_by_id = {row["probe_id"]: row for row in results}
    return results, result_by_id


def group_samples_by_domain(samples):
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample.get("domain", "unknown")].append(sample)
    return dict(grouped)


def compute_forgetting_scores(before_by_id, after_by_id, probe_by_domain, tau):
    """
    Compute per-domain forgetting from before/after teacher-forced log-probs.

    Delta is logp_after - logp_before. Larger F_avg means larger loss of probe
    likelihood; F_ratio is the fraction of probe samples with delta < -tau.
    """
    f_avg_by_domain = {}
    f_ratio_by_domain = {}

    for domain, probe_items in probe_by_domain.items():
        deltas = []
        for local_index, item in enumerate(probe_items):
            probe_id = _sample_id(item, default=local_index)
            if probe_id not in before_by_id or probe_id not in after_by_id:
                raise KeyError(f"Missing before/after probe result for probe_id={probe_id!r}.")
            delta = after_by_id[probe_id]["total_logp"] - before_by_id[probe_id]["total_logp"]
            deltas.append(delta)

        if not deltas:
            f_avg_by_domain[domain] = None
            f_ratio_by_domain[domain] = None
            continue

        f_avg_by_domain[domain] = -sum(deltas) / len(deltas)
        f_ratio_by_domain[domain] = sum(delta < -tau for delta in deltas) / len(deltas)

    return f_avg_by_domain, f_ratio_by_domain


class UpdateResultWriter:
    def __init__(self, domains):
        self.domains = list(domains)
        self.summary_rows = []
        self.domain_rows = []

    def add_update(
        self,
        update_id,
        algo,
        sample_id,
        token_pos,
        sop,
        entropy,
        aop_l2,
        aop_kl,
        f_avg_by_domain,
        f_ratio_by_domain,
        extra=None,
    ):
        extra = extra or {}
        row = {
            "update_id": update_id,
            "algo": algo,
            "sample_id": sample_id,
            "token_pos": token_pos,
            "sop": sop,
            "entropy": entropy,
            "aop_l2": aop_l2,
            "aop_kl": aop_kl,
            "f_avg_mean": self._safe_mean(f_avg_by_domain),
            "f_ratio_mean": self._safe_mean(f_ratio_by_domain),
        }

        for domain in self.domains:
            row[f"f_avg_{domain}"] = f_avg_by_domain.get(domain)
            row[f"f_ratio_{domain}"] = f_ratio_by_domain.get(domain)

        row.update(extra)
        self.summary_rows.append(row)

        for domain in self.domains:
            self.domain_rows.append(
                {
                    "update_id": update_id,
                    "algo": algo,
                    "sample_id": sample_id,
                    "token_pos": token_pos,
                    "domain": domain,
                    "sop": sop,
                    "entropy": entropy,
                    "aop_l2": aop_l2,
                    "aop_kl": aop_kl,
                    "f_avg": f_avg_by_domain.get(domain),
                    "f_ratio": f_ratio_by_domain.get(domain),
                    **extra,
                }
            )

    def to_dataframes(self):
        return pd.DataFrame(self.summary_rows), pd.DataFrame(self.domain_rows)

    def save(self, out_dir, config=None, save_csv=False):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        df_summary, df_domain = self.to_dataframes()
        df_summary.to_parquet(out_dir / "updates_summary.parquet", index=False)
        df_domain.to_parquet(out_dir / "updates_by_domain.parquet", index=False)

        if save_csv:
            df_summary.to_csv(out_dir / "updates_summary.csv", index=False)
            df_domain.to_csv(out_dir / "updates_by_domain.csv", index=False)

        if config is not None:
            with open(out_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _safe_mean(values_by_domain):
        values = [value for value in values_by_domain.values() if value is not None]
        return sum(values) / len(values) if values else None


def _dry_test():
    class ShiftCheckModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def forward(
            self,
            input_ids,
            attention_mask=None,
            labels=None,
            output_hidden_states=False,
            output_attentions=False,
        ):
            batch_size, seq_len = input_ids.shape
            logits = torch.full(
                (batch_size, seq_len, 32),
                -10.0,
                dtype=torch.float32,
                device=input_ids.device,
            )
            # Position 0 strongly predicts label 10. Position 1 strongly predicts
            # a different token, so this catches accidental logits[pos] scoring.
            logits[:, 0, 10] = 10.0
            logits[:, 1, 10] = -10.0
            logits[:, 1, 11] = 10.0
            logits = logits + self.anchor
            loss = None
            if labels is not None:
                loss = F.cross_entropy(
                    logits[:, :-1, :].reshape(-1, 32),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100,
                )
            return type("ToyOutput", (), {"logits": logits, "loss": loss})()

    position_sample = {
        "input_ids": torch.tensor([1, 2, 3, 4]),
        "attention_mask": torch.tensor([1, 1, 1, 0]),
        "labels": torch.tensor([-100, 10, -100, 20]),
        "probe_id": "position_case",
        "domain": "dry",
    }
    positions = list(iter_supervised_positions(position_sample))
    assert positions == [1], positions
    assert list(iter_supervised_positions(position_sample, max_positions=0)) == []
    print(f"iter_supervised_positions: {positions}")

    model = ShiftCheckModel()
    score = eval_supervised_logprob(model, position_sample)
    logits = model(position_sample["input_ids"].unsqueeze(0)).logits
    expected_prev = F.log_softmax(logits, dim=-1)[0, 0, 10].item()
    wrong_current = F.log_softmax(logits, dim=-1)[0, 1, 10].item()
    assert abs(score["total_logp"] - expected_prev) < 1e-7
    assert abs(score["total_logp"] - wrong_current) > 1.0
    assert score["num_scored_tokens"] == 1
    print(
        "eval_supervised_logprob: "
        f"total_logp={score['total_logp']:.6f}, "
        f"logits[pos-1]={expected_prev:.6f}, logits[pos]={wrong_current:.6f}"
    )

    before = {"position_case": {"total_logp": -10.0}}
    after = {"position_case": {"total_logp": -11.0}}
    f_avg, f_ratio = compute_forgetting_scores(
        before,
        after,
        {"dry": [position_sample]},
        tau=0.1,
    )
    assert f_avg["dry"] == 1.0
    assert f_ratio["dry"] == 1.0
    print(f"compute_forgetting_scores: f_avg={f_avg['dry']}, f_ratio={f_ratio['dry']}")

    writer = UpdateResultWriter(domains=["dry"])
    writer.add_update(
        update_id="0_1",
        algo="sft",
        sample_id=0,
        token_pos=1,
        sop=0.0,
        entropy=0.0,
        aop_l2=0.0,
        aop_kl=0.0,
        f_avg_by_domain=f_avg,
        f_ratio_by_domain=f_ratio,
    )
    summary, by_domain = writer.to_dataframes()
    assert len(summary) == 1
    assert len(by_domain) == 1
    print("one_step_forgetting dry test passed")


if __name__ == "__main__":
    _dry_test()
