#!/usr/bin/env python
"""Score translated candidates and report top-4 source retrieval metrics."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys

COMMON_DIR = Path(__file__).resolve().parents[2] / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

import numpy as np
import torch
from datasets import Dataset
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from forvalue_streaming_ghrh import (
    compute_train_representations,
    get_lm_head_weight,
    normalize_readout_channels,
    parse_gh_embedding_layers,
    resolve_gh_embedding_layers,
    score_test_streaming,
)
from utils import GRPO_dataset


LANGUAGES = ("zh", "fr", "ko", "es")
LABELS = {
    "en": {"question": "Question", "answer": "Answer"},
    "zh": {"question": "问题", "answer": "答案"},
    "fr": {"question": "Question", "answer": "Réponse"},
    "ko": {"question": "문제", "answer": "정답"},
    "es": {"question": "Pregunta", "answer": "Respuesta"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use forvalue_streaming_ghrh readout scores to check whether each English "
            "sample's top-4 translation candidates come from the same original sample."
        )
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Manual translation JSON with source and translations.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max_length", type=int, default=192)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--prediction_topk", type=int, default=16)
    parser.add_argument("--train_score_chunk", type=int, default=8)
    parser.add_argument("--embed_device", type=str, default="auto")
    parser.add_argument("--score_device", type=str, default="auto")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument(
        "--scoring_method",
        "--score_method",
        dest="scoring_method",
        choices=("forvalue", "native_last_embedding"),
        default="forvalue",
        help=(
            "`forvalue` uses the RH/GH readout proposed score. "
            "`native_last_embedding` uses cosine similarity between pooled native "
            "last-layer hidden states as a retrieval baseline."
        ),
    )
    parser.add_argument(
        "--native_pooling",
        choices=("mean", "last_token"),
        default="mean",
        help=(
            "Pooling for --scoring_method native_last_embedding. `mean` averages "
            "non-padding last-layer token embeddings; `last_token` uses the final "
            "non-padding token embedding."
        ),
    )
    parser.add_argument(
        "--readout_channel",
        type=str,
        default=None,
        choices=("rh", "gh", "both"),
        help="Single-channel alias. Use `both` to score RH plus GH.",
    )
    parser.add_argument(
        "--readout_channels",
        nargs="+",
        default=None,
        help=(
            "Readout score channel(s): `rh`, `gh`, or `both`. Also accepts "
            "comma-separated values such as `rh,gh`. Default is `rh`."
        ),
    )
    parser.add_argument(
        "--gh_embedding_layers",
        nargs="+",
        default=["all"],
        help=(
            "GH hidden layers to concatenate before scoring. Default `all` resolves "
            "to layers 1..L-1 with --gh_layer_index_mode bottom."
        ),
    )
    parser.add_argument(
        "--gh_layer_index_mode",
        choices=("bottom", "top"),
        default="bottom",
    )
    parser.add_argument(
        "--gh_use_input_layernorm",
        "--gh_input_layernorm",
        dest="gh_use_input_layernorm",
        action="store_true",
        help="GH-only: apply each selected decoder block's input_layernorm.",
    )
    parser.add_argument(
        "--query_dataset",
        type=str,
        default=None,
        help="If set, run only one English query from this dataset.",
    )
    parser.add_argument(
        "--query_index",
        type=int,
        default=None,
        help="If set with --query_dataset, run only this English query index.",
    )
    parser.add_argument(
        "--retrieval_query_field",
        choices=("qa", "question"),
        default="qa",
        help=(
            "Text used for English queries when computing retrieval/value scores. "
            "`qa` keeps the previous question+answer behavior; `question` removes "
            "the English gold answer while translated candidates still use question+answer."
        ),
    )
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device


def format_qa(fields: Dict[str, str], target: str, language: str = "en") -> str:
    labels = LABELS[language]
    return (
        f"{labels['question']}: {fields['input']}\n"
        f"A. {fields['A']}\n"
        f"B. {fields['B']}\n"
        f"C. {fields['C']}\n"
        f"D. {fields['D']}\n"
        f"{labels['answer']}: {target}"
    )


def format_question(fields: Dict[str, str], language: str = "en") -> str:
    labels = LABELS[language]
    return (
        f"{labels['question']}: {fields['input']}\n"
        f"A. {fields['A']}\n"
        f"B. {fields['B']}\n"
        f"C. {fields['C']}\n"
        f"D. {fields['D']}"
    )


def format_gsm8k(fields: Dict[str, str], language: str = "en") -> str:
    labels = LABELS[language]
    return (
        f"{labels['question']}: {fields['question']}\n"
        f"{labels['answer']}: {fields['answer']}"
    )


def format_gsm8k_question(fields: Dict[str, str], language: str = "en") -> str:
    labels = LABELS[language]
    return f"{labels['question']}: {fields['question']}"


def build_medical_records(
    data: Dict[str, object],
    query_dataset: str | None,
    query_index: int | None,
    retrieval_query_field: str = "qa",
) -> Tuple[List[Dict[str, str]], List[Dict[str, object]], List[Dict[str, str]], List[Dict[str, object]]]:
    if (query_dataset is None) != (query_index is None):
        raise ValueError("--query_dataset and --query_index must be set together.")

    train_records: List[Dict[str, str]] = []
    train_meta: List[Dict[str, object]] = []
    test_records: List[Dict[str, str]] = []
    test_meta: List[Dict[str, object]] = []

    for dataset_name, rows in data["datasets"].items():
        for row in rows:
            source_id = f"{dataset_name}:{row['index']}"
            include_query = (
                query_dataset is None
                or (dataset_name == query_dataset and row["index"] == query_index)
            )
            if include_query:
                if retrieval_query_field == "question":
                    query_text = format_question(row["source"], "en")
                else:
                    query_text = format_qa(row["source"], row["target"], "en")
                test_records.append({"text": query_text})
                test_meta.append(
                    {
                        "test_index": len(test_meta),
                        "dataset": dataset_name,
                        "index": row["index"],
                        "source_id": source_id,
                        "target": row["target"],
                        "retrieval_query_field": retrieval_query_field,
                    }
                )

            for language in LANGUAGES:
                train_records.append(
                    {"text": format_qa(row["translations"][language], row["target"], language)}
                )
                train_meta.append(
                    {
                        "train_index": len(train_meta),
                        "dataset": dataset_name,
                        "index": row["index"],
                        "source_id": source_id,
                        "language": language,
                        "target": row["target"],
                    }
                )

    if not test_records:
        raise ValueError(f"No query found for {query_dataset}:{query_index}.")

    return train_records, train_meta, test_records, test_meta



def build_gsm8k_records(
    data: Dict[str, object],
    query_dataset: str | None,
    query_index: int | None,
    retrieval_query_field: str = "qa",
) -> Tuple[List[Dict[str, str]], List[Dict[str, object]], List[Dict[str, str]], List[Dict[str, object]]]:
    if query_dataset is not None and query_dataset not in {"gsm8k", "openai/gsm8k"}:
        raise ValueError(
            "For GSM8K data, use --query_dataset gsm8k or --query_dataset openai/gsm8k."
        )

    train_records: List[Dict[str, str]] = []
    train_meta: List[Dict[str, object]] = []
    test_records: List[Dict[str, str]] = []
    test_meta: List[Dict[str, object]] = []
    dataset_name = data.get("source_dataset", "openai/gsm8k")

    for row in data["records"]:
        source_id = f"{dataset_name}:{row['index']}"
        include_query = query_index is None or row["index"] == query_index
        if include_query:
            source = row["translations"].get("en", row["source"])
            if retrieval_query_field == "question":
                query_text = format_gsm8k_question(source, "en")
            else:
                query_text = format_gsm8k(source, "en")
            test_records.append({"text": query_text})
            test_meta.append(
                {
                    "test_index": len(test_meta),
                    "dataset": dataset_name,
                    "index": row["index"],
                    "source_id": source_id,
                    "retrieval_query_field": retrieval_query_field,
                }
            )

        for language in LANGUAGES:
            train_records.append(
                {"text": format_gsm8k(row["translations"][language], language)}
            )
            train_meta.append(
                {
                    "train_index": len(train_meta),
                    "dataset": dataset_name,
                    "index": row["index"],
                    "source_id": source_id,
                    "language": language,
                }
            )

    if not test_records:
        raise ValueError(f"No GSM8K query found for index {query_index}.")

    return train_records, train_meta, test_records, test_meta


def build_records(
    data: Dict[str, object],
    query_dataset: str | None,
    query_index: int | None,
    retrieval_query_field: str = "qa",
) -> Tuple[List[Dict[str, str]], List[Dict[str, object]], List[Dict[str, str]], List[Dict[str, object]]]:
    if "datasets" in data:
        return build_medical_records(data, query_dataset, query_index, retrieval_query_field)
    if "records" in data:
        return build_gsm8k_records(data, query_dataset, query_index, retrieval_query_field)
    raise ValueError("Unsupported translation JSON schema: expected `datasets` or `records`.")

def load_model_and_tokenizer(args: argparse.Namespace, embed_device: str):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    dtype = torch.float16 if embed_device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    model.to(embed_device)
    model.eval()
    return tokenizer, model



def forward_native_last_hidden(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    if hasattr(model, "model"):
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return outputs.last_hidden_state

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    return outputs.hidden_states[-1]


@torch.no_grad()
def compute_native_last_embeddings(
    dataloader: DataLoader,
    model: AutoModelForCausalLM,
    embed_device: str,
    pooling: str,
) -> torch.Tensor:
    embeddings = []
    use_amp = embed_device.startswith("cuda")

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(embed_device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(embed_device, non_blocking=True)
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                hidden = forward_native_last_hidden(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).float()
        else:
            hidden = forward_native_last_hidden(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).float()

        if pooling == "mean":
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        elif pooling == "last_token":
            lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
            pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]
        else:
            raise ValueError(f"Unsupported native_pooling: {pooling}")

        pooled = F.normalize(pooled.float(), p=2, dim=-1)
        embeddings.append(pooled.detach().cpu())

        if (step + 1) % 20 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return torch.cat(embeddings, dim=0)


def score_native_last_embedding_similarity(
    train_loader: DataLoader,
    test_loader: DataLoader,
    model: AutoModelForCausalLM,
    embed_device: str,
    score_device: str,
    pooling: str,
    train_score_chunk: int,
    batch_size: int,
) -> torch.Tensor:
    print(f"native embedding pooling: {pooling}")
    print("building native train embeddings...")
    train_embeddings = compute_native_last_embeddings(
        dataloader=train_loader,
        model=model,
        embed_device=embed_device,
        pooling=pooling,
    )
    print("building native test embeddings...")
    test_embeddings = compute_native_last_embeddings(
        dataloader=test_loader,
        model=model,
        embed_device=embed_device,
        pooling=pooling,
    )

    chunk_size = max(1, train_score_chunk * max(1, batch_size))
    test_embeddings = test_embeddings.to(score_device, dtype=torch.float32, non_blocking=True)
    score_parts = []
    for start in range(0, train_embeddings.shape[0], chunk_size):
        train_chunk = train_embeddings[start : start + chunk_size].to(
            score_device,
            dtype=torch.float32,
            non_blocking=True,
        )
        score_parts.append(torch.matmul(test_embeddings, train_chunk.transpose(0, 1)).cpu())
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return torch.cat(score_parts, dim=1)

def summarize_scores(
    score_matrix: np.ndarray,
    train_meta: List[Dict[str, object]],
    test_meta: List[Dict[str, object]],
) -> Dict[str, object]:
    per_query = []
    counts = []

    for test_index, score_row in enumerate(score_matrix):
        order = np.argsort(score_row)[::-1]
        expected_source_id = test_meta[test_index]["source_id"]
        top4 = []
        same_source_count = 0

        for rank, train_index in enumerate(order[:4], start=1):
            meta = dict(train_meta[int(train_index)])
            meta["rank"] = rank
            meta["score"] = float(score_row[int(train_index)])
            meta["same_source"] = meta["source_id"] == expected_source_id
            same_source_count += int(meta["same_source"])
            top4.append(meta)

        counts.append(same_source_count)
        per_query.append(
            {
                "query": test_meta[test_index],
                "top4_same_source_count": same_source_count,
                "top4_same_source_accuracy": same_source_count / 4.0,
                "top4_all_same_source": same_source_count == 4,
                "top4_languages": sorted({item["language"] for item in top4}),
                "top4": top4,
            }
        )

    counts_array = np.array(counts, dtype=np.float32)
    all_four = int(np.sum(counts_array == 4))
    summary = {
        "num_queries": len(test_meta),
        "num_translation_candidates": len(train_meta),
        "top4_item_accuracy_mean": float(np.mean(counts_array / 4.0)),
        "top4_all_same_source_rate": float(all_four / len(test_meta)),
        "queries_with_all_4_same_source": all_four,
        "queries_with_all_4_same_source_and_all_languages": int(
            sum(
                query["top4_all_same_source"]
                and set(query["top4_languages"]) == set(LANGUAGES)
                for query in per_query
            )
        ),
        "top4_same_source_count_histogram": {
            str(count): int(np.sum(counts_array == count)) for count in range(5)
        },
    }
    return {"summary": summary, "per_query": per_query}


def main() -> None:
    args = parse_args()
    embed_device = resolve_device(args.embed_device)
    score_device = resolve_device(args.score_device)
    readout_channels = (
        normalize_readout_channels(args.readout_channels, args.readout_channel)
        if args.scoring_method == "forvalue"
        else tuple()
    )

    data = json.loads(args.data_path.read_text(encoding="utf-8"))
    train_records, train_meta, test_records, test_meta = build_records(
        data=data,
        query_dataset=args.query_dataset,
        query_index=args.query_index,
        retrieval_query_field=args.retrieval_query_field,
    )

    print(f"train translations: {len(train_records)}")
    print(f"test english samples: {len(test_records)}")

    tokenizer, model = load_model_and_tokenizer(args, embed_device)
    train_loader = DataLoader(
        GRPO_dataset(Dataset.from_list(train_records), tokenizer, max_length=args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        GRPO_dataset(Dataset.from_list(test_records), tokenizer, max_length=args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
    )

    gh_embedding_layers = None
    if args.scoring_method == "native_last_embedding":
        print("scoring method: native_last_embedding cosine similarity")
        scores = score_native_last_embedding_similarity(
            train_loader=train_loader,
            test_loader=test_loader,
            model=model,
            embed_device=embed_device,
            score_device=score_device,
            pooling=args.native_pooling,
            train_score_chunk=args.train_score_chunk,
            batch_size=args.batch_size,
        )
    else:
        lm_head_weight = get_lm_head_weight(model) if "gh" in readout_channels else None
        if "gh" in readout_channels:
            gh_embedding_layers = resolve_gh_embedding_layers(
                parse_gh_embedding_layers(args.gh_embedding_layers),
                model,
                args.gh_layer_index_mode,
            )
            print(f"GH embedding layers: {gh_embedding_layers}")
            print(f"GH layer index mode: {args.gh_layer_index_mode}")
            print(f"GH use input layernorm: {args.gh_use_input_layernorm}")

        print(f"readout channels: {', '.join(readout_channels)}")
        print("building train representations...")
        train_representations = compute_train_representations(
            dataloader_train=train_loader,
            model=model,
            embed_device=embed_device,
            prediction_topk=args.prediction_topk,
            vocab_mode="topk_unique",
            lowest_likelihood_ratio=1.0,
            global_vocab_ids_cpu=None,
            readout_channels=readout_channels,
            lm_head_weight=lm_head_weight,
            gh_embedding_layers=gh_embedding_layers,
            gh_layer_index_mode=args.gh_layer_index_mode,
            gh_use_input_layernorm=args.gh_use_input_layernorm,
            compute_proposed=True,
        )

        print("scoring english samples...")
        scores = score_test_streaming(
            dataloader_test=test_loader,
            model=model,
            train_representations=train_representations,
            embed_device=embed_device,
            score_device=score_device,
            prediction_topk=args.prediction_topk,
            train_score_chunk=args.train_score_chunk,
            vocab_mode="topk_unique",
            lowest_likelihood_ratio=1.0,
            global_vocab_ids_cpu=None,
            readout_channels=readout_channels,
            lm_head_weight=lm_head_weight,
            gh_embedding_layers=gh_embedding_layers,
            gh_layer_index_mode=args.gh_layer_index_mode,
            gh_use_input_layernorm=args.gh_use_input_layernorm,
            approximate_proposed=False,
        )

    result = summarize_scores(
        score_matrix=scores.detach().cpu().float().numpy(),
        train_meta=train_meta,
        test_meta=test_meta,
    )
    result["summary"].update(
        {
            "method": (
                "native last-layer embedding cosine similarity"
                if args.scoring_method == "native_last_embedding"
                else "forvalue_streaming_ghrh imported functions, selected readout proposed score"
            ),
            "scoring_method": args.scoring_method,
            "native_pooling": (
                args.native_pooling if args.scoring_method == "native_last_embedding" else None
            ),
            "readout_channels": list(readout_channels),
            "gh_embedding_layers": gh_embedding_layers,
            "gh_layer_index_mode": args.gh_layer_index_mode if "gh" in readout_channels else None,
            "gh_use_input_layernorm": (
                bool(args.gh_use_input_layernorm) if "gh" in readout_channels else False
            ),
            "model_name": args.model_name,
            "retrieval_query_field": args.retrieval_query_field,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "prediction_topk": args.prediction_topk,
        }
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {args.output_path}")


if __name__ == "__main__":
    main()
