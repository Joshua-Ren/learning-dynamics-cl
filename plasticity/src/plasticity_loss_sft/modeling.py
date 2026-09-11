from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


SUPPORTED_MODELS = (
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "EleutherAI/pythia-160m",
    "allenai/OLMo-1B-hf",
    "meta-llama/Llama-3.2-3B-Instruct",
)


@dataclass(frozen=True)
class ModelAccessReport:
    trainable_parameters: int
    total_parameters: int
    lm_head_shape: tuple[int, ...]
    embeddings_tied: bool


def load_model_and_tokenizer(
    model_name: str,
    use_bf16: bool,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    if model_name not in SUPPORTED_MODELS:
        supported = ", ".join(SUPPORTED_MODELS)
        raise ValueError(f"Unsupported model {model_name!r}. Supported models: {supported}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if use_bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    return model, tokenizer


def assistant_token_mask(tokenizer: PreTrainedTokenizerBase) -> list[int] | None:
    if tokenizer.chat_template is None:
        return None

    messages = [
        {"role": "user", "content": "Say one word."},
        {"role": "assistant", "content": "hello"},
    ]
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    mask = encoded.get("assistant_masks")
    if mask is None:
        mask = encoded.get("assistant_tokens_mask")
    return mask


def supports_assistant_token_mask(tokenizer: PreTrainedTokenizerBase) -> bool:
    mask = assistant_token_mask(tokenizer)
    return mask is not None and bool(torch.as_tensor(mask).any().item())


def assert_assistant_mask_supported(tokenizer: PreTrainedTokenizerBase) -> None:
    if not supports_assistant_token_mask(tokenizer):
        raise RuntimeError(
            "The tokenizer chat template did not return a positive assistant-token mask. "
            "assistant_only_loss=True requires a generation-aware chat template."
        )


def get_lm_head_weight(model: PreTrainedModel) -> torch.Tensor:
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None or not hasattr(output_embeddings, "weight"):
        raise AttributeError("Expected model.get_output_embeddings().weight for readout access.")
    return output_embeddings.weight


def inspect_model_access(model: PreTrainedModel) -> ModelAccessReport:
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    lm_head_weight = get_lm_head_weight(model)
    tied = bool(
        input_embeddings is not None
        and output_embeddings is not None
        and hasattr(input_embeddings, "weight")
        and hasattr(output_embeddings, "weight")
        and input_embeddings.weight.data_ptr() == output_embeddings.weight.data_ptr()
    )

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return ModelAccessReport(
        trainable_parameters=trainable,
        total_parameters=total,
        lm_head_shape=tuple(lm_head_weight.shape),
        embeddings_tied=tied,
    )
