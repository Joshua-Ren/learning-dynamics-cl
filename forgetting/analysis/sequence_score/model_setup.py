from dataclasses import dataclass
from pathlib import Path
import sys


def ensure_repo_imports(repo_root: str | Path) -> Path:
    root = Path(repo_root).resolve()
    for path in (root / "src", root, root / "eval"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    return root


@dataclass
class LoadedBaseModel:
    model: object
    tokenizer: object
    processor: object | None
    template: object
    model_args: object
    data_args: object
    finetuning_args: object


def load_base_model_and_template(
    repo_root: str | Path,
    model_name_or_path: str,
    template_name: str,
    cache_dir: str | None = None,
    cutoff_len: int = 2048,
    trust_remote_code: bool = True,
) -> LoadedBaseModel:
    """Load model/tokenizer/template through LLaMA-Factory's normal path."""
    ensure_repo_imports(repo_root)

    from llamafactory.data import get_template_and_fix_tokenizer
    from llamafactory.hparams import DataArguments, FinetuningArguments, ModelArguments
    from llamafactory.model import load_model, load_tokenizer

    model_args = ModelArguments(
        model_name_or_path=model_name_or_path,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )
    data_args = DataArguments(template=template_name, cutoff_len=cutoff_len)
    finetuning_args = FinetuningArguments(stage="sft", finetuning_type="full")

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module["processor"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False)

    return LoadedBaseModel(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        template=template,
        model_args=model_args,
        data_args=data_args,
        finetuning_args=finetuning_args,
    )


def model_architecture_summary(model: object) -> dict:
    config = getattr(model, "config", None)
    output_embeddings = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    input_embeddings = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None

    tied_by_storage = None
    if input_embeddings is not None and output_embeddings is not None:
        try:
            tied_by_storage = (
                input_embeddings.weight.detach().data_ptr() == output_embeddings.weight.detach().data_ptr()
            )
        except Exception:
            tied_by_storage = None

    return {
        "model_class": model.__class__.__name__,
        "model_type": getattr(config, "model_type", None),
        "architectures": getattr(config, "architectures", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "vocab_size": getattr(config, "vocab_size", None),
        "tie_word_embeddings": getattr(config, "tie_word_embeddings", None),
        "tied_by_storage": tied_by_storage,
        "rms_norm_eps": getattr(config, "rms_norm_eps", None),
        "output_head": output_embeddings.__class__.__name__ if output_embeddings is not None else None,
    }
