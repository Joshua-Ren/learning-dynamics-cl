import argparse
from pathlib import Path
import yaml
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from utils.config_read import load_config
from utils.utils import extract_solution
from utils.gsm8k_tokenizer import GSM8KTokenizerProcessor


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="interaction/configs/train_basic.yaml")
    parser.add_argument("--output_dir", type=str, required=True)

    # common overrides
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=None)

    # ---- RL related
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--model_name",type=str, default=None)

    # mode: greedy / sample
    parser.add_argument("--mode", type=str, default="greedy",
                        choices=["greedy", "sample"])

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()

def apply_overrides(config, args):
    if args.lr is not None:
        config["train"]["learning_rate"] = args.lr
    if args.epochs is not None:
        config["train"]["num_train_epochs"] = args.epochs
    if args.batch_size is not None:
        config["train"]["per_device_train_batch_size"] = args.batch_size
    if args.save_steps is not None:
        config["train"]["save_steps"] = args.save_steps
    if args.logging_steps is not None:
        config["train"]["logging_steps"] = args.logging_steps
    if args.model_name is not None:
        config["model"]["name"] = args.model_name
    return config

def save_yaml(obj, path: Path):
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def get_modes(mode: str, temperature: float):
    if mode == "greedy":
        return [("rl_greedy", False, temperature)]
    elif mode == "sample":
        return [(f"rl_tmp{temperature}", True, temperature)]
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    config = apply_overrides(config, args)
    save_yaml(vars(args), output_dir / "extract_rl_args.yaml")
    save_yaml(config, output_dir / "extract_rl_config.yaml")

    model_name = config["model"]["name"]
    max_length = config["model"]["max_length"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    dataset = load_dataset(
        config["dataset"]["name"],
        config["dataset"]["config"],
        split="train"
    )

    processor = GSM8KTokenizerProcessor(
        model_name_or_path=model_name,
        max_length=max_length
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    processed_dataset = processor.process_dataset(dataset)
    n_samples = min(args.n_samples, len(processed_dataset))
    run_modes = get_modes(args.mode, args.temperature)

    for exp_name, do_sample, temperature in run_modes:
        batch_valid_logits = []
        batch_valid_labels = []
        batch_responses = []
        batch_is_correct = []
        batch_sample_indices = []

        with torch.no_grad():
            for i in tqdm(range(n_samples), desc=f"{exp_name}"):
                sample = processed_dataset[i]

                inputs = {
                    "input_ids": sample["input_ids_rl"].unsqueeze(0).to(device),
                    "attention_mask": sample["attention_mask_rl"].unsqueeze(0).to(device),
                }

                prompt_len = inputs["input_ids"].shape[1]

                generate_outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    top_p=args.top_p,
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

                full_generated_ids = generate_outputs.sequences[0]   # [full_len]
                response_start_idx = prompt_len

                # scores: tuple of length generated_len, each [B, V]
                response_logits = torch.cat(generate_outputs.scores, dim=0)  # [generated_len, V]

                # Align with SFT shift convention:
                # logits at step t predict token at position response_start_idx + t
                response_ids = full_generated_ids[response_start_idx - 1 : -1]

                response_text = tokenizer.decode(
                    full_generated_ids[response_start_idx:],
                    skip_special_tokens=True
                ).strip()

                gen_solution = extract_solution(response_text)
                gt_solution = sample["ground_truth"]
                is_correct = (gen_solution == gt_solution)

                batch_valid_logits.append(response_logits.cpu())
                batch_valid_labels.append(response_ids.cpu())
                batch_responses.append(response_text)
                batch_is_correct.append(bool(is_correct))
                batch_sample_indices.append(i)

        save_path = output_dir / f"{exp_name}_logits.pt"
        torch.save(
            {
                "rl_logits": batch_valid_logits,
                "rl_labels": batch_valid_labels,
                "responses": batch_responses,
                "is_correct": batch_is_correct,
                "sample_indices": batch_sample_indices,
                "do_sample": do_sample,
                "temperature": temperature,
                "max_new_tokens": args.max_new_tokens,
                "model_name": model_name,
            },
            save_path,
        )

        print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()