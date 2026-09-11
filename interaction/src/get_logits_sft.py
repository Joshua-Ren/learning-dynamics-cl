import os
import json
import yaml
import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

from utils.config_read import load_config
from utils.gsm8k_tokenizer import GSM8KTokenizerProcessor


def parse_args():
    parser = argparse.ArgumentParser()

    # basic
    parser.add_argument("--config", type=str, default="interaction/configs/train_basic.yaml")
    parser.add_argument("--output_dir", type=str, required=True)

    # common overrides
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=None)

    parser.add_argument("--model_name",type=str, default=None)

    # Train & evaluations
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_extract_logits", action="store_true")
    parser.add_argument("--extract_n_samples", type=int, default=50)

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


def save_yaml(obj, path):
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def main():
    args = parse_args()
    config = load_config(args.config)
    config = apply_overrides(config, args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # save resolved config
    save_yaml(config, output_dir / "resolved_config.yaml")

    model_name = config["model"]["name"]
    max_length = config["model"]["max_length"]

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
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    processed_dataset = processor.process_dataset(dataset)
    # -------------------------
    # Training
    # -------------------------
    if args.do_train:
        train_args = TrainingArguments(
            output_dir=str(output_dir / "trainer_output"),
            learning_rate=config["train"]["learning_rate"],
            num_train_epochs=config["train"]["num_train_epochs"],
            per_device_train_batch_size=config["train"]["per_device_train_batch_size"],
            logging_steps=config["train"].get("logging_steps", 10),
            save_steps=config["train"].get("save_steps", 100),
            save_total_limit=config["train"].get("save_total_limit", 2),
            bf16=config["train"].get("bf16", True),
            fp16=config["train"].get("fp16", False),
            report_to=config["train"].get("report_to", "none"),
            remove_unused_columns=False,
        )

        trainer = SFTTrainer(
            model=model,
            args=train_args,
            train_dataset=processed_dataset,
            processing_class=tokenizer,
        )

        trainer.train()
        # trainer.save_model(str(output_dir / "final_model"))
        # tokenizer.save_pretrained(str(output_dir / "final_model"))

    # -------------------------
    # Extract logits
    # -------------------------
    if args.do_extract_logits:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        batch_valid_logits = []
        batch_valid_labels = []
        batch_responses = []
        n_samples = min(args.extract_n_samples, len(processed_dataset))

        with torch.no_grad():
            for i in tqdm(range(n_samples)):
                sample = processed_dataset[i]
                inputs = {
                    "input_ids": sample["input_ids"].unsqueeze(0).to(device), # [1, max_length]
                    "attention_mask": sample["attention_mask"].unsqueeze(0).to(device), # [1, max_length]
                    "labels": sample["labels"].unsqueeze(0).to(device),
                }

                outputs = model(**inputs, output_hidden_states=False, output_attentions=False)
                logits = outputs.logits #[B, L, V]

                labels = inputs["labels"]
                valid_mask = ((labels != -100) * inputs["attention_mask"]).bool()
                # import pdb
                # breakpoint()
                for bb in range(logits.shape[0]):
                    batch_mask = valid_mask[bb]
                    batch_logits = logits[bb, batch_mask, :]
                    batch_valid_logits.append(batch_logits.cpu())
                    batch_label = labels[bb, batch_mask]
                    batch_valid_labels.append(batch_label.cpu())
                    response_text = tokenizer.decode(
                        batch_label.cpu(),
                        skip_special_tokens=True
                    ).strip()
                    batch_responses.append(response_text)
        torch.save(
            {
                "sft_logits": batch_valid_logits,
                "sft_labels": batch_valid_labels,
                "responses": batch_responses,
                "model_name": model_name,
            },
            output_dir / "sft_logits.pt"
        )


if __name__ == "__main__":
    main()