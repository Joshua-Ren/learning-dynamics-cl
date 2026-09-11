from __future__ import annotations

from typing import Any


MMLU_CHOICES = ("A", "B", "C", "D")


def format_mmlu_prompt(row: dict[str, Any]) -> tuple[str, str]:
    """Format MMLU exactly as in the experiments in this repository."""
    choices = list(row["choices"])
    answer = row["answer"]
    target = MMLU_CHOICES[answer] if isinstance(answer, int) else str(answer).strip().upper()
    prompt = "\n".join(
        [
            "Answer the following multiple-choice question.",
            "Give only the letter A, B, C, or D.",
            "",
            f"Question: {row['question']}",
            f"A. {choices[0]}",
            f"B. {choices[1]}",
            f"C. {choices[2]}",
            f"D. {choices[3]}",
            "",
            "Answer:",
        ]
    )
    return prompt, target
