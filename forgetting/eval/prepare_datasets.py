from __future__ import annotations

from evaluation.prompts import format_mmlu_prompt


def _format_mmlu_prompt(row: dict):
    return format_mmlu_prompt(row)
