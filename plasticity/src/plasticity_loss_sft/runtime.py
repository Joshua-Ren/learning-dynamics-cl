from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass

import torch


PACKAGE_NAMES = ("torch", "transformers", "trl", "datasets", "accelerate", "wandb")


@dataclass(frozen=True)
class GpuReport:
    name: str
    peak_allocated_gb: float


def package_versions() -> dict[str, str]:
    versions = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def bf16_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def gpu_report() -> GpuReport:
    if not torch.cuda.is_available():
        return GpuReport(name="cuda unavailable", peak_allocated_gb=0.0)
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    return GpuReport(name=torch.cuda.get_device_name(), peak_allocated_gb=peak_gb)
