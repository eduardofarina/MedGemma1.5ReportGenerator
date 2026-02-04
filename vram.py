"""Estimativa de VRAM e sugestão automática de parâmetros (auto-fit).

Este modulo e heuristico por definicao: uso real de memoria varia com driver,
versao do torch, implementacao de atencao e tamanho do prompt/imagens.
O objetivo aqui e dar uma estimativa "boa o suficiente" e sugerir parametros
para evitar OOM (out-of-memory) sem exigir tentativa-e-erro manual.

Dependencias: apenas torch (opcional). Se torch nao existir, o modulo continua
funcionando com informacoes parciais.
"""


from __future__ import annotations

# Heurísticas de VRAM: não é ciência exata, mas evita tentativa-e-erro.
# Use como guia para escolher resolução, número de slices e tamanho de chunk.

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, List, Dict, Any

import math


QuantizationMode = Literal["auto", "none", "8bit", "4bit"]


@dataclass(frozen=True)
class GpuInfo:
    available: bool
    device_name: str = "CPU"
    total_gb: float = 0.0
    free_gb: float = 0.0

    @property
    def used_gb(self) -> float:
        if not self.available:
            return 0.0
        used = self.total_gb - self.free_gb
        return max(0.0, used)


def _bytes_to_gb(num_bytes: int) -> float:
    return float(num_bytes) / (1024.0 ** 3)


def get_gpu_info() -> GpuInfo:
    """Obtém informacoes basicas da GPU (nome, VRAM total, device) usando torch quando disponivel."""
    try:
        import torch  # local import (optional)
    except Exception:
        return GpuInfo(available=False)

    if not torch.cuda.is_available():
        return GpuInfo(available=False)

    device_idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_idx)
    device_name = props.name

    # torch.cuda.mem_get_info retorna (livre, total) em bytes
    try:
        free_b, total_b = torch.cuda.mem_get_info(device_idx)
        return GpuInfo(
            available=True,
            device_name=device_name,
            total_gb=_bytes_to_gb(int(total_b)),
            free_gb=_bytes_to_gb(int(free_b)),
        )
    except Exception:
        # Fallback: só total
        total_b = int(props.total_memory)
        return GpuInfo(
            available=True,
            device_name=device_name,
            total_gb=_bytes_to_gb(total_b),
            free_gb=0.0,
        )


def supports_bfloat16() -> bool:
    """Best-effort BF16 support detection."""
    try:
        import torch
    except Exception:
        return False

    if not torch.cuda.is_available():
        return False

    # Prefere o helper dedicado se existir.
    fn = getattr(torch.cuda, "is_bf16_supported", None)
    if callable(fn):
        try:
            return bool(fn())
        except Exception:
            pass

    # Heurística por compute capability.
    try:
        major, minor = torch.cuda.get_device_capability()
        # Ampere+ generally supports BF16
        return (major, minor) >= (8, 0)
    except Exception:
        return False


def estimate_model_vram_gb(
    *,
    quantization: QuantizationMode,
    dtype: str,
) -> float:
    """
    Estimativa bem aproximada de VRAM do modelo (MedGemma 1.5 4B, vision-language).

    dtype: "bf16" | "fp16" | "fp32"
    """
    q = quantization
    if q == "auto":
        q = "none"

    # These values are intentionally conservative.
    if q == "4bit":
        return 3.5
    if q == "8bit":
        return 5.5

    # No quantization
    if dtype == "fp32":
        return 14.0
    # bf16/fp16
    return 8.5


def estimate_images_vram_gb(
    *,
    num_images: int,
    image_size: int,
) -> float:
    """
    Heurística de overhead por imagem.
    Base: ~50 MB por imagem 896x896 (observação empírica),
    escalada pela área.
    """
    if num_images <= 0:
        return 0.0

    base_per_image_mb = 50.0
    size_factor = (float(image_size) / 896.0) ** 2
    per_image_mb = base_per_image_mb * size_factor
    return (num_images * per_image_mb) / 1024.0


def estimate_total_vram_gb(
    *,
    num_images: int,
    image_size: int,
    quantization: QuantizationMode,
    dtype: str,
    safety_overhead_gb: float = 1.0,
) -> float:
    """Model + images + small overhead."""
    return (
        estimate_model_vram_gb(quantization=quantization, dtype=dtype)
        + estimate_images_vram_gb(num_images=num_images, image_size=image_size)
        + max(0.0, safety_overhead_gb)
    )


@dataclass(frozen=True)
class AutoFitSuggestion:
    image_size: int
    max_slices_per_series: int
    estimated_total_vram_gb: float

    def as_gradio_updates(self) -> Dict[str, Any]:
        """Helper to be used by Gradio callback."""
        return {
            "image_size": self.image_size,
            "max_slices_per_series": self.max_slices_per_series,
            "estimated_total_vram_gb": self.estimated_total_vram_gb,
        }


def suggest_processing_params(
    *,
    series_count: int,
    target_vram_gb: float,
    quantization: QuantizationMode,
    dtype: str,
    candidate_image_sizes: Optional[List[int]] = None,
    min_slices_per_series: int = 4,
    max_slices_per_series_cap: int = 50,
) -> Optional[AutoFitSuggestion]:
    """
    Procura um par (image_size, slices_per_series) que caiba dentro da VRAM alvo.

    Prioriza:
    1) higher slices_per_series
    2) higher image_size
    """
    if series_count <= 0:
        return None

    candidate_image_sizes = candidate_image_sizes or list(range(224, 1025, 32))

    best: Optional[AutoFitSuggestion] = None

    for image_size in candidate_image_sizes:
        per_image_gb = estimate_images_vram_gb(num_images=1, image_size=image_size)
        model_gb = estimate_model_vram_gb(quantization=quantization, dtype=dtype)

        # Reserve some overhead for KV cache etc
        overhead_gb = 1.0

        remaining_gb = target_vram_gb - model_gb - overhead_gb
        if remaining_gb <= 0:
            continue

        # Max images we can afford (integer)
        max_images = int(math.floor(remaining_gb / max(per_image_gb, 1e-6)))
        if max_images <= 0:
            continue

        # Translate max_images into per-series budget.
        slices_per_series = max(1, int(max_images // series_count))
        slices_per_series = min(slices_per_series, max_slices_per_series_cap)

        if slices_per_series < min_slices_per_series:
            continue

        est_total = estimate_total_vram_gb(
            num_images=slices_per_series * series_count,
            image_size=image_size,
            quantization=quantization,
            dtype=dtype,
            safety_overhead_gb=overhead_gb,
        )

        if est_total > target_vram_gb:
            continue

        candidate = AutoFitSuggestion(
            image_size=image_size,
            max_slices_per_series=slices_per_series,
            estimated_total_vram_gb=est_total,
        )

        if best is None:
            best = candidate
            continue

        if (
            candidate.max_slices_per_series > best.max_slices_per_series
            or (
                candidate.max_slices_per_series == best.max_slices_per_series
                and candidate.image_size > best.image_size
            )
        ):
            best = candidate

    return best


def format_vram_summary(
    *,
    gpu: GpuInfo,
    est_total_gb: float,
) -> str:
    """Formata um resumo humano (para UI) com estimativas e sugestoes de VRAM."""
    if not gpu.available:
        return f"VRAM Estimate (no GPU detected): ~{est_total_gb:.1f} GB"

    pct_total = (est_total_gb / max(gpu.total_gb, 1e-6)) * 100.0
    pct_free = (est_total_gb / max(gpu.free_gb, 1e-6)) * 100.0 if gpu.free_gb > 0 else float("nan")

    lines = [
        f"GPU: {gpu.device_name}",
        f"VRAM: total {gpu.total_gb:.1f} GB | free {gpu.free_gb:.1f} GB",
        f"Estimated required: ~{est_total_gb:.1f} GB ({pct_total:.0f}% of total)",
    ]
    if gpu.free_gb > 0:
        lines.append(f"Estimated required vs free: ~{pct_free:.0f}% of free")
        if est_total_gb > gpu.free_gb:
            lines.append("⚠️ Likely OOM with current settings. Reduce slices, reduce image size, or use chunked pipeline.")
    return "\n".join(lines)
