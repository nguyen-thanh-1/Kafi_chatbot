from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class VramSnapshot:
    allocated_mb: float
    reserved_mb: float
    max_allocated_mb: float
    max_reserved_mb: float
    total_mb: float


def _bytes_to_mb(x: int) -> float:
    return float(x) / (1024.0 * 1024.0)


def get_vram_snapshot() -> Optional[VramSnapshot]:
    if not torch.cuda.is_available():
        return None

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)

    return VramSnapshot(
        allocated_mb=_bytes_to_mb(torch.cuda.memory_allocated(device)),
        reserved_mb=_bytes_to_mb(torch.cuda.memory_reserved(device)),
        max_allocated_mb=_bytes_to_mb(torch.cuda.max_memory_allocated(device)),
        max_reserved_mb=_bytes_to_mb(torch.cuda.max_memory_reserved(device)),
        total_mb=_bytes_to_mb(props.total_memory),
    )


def format_vram(s: Optional[VramSnapshot]) -> str:
    if s is None:
        return "CUDA not available"
    return (
        f"allocated={s.allocated_mb:.0f}MB reserved={s.reserved_mb:.0f}MB "
        f"max_allocated={s.max_allocated_mb:.0f}MB total={s.total_mb:.0f}MB"
    )


def delta_vram(before: Optional[VramSnapshot], after: Optional[VramSnapshot]) -> str:
    if before is None or after is None:
        return "n/a"
    return f"+{(after.allocated_mb - before.allocated_mb):.0f}MB alloc, +{(after.reserved_mb - before.reserved_mb):.0f}MB reserv"

