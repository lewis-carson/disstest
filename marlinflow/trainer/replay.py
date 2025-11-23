from __future__ import annotations

import random
from collections import deque
from typing import Deque

import torch

from dataloader import Batch


class ReplayBuffer:
    """A minimal FIFO replay buffer that stores full batches on a target device."""

    def __init__(
        self,
        capacity: int,
        device: torch.device,
        priority_metric: str | None = None,
        priority_eps: float = 1e-6,
    ) -> None:
        if capacity <= 0:
            raise ValueError("Replay buffer capacity must be positive")
        self._buffer: Deque[tuple[Batch, float]] = deque(maxlen=capacity)
        self._device = device
        self._priority_metric = priority_metric
        self._priority_eps = priority_eps

    def __len__(self) -> int:
        return len(self._buffer)

    def add(self, batch: Batch) -> None:
        """Store a detached clone of the batch on the replay device."""
        stored = batch.to_device(self._device).detach_clone()
        priority = self._compute_priority(stored)
        self._buffer.append((stored, priority))

    def can_sample(self, minimum: int) -> bool:
        return len(self._buffer) >= minimum

    def sample(self) -> Batch:
        if not self._buffer:
            raise RuntimeError("Cannot sample from an empty replay buffer")

        entries: tuple[tuple[Batch, float], ...] = tuple(self._buffer)
        if self._priority_metric is None:
            batch, _ = random.choice(entries)
            return batch

        batches, priorities = zip(*entries)
        batch = random.choices(batches, weights=priorities, k=1)[0]
        return batch

    def sample_to_device(self, device: torch.device) -> Batch:
        batch = self.sample()
        if device == self._device:
            return batch
        return batch.to_device(device)

    def _compute_priority(self, batch: Batch) -> float:
        if self._priority_metric is None:
            return 1.0

        metric = self._priority_metric.lower()
        if metric == "cp_norm":
            value = torch.linalg.norm(batch.cp.float()).item()
        elif metric == "cp_abs_mean":
            value = batch.cp.float().abs().mean().item()
        elif metric == "cp_var":
            value = batch.cp.float().var(unbiased=False).item()
        else:
            raise ValueError(f"Unknown priority metric: {self._priority_metric}")

        return max(value + self._priority_eps, self._priority_eps)
