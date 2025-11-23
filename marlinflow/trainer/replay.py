from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque

import torch

from dataloader import Batch


@dataclass
class ReplayEntry:
    batch: Batch
    priority: float


class ReplayBuffer:
    """A minimal prioritized FIFO replay buffer stored on a target device."""

    def __init__(self, capacity: int, device: torch.device, eps: float = 1e-6) -> None:
        if capacity <= 0:
            raise ValueError("Replay buffer capacity must be positive")
        self._capacity = capacity
        self._buffer: Deque[ReplayEntry] = deque()
        self._device = device
        self._total_priority = 0.0
        self._max_priority = 1.0
        self._eps = eps

    def __len__(self) -> int:
        return len(self._buffer)

    def add(self, batch: Batch) -> None:
        """Store a detached clone of the batch on the replay device with max priority."""
        if len(self._buffer) >= self._capacity:
            removed = self._buffer.popleft()
            self._total_priority -= removed.priority

        stored = batch.to_device(self._device).detach_clone()
        priority = max(self._max_priority, self._eps)
        entry = ReplayEntry(stored, priority)
        self._buffer.append(entry)
        self._total_priority += priority
        self._max_priority = max(self._max_priority, priority)

    def can_sample(self, minimum: int) -> bool:
        return len(self._buffer) >= minimum

    def sample_entry(self) -> ReplayEntry:
        if not self._buffer:
            raise RuntimeError("Cannot sample from an empty replay buffer")
        total = max(self._total_priority, self._eps)
        r = random.random() * total
        upto = 0.0
        for entry in self._buffer:
            upto += entry.priority
            if upto >= r:
                return entry
        return self._buffer[-1]

    def update_priority(self, entry: ReplayEntry, priority: float) -> None:
        priority = max(priority, self._eps)
        self._total_priority += priority - entry.priority
        entry.priority = priority
        self._max_priority = max(self._max_priority, priority)
