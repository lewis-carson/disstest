from __future__ import annotations

import random
from collections import deque
from typing import Deque

import torch

from dataloader import Batch


class ReplayBuffer:
    """A minimal FIFO replay buffer that stores full batches on a target device."""

    def __init__(self, capacity: int, device: torch.device) -> None:
        if capacity <= 0:
            raise ValueError("Replay buffer capacity must be positive")
        self._buffer: Deque[Batch] = deque(maxlen=capacity)
        self._device = device

    def __len__(self) -> int:
        return len(self._buffer)

    def add(self, batch: Batch) -> None:
        """Store a detached clone of the batch on the replay device."""
        stored = batch.to_device(self._device).detach_clone()
        self._buffer.append(stored)

    def can_sample(self, minimum: int) -> bool:
        return len(self._buffer) >= minimum

    def sample(self) -> Batch:
        if not self._buffer:
            raise RuntimeError("Cannot sample from an empty replay buffer")
        return random.choice(self._buffer)

    def sample_to_device(self, device: torch.device) -> Batch:
        batch = self.sample()
        if device == self._device:
            return batch
        return batch.to_device(device)
