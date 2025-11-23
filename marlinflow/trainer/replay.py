from __future__ import annotations

import random
from collections import deque
from typing import Deque

from dataloader import Batch


class ReplayBuffer:
    """A minimal FIFO replay buffer that stores full batches on CPU."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("Replay buffer capacity must be positive")
        self._buffer: Deque[Batch] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buffer)

    def add(self, batch: Batch) -> None:
        """Store a detached CPU copy of the batch."""
        self._buffer.append(batch.detach_cpu())

    def can_sample(self, minimum: int) -> bool:
        return len(self._buffer) >= minimum

    def sample(self) -> Batch:
        if not self._buffer:
            raise RuntimeError("Cannot sample from an empty replay buffer")
        return random.choice(self._buffer)

    def sample_to_device(self, device) -> Batch:
        return self.sample().to_device(device)
