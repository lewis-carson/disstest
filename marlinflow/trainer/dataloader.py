from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import torch

try:
    import binpack_loader
except ImportError:
    print("binpack_loader not found. Please install it.")


class InputFeatureSet(IntEnum):
    BOARD_768 = 0
    HALF_KP = 1
    HALF_KA = 2
    BOARD_768_CUDA = 3
    HALF_KP_CUDA = 4
    HALF_KA_CUDA = 5

    def max_features(self) -> int:
        if self == InputFeatureSet.HALF_KP or self == InputFeatureSet.HALF_KP_CUDA:
            return 32
        raise NotImplementedError("Only HalfKP is supported with binpack loader")

    def indices_per_feature(self) -> int:
        return 2


@dataclass
class Batch:
    stm_indices: torch.Tensor
    nstm_indices: torch.Tensor
    values: torch.Tensor
    cp: torch.Tensor
    wdl: torch.Tensor
    size: int


class BatchLoader:
    def __init__(
        self, files: list[str], feature_set: InputFeatureSet, batch_size: int
    ) -> None:
        assert files
        if feature_set not in [InputFeatureSet.HALF_KP, InputFeatureSet.HALF_KP_CUDA]:
            raise ValueError("Only HalfKP feature set is supported")

        self._files = files
        self._batch_size = batch_size
        self._feature_set_name = "HalfKP"
        self._stream = self._create_stream()

    def _create_stream(self):
        return binpack_loader.SparseBatchStream(
            self._feature_set_name,
            self._files,
            self._batch_size,
            None,  # skip_config
            False,  # cyclic
            1,  # num_workers
        )

    def read_batch(self, device: torch.device) -> tuple[bool, Batch]:
        data = self._stream.next_batch()
        new_epoch = False

        if data is None:
            new_epoch = True
            self._stream = self._create_stream()
            data = self._stream.next_batch()
            if data is None:
                raise StopIteration("No data available in files")

        (
            us,
            them,
            white_idx,
            white_val,
            black_idx,
            black_val,
            outcome,
            score,
            psqt,
            layer_stack,
        ) = data

        # Convert to torch tensors on device
        # Note: binpack returns numpy arrays
        us = torch.from_numpy(us).to(device)
        white_idx = torch.from_numpy(white_idx).to(device)
        black_idx = torch.from_numpy(black_idx).to(device)
        outcome = torch.from_numpy(outcome).to(device)
        score = torch.from_numpy(score).to(device)

        batch_size = us.shape[0]

        # Determine STM and NSTM indices
        # us is (B, 1), 1.0 if white is STM
        us_bool = us > 0.5

        stm_idx_dense = torch.where(us_bool, white_idx, black_idx)
        nstm_idx_dense = torch.where(us_bool, black_idx, white_idx)

        if self._feature_set == InputFeatureSet.HALF_KP_CUDA:
            stm_indices = stm_idx_dense.flatten()
            nstm_indices = nstm_idx_dense.flatten()
            values = torch.ones_like(stm_indices, dtype=torch.float32)
        else:
            def to_coo_flat(dense_idx):
                mask = dense_idx != -1
                # indices is (N, 2) -> (batch_idx, feature_pos_idx)
                indices = torch.nonzero(mask)
                batch_indices = indices[:, 0]
                feature_indices = dense_idx[mask]
                # Stack (batch_idx, feature_idx) and flatten
                return torch.stack(
                    (batch_indices.int(), feature_indices.int()), dim=1
                ).flatten()

            stm_indices = to_coo_flat(stm_idx_dense)
            nstm_indices = to_coo_flat(nstm_idx_dense)

            # Values are all 1.0 for HalfKP
            # We assume stm and nstm have same number of features per batch (true for HalfKP)
            num_features = stm_indices.shape[0] // 2
            values = torch.ones(num_features, dtype=torch.float32, device=device)

        return new_epoch, Batch(
            stm_indices, nstm_indices, values, score, outcome, batch_size
        )

    def drop(self) -> None:
        pass

    def __enter__(self) -> BatchLoader:
        return self

    def __exit__(self) -> None:
        self.drop()

