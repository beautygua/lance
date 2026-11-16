# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from dataclasses import dataclass


@dataclass(frozen=True)
class PcaParams:
    """Configuration for a PCA vector pre-transform.

    **Experimental.** This pre-transform may change incompatibly, in which
    case the index must be rebuilt. The source data is not rewritten.

    Parameters
    ----------
    dimension : int
        Number of dimensions produced by the PCA transform.
    sample_size : int, default 65536
        Target number of valid rows used to train PCA. Null rows and rows
        containing NaN or infinity are skipped and do not count toward this
        target. Training stops early when the dataset has fewer valid rows.
        This sample is separate from IVF centroid sampling.
    sampling_seed : int, default 42
        Seed used to sample vectors for PCA training.
    """

    dimension: int
    sample_size: int = 65536
    sampling_seed: int = 42
