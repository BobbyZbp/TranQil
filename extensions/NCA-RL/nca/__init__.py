"""NCA-T core library.

Paper: Necessity-aware Credit Assignment for Offline-to-Online RL (ICML 2026
O2O workshop). See `paper/ncat.tex`.
"""
__version__ = "0.0.1"

from .kernels import PolicySupportedKernel
from .necessity import (
    NecessityHead,
    NecessityEnsemble,
    compute_offline_proxy,
    rank_normalise,
)
from .droptest import DropTestRecord, DropTestResult, paired_rollout

__all__ = [
    "PolicySupportedKernel",
    "NecessityHead",
    "NecessityEnsemble",
    "compute_offline_proxy",
    "rank_normalise",
    "DropTestRecord",
    "DropTestResult",
    "paired_rollout",
]
