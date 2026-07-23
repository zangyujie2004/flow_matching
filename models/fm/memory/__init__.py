"""Memory condition encoders for Flow Matching.

Layout
------
types.py       MemoryOutput (tokens, memory_global)
fusion.py      temporal visual pooling + Conv fusion → single token (N=1)
multi_scale.py FastQuery range/state queries → (N=num_queries+1)
factory.py     build_memory_encoder(method=...)
"""

from .factory import build_memory_encoder, normalize_memory_method
from .fusion import MemoryEncoder, StateConvMemoryEncoder, VisualTemporalMemoryEncoder
from .multi_scale import DepthwiseTemporalBlock, FastQueryMemoryEncoder, build_range_attn_mask
from .types import MemoryOutput

__all__ = [
    "MemoryOutput",
    "MemoryEncoder",
    "StateConvMemoryEncoder",
    "VisualTemporalMemoryEncoder",
    "DepthwiseTemporalBlock",
    "FastQueryMemoryEncoder",
    "build_range_attn_mask",
    "build_memory_encoder",
    "normalize_memory_method",
]
