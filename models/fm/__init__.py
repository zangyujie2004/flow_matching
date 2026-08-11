"""Flow-matching exports for both upstream and namespaced LIP imports."""

from .latent_dit import LatentDiT, LatentDiTBlock
from .latent_flow_policy import LatentFlowMatchingPolicy

__all__ = ["LatentDiT", "LatentDiTBlock", "LatentFlowMatchingPolicy"]

# The upstream repository historically imports ``models.*`` as a top-level
# package.  Preserve that API when the vendor root is on sys.path, while still
# allowing LIP to import the new latent modules through its namespaced vendor.
try:
    from .condition_encoder import ConditionEncoder
    from .flow_policy import FlowMatchingPolicy, build_flow_policy
except ModuleNotFoundError as exc:
    if exc.name != "models":
        raise
else:
    __all__ += ["ConditionEncoder", "FlowMatchingPolicy", "build_flow_policy"]
