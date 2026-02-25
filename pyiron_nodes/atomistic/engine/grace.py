from core import as_function_node
from functools import lru_cache


@as_function_node("engine")
@lru_cache
def GRACE(model: str = "GRACE-FS-OAM"):
    """Universal Graph Atomic Cluster Expansion models."""
    from tensorpotential.calculator import grace_fm

    from pyiron_nodes.atomistic.engine.generic import OutputEngine

    out = OutputEngine(calculator=grace_fm(model))
    return out