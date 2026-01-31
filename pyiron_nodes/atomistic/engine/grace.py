from pyiron_core import as_function_node
from functools import lru_cache


@as_function_node
@lru_cache
def Grace(model: str = "GRACE-FS-OAM",  use_symmetry=True):
    """Universal Graph Atomic Cluster Expansion models."""
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    from tensorpotential.calculator import grace_fm
    grace_obj = grace_fm(model)
    return grace_obj