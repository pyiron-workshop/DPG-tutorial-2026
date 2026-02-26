from core import as_function_node

@as_function_node
def EAM(
    potential_name: str
):
    from logging import ERROR
    from pyiron_lammps.potential import get_potential_by_name
    from pyiron_nodes.atomistic.engine.generic import OutputEngine
    from pyiron_snippets.logger import logger
    
    potential_dataframe = get_potential_by_name(potential_name)
    
    out = OutputEngine(calculator=potential_dataframe)

    logger.setLevel(ERROR)
    return potential_dataframe