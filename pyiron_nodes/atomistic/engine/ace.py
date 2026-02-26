from core import as_function_node


@as_function_node
def Ace(potential_file, use_symmetry:bool=True):
    from pyace import PyACECalculator
    calc = PyACECalculator(potential_file)
    from logging import ERROR
    from pyiron_snippets.logger import logger

    from pyiron_nodes.atomistic.engine.generic import OutputEngine

    out = OutputEngine(calculator=calc)
    logger.setLevel(ERROR)
    return out