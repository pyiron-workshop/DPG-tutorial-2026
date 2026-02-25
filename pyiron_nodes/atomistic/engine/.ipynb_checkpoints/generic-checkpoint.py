from __future__ import annotations

from dataclasses import field, dataclass
from typing import Optional

def wf_data_class(*args, doc_func=None, **kwargs):
    """
    Extension of the python default dataclass to include methods and functionality needed for pyiron_core.pyiron_workflows

    :param args: pass to dataclass decorator
    :param doc_func: function from which to copy docstring
    # :param keys_to_store:
    :param kwargs: pass to dataclass decorator
    :return: dataclass like object with enhanced workflow features
    """

    def wrapper(cls):
        cls = dataclass(*args, **kwargs)(cls)

        # Add/modify a variable
        if doc_func is not None:
            cls.__doc__ = doc_func.__doc__

        # Add new methods
        def keys(self):
            return self.__dict__.keys()

        def items(self):
            return [(k, self[k]) for k in self.keys()]

        def __getitem__(self, key):
            return self.__dict__[key]

        def __setitem__(self, key, value):
            if key in self.keys():
                self.__dict__[key] = value

        def select(self, keys_to_store=None):
            if keys_to_store is None:
                keys_to_store = self.keys()  # cls._keys_to_store
            return {k: self[k] for k in keys_to_store}

        cls.keys = keys
        cls.items = items
        cls.__getitem__ = __getitem__
        cls.__setitem__ = __setitem__
        cls.select = select

        return cls

    return wrapper


@wf_data_class()
class wfMetaData:
    log_level: int = 0
    doc: Optional[str] = None


@wf_data_class()
class OutputEngine:
    calculator: Optional[callable] = field(
        default=None, metadata=wfMetaData(log_level=0)
    )
    engine_id: Optional[int] = field(default=None, metadata=wfMetaData(log_level=0))
    parameters: Optional[wf_data_class] = field(
        default=None, metadata=wfMetaData(log_level=10)
    )
    _do_not_serialize: bool = True
