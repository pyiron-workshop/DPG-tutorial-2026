"""
Utility to scan a directory tree that contains pyiron‑workflow function nodes,
extract their signatures and metadata, and write a JSON catalog.

Typical usage:

    from generate_node_catalog import generate_node_catalog
    generate_node_catalog(
        pyiron_nodes_path="path/to/your/pyiron_nodes",
        output_file="pyiron_node_catalog.json",
    )
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional, Union, get_type_hints, Literal
import textwrap
import pathlib
import time

from dataclasses import dataclass
import json
import warnings

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from core import Workflow

_CONFIG_PATH = pathlib.Path.home() / ".pyiron_aiflow_config.json"

MODEL = "openai/gpt-oss-120b" # "gpt-oss:120b" for Ollama
API_KEY = "d03af2a76579d513f794529dc466e6e6"
API_BASE = "http://192.168.0.112:8000/v1"


@dataclass
class paths:
    AI_STORAGE: str = pathlib.Path.home() / "pyiron_core_data" / "ai_storage"

def _save_default_paths():
    """Add AI_STORAGE to the config if missing, without altering existing entries."""
    try:
        if _CONFIG_PATH.is_file():
            # Load existing config
            with open(_CONFIG_PATH) as f:
                data = json.load(f)
            # Only add AI_STORAGE if it does not already exist
            if "AI_STORAGE" not in data:
                data["AI_STORAGE"] = str(paths.AI_STORAGE)
                with open(_CONFIG_PATH, "w") as f:
                    json.dump(data, f, indent=2)
            # If AI_STORAGE is already present, leave the file unchanged
        else:
            # No config yet, create one with AI_STORAGE only
            with open(_CONFIG_PATH, "w") as f:
                json.dump({"AI_STORAGE": str(paths.AI_STORAGE)}, f, indent=2)
    except Exception as e:
        warnings.warn(f"Failed to write default paths config: {e}")

def _load_paths():
    """Load path values from the config file, creating it with defaults if missing."""
    if not _CONFIG_PATH.is_file():
        _save_default_paths()
        return
    try:
        with open(_CONFIG_PATH) as f:
            data = json.load(f)
        # Preserve existing AI_STORAGE entry; if missing, add it without affecting other keys.
        if "AI_STORAGE" in data:
            paths.AI_STORAGE = pathlib.Path(data["AI_STORAGE"])
        else:
            # Ensure AI_STORAGE is present in the config file.
            _save_default_paths()
    except Exception as e:
        warnings.warn(f"Failed to load paths config: {e}")

# Initialize paths from configuration on module import
_load_paths()

# Ensure AI_STORAGE directory exists and copy markdown prompt files into it if missing.
import shutil

def _ensure_ai_storage():
    """Create AI_STORAGE directory and copy markdown prompt files if they are not already present."""
    # Create the directory if it does not exist
    paths.AI_STORAGE.mkdir(parents=True, exist_ok=True)
    # Source directory: the directory containing this file
    src_dir = pathlib.Path(__file__).parent
    # List of markdown files expected by the module
    md_files = ["ask_llm_for_nodes.md", "generate_workflow_prompt.md"]
    for md in md_files:
        src_path = src_dir / md
        dst_path = paths.AI_STORAGE / md
        if src_path.is_file() and not dst_path.is_file():
            try:
                shutil.copy2(src_path, dst_path)
            except Exception as copy_err:
                warnings.warn(f"Failed to copy {md} to AI_STORAGE: {copy_err}")

# Run the sync routine
_ensure_ai_storage()


# ------------------------------------------------------------------
# Ensure the node catalog exists in AI_STORAGE
# ------------------------------------------------------------------
def _ensure_node_catalog():
    """Generate the node catalog JSON in AI_STORAGE if it does not already exist."""
    catalog_path = pathlib.Path(paths.AI_STORAGE) / "pyiron_node_catalog.json"
    if catalog_path.is_file():
        return
    try:
        import pyiron_nodes
        pyiron_nodes_path = pyiron_nodes.__path__[0]
        # Generate the catalog; do not overwrite if it somehow appears concurrently.
        generate_node_catalog(
            pyiron_nodes_path=pyiron_nodes_path,
            output_file=catalog_path,
            update=False,
        )
    except Exception as e:
        warnings.warn(f"Failed to generate node catalog in AI_STORAGE: {e}")

# Call the routine after the module has been fully defined (post‑definition hook)
def _post_module_init():
    # This function is invoked at the end of the file to perform any lazy
    # initialisation that depends on functions defined later.
    _ensure_node_catalog()


# Initialise AI module on import so that the AI_STORAGE directory and node
# catalog are created automatically.
_post_module_init()


# Load prompt templates from separate Markdown files located in AI_STORAGE.
_PROMPT_FILES = {
    "ask_llm_for_nodes": pathlib.Path(str(paths.AI_STORAGE)).joinpath("ask_llm_for_nodes.md"),
    "generate_workflow_prompt": pathlib.Path(str(paths.AI_STORAGE)).joinpath("generate_workflow_prompt.md"),
}

def _load_prompt(file_path: pathlib.Path) -> str:
    """Read a Markdown file and return its raw text."""
    return file_path.read_text(encoding="utf-8")

# Build the prompts dictionary mapping names to their contents.
_PROMPTS = {
    name: _load_prompt(path) for name, path in _PROMPT_FILES.items()
}

# from node_store import get_nodes_for_task, _ollama_generate


# ------------------------------------------------------------------
# Helper to turn a Python type into a JSON‑friendly string
# ------------------------------------------------------------------
def _type_to_str(tp: Any) -> str:
    """Return a readable representation for a type hint."""
    if hasattr(tp, "__origin__"):
        # Handle Optional[Literal[...]] etc.
        origin = tp.__origin__
        if hasattr(tp, "__args__"):
            args = tp.__args__
        else:
            print("_type_to_str failed: ", tp, origin)
            return str(tp)
        if origin is Union and type(None) in args:  # Optional
            inner = [a for a in args if a is not type(None)][0]
            return f"Optional[{_type_to_str(inner)}]"
        if origin is Literal:
            literals = ", ".join(repr(l) for l in args)
            return f"Literal[{literals}]"
        # Fallback for generic containers (e.g. List[int], Tuple[float, …])
        inner = ", ".join(_type_to_str(a) for a in args)
        return f"{origin.__name__}[{inner}]"
    elif isinstance(tp, type):
        return tp.__name__
    else:
        return str(tp)


# ------------------------------------------------------------
# 2️⃣  Helper: make a default value JSON‑serialisable
# ------------------------------------------------------------
def _json_compatible_default(value: Any) -> Any:
    """
    Convert a possibly‑complex default value (e.g. a pyiron Node object)
    into something that ``json.dump`` can serialise.

    Rules
    -----
    * Primitive JSON types (str, int, float, bool, None) are returned unchanged.
    * Lists, tuples and dicts are processed recursively.
    * ``Node`` instances are replaced by a minimal descriptor:
          {"_node_ref": "<node_label>", "type": "Node"}
    * ``dataclass`` instances are turned into plain dicts via ``asdict``.
    * Anything else falls back to ``repr(value)``.
    """
    # Primitive JSON types
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    # Containers – recurse
    if isinstance(value, (list, tuple)):
        return [_json_compatible_default(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_compatible_default(v) for k, v in value.items()}

    # Try to recognise a pyiron Node object
    try:
        # Import lazily to avoid circular imports
        from core import Node  # type: ignore

        if isinstance(value, Node):
            # Use the node's label (or a placeholder) as a readable reference
            label = getattr(value, "label", "<unnamed>")
            return {"_node_ref": label, "type": "Node"}
    except Exception:
        pass  # Import failed or not a Node – continue

    # Dataclass instances – turn into dicts
    if hasattr(value, "__dataclass_fields__"):
        try:
            from dataclasses import asdict

            return asdict(value)
        except Exception:
            pass

    # Fallback – string representation (always JSON‑serialisable)
    return repr(value)


# ------------------------------------------------------------------
# Extract metadata from a single function node
# ------------------------------------------------------------------
def _node_info(wrapped_func: Callable, root_path: Path, debug: bool = False) -> dict:
    """Collect the information we need from a function node.
    Includes source code of the original function and module path relative to the pyiron_nodes root.
    """
    if debug:
        print(f"Extracting metadata for function node: {wrapped_func}")
    try:
        func = inspect.unwrap(wrapped_func().func)
    except Exception as e:
        print("exception: ", e)
        func = inspect.unwrap(wrapped_func)
    sig = inspect.signature(func)
    hints = get_type_hints(func)

    # Source code of the original (non‑decorated) function
    try:
        source_code = inspect.getsource(func)
    except Exception:
        source_code = ""

    # Compute module path relative to the provided root directory
    try:
        source_file = inspect.getsourcefile(func)
        if source_file:
            rel_path = Path(source_file).resolve().relative_to(root_path.resolve())
            module_rel = ".".join(rel_path.with_suffix("").parts)
        else:
            module_rel = func._module_path   # func.__module__
    except Exception:
        module_rel = func._module_path   # func.__module__

    # ---------- Gather raw input metadata (including the *original* defaults) ----------
    raw_inputs: List[dict] = []
    for name, param in sig.parameters.items():
        ann = hints.get(name, Any)
        default = (
            param.default if param.default is not inspect.Parameter.empty else None
        )
        # Detect Optional[Literal[...]] → expose the literal choices
        choices: Optional[List[Any]] = None
        if getattr(ann, "__origin__", None) is Union:
            literal_args = [
                a for a in ann.__args__ if getattr(a, "__origin__", None) is Literal
            ]
            if literal_args:
                lit = literal_args[0]
                choices = [c for c in lit.__args__ if c is not type(None)]
        raw_inputs.append(
            {
                "name": name,
                "type": _type_to_str(ann),
                "default": default,
                "optional": default is not None,
                "choices": choices,
            }
        )

    # ---------- Make the defaults JSON‑compatible **without** losing the original ----------
    json_inputs: List[dict] = []
    for inp in raw_inputs:
        json_inputs.append(
            {
                "name": inp["name"],
                "type": inp["type"],
                "default": _json_compatible_default(inp["default"]),
                "optional": inp["optional"],
                "choices": inp["choices"],
            }
        )

    # ---------- Output type handling (unchanged) ----------
    out_ann = hints.get("return", Any)
    outputs = [{"type": _type_to_str(out_ann)}]
    # print("outlabels: ", func.__name__, module_rel)
    try:
        output_labels = get_output_ports(
            func.__name__, module_rel, pyiron_nodes="pyiron_nodes"
        )
    except Exception:
        # TODO: ensire creation of macro istances when no input default exists
        print(f"⚠️  Could not get output ports for {func.__name__} in {module_rel}")
        output_labels = []

    # ---------- Build the full return dictionary ----------
    return {
        "name": func.__name__,
        "module": module_rel,
        "description": inspect.getdoc(func) or "",
        "inputs": json_inputs,
        "outputs": outputs,
        "output_labels": output_labels,
        "tags": getattr(func, "_node_tags", []),
        "source": source_code,
        "example": f"{func.__name__}(" + ", ".join(f"{i['name']}={i['default']!r}" for i in raw_inputs if i['default'] is not None) + ")",
    }


# ------------------------------------------------------------------
# Import a module from an arbitrary file path
# ------------------------------------------------------------------
def _import_module_from_path(file_path: Path, base_pkg: str = "") -> Any:
    """
    Import a Python file as a module.

    Parameters
    ----------
    file_path : Path
        Absolute path to the ``.py`` file.
    base_pkg : str, optional
        If you want the imported module to appear under a pseudo‑package
        (e.g. ``my_pkg.submodule``), pass the dotted name here.
        The function will prepend it to the generated module name.

    Returns
    -------
    module
        The imported module object.
    """
    # Build a unique module name – we use the relative path with dots.
    rel_path = file_path.with_suffix("")  # drop .py
    parts = rel_path.parts
    if base_pkg:
        parts = (base_pkg, *parts)
    module_name = ".".join(parts)

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module


# ------------------------------------------------------------------
# Walk a directory tree and yield all callables that have been
# decorated with @as_function_node.
# ------------------------------------------------------------------
def _walk_directory(root_dir: Path, debug: bool = False) -> List[Callable]:
    """
    Recursively import every ``.py`` file under *root_dir* and collect
    objects that have the ``_is_function_node`` attribute set to ``True``.

    Returns
    -------
    List[Callable]
        All discovered function‑node callables.
    """
    nodes: List[Callable] = []

    # Ensure the root directory is on ``sys.path`` so that relative imports
    # inside the discovered modules keep working.
    root_str = str(root_dir.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    for py_file in root_dir.rglob("*.py"):
        # Skip hidden files and typical “private” modules (e.g. __init__.py)
        if py_file.name.startswith("_"):
            continue

        try:
            module = _import_module_from_path(py_file, base_pkg="")
        except Exception as exc:  # pragma: no cover
            # We do not want a single broken file to abort the whole scan.
            print(f"⚠️  Could not import {py_file}: {exc}", file=sys.stderr)
            continue

        module_path = module.__file__.split("pyiron_nodes/")[-1].split(".")[0].replace("/", ".")
        for name, obj in vars(module).items():
            # print(f"Discovered function node: {name}", callable(obj), hasattr(obj, "node_type"), type(obj))
            if callable(obj) and hasattr(obj, "node_type"):
                if debug:
                    print(f"✅  Found function node: {name}")
                obj._module_path = module_path
                nodes.append(obj)

    return nodes


from core.node import get_node_from_path


def get_output_ports(node_name: str, module: str, pyiron_nodes="pyiron_nodes") -> list:
    """
    Retrieve the output ports for a specific node from the Chroma collection.

    Returns a list of output port names.
    """
    node = get_node_from_path(f"{pyiron_nodes}.{module}.{node_name}")
    output_labels = [v.label for v in node.outputs]
    # print("get_outputs: ", output_labels)
    return output_labels


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------
def generate_node_catalog(
    pyiron_nodes_path: Union[str, Path],
    output_file: Union[str, Path] = None,
    update: bool = False,
    debug: bool = False,
    start_dir: str = "."
) -> int:
    """
    Scan a directory tree that contains pyiron‑workflow function nodes,
    build a JSON‑serialisable catalog and write it to *output_file*.

    Parameters
    ----------
    pyiron_nodes_path : str or Path
        Path to the root directory that holds the ``.py`` files with the
        ``@as_function_node``‑decorated functions.  The directory may
        contain sub‑directories; they are processed recursively.
    output_file : str or Path, optional
        Destination JSON file.  The parent directory will be created if it
        does not exist.  Default: ``pyiron_node_catalog.json``.
    update : bool, optional
        If ``True``, an existing ``output_file`` will be overwritten with the
        newly generated catalog.  If ``False`` (default) and the file already
        exists, a ``FileExistsError`` is raised to avoid accidental data loss.

    Returns
    -------
    int
        Number of node entries written to the JSON file.

    Example
    -------
    >>> from generate_node_catalog import generate_node_catalog
    >>> n = generate_node_catalog("src/pyiron_nodes", "catalog.json", update=True)
    >>> print(f"catalog contains {n} nodes")
    """
    root = Path(pyiron_nodes_path).expanduser().resolve() / start_dir
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")

    # Determine default output location based on config file presence
    if output_file is None:
        if _CONFIG_PATH.is_file():
            output_file = pathlib.Path(paths.AI_STORAGE) / "pyiron_node_catalog.json"
        else:
            output_file = _CONFIG_PATH / "pyiron_node_catalog.json"
    out_path = Path(output_file).expanduser().resolve()
    # If the file already exists and we are not allowed to overwrite, abort early.
    if out_path.is_file():
        if update:
            print(f"⚠️  Overwriting existing file {out_path}")
            import os

            os.remove(out_path)
        else:
            print("existing file will be used as is")
            return 0

            # raise FileExistsError(f"{out_path} already exists. Use update=True to overwrite.")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ------️⃣  Discover all function nodes
    node_funcs = _walk_directory(root, debug=debug)
    print(f"🔍 Found {len(node_funcs)} function nodes in {root}")

    # ------️⃣  Build the catalog (list of dicts)
    if start_dir == ".":
        path_root = root
    else:
        path_root = root.parent

    catalog = [_node_info(fn, path_root, debug=debug) for fn in node_funcs]
    print(f"🗂  Built catalog with {len(catalog)} entries")

    # ------🔧  Write JSON – with better error reporting
    with out_path.open("w", encoding="utf-8") as f:
        try:
            json.dump(catalog, f, indent=2, ensure_ascii=False)
        except TypeError as exc:
            # Find the first entry that cannot be JSON‑encoded
            for idx, entry in enumerate(catalog):
                try:
                    json.dumps(entry, ensure_ascii=False)  # test serialisation
                except TypeError as inner_exc:
                    node_name = entry.get("name", f"<unknown #{idx}>")
                    module = entry.get("module", "<unknown module>")
                    print(
                        f"❌  JSON‑serialization failed for node "
                        f"'{node_name}' (module: {module}) – {inner_exc}",
                        file=sys.stderr,
                    )
                    raise TypeError(
                        f"Node '{node_name}' (module: {module}) is not JSON serializable: "
                        f"{inner_exc}"
                    ) from exc
            raise

    print(f"✅ Wrote {len(catalog)} node entries to {out_path}")
    return len(catalog)


def transfer_json_to_chroma(
    json_path: str, chroma_client, collection_name: str = "node_catalog"
):
    """Transfer the node catalog JSON into a Chroma collection.

    Args:
        json_path: Path to the JSON file containing the node catalog.
        chroma_client: An instance of `chromadb.Client`.
        collection_name: Name of the Chroma collection to create or use.
    """
    import json
    from pathlib import Path

    # Load JSON data
    with open(Path(json_path), "r", encoding="utf-8") as f:
        data = json.load(f)

    # Ensure data is a list of node entries
    if isinstance(data, dict):
        # If the JSON is a dict with a top‑level key, try to extract the list
        # of nodes assuming a common key like "nodes" or "catalog".
        for key in ("nodes", "catalog", "items"):
            if key in data and isinstance(data[key], list):
                nodes = data[key]
                break
        else:
            # Fallback: treat the dict values as node entries
            nodes = list(data.values())
    else:
        nodes = data

    # Prepare documents and metadata for Chroma
    documents = []
    metadatas = []
    ids = []
    for i, node in enumerate(nodes):
        # Expect each node to have a textual description field; fall back to str(node)
        description = (
            node.get("description", str(node)) if isinstance(node, dict) else str(node)
        )
        documents.append(description)
        # Store the whole node as metadata (JSON‑serialisable)
        metadatas.append(node if isinstance(node, dict) else {"value": node})
        ids.append(str(i))

    # Create or get the collection
    if collection_name in chroma_client.list_collections():
        collection = chroma_client.get_collection(name=collection_name)
    else:
        collection = chroma_client.create_collection(name=collection_name)

    # Add entries to the collection
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )
    return collection


def transfer_node_catalog_to_chroma(
    json_path: str, chroma_client, collection_name: str = "node_catalog"
):
    """Load a node catalog from a JSON file and store it in a Chroma collection.

    Parameters
    ----------
    json_path: str
        Path to the JSON file containing the node catalog.
    chroma_client: Any
        An initialized Chroma client (e.g., ``chromadb.Client``).
    collection_name: str, optional
        Name of the collection to create or update. Defaults to ``"node_catalog"``.
    """
    import json
    from pathlib import Path

    # Load the JSON file
    if not Path(json_path).exists():
        import pyiron_nodes

        pyiron_nodes_path = pyiron_nodes.__path__[0]
        generate_node_catalog(
            pyiron_nodes_path=pyiron_nodes_path,
            output_file=json_path,
            update=True,
        )

    with open(Path(json_path), "r", encoding="utf-8") as f:
        data = json.load(f)

    # Normalise the data to a list of node dictionaries
    if isinstance(data, dict):
        # Try common container keys, otherwise use the dict values
        for key in ("nodes", "catalog", "items"):
            if key in data and isinstance(data[key], list):
                nodes = data[key]
                break
        else:
            nodes = list(data.values())
    else:
        nodes = data

    # Prepare the fields required by Chroma
    ids = []
    documents = []
    metadatas = []
    for idx, node in enumerate(nodes):
        # Ensure each node is a dict
        if not isinstance(node, dict):
            node = {"value": node}
        # Use an explicit id if present, otherwise generate one
        node_id = str(node.get("id", idx))
        ids.append(node_id)
        # Choose a textual field for the document – fall back to the whole dict as string
        doc = (
            node.get("description")
            or node.get("content")
            or json.dumps(node, ensure_ascii=False)
        )
        documents.append(doc)
        # Store the remaining fields as metadata (excluding the large text field)
        raw_meta = {
            k: v for k, v in node.items() if k not in ("description", "content", "id")
        }
        # Convert any non‑primitive values (list, dict) to JSON strings to satisfy Chroma's metadata requirements
        meta = {}
        for k, v in raw_meta.items():
            if isinstance(v, (list, dict)):
                meta[k] = json.dumps(v, ensure_ascii=False)
            else:
                meta[k] = v
        metadatas.append(meta)

    # Get or create the collection
    if collection_name in [c.name for c in chroma_client.list_collections()]:
        collection = chroma_client.get_collection(name=collection_name)
    else:
        collection = chroma_client.create_collection(name=collection_name)

    # Add the entries to the collection
    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return collection


def get_node_catalog_collection(
    json_path: str,
    chroma_client,
    collection_name: str = "node_catalog",
    update: bool = False,
):
    """
    Load the node catalog JSON and return the Chroma collection.

    This function now resolves ``json_path`` relative to ``paths.AI_STORAGE`` if
    a relative path is given, ensuring all AI‑related files are stored under the
    dedicated AI storage directory.
    """
    # Resolve the catalog path – if a relative path is provided, place it in AI_STORAGE.
    json_path_obj = Path(json_path)
    if not json_path_obj.is_absolute():
        json_path_obj = pathlib.Path(paths.AI_STORAGE) / json_path_obj
    # Update the variable used later.
    json_path = str(json_path_obj)
    print("json_path: ", json_path)

    # Check whether the collection already exists
    existing_names = [c.name for c in chroma_client.list_collections()]

    if collection_name in existing_names:
        if not update:
            # Return the already‑populated collection without touching it
            return chroma_client.get_collection(name=collection_name)
        else:
            # Remove the old collection so we can recreate it fresh
            try:
                chroma_client.delete_collection(name=collection_name)
            except Exception as exc:  # pragma: no cover
                # If the client does not support delete_collection, fall back to get and clear
                print(
                    f"⚠️  Could not delete collection '{collection_name}': {exc}",
                    file=sys.stderr,
                )
                # Attempt to get the collection and remove all entries
                coll = chroma_client.get_collection(name=collection_name)
                coll.delete(ids=coll.get()["ids"])  # type: ignore[attr-defined]

    # (Re)create and populate the collection from the JSON file

    collection = transfer_node_catalog_to_chroma(
        json_path, chroma_client, collection_name
    )
    return collection


def _list_available_models(
    *,
    api_key: str | None = None,
    api_base: str | None = "https://chat-ai.academiccloud.de/v1",
) -> list[str]:
    """
    Return a list of model identifiers that the provider exposes.

    The function uses the ``/models`` endpoint (the same call the official
    OpenAI Python client uses).  It works for any OpenAI‑compatible service.
    """
    try:
        import openai
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install `openai>=1.0`") from exc

    client_kwargs = {}
    if api_key is not None:
        client_kwargs["api_key"] = api_key
    if api_base is not None:
        client_kwargs["base_url"] = api_base.rstrip("/")

    client = openai.OpenAI(**client_kwargs)

    # The SDK returns a ``ModelList`` object; we only need the ``id`` field.
    try:
        model_list = client.models.list()
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to fetch model list: {exc}") from exc

    return [m.id for m in model_list.data]


def _openai_generate(
    model: str,
    prompt: str,
    max_tokens: int,
    *,
    stream: bool = False,
    temperature: float = 0.0,
    api_key: Optional[str] = API_KEY,
    api_base: Optional[str] = API_BASE,
    debug: bool = False,
) -> str:
    """
    Generate text using an OpenAI‑compatible provider (OpenAI ≥ 1.0 SDK).

    The function is tolerant to a few different response schemas:
      * ``choices[0].message.content``          – normal chat response
      * ``choices[0].message.reasoning``       – provider‑specific field
      * ``choices[0].message.reasoning_content`` – provider‑specific field
      * ``choices[0].text``                    – classic completion endpoint

    Parameters
    ----------
    model, prompt, max_tokens, stream, temperature – same as before.
    api_key   – optional API key (falls back to ``OPENAI_API_KEY``).
    api_base  – base URL of the provider (default is your academic‑cloud URL).
    debug     – if ``True`` prints the raw JSON payload for each request/stream.

    Returns
    -------
    str
        The generated text.  Raises ``RuntimeError`` if the text cannot be
        located.
    """
    # ------------------------------------------------------------------
    # 1️⃣  Import the new SDK (lazy import for a nicer error message)
    # ------------------------------------------------------------------
    try:
        import openai  # version >= 1.0
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The `openai` package (>= 1.0) is required. Install it with "
            "`pip install openai`."
        ) from exc

    # ------------------------------------------------------------------
    # 2️⃣  Build the client (thread‑safe, no global mutation)
    # ------------------------------------------------------------------
    client_kwargs: dict[str, Any] = {}
    if api_key is not None:
        client_kwargs["api_key"] = api_key
    if api_base is not None:
        client_kwargs["base_url"] = api_base.rstrip("/")   # no trailing slash

    client = openai.OpenAI(**client_kwargs)

    # ------------------------------------------------------------------
    # 3️⃣  Payload – we use the chat endpoint (single‑turn)
    # ------------------------------------------------------------------
    messages = [{"role": "user", "content": prompt}]

    # ------------------------------------------------------------------
    # 4️⃣  Helper to pull the text out of a response object
    # ------------------------------------------------------------------
    def _extract_text_from_choice(choice: Any) -> Optional[str]:
        """
        Try the known locations in order of preference.
        Returns ``None`` if nothing is found.
        """
        # 1️⃣  chat‑style: message.content
        try:
            content = choice.message.content
            if content:
                return content
        except Exception:
            pass

        # 2️⃣  provider‑specific: reasoning / reasoning_content
        try:
            # ``choice.message`` may be a pydantic model or a dict‑like object.
            msg = choice.message
            # attribute access first
            reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
            if reasoning:
                return reasoning
            # fallback to dict‑style access
            if isinstance(msg, dict):
                for key in ("reasoning", "reasoning_content"):
                    if msg.get(key):
                        return msg[key]
        except Exception:
            pass

        # 3️⃣  classic completion endpoint: text
        try:
            text = choice.text
            if text:
                return text
        except Exception:
            pass

        return None

    # ------------------------------------------------------------------
    # 5️⃣  Streaming path
    # ------------------------------------------------------------------
    if stream:
        generated_parts: List[str] = []

        try:
            chunk_iter = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"OpenAI request failed: {exc}") from exc

        for chunk in chunk_iter:
            if debug:
                print("🔹 streaming chunk →", json.dumps(chunk.model_dump(), ensure_ascii=False))

            # Each chunk is a ``ChatCompletionChunk`` with ``choices`` list.
            try:
                delta = chunk.choices[0].delta
                # ``delta`` may contain only a ``role`` field on the first chunk.
                # The actual text (or reasoning) lives in ``delta.content`` or
                # ``delta.reasoning``.
                # We reuse the same extraction helper on the delta object.
                part = _extract_text_from_choice(delta)
                if part:
                    generated_parts.append(part)
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    f"Unexpected streaming chunk format: {json.dumps(chunk.model_dump(), ensure_ascii=False)}"
                ) from exc

        return "".join(generated_parts)

    # ------------------------------------------------------------------
    # 6️⃣  Non‑streaming path
    # ------------------------------------------------------------------
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"OpenAI request failed: {exc}") from exc

    if debug:
        print("🔹 full response →", json.dumps(resp.model_dump(), ensure_ascii=False))

    # The response contains a ``choices`` list – we look at the first one.
    try:
        first_choice = resp.choices[0]
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Response does not contain a `choices` array.") from exc

    text = _extract_text_from_choice(first_choice)
    if text is not None:
        return text

    # ------------------------------------------------------------------
    # 7️⃣  If we get here we really don’t know where the text lives.
    # ------------------------------------------------------------------
    raise RuntimeError(
        "Could not locate generated text in the provider's response. "
        "Enable `debug=True` to see the raw payload and adjust the extractor."
    )


def _ollama_generate(
    model: str,
    prompt: str,
    max_tokens: int,
    *,
    stream: bool = True,
    temperature: float = 0,
) -> str:
    """
    Call the local Ollama server and return the generated text.

    Parameters
    ----------
    model : str
        Model name, e.g. ``"gpt-oss:120b"``.
    prompt : str
        Prompt to send to the model.
    max_tokens : int
        Maximum number of tokens to generate.
    stream : bool, optional
        If ``True`` (default) the endpoint returns a stream of JSON lines.
        If ``False`` a single JSON object is returned.

    Returns
    -------
    str
        The concatenated generated text.
    """
    import json, requests

    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }

    resp = requests.post(
        "http://localhost:11434/api/generate",
        json=payload,
        timeout=120,  # avoid hanging notebooks
        stream=stream,  # let ``requests`` keep the connection open if streaming
    )
    resp.raise_for_status()

    if stream:
        # Ollama streams a sequence of JSON objects, one per line.
        # We collect the ``response`` field from each chunk.
        generated_parts = []
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue  # skip empty keep‑alive lines
            try:
                chunk = json.loads(line)
                generated_parts.append(chunk.get("response", ""))
            except json.JSONDecodeError:
                # If a line cannot be parsed, raise a clear error.
                raise RuntimeError(f"Failed to decode Ollama stream line: {line!r}")
        return "".join(generated_parts)
    else:
        # Non‑streaming mode – a single JSON document.
        data = resp.json()
        return data.get("response", "")


def ask_llm_for_nodes(
    task_description: str, model: str = MODEL, max_tokens: int = 5000
) -> str:
    """Return the raw description string produced by the LLM."""
    prompt = _PROMPTS["ask_llm_for_nodes"].format(task_description=task_description)
    raw = _openai_generate(model=model, prompt=prompt, max_tokens=max_tokens).strip()
    print("LLM raw response:", raw)
    return raw


# --------------------------------------------------------------
# 1️⃣  Helper to decode JSON‑encoded fields (add once, near the top)
# --------------------------------------------------------------


def _decode_meta_fields(meta: dict) -> dict:
    """Convert JSON‑encoded string fields (inputs, outputs, etc.) to Python objects."""
    for key in ("inputs", "outputs", "parameters", "description"):
        val = meta.get(key)
        if isinstance(val, str):
            try:
                meta[key] = json.loads(val)
            except json.JSONDecodeError:
                # keep the original string if it is not valid JSON
                pass
    return meta


def get_nodes_for_task(task, collection, num_nodes_per_task=2):
    # 2️⃣ Ask the LLM which nodes are needed (raw string)
    raw_node_output = ask_llm_for_nodes(task)

    # 3️⃣ Parse it into a list of node blocks + optional workflow description
    # node_descriptions, typical_workflow = split_llm_output(raw_node_output)
    nodes, typical_workflow = raw_node_output.split("Typical Workflow:")
    node_descriptions = nodes.split("Node Name:")[1:]

    # print("LLM suggested nodes (list):", node_descriptions)
    # print("\nTypical Workflow excerpt:\n", typical_workflow)

    nodes_map = {}
    for node in node_descriptions:
        node_task_name = node.split()[0]
        # query the collection with the *raw* LLM description
        results = collection.query(
            query_texts=node, n_results=10, include=["metadatas", "distances"]
        )
        # `results["metadatas"][0]` is a list of candidate metadata dicts.
        # We pick the *first* candidate that matches the node name.
        for i in range(num_nodes_per_task):
            candidate_meta = results["metadatas"][0][i]  # first dict in the list
            candidate_meta = _decode_meta_fields(candidate_meta)

            node_name = candidate_meta.get("name")
            nodes_map[node_name] = candidate_meta  # store a **single** dict, not a list
            print("✔️  Added node:", node_name)

    return nodes_map


def generate_workflow_prompt(task, node_meta_data) -> str:
    """Generate workflow prompt from markdown template."""
    template = _PROMPTS["generate_workflow_prompt"]
    meta_json = json.dumps(node_meta_data, indent=2)
    return template.format(task=task, node_meta_data=meta_json)
   

from dataclasses import dataclass
from typing import Any, Optional
import re


@dataclass
class GenOutput:
    raw: Optional[str] = None    # raw LLM output   
    workflow: Any = None         # pyiron workflow
    gui: Any = None              # gui widget
    prompt: Optional[str] = ""   # input prompt to LLM


def extract_code(text):
    """Extract Python code from triple backticks, or return full text if no blocks."""
    match = re.search(r"```python(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match2 = re.search(r"```(.*?)```", text, re.DOTALL)  # without python hint
    if match2:
        return match2.group(1).strip()
    return text.strip()


def generate_workflow(
    task: str,
    collection,
    max_tokens: int = 1000,
    temperature: float = 0,
    model: str = MODEL,
    print_prompt: bool = False,
    prompt_only: bool = False,
    as_gui: bool = True,
    debug: bool = False,
    num_nodes_per_task=2
):

    node_meta_data = get_nodes_for_task(task, collection, num_nodes_per_task=num_nodes_per_task)
    prompt = generate_workflow_prompt(task, node_meta_data)

    if print_prompt:
        print(prompt)

    out = GenOutput()
    out.prompt = prompt
    if prompt_only:
        return out

    # Generate the workflow using the LLM
    out.raw = _openai_generate(
        model=model, prompt=prompt, max_tokens=max_tokens, temperature=temperature
    ).strip()

    local_ns: dict = {}
    if debug:
        print(out.raw)

    out.error = None
    try:
        exec(extract_code(out.raw), {}, local_ns)
    except Exception as e:
        print("LLM generated code fails: ", e)
        out.error = e
        out.wf = None
        return out

    out.wf = local_ns["wf"]

    if as_gui:
        from core.gui import PyironFlow

        out.gui = PyironFlow([out.wf]).gui
        return out

    return out

def handle_ai_send(widget):
    """Delegate AI handling from the GUI to this module."""
    # Clear the AI output window
    widget.ai_output.clear_output()
    if widget.node_collection is None:
        import chromadb

        client = chromadb.Client()
        widget.node_collection = get_node_catalog_collection(
            "pyiron_node_catalog.json", chroma_client=client
        )

    prompt = widget.ai_input.value

    # Display prompt and indicate processing (content persists across sends)
    with widget.ai_output:
        print("\n---")
        print(f"User prompt: {prompt}")
        print("Generating answer...")

    # Record start time and generate workflow
    start_time = time.time()
    with widget.ai_output:
        out = generate_workflow(collection=widget.node_collection, task=prompt, max_tokens=30000, debug=True)
    elapsed = time.time() - start_time

    # Show answer with runtime (does not clear previous messages)
    with widget.ai_output:
        print(f"AI answer (took {elapsed:.2f}s): {getattr(out, 'raw', out)}")

    # If AI generated a workflow, load it into the current tab
    if hasattr(out, "wf") and out.wf is not None:
        # Replace current graph with the generated workflow
        if isinstance(out.wf, Workflow):
            widget.graph = out.wf._graph.copy()
        elif isinstance(out.wf, Graph):
            widget.graph = out.wf.copy()
        else:
            widget.graph = out.wf
        # Update tab title to reflect the new workflow label (if available)
        new_label = getattr(out.wf, "label", getattr(widget.graph, "label", None))
        if new_label is not None:
            widget.main_widget.tab_widget.set_title(
                widget.main_widget.tab_widget.selected_index, new_label
            )
        # Refresh the GUI to reflect the new workflow
        widget.update_gui(export_data=True, sleep_time=0.5)

    with widget.ai_output:
        if out.error is not None:
            print("Invalid code - workflow creation failed:", out.error)
