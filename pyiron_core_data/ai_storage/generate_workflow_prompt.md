# PyIron Workflow Creation Guide  

This document explains how to write a **pyiron** workflow for a given task using specified nodes. It is formatted for both human readers and LLMs, with clear sections and properly fenced code blocks.

---  

## 1. Task & Node Metadata  

- **Task**: `{task}`  
- **Node metadata**: `{node_meta_data}`  

*(Replace the placeholders with the actual task description and node parameters.)*  

---  

## 2. Basic Workflow Template  

Construct the import path of a node from the metadata:
    "name": "IterToDataFrame",
    "module": "dpg2026.basic.loop",

The corresponding import statement is:

```python
from pyiron_nodes.dpg2026.basic.loop import IterToDataFrame
```

__Procedure__

1. Prefix the `module` value with `pyiron_nodes.` → `pyiron_nodes.dpg2026.basic.loop`.
2. Append the `name` value after `import`.
3. Verify that both parts match the metadata exactly (no extra spaces, trailing dots, or case changes).


Below is the minimal template you should follow when constructing a workflow. 

```python
from core import Workflow

# Import all required nodes
from pyiron_nodes.dpg2026.atomistic.structure.build import Bulk
from pyiron_nodes.dpg2026.atomistic.structure.transform import CreateVacancy

# Create the workflow instance
wf = Workflow("myWorkflow")          # or Workflow(label="myWorkflow")

# Add nodes to the workflow
wf.bulk = Bulk(name="Al", cubic=True)
wf.vacancy = CreateVacancy(
    structure=wf.bulk.outputs.structure,
    index=0,
)
```

**Important:**  
- Do **not** create a `Project` object (e.g., `proj = Project("...")`).  
- Only import nodes that are explicitly listed in the metadata.  
- Do **not** write nodes, e.g. by using @as_function_node 

---  



## 3. Goal – Output Only the Python Source Code  

When you generate the final workflow (or custom node), wrap **all** Python code in a single fenced block:

\`\`\`python  
# ... complete, runnable code ...  
\`\`\`

- No extra text, headings, or stray back‑ticks should appear **inside** the fenced block.  
- The block must be syntactically correct so that an extraction utility can retrieve it without modification.

---  

## 4. Example: Full Workflow  

```python
from core import Workflow, as_function_node
from pyiron_nodes.dpg2026.atomistic.structure.build import Bulk
from pyiron_nodes.dpg2026.atomistic.structure.transform import CreateVacancy



# ----------------------------------------------------------------------
# Workflow definition
# ----------------------------------------------------------------------
wf = Workflow("myWorkflow")

# Node 1: Build bulk structure
wf.bulk = Bulk(name="Al", cubic=True)


# Node 2: Create a vacancy using the (possibly) modified structure
wf.vacancy = CreateVacancy(
    structure=wf.bulk.outputs.structure,
    index=0,
)

# End of workflow
```

---  

*Follow the guidelines above to construct any pyiron workflow while ensuring the generated code is clean, deterministic, and ready for automated extraction.*