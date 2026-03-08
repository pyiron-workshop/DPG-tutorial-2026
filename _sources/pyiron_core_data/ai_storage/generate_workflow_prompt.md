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
    "module": "basic.loop",

The corresponding import statement is:

```python
from pyiron_nodes.basic.loop import IterToDataFrame
```

__Procedure__

1. Prefix the `module` value with `pyiron_nodes.` → `pyiron_nodes.basic.loop`.
2. Append the `name` value after `import`.
3. Verify that both parts match the metadata exactly (no extra spaces, trailing dots, or case changes).


Below is the minimal template you should follow when constructing a workflow. 

```python
from core import Workflow

# Import all required nodes
from pyiron_nodes.atomistic.structure.build import Bulk
from pyiron_nodes.atomistic.structure.transform import CreateVacancy

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
- Do **not** assume port labels that are not explicitly given in the metadata for a node (avoid errors such as: Invalid code - workflow creation failed: Input port "temperature_step" does not exist on node "InputClass")
- Double check that a port label that you want to use is explicitly listed in the node metadata. Be conservative, it is better to have a missing port label rather than a wrong one! 

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
from pyiron_nodes.atomistic.structure.build import Bulk
from pyiron_nodes.atomistic.structure.transform import CreateVacancy



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

## 5. Example Workflows 

### Free energy calculation of a solid unary phase
```python
from pyiron_nodes.atomistic.calculator.calphy import InputClass, PlotFreeEnergy, SolidFreeEnergyWithTemp
from pyiron_nodes.atomistic.engine.eam import EAM
from pyiron_nodes.atomistic.engine.lammps import ListPotentials
from pyiron_nodes.atomistic.structure.build import Bulk
from pyiron_nodes.atomistic.structure.transform import Repeat
from pyiron_nodes.basic.list import PickElement
from core import Workflow

wf = Workflow("example")

wf.BulkStructure = Bulk(name='Al', cubic=True)
wf.CalphyInputClass = InputClass()
wf.RepeatStructure = Repeat(structure=wf.BulkStructure.outputs.structure, repeat_scalar=3)
wf.ListPotentials = ListPotentials(structure=wf.BulkStructure.outputs.structure)
wf.PickElement = PickElement(lst=wf.ListPotentials.outputs.potentials, index=20)
wf.EAM = EAM(potential_name=wf.PickElement.outputs.element)
wf.SolidFreeEnergyWithTemp = SolidFreeEnergyWithTemp(input_class=wf.CalphyInputClass.outputs.output, structure=wf.RepeatStructure.outputs.structure, potential=wf.EAM.outputs.engine)
wf.PlotUnaryFreeEnergy = PlotFreeEnergy(temperature=wf.SolidFreeEnergyWithTemp.outputs.temperature, free_energy=wf.SolidFreeEnergyWithTemp.outputs.free_energy)
```

*Follow the guidelines above to construct any pyiron workflow while ensuring the generated code is clean, deterministic, and ready for automated extraction.*
