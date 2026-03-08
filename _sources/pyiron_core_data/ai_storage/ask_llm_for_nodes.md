You are an expert in pyiron node‑based workflows.  
Given the following scientific task description, write a concise but detailed description of the node(s) required to accomplish the task.

For each node you describe, start a new section with the marker  

Node Name: <node name>  

In that section include:

- Purpose: a brief statement of what the node does.  
- Required Input Ports: list each needed input with its data type. Since the exact port names are not known at this stage, use descriptive placeholders (e.g., “input_structure”, “input_parameters”, etc.).  
- Output Ports: list each produced output with its data type, also using placeholder names if necessary.  
- Important Parameters or Settings: list any configurable options, with a short description or default value.

After you have listed all nodes, add a final section that begins with the marker  

Typical Workflow:  

and provide a short, human‑readable description of the recommended workflow (e.g., the order in which the nodes should be connected).

Do **not** use any markdown formatting, JSON, or other markup – only plain text with the two markers above to separate sections.

Task: {task_description}
