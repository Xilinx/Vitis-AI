# L2: Kernels

## Data-Mover Generator

This layer provides a tool (`L2/scripts/generate_kernels`) to create data-mover kernel source files
form description written in JSON format.
This tool aims to help developers to quickly create helper kernels to verify, benchmark and deploy their design
in hardware. No HLS kernel development experience is required to use this tool.

The `src/xf_datamover` subfolder contains the template of kernels used by the tool,
in [Jinja2](https://jinja.palletsprojects.com/en/2.11.x/) format.


