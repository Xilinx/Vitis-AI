# Usage
1. install vai_q_tensorflow tensorflow_gpu-1.15.2-cp36-cp36m-linux_x86_64.whl

2. edit `run_quantize.sh` and `run_dump.sh`, fill the inpformation below
  - float_pb
  - input_nodes
  - output_nodes
  - height
  - width
  - channel

3. run the shell script
```
bash run_quantize.sh
bash run_dump.sh
```
