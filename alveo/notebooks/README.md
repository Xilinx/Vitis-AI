# Jupyter Notebooks

The Jupyter Notebooks provide tutorials on how to run ML inference with Xilinx Vitis-AI.  
Jupyter is preinstalled in the Xilinx Vitis-AI Docker image.

## Notebook Setup

Follow these instructions from inside a running container.

1. Setup the environment
  ```sh
  . /workspace/alveo/overlaybins/setup.sh
  ```

2. Install the necessary dataset (Proceed to next step, if already done)
  ```sh
  # For Caffe Notebooks
  cd /workspace/alveo/examples/caffe
  conda activate vitis-ai-caffe
  python -m ck pull repo:ck-env
  python -m ck install package:imagenet-2012-val-min
  python -m ck install package:imagenet-2012-aux
  head -n 500 $HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-aux/val.txt > \
  $HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/val_map.txt
  # Resize all the images to a common dimension for Caffe
  python resize.py $HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min 256 256
  # Get the necessary models
  python getModels.py
  python replace_mluser.py --modelsdir models
  
  # For Tensorflow Notebooks
  cd /workspace/alveo/examples/tensorflow
  conda activate vitis-ai-tensorflow
  python -m ck pull repo:ck-env
  python -m ck install package:imagenet-2012-val-min
  python -m ck install package:imagenet-2012-aux
  head -n 500 $HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-aux/val.txt > \
  $HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/val.txt
  # Get the necessary models
  python getModels.py
  ```
  
3. Launch Jupyter notebook server
  ```sh
  cd /workspace/alveo/notebooks
  jupyter notebook --no-browser --ip=0.0.0.0 --NotebookApp.token='' --NotebookApp.password=''
  ```
  
4. Open a broswer, and navigate to one of:
  - `<yourpublicipaddress>:8888`
  - `<yourdns>:8888`
  - `<yourhostname>:8888`
