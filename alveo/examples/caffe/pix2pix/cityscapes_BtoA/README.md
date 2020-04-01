

## Running Caffe pix2pix (cityscapes_BtoA) Model:
### Data Preparation

Download cityscapes-dataset from https://www.cityscapes-dataset.com/downloads/

Or Download cityscapes-dataset fro https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/cityscapes.tar.gz

The format of the input image is [256,256,3]

The input image is supposed to be cityscape semantic segmenation image.



### Prepare a model for inference
** MUST BE DONE FIRST **


To run a Caffe model on the FPGA, it needs to be quantized, compiled, and a new graph needs to be generated. The new graph is similar to the original, with the FPGA subgraph removed, and replaced with a custom Python layer.
```
cd /workspace/alveo/examples/caffe/cityscapes_BtoA 
source run.sh deploy
```

### Run Inference for a single image
```
cd /workspace/alveo/examples/caffe/cityscapes_BtoA 
python cityscapes_BtoA_fpga.py --image imagefilename 
```



