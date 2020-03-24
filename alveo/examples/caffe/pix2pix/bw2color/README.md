

## Running Caffe pix2pix (b/w to color) Model:
### Data Preparation

The format of the input image is [256,256,3]

The input image is supposed to be black/white image.


### Prepare a model for inference
** MUST BE DONE FIRST **



To run a Caffe model on the FPGA, it needs to be quantized, compiled, and a new graph needs to be generated. The new graph is similar to the original, with the FPGA subgraph removed, and replaced with a custom Python layer.
```
cd /workspace/alveo/examples/caffe/bw2color 
source run.sh deploy
```

### Run Inference for a single image
```
cd /workspace/alveo/examples/caffe/bw2color 
python bw2color_fpga.py --image imagefilename 
```



