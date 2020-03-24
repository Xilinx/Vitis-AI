

## Running Caffe FPN segmentation Model:
### Data Preparation

Download cityscapes-dataset from https://www.cityscapes-dataset.com/downloads/

You need to register for the website to download the dataset.

The format of the input image is [256,256,3]

The input image is supposed to be cityscape photo image.


### Prepare a model for inference
** MUST BE DONE FIRST **

Get the caffe model. 
```
cd /workspace/alveo/examples/caffe 
python getModels.py
```

To run a Caffe model on the FPGA, it needs to be quantized, compiled, and a new graph needs to be generated. The new graph is similar to the original, with the FPGA subgraph removed, and replaced with a custom Python layer.
```
cd /workspace/alveo/examples/caffe/FPN_CityScapes 
source run.sh deploy
```


### Run Inference for a single image
```
cd /workspace/alveo/examples/caffe/FPN_CityScapes
python FPN_caffe.py --image imagefilename 
```



