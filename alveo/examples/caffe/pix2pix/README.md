## Pix2Pix Caffe Models

### Model list

- bw2color
- cityscapes_AtoB
- cityscapes_BtoA
- facades_BtoA
- maps_AtoB
- maps_BtoA


## Activate Caffe conda environment

Please activate Caffe conda environment using the following commands.

```
conda activate vitis-ai-caffe
source /workspace/alveo/overlaybins/setup.sh
```


### Get the caffe models 

Download the caffe models. 

```
cd /workspace/alveo/examples/caffe 
python getModels.py
```