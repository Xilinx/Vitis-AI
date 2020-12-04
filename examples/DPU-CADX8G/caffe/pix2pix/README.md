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
source <path-to-vitis-ai>/setup/alveo/DPU-CADX8G/overlaybins/setup.sh
```


### Get the caffe models 

Download the caffe models. 

```
cd ${VAI_ALVEO_ROOT}/DPU-CADX8G/caffe
python getModels.py
```