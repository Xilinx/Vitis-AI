## Pix2Pix Caffe Models


### Activate Caffe conda environment

Please activate Caffe conda environment using the following commands.

```
conda activate vitis-ai-caffe
# Typically, <path-to-vitis-ai> is `/workspace`
source <path-to-vitis-ai>/setup/alveo/DPU-CADX8G/overlaybins/setup.sh
```


### Get the caffe models

Download the caffe models.

```
cd ${VAI_ALVEO_ROOT}/DPU-CADX8G/caffe
python getModels.py
```

### Model list

Follow below links to try out different models

- [bw2color](bw2color)
- [cityscapes_AtoB](cityscapes_AtoB)
- [cityscapes_BtoA](cityscapes_BtoA)
- [facades_BtoA](facades_BtoA)
- [maps_AtoB](maps_AtoB)
- [maps_BtoA](maps_BtoA)
