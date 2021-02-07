## Pix2Pix Caffe Models


### Activate Caffe conda environment

Please activate Caffe conda environment using the following commands.

```
conda activate vitis-ai-caffe
source /vitis_ai_home/setup/alveo/u200_u250/overlaybins/setup.sh
```


### Get the caffe models

Download the caffe models.

```
cd ${VAI_HOME}/examples/DPUCADX8G/caffe
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
