All parameters that can be configed are write into file `config.ini`.
In general, `quantize.sh` and `evaluate_quantize_model.sh` do not need to be modified.

## prepare data
As we do not need to use all validation dataset, we collects 1000 images to
quantize model. File [calib_list](../../data/calib_list.txt) contains these images.
Other dataset is the same as float model.

## quantize model
Config the `config.ini` to your own situation. The product of `CALIB_ITER` and `CALIB_BATCH_SIZE` need to be equal to
number of `calib_list` images. Then run 
```
bash quantize.sh
```
After runing `quantize.sh`, the quantized model are in `QUANTIZE_DIR` which are
set in `config.ini`

## evaluate quantize model
Same as evaluate float model, config the parameters associated with
evaluation. Then run 
```
bash evaluate_quantize_model.sh
```
If everything runs correctly, accuracy of quantized model are:

```
evaluate 4952 images
dog AP: 0.8478591925010324
person AP: 0.8119346474202439
train AP: 0.8476080433490066
sofa AP: 0.7768383058543032
chair AP: 0.6120402485269938
car AP: 0.8527281000680158
pottedplant AP: 0.48435638792414626
diningtable AP: 0.6993111325107773
horse AP: 0.8635459402139203
cat AP: 0.8754112057227865
cow AP: 0.8114350136367143
bus AP: 0.8590487538760534
bicycle AP: 0.8609449758425377
motorbike AP: 0.8420732554703096
bird AP: 0.7565624443782338
tvmonitor AP: 0.7816398202217327
sheep AP: 0.7624726586591455
aeroplane AP: 0.776311288746124
boat AP: 0.7076766138290085
bottle AP: 0.628253805089068
mAP: 0.7729025916920076
```
