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
dog AP: 0.8572931074327245
person AP: 0.8389232879457006
train AP: 0.8899025092111869
sofa AP: 0.7974507551409522
chair AP: 0.6354034320881893
car AP: 0.8832790892172905
pottedplant AP: 0.5608038019385727
diningtable AP: 0.7718230139713811
horse AP: 0.8825828779277769
cat AP: 0.8804525587134284
cow AP: 0.837611574165702
bus AP: 0.8764368638914524
bicycle AP: 0.8751391567231396
motorbike AP: 0.8753000644453839
bird AP: 0.7860377180077252
tvmonitor AP: 0.7870921946379476
sheep AP: 0.8141852284463258
aeroplane AP: 0.8515871669510044
boat AP: 0.7030225316976867
bottle AP: 0.5931306907147502
mAP: 0.7998728811634159
```
