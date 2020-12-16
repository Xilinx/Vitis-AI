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
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.236
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.396
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.239
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.036
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.209
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.455
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.231
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.342
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.362
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.081
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.359
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.648
```
