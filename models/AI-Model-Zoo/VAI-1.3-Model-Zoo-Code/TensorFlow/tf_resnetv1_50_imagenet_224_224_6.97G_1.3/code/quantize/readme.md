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

Recall_1 = [0.74106]

Recall_5 = [0.91668]
