# EfficientDet-D2


### Contents
1. [Use Case and Application](#Use-Case-and-Application)
2. [Specification](#Specification)
3. [Paper and Architecture](#Paper-and-Architecture)
4. [Dataset Preparation](#Dataset-Preparation)
5. [Use Guide](#Use-Guide)
6. [License](#License)
7. [Note](#Note)


### Use Case and Application

   - Classic Object Detection
   - Trained on COCO dataset   
   
### Specification

| Metric             | Value                                   |
| :----------------- | :-------------------------------------- |
| Framework          | TensorFlow                              |
| Prune Ratio        | 0%                                      |
| FLOPs              | 11.06G                                  |
| Input Dims (H W C) | 768,768,3                               |
| FP32 Accuracy      | 0.413 mAP                               |
| INT8 Accuracy      | 0.327 mAP                               |
| Train Dataset      | COCO2017                                |
| Test Dataset       | COCO2017                                |
| Supported Platform | GPU                                     |
  

### Paper and Architecture 

1. Network Architecture: EfficientDet-D2
 
2. Paper link: https://arxiv.org/abs/1911.09070

  
### Dataset Preparation

1. Dataset preparation
```bash
cd code/efficientdet
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip

mkdir tfrecord
PYTHONPATH=".:$PYTHONPATH"  python dataset/create_coco_tfrecord.py \
    --image_dir=train2017 \
    --caption_annotations_file=annotations/captions_train2017.json \
    --output_file_prefix=tfrecord/train \
    --num_shards=32
PYTHONPATH=".:$PYTHONPATH"  python dataset/create_coco_tfrecord.py \
    --image_dir=val2017 \
    --caption_annotations_file=annotations/captions_val2017.json \
    --output_file_prefix=tfrecord/val \
    --num_shards=32
```

2. Dataset diretory structure
   ```
   + tfrecord
     + train-00000-of-00032.tfrecord
     + ...
     + train-00031-of-00032.tfrecord
     + val-00000-of-00032.tfrecord
     + ...
     + val-00031-of-00032.tfrecord

    ```


### Use Guide

1. Load origianl model and finetune (replace sigmoid to hard_sigmoid)
    ```shell
    cd code/efficientdet
    python main.py --mode=train_and_eval --train_file_pattern=tfrecord/train-* --val_file_pattern=tfrecord/val-* --model_name=efficientdet-d2 --model_dir=efficientdet-d2-finetune --ckpt=efficientdet-d2 --train_batch_size=16 --eval_batch_size=16 --num_examples_per_epoch 118287 --hparams="use_keras_model=False" --val_json_file=path/to/instances_val2017.json --num_epochs=50
    ```

2. Do constant folding and freeze the model (some operations in fast_attention is not supported now, we replace it by constant value)
    ```shell
    cd code/efficientdet
    python model_inspect.py --runmode=constant_folding --model_name=efficientdet-d2 --ckpt_path=efficientdet-d2-finetune --saved_model_dir=savedmodeldir --hparams="use_keras_model=False" --logdir=savedmodeldir

    python model_inspect.py --runmode=freeze --model_name=efficientdet-d2 --ckpt_path=efficientdet-d2-finetune --saved_model_dir=savedmodeldir --hparams="use_keras_model=False,constant_folding='savedmodeldir/bifpn_wsm.pkl'" --logdir=savedmodeldir --delete_logdir=False
    ```
3. Evaluation
    ```shell
    cd code/efficientdet
    python main.py --mode=eval --model_name=efficientdet-d2 --model_dir=efficientdet-d2-finetune --val_file_pattern=tfrecord/val* --val_json_file=path/to/instances_val2017.json --hparams="constant_folding='savedmodeldir/bifpn_wsm.pkl'"
    ```
 
### Post training Quantize

1. Quantization
    ```shell
    cd code/efficientdet
    bash run_quantize.sh
    ```

2. Evaluation quantized model(optional)
    ```shell
    cd code/efficientdet
    bash run_eval_quantized_pb.sh
    ```

### License

Apache License 2.0

For details, please refer to **[Vitis-AI License](https://github.com/Xilinx/Vitis-AI/blob/master/LICENSE)**

