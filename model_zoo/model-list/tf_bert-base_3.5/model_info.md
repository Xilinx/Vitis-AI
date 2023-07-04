# Bert base model


### Contents
1. [Use Case and Application](#Use-Case-and-Application)
2. [Specification](#Specification)
3. [Paper and Architecture](#Paper-and-Architecture)
4. [Dataset Preparation](#Dataset-Preparation)
5. [Use Guide](#Use-Guide)
6. [License](#License)
7. [Note](#Note)


### Use Case and Application

   - BERT is a transformer model pretrained on a large corpus of English data in a self-supervised fashion
   - For either masked language modeling or next sentence prediction, but it's mostly intended to be fine-tuned on a downstream task
   - Trained on SQuADv1.1 dataset
   
   
### Specification

| Metric             | Value                                   |
| :----------------- | :-------------------------------------- |
| Framework          | TensorFlow                              |
| Prune Ratio        | 0%                                      |
| FLOPs              | 22.34G                                  |
| sequence length    | 128                                     |
| FP32 Accuracy      | 0.8694 F1-Score                         |
| INT8 Accuracy      | 0.8656 F1-Score                         |
| Train Dataset      | SQuADv1.1                               |
| Test Dataset       | SQuADv1.1                               |
| Supported Platform | GPU                                     |
  

### Paper and Architecture 

1. Network Architecture: BERT
 
2. Paper link: https://arxiv.org/abs/1810.04805

  
### Dataset Preparation

1. Dataset decription

    - Download SQuADv1.1 dataset and evaluation scripts from [BERT github](https://github.com/google-research/bert#squad-11). And put SQuADv1.1 dataset into `./data/SQuADv1.1` folder.

2. Dataset directory structure.
   ```
   + data
     + SQuADv1.1
       + dev-v1.1.json
       + train-v1.1.json
       + evaluate-v1.1.py
    ```
3. Download pre-trained model files
    - Pre-Norm BERT Base xcd path: `/group/modelzoo/sequence_learning/weights/nlp-pretrained-model/pre_norm_bert_base`.
    - Put model files in folder `./weights/pre_norm_bert_base`.

### Use Guide

1. FP32 Training(Fine tuning)
    ```shell
    bash run_train.sh
    ```
2. FP32 Evaluation
    ```shell
    bash run_eval_fp32.sh
    ```
3. INT8 Quantization warmup
    ```shell
    bash run_quant_warmup.sh
    ```
4. INT8 QAT
    ```shell
    bash run_quant.sh
    ```
5. INT8 Evaluation
    ```shell
    bash run_eval_int8.sh
    ```
6. Dump pb file
    ```shell
    bash run_dump_pb.sh
    ```


### License

Apache License 2.0

For details, please refer to **[Vitis-AI License](https://github.com/Xilinx/Vitis-AI/blob/master/LICENSE)**


### Note

Data preprocess
  ```
  word.lower()
  Tokenize.
  If input sequence length is more then max_seq_length, cut input document into slices according doc_stride and max_seq_length.
```
