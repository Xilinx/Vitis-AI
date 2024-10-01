# Bert tiny model


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
| Framework          | PyTorch                                 |
| Prune Ratio        | 0%                                      |
| FLOPs              | 453M                                    |
| sequence length    | 384                                     |
| FP32 Accuracy      | 0.5231 F1-Score                         |
| INT8 Accuracy      | 0.5272 F1-Score                         |
| Train Dataset      | SQuADv1.1                               |
| Test Dataset       | SQuADv1.1                               |
| Supported Platform | GPU                                     |
  

### Paper and Architecture 

1. Network Architecture: BERT
 
2. Paper link: https://arxiv.org/abs/1810.04805

  
### Dataset Preparation

1. Dataset description

    - Load datasets offline. Firstly download the data to local disk by following python scripts. Then upload or copy to `./data/squad_offline`
    ``` bash
    bash download_dataset.sh
    ```

2. Dataset directory structure.
   ```
   + data
     + squad_offline
       + train
         + dataset_info.json
         + dataset.arrow
         + state.json
       + validation
         + dataset_info.json
         + dataset.arrow
         + state.json
       + dataset_dict.json
    ```

3. Download the Huggingface Transformers.
    ``` bash
    git clone https://github.com/huggingface/transformers huggingface_transformers
    cd huggingface_transformers
    git checkout 39b4aba54d349f35e2f0bd4addbe21847d037e9e
    cd ..
    mv huggingface_transformers/src/transformers ./code/
    cp ./code/quant_code/modeling_bert.py ./code/transformers/models/bert/modeling_bert.py
    cp ./code/quant_code/modeling_utils.py ./code/transformers/modeling_utils.py
    ```

4. Download model files
    - Download BERT model from huggingface (https://huggingface.co/bert-base-uncased). And put then into `./float/pretrained_model` folder.


### Use Guide

1. Training
    ```shell
    bash run_train.sh
    ```

2. Evaluation FP32 model
    ```shell
    bash run_eval.sh
    ```

3. Quantization
    ```shell
    bash run_quant.sh
    ```

4. Evaluation INT8 quantized model
    ```shell
    bash run_eval_quant.sh
    ```

### License

Apache License 2.0

For details, please refer to **[Vitis-AI License](https://github.com/Xilinx/Vitis-AI/blob/master/LICENSE)**


### Note

Data preprocess
  ```
  Tokenize.
  If input sequence length is more then max_seq_length, cut input document into slices according doc_stride and max_seq_length.
  ```
  