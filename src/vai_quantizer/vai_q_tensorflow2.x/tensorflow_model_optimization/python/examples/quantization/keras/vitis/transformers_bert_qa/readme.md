### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Train/Eval](#traineval)
4. [Quantization](#quantization)
5. [Performance](#performance)
6. [Model_info](#model_info)

### Installation

1. If you use Vitis-AI docker, you could use preset environment by input 'conda activate vitis-ai-tensorflow2'. 
   Don't need install environment manually like the following steps.
   
2. Installation
   - Create virtual envrionment and activate it:
   ```shell
   conda create -n tf2_dev python=3.8
   conda activate tf2_dev
   ```
   - Install all the python dependencies using pip:
   ```shell
   pip install -r requirements.txt
   ```

### Preparation

1. Dataset decription

    - This repo supports loading dataset automatically when you have internet connection.
    - Load by datasets offline. Firstly download the data to local disk. Then upload or copy to `./data/squad_offline`
    ``` bash
    import datasets
    data = datasets.load_dataset('squad')
    data.save_to_disk('./squad_offline')
    ```

2. Dataset directory structure.
   ```
   + data
     + squad_offline
       + metric
         + evaluate.py
         + squad.py
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

3. Download model files  
  Download [bert-large-uncased-whole-word-masking-finetuned-squad](https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad) and copy to `./float/bert-large-uncased-whole-word-masking-finetuned-squad`

4. Download evaluation files  
  Download [SQuAD 1.1](https://github.com/google-research/bert#squad-11) and copy the 3 files to `./data/SQuADv1.1/`


### Eval & Training

1. Evaluation
    ```shell
    bash run_eval.sh
    ```
2. Training
    ```shell
    bash run_train.sh
    ```

### Quantization
**vai_q_tensorflow2** is required, see 
[Vitis AI User Document](https://www.xilinx.com/products/design-tools/vitis/vitis-ai.html#documentation) for installation guide.

1. Quantize model.  
  For PTQ (Post Training Quantization)
  ```shell
  bash run_quantize.sh
  ```
  For QAT (Quantization-Aware Training)
  ```shell
  bash run_qatrain.sh
  ```
2. Evaluate quantized model.
  ```shell
  bash run_quantize_eval.sh
  ```

### Performance
| model | sequence length | exact_match | f1 |
|-------|------------|--------------|-------|
| float | 384 | 86.925% | 93.158% |
| PTQ | 384 | 84.153% | 91.171% |
| QAT | 384 | 85.705% | 92.178% |

### Model_info

1. Data preprocess
  ```
  Tokenize.
  If input sequence length is more then max_seq_length, cut input document into slices according doc_stride and max_seq_length.
  ```

2.  System Environment

The operation and accuracy provided above are verified in Ubuntu20.04.1 LTS, cuda-11.8, Driver Version: 510.47.03, GPU NVDIA V100
