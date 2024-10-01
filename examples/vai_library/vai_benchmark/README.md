# Benchmark runner for vai_library samples

The vai_benchmark is a utility tool provided by Vitis AI. It allows users to perform comprehensive performance testing, both end-to-end and DPU-only, by utilizing supported sample programs in Vitis AI library and models from the Vitis AI model zoo. By using vai_benchmark, users can conveniently assess the performance of their models and gain valuable insights into the capabilities of the DPU (Deep Learning Processing Unit).

> For End2End performance results on all targets, please have a look at [Performance-On-Different-Targets](https://docs.xilinx.com/r/en-US/ug1354-xilinx-ai-sdk/Performance)

> Before running End2End performance, please get testing images by steps in [Running-Vitis-AI-Library-Examples](https://docs.xilinx.com/r/en-US/ug1354-xilinx-ai-sdk/Running-Vitis-AI-Library-Examples)


#### Usage
``` 
# vai_benchmark.py --help
usage: vai_benchmark.py [-h] [--query | --query_all] [-func | -e2e | -dpu | -full] [-m MODEL_NAME | -g {KeyModel,all,typical}] [--debug] [--summary] [-models_path MODELS_PATH] [-t THREADS] [-s SECONDS] [-example_path EXAMPLE_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --query               Query the target, show key elements.
  --query_all           Query the target, show all information.
  -func, --functionality
                        Test model's functionality, which needs testing image.
  -e2e, --e2e_benchmark
                        Test model's E2E benchmark, which needs testing image.
  -dpu, --dpu_benchmark
                        Test model's DPU benchmark, only xmodel is necessary.
  -full, --full         Full test of model, functionality and DPU E2E benchmark
  -m MODEL_NAME, --model_name MODEL_NAME
                        Only run specified models, use ',' to separate multiple models.
  -g {KeyModel,all,typical}, --group {KeyModel,all,typical}
                        Run models in specified group, default is KeyModel.
  --debug               Enable debug logs.
  --summary             Integrate DPU and E2E benchmark results in one table.
  -models_path MODELS_PATH
                        The path of saved models.
  -t THREADS, --threads THREADS
                        Assigned benchmark threads.
  -s SECONDS, --seconds SECONDS
                        The number of running seconds for every model, default is 60s.
  -example_path EXAMPLE_PATH
                        The path of saved vitis_ai_library example.
```


#### Running benchmark runner directly
1. Running assigned models. <br/>
``` vai_benchmark.py -func -m resnet50_tf2,mlperf_ssd_resnet34_tf ```  <br>
``` vai_benchmark.py -dpu -m resnet50_tf2,mlperf_ssd_resnet34_tf -s 30 -t 1,3,6 ```   <br>
``` vai_benchmark.py -e2e -m resnet50_tf2,mlperf_ssd_resnet34_tf -t 1,2,6```   <br>
``` vai_benchmark.py -full -m resnet50_tf2,mlperf_ssd_resnet34_tf --summary```  <br>

2. Running models in specified group, whose default value is "KeyModel". <br>
``` vai_benchmark.py -full -g typical```  <br>
``` vai_benchmark.py -dpu -g all```  <br>
``` vai_benchmark.py -full --summary```  <br>

3. Query the target information <br>
``` vai_benchmark.py --query```      <br>
``` vai_benchmark.py --query_all```  <br>


#### Sample Output
``` vai_benchmark.py -dpu -m resnet50_tf2,mlperf_ssd_resnet34_tf,multi_task_v3_pt -s 30 -t 1,3,6 --summary```   <br>

| **Model Name**         |  **Thread**   | **DPU(FPS)**                    |
|:-----------------------|:-------------:|:--------------------------------|
| multi_task_v3_pt       | 1<br> 3<br> 6 | 302.586<br> 444.517<br> 444.126 |
| resnet50_tf2           | 1<br> 3<br> 6 | 2499.61<br> 3129.31<br> 3124.43 |
| mlperf_ssd_resnet34_tf | 1<br> 3<br> 6 | 64.8063<br> 70.1129<br> 70.1163 |


#### Installing and Running vai_benchmark as a module 
1. Step 1: Installing vai_benchmark. <br>
``` python3 setup.py install```      <br>
2. Step 2: Running vai_benchmark as a module anywhere. <br>
``` vai_benchmark --query```  <br>
``` vai_benchmark -dpu -m resnet50_tf2```  <br><br>
