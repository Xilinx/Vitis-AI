## Getting Started to prepare and deploy a ofa/searched model on Vitis-AI docker container


### Setup
> **Note:** Skip, If you have already run the below steps.

Once-for-all network is built on PyTorch Framework. 

Vitis AI Quantizer supports PyTorch Framework.

You can find detail guide how to quantize PyTorch model using Vitis-AI Quantizer in https://docs.xilinx.com/r/en-US/ug1414-vitis-ai/Inspect-Float-Model-Before-Quantization

 

  Activate Conda Environment
  ```sh
  conda activate vitis-ai-pytorch
  ```

  Install the necessary dataset (Proceed to next step, if already done)
  
  Download validation set for [Imagenet2012](http://www.image-net.org/challenges/LSVRC/2012).
   > **Note:** User is responsible for the use of the downloaded content and compliance with any copyright licenses.

  The downloaded validation dataset is supposed to be located '/workspace/imagenet/'.
  
  The validation dataset has 50k image files. There are 1000 categories and 50 image files for each category. 
  
  We need 1 - 5 image files for each category. 
  
  And the imagenet folder should be something like this:

    .imagenet/
    ├── val
    │   ├── n01440764
    │   ├── n01443537
    │   ├── n01484850
    │   ├── n01491361
    │   ├── n01494475
    ………….
    │   └── n15075141
    └── val.txt
    



  Setup the Environment for once-for-all(ofa)

  ```sh
  # install ofa package
  
  pip install ofa
  
  # modify resnet50 design space to support DPUCZDX8G compiler
  # we need to check where ofa folder is located. 
  # if ofa foler is located in /opt/vitis_ai/conda/envs/vitis-ai-pytorch/lib/python3.7/site-packages/  
  chmod o+w /opt/vitis_ai/conda/envs/vitis-ai-pytorch/lib/python3.7/site-packages/ofa/imagenet_classification/networks/resnets.py
  vim /opt/vitis_ai/conda/envs/vitis-ai-pytorch/lib/python3.7/site-packages/ofa/imagenet_classification/networks/resnets.py
  
  # if ofa foler is located in /home/vitis-ai-user/.local/lib/python3.7/site-packages/
  chmod o+w /home/vitis-ai-user/.local/lib/python3.7/site-packages/ofa/imagenet_classification/networks/resnets.py
  vim /home/vitis-ai-user/.local/lib/python3.7/site-packages/ofa/imagenet_classification/networks/resnets.py
  
    at line 2: add the following
            import torch

    at line 28: modify the following
            from: self.global_avg_pool = MyGlobalAvgPool2d(keep_dim=False) 
            to: self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

    after line 37: add new line 38 as the following 
             x = torch.flatten(x, start_dim=1, end_dim=-1)

    # setup tab space to match the other indent space
    :retab! 16
    # replace indent using 'tab'key instead of 'space bar'key on new line 38


  # modify imagenet folder for ofa design space
  chmod o+w /home/vitis-ai-user/.local/lib/python3.7/site-packages/ofa/imagenet_classification/data_providers/imagenet.py
  vim /home/vitis-ai-user/.local/lib/python3.7/site-packages/ofa/imagenet_classification/data_providers/imagenet.py
  
    at line 20: modify the following
            from: DEFAULT_PATH = '/dataset/imagenet'
            to: DEFAULT_PATH = '/workspace/imagenet'
            
    at line 183: modify the following
            from: return os.path.join(self.save_path, 'train')
            to: return os.path.join(self.save_path, 'val')
          
  # modify my_random_resize_crop.py to fix bug in the original repo.          
  chmod o+w /home/vitis-ai-user/.local/lib/python3.7/site-packages/ofa/utils/my_dataloader/my_random_resize_crop.py
  vim /home/vitis-ai-user/.local/lib/python3.7/site-packages/ofa/utils/my_dataloader/my_random_resize_crop.py
  
    after line 39: add new line 40 as the following 
	    self.interpol = interpolation
          
    at line 86: modify the following
            from: interpolate_str = _pil_interpolation_to_str[self.interpolation]
            to: interpolate_str = _pil_interpolation_to_str[self.interpol]

  ```




### End-to-End Example using Jupyter Notebook

The Jupyter Notebooks provide tutorial on how to run ofa/resnet50 with Xilinx Vitis-AI.

ofa_resnet50_search_byOps is End-to-End example of searching a subnet from ofa/resnet50 design space with constraint of Ops, then quantizing/compiling it as target device of ZCU102 board. 

Jupyter is preinstalled in the Xilinx Vitis-AI Docker image.


  ```sh
  # create model folder to store the searched model
  cd /workspace/examples/DPUCZDX8G/ofa_resnet50
  mkdir models
  ```
  
  

  ```sh
  # Launch Jupyter notebook server
  cd /workspace/examples/DPUCZDX8G/ofa_resnet50
  jupyter notebook --no-browser --ip=0.0.0.0 --NotebookApp.token='' --NotebookApp.password=''
  ```
  
Open a broswer, and navigate to one of:
  - `<yourpublicipaddress>:8888`
  - `<yourdns>:8888`
  - `<yourhostname>:8888`
  


### Deploy compiled model on target hardware(ZCU102 board)

Copy Vitis-AI sample images to target hardware

Download the [vitis_ai_library_r1.3.x_images.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_library_r1.3.1_images.tar.gz).

Copy it from host to the target using scp with the following command.
  ```sh 
  [Host]scp vitis_ai_library_r1.3.x_images.tar.gz userid@IP_OF_BOARD:~/
  [Remote]tar -xzvf vitis_ai_library_r1.3.*_images.tar.gz -C Vitis-AI/demo/Vitis-AI-Library
  ```

Create prototxt file corresponding to the compiled model
   > **Note:** As an example, we assume the filename of the compiled model is resnet50_fp2000.
  ```sh 
  # create prototxt which name is same as compiled model name
  cp resnet50_dump.prototxt resnet50_fp2000.prototxt
  
  # update the prototxt
  vim resnet50_fp2000.prototxt
    at line 2: modify the following
            from:    name : "resnet50_dump"
            to:    name : "resnet50_fp2000"
  ```
   > **Note:** 

        mean_in_prototxt = mean_in_python *255;
        scale_in_prototxt = 1/(scale_in_python *255)
        In python, the RGB channel order is : R-G-B
        In prototxt, the RGB channel order is: B-G-R


Copy compiled xmodel to board(remote)
  ```sh 
    # connect zcu102 board(remote)
    [Host]ssh userid@IP_OF_BOARD
    [Remote]cd ~/Vitis-AI/demo/Vitis-AI-Library/samples/classification
    # create folder which name is same as the compiled model name
    [Remote]mkdir resnet50_fp2000  

    [Host]cd /workspace/ofa/compiled
    [Host]scp resnet50_fp2000.xmodel userid@IP_OF_BOARD:/home/root/Vitis-AI/demo/Vitis-AI-Library/samples/classification/resnet50_fp2000/
    [Host]scp resnet50_fp2000.prototxt userid@IP_OF_BOARD:/home/root/Vitis-AI/demo/Vitis-AI-Library/samples/classification/resnet50_fp2000/
  ```


  
Evaulate the compiled model with an image file
  ```sh 
    [Remote]cd ~/Vitis-AI/demo/Vitis-AI-Library/samples/classification
    [Remote]./test_jpeg_classification resnet50_fp2000 sample_classification.jpg
  ```

Evaluate the performance(inference time) of the compiled model with thread 1
  ```sh 
    [Remote]cd ~/Vitis-AI/demo/Vitis-AI-Library/samples/classification
    [Remote]./test_performance_classification resnet50_fp2000 test_performance_classification.list -t 1
  ```
  



