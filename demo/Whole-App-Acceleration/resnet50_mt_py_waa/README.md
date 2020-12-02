## 1. For ZCU102
 ### Run resnet50 without waa
  ```
  cd ~/resnet50_waa
  python3 resnet50.py 4 /usr/share/vitis_ai_library/models/resnet50/resnet50.xmodel
  ```
 ### Run resnet50 with waa
 ```
  cd ~/resnet50_waa
  env XILINX_XRT=/usr python3 resnet50_waa.py 4 /usr/share/vitis_ai_library/models/resnet50/resnet50.xmodel
```
  ## 2. For U50
 ### Run resnet50 without waa
  ```
  cd ./resnet50_waa
  /usr/bin/python3 resnet50.py 4 /usr/share/vitis_ai_library/models/resnet50/resnet50.xmodel
  ```
  ### Run resnet50 with waa
  ```
  cd ./resnet50_waa
  /usr/bin/python3 resnet50_waa.py 4 /usr/share/vitis_ai_library/models/resnet50/resnet50.xmodel  