<table class="sphinxhide">
 <tr>
   <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis AI</h1><h0>Adaptable & Real-Time AI Inference Acceleration</h0>
   </td>
 </tr>
</table>

## QuickStart guide

The step-by-step instructions are made for easy start with the vai quantization tool for model optimization via neural network deep compression.
Follow the guide to quickly launch the tool, the steps are operated with the pytorch model framework as the example. 
The procedure for all frameworks is similar to each other, you may find the desired framework in [this directory](https://github.com/Xilinx/Vitis-AI/tree/master/src/vai_quantizer) 
and replace correspondingly. 
1. Prerequisite and installation. <br>
Make sure that you have the latest version of Vitis-AI and the model quantization Python package is installed. 
The packages are different depending on what the model framework is. <br>
For Pytorch - `pytorch_nndct` must be installed. You may find the installation options in [this document](https://github.com/Xilinx/Vitis-AI/tree/master/src/vai_quantizer/vai_q_pytorch#install-from-source-code). <br>
If the following command line does not report error, the installation is done. <br>

    ```
    python -c "import pytorch_nndct"
    ```
   
    > **_NOTE:_**  The further steps with code-snippets can be combined in the single jupyter notebook or the python script.

2. Import the modules. <br>

    ```
    import torch
    from torch.utils.data import DataLoader, Dataset
    import torchvision.transforms as transforms
    from PIL import Image
    import os
    from pytorch_nndct.apis import torch_quantizer
    ```
   
3. Prepare the dataset, dataloader, model and evaluation function. <br>   
For the example, the pretrained resnet18 classification model is used from the torchvision. <br>
Dataset and DataLoader from the custom data: <br>

    ```
    # Define your custom dataset class
    class CustomDataset(Dataset):
        def __init__(self, data_dir, transform=None):
            self.data_dir = data_dir
            self.transform = transform
            self.data = []  # List to store image filenames and labels
            
            # Assuming directory structure: data_dir/class_name/image.jpg
            classes = os.listdir(data_dir)
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
            
            for class_name in classes:
                class_dir = os.path.join(data_dir, class_name)
                if os.path.isdir(class_dir):
                    for file_name in os.listdir(class_dir):
                        if file_name.endswith(".jpg"):
                            self.data.append((os.path.join(class_dir, file_name), self.class_to_idx[class_name]))
    
        def __len__(self):
            return len(self.data)
    
        def __getitem__(self, idx):
            img_path, label = self.data[idx]
            image = Image.open(img_path).convert("RGB")
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
    
    # Define transformations to apply to the images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create an instance of your custom dataset
    data_dir = "/path/to/your/dataset"
    custom_dataset = CustomDataset(data_dir, transform=transform)
    
    # Create a custom dataloader
    batch_size = 8
    dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)
    # val_dataloader - you may also declare the dataloader with validation data  
    ```
   
Evaluation function with top1 accuracy: <br>

    ```
    def evaluate_model(model, dataloader):
        model.eval()
        device = next(model.parameters()).device
        
        total_samples = 0
        correct_predictions = 0
    
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
    
        top1_accuracy = correct_predictions / total_samples
        return top1_accuracy
    ```

The pretrained model object: <br>

    ```
    from torchvision.models.resnet import resnet18
    model = resnet18(pretrained=True)
    ```

4. Generate a quantizer with quantization needed input and get converted model. <br>

   ```
    quant_mode = "calib"
    input = torch.randn([batch_size, 3, 224, 224])
    quantizer = torch_quantizer(quant_mode, model, (input))
    quant_model = quantizer.quant_model
   ```
   
   As we have pretrained model, the training process is omitted and here only the quantization of the model weights is presented.
    > **_NOTE:_**  quant_mode: An integer that indicates which quantization mode the process is using. "calib" for calibration of quantization. "test" for evaluation of quantized model.

5. Forward with converted model by evaluating on the validation data. <br>

   ```
    top_acc1 = evaluate(quant_model, val_dataloader)
   ```

6. Output the quantization result and deploy model. <br>

   ```
   if quant_mode == 'calib':
     quantizer.export_quant_config()
   if deploy:
     quantizer.export_torch_script()
     quantizer.export_onnx_model()
     quantizer.export_xmodel()
   ```

Xmodel file for Vitis AI compiler and other artifacts will be generated under output directory “./quantize_result”. 
It will be further used to deploy this model to the DPU device. 
```
    ResNet_int.xmodel: deployed XIR format model
    ResNet_int.onnx:   deployed onnx format model
    ResNet_int.pt:     deployed torch script format model
```
