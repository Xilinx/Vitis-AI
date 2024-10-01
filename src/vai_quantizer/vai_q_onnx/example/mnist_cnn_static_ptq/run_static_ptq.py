#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import vai_q_onnx
from torch.utils.data import DataLoader
from onnxruntime.quantization import CalibrationDataReader, QuantFormat
from torchvision import datasets, transforms


class MnistDataReader(CalibrationDataReader):

    def __init__(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))])
        self.dataset = datasets.MNIST('./data',
                                      train=False,
                                      transform=transform)
        self.iterator = iter(DataLoader(self.dataset, batch_size=1))

    def get_next(self) -> dict:
        try:
            return {"input": next(self.iterator)[0].numpy()}
        except Exception:
            return None


model_input = "./mnist_cnn.onnx"
model_output = "./mnist_cnn_quantized.onnx"
vai_q_onnx.quantize_static(model_input=model_input,
                           model_output=model_output,
                           calibration_data_reader=MnistDataReader(),
                           quant_format=QuantFormat.QOperator)
print("Quantize finish, quantized models saved in ", model_output)
