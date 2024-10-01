#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import torch
from onnxruntime.quantization.calibrate import CalibrationDataReader
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10


import onnx
import onnxruntime
from onnxruntime.quantization import CalibrationDataReader, QuantType, QuantFormat, CalibrationMethod, quantize_static

import vai_q_onnx


class CIFAR10DataSet:
    def __init__(
        self,
        data_dir,
        **kwargs,
    ):
        super().__init__()
        self.train_path = data_dir
        self.vld_path = data_dir
        self.setup("fit")

    def setup(self, stage: str):
        transform = transforms.Compose(
            [transforms.Pad(4), transforms.RandomHorizontalFlip(), transforms.RandomCrop(32), transforms.ToTensor()]
        )
        self.train_dataset = CIFAR10(root=self.train_path, train=True, transform=transform, download=False)
        self.val_dataset = CIFAR10(root=self.vld_path, train=True, transform=transform, download=False)


class PytorchResNetDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        input_data = sample[0]
        label = sample[1]
        return input_data, label


def create_dataloader(data_dir, batch_size):
    cifar10_dataset = CIFAR10DataSet(data_dir)
    _, val_set = torch.utils.data.random_split(cifar10_dataset.val_dataset, [49000, 1000])
    benchmark_dataloader = DataLoader(PytorchResNetDataset(val_set), batch_size=batch_size, drop_last=True)
    return benchmark_dataloader


class ResnetCalibrationDataReader(CalibrationDataReader):
    def __init__(self, data_dir: str, batch_size: int = 16):
        super().__init__()
        self.iterator = iter(create_dataloader(data_dir, batch_size))

    def get_next(self) -> dict:
        try:
            images, labels = next(self.iterator)
            return {"input": images.numpy()}
        except Exception:
            return None


def resnet_calibration_reader(data_dir, batch_size=16):
    return ResnetCalibrationDataReader(data_dir, batch_size=batch_size)



def main():
    # `input_model_path` is the path to the original, unquantized ONNX model.
    input_model_path = "models/resnet_trained_for_cifar10.onnx"

    # `output_model_path` is the path where the quantized model will be saved.
    output_model_path = "models/resnet.qop.U8S8.onnx"

    # `calibration_dataset_path` is the path to the dataset used for calibration during quantization.
    calibration_dataset_path = "data/"

    # `dr` (Data Reader) is an instance of ResNet50DataReader, which is a utility class that 
    # reads the calibration dataset and prepares it for the quantization process.
    dr = resnet_calibration_reader(calibration_dataset_path)

    # `quantize_static` is a function that applies static quantization to the model.
    # The parameters of this function are:
    # - `input_model_path`: the path to the original, unquantized model.
    # - `output_model_path`: the path where the quantized model will be saved.
    # - `dr`: an instance of a data reader utility, which provides data for model calibration.
    # - `quant_format`: the format of quantization operators. Need to set to QDQ or QOperator.
    # - `activation_type`: the data type of activation tensors after quantization. In this case, it's QUInt8 (Quantized Unsigned Int 8).
    # - `weight_type`: the data type of weight tensors after quantization. In this case, it's QInt8 (Quantized Int 8).
    vai_q_onnx.quantize_static(
        input_model_path,
        output_model_path,
        dr,
        quant_format=QuantFormat.QOperator,
        calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
    )
    print('Calibrated and quantized model saved at:', output_model_path)

if __name__ == '__main__':
    main()
