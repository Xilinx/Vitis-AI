"""
Copyright 2021 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pdb
import time
import argparse
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import vart
import xir
from tensorflow import keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from torch.utils.data import DataLoader, TensorDataset
from threading import Thread

DEVICE = torch.device("cpu")

top_words = 5000  # vocab size
max_review_length = 500
embedding_vector_length = 32
total_vec_length = max_review_length * embedding_vector_length
hidden_vector_length = 100


class PreProcModel(nn.Module):
    def __init__(self, max_words, emb_size, hid_size):
        super(PreProcModel, self).__init__()
        self.max_words = max_words
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.Embedding = nn.Embedding(self.max_words, self.emb_size)

    def forward(self, x):
        print("Running Preproc ...")
        return self.Embedding(x)


class GRUModel(nn.Module):
    def __init__(self, max_words, emb_size, hid_size):
        super(GRUModel, self).__init__()
        self.max_words = max_words
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.GRU = nn.GRU(self.emb_size, self.hid_size, batch_first=True)

    def forward(self, x):
        print("Running GRU on CPU ...")
        y, _ = self.GRU(x)
        z = y[:, -1, :]
        return z


class PostProcModel(nn.Module):
    def __init__(self, max_words, emb_size, hid_size):
        super(PostProcModel, self).__init__()
        self.max_words = max_words
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.fc = nn.Linear(self.hid_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        print("Running Postproc ...")
        #  x = x[:, -1, :]
        x = self.fc(x)
        out = self.sigmoid(x)
        return out.view(-1)

def run_impl(in_np, begin, end, num_sequences, output_seq_dim, runner, out_np):
    start_id = begin
    inputTensors = runner.get_input_tensors()
    outputTensors = runner.get_output_tensors()
    batch_size, num_frames, runner_in_seq_len = tuple(inputTensors[0].dims)
    _, _, runner_out_seq_len = tuple(outputTensors[0].dims)
    output_buf = np.empty((end - start_id, 1, output_seq_dim), dtype=np.int16)

    while(begin < end):
        eff_batch = min(end-begin, batch_size)
        input_data = in_np[begin: begin + eff_batch].reshape(-1, num_sequences, runner_in_seq_len)
        output_data = np.empty((eff_batch, num_sequences, runner_out_seq_len), dtype=np.int16)
        job_id = runner.execute_async([input_data], [output_data], True)
        runner.wait(job_id)
        offset = begin - start_id
        output_buf[offset:offset+eff_batch, ...] = output_data[:, -1, :output_seq_dim].reshape(
                  eff_batch, 1, output_seq_dim)
        begin += eff_batch
    out_np[start_id:end, ...] = output_buf


class HWRun:
    def __init__(self, num_cores=2, num_sequences=max_review_length, output_seq_dim=hidden_vector_length):
        # Setup the Runners
        self.num_sequences = num_sequences
        self.output_seq_dim = output_seq_dim
        self.num_cores = num_cores
        models = [
            "data/compiled_batch_3.xmodel",
            "data/compiled_batch_4.xmodel",
        ]
        self.runners = []
        self.batches = []
        for i in range(self.num_cores):
            model = models[i % len(models)]
            print("Initiating Runner-{} from {}".format(i, model))
            graph = xir.Graph.deserialize(model)
            self.runners.append(vart.Runner.create_runner(graph.get_root_subgraph(), "run"))
            inputTensors = self.runners[i].get_input_tensors()
            outputTensors = self.runners[i].get_output_tensors()
            batch_size, num_frames, runner_in_seq_len = tuple(inputTensors[0].dims)
            _, _, runner_out_seq_len = tuple(outputTensors[0].dims)
            self.batches.append(batch_size)
        self.out_pos = graph.get_root_subgraph().get_attr("output_fix2float")
        self.in_pos = graph.get_root_subgraph().get_attr("input_float2fix")

    def run(self, xin_pt):
        print("Running GRU on DPU ...")
        inputTensors = self.runners[0].get_input_tensors()
        outputTensors = self.runners[0].get_output_tensors()
        batch_size, num_frames, runner_in_seq_len = tuple(inputTensors[0].dims)
        _, _, runner_out_seq_len = tuple(outputTensors[0].dims)
        xin_np = xin_pt.cpu().numpy()
        num_records = xin_np.shape[0]
        t1 = time.time()
        quantized_lstm_input = quanti_convert_float_to_int16(xin_np, self.in_pos).reshape(
            (num_records, total_vec_length)
        )
        t2 = time.time()

        lstm_output = np.zeros((quantized_lstm_input.shape[0], 1, self.output_seq_dim), dtype=np.int16)

        quantized_input = quantized_lstm_input.view()
        out_np = lstm_output.view()
        threads = []
        begin = 0
        nrec = math.ceil(num_records / sum(self.batches))
        for i, runner in enumerate(self.runners):
            end = min(num_records, begin + nrec * self.batches[i])
            print("Runner : {}, Batch Size : {}, #Batches : {}".format(
                i, self.batches[i], math.ceil((end-begin)/self.batches[i]))
            )
            threads.append(Thread(target=run_impl,
                                  args=(quantized_input, begin, end,
                                        self.num_sequences, self.output_seq_dim,
                                        runner, out_np)))
            begin = end

        t3 = time.time()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        t4 = time.time()

        lstmlayerout = quanti_convert_int16_to_float(lstm_output, self.out_pos)
        t5 = time.time()
        yout_pt = torch.from_numpy(lstmlayerout).cpu()
        return yout_pt, ((t2-t1)*1000, (t4-t3)*1000, (t5-t4)*1000)

    def __del__(self):
        while self.runners:
            del self.runners[0]


def test(preprocModel, postprocModel, gruModel, hwModel, test_loader, device="CPU"):
    preprocModel.eval()
    gruModel.eval()
    postprocModel.eval()
    #  criterion = nn.BCELoss()
    acc = 0
    for batch_idx, (x, y) in enumerate(test_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.no_grad():
            t1 = time.time()
            pre_y_ = preprocModel(x)
            t2 = time.time()
            if(device == "CPU"):
                gru_y_ = gruModel(pre_y_)
            elif(device == "U50LV"):
                gru_y_, (tq, td, tdq) = hwModel.run(pre_y_)
            t3 = time.time()
            y_ = postprocModel(gru_y_)
            t4 = time.time()
        #  test_loss += criterion(y_, y)
        acc += (
            y.cpu().numpy().astype(int) == (y_ > 0.5).cpu().numpy().astype(int)
        ).sum()
    #  test_loss /= len(test_loader.dataset)
    print("Preprocess time : {:.2f} ms".format((t2-t1) * 1000))
    if(device == "CPU"):
        print("GRU (CPU) time : {:.2f} ms".format((t3-t2) * 1000))
    else:
        print("Quantize time : {:.2f} ms".format(tq))
        print("GRU ({}) time : {:.2f} ms".format(device, td))
        print("Dequantize time : {:.2f} ms".format(tdq))
    print("Postprocess time : {:.2f} ms".format((t4-t3) * 1000))
    print("-----------------------------------------")
    print("Total time : {:.2f} ms".format((t4-t1) * 1000))
    print(
        "\nTest set: Accuracy: {}/{} ({:.0f}%)".format(
            acc,
            len(test_loader.dataset),
            100.0 * acc / len(test_loader.dataset),
        )
    )
    return acc / len(test_loader.dataset)


def quanti_convert_float_to_int16(data, fix_pos):
    amp = 2 ** fix_pos
    max = 2 ** (16 - 1)

    output = data * amp
    output = np.clip(output, -max, max - 1)
    #  mask = 0.5 * np.ones_like(output)
    #  mask[output < 0] = -0.5
    mask = 0
    output = np.floor(output + mask)
    output = output.astype(np.int16)
    return output


def quanti_convert_int16_to_float(data, fix_pos):
    amp = 2 ** fix_pos
    output = data.astype(np.float32)
    output = data / amp
    return output.astype(np.float32)


def main(args):
    print("Loading data ...")
    (_, _), (X_test, y_test) = imdb.load_data(path="imdb.npz", num_words=top_words)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

    test_data = TensorDataset(torch.LongTensor(X_test), torch.FloatTensor(y_test))
    test_loader = DataLoader(test_data, batch_size=25000, shuffle=False)
    PATH = "data/model.best.pth"

    preprocmodel = PreProcModel(
        top_words, embedding_vector_length, hidden_vector_length
    ).to(DEVICE)
    print(preprocmodel)
    preprocmodel.load_state_dict(
        torch.load(PATH, map_location=torch.device("cpu")), strict=False
    )

    grumodel = GRUModel(
        top_words, embedding_vector_length, hidden_vector_length
    ).to(DEVICE)
    print(grumodel)
    grumodel.load_state_dict(
        torch.load(PATH, map_location=torch.device("cpu")), strict=False
    )

    # PostProc
    postprocmodel = PostProcModel(
        top_words, embedding_vector_length, hidden_vector_length
    ).to(DEVICE)
    print(postprocmodel)
    postprocmodel.load_state_dict(
        torch.load(PATH, map_location=torch.device("cpu")), strict=False
    )

    # HWModel
    hwModel = HWRun(args.num_runners) if args.device != "CPU" else None
    #  optimizer = optim.Adam(postprocmodel.parameters())

    _ = test(preprocmodel, postprocmodel, grumodel, hwModel, test_loader, args.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", required=True,
                        help="specify the device for inference",
                        type=str, choices=("CPU", "U50LV"))
    parser.add_argument("-n", "--num_runners", default=2, type=int,
                        help="number of parallel runners")
    args = parser.parse_args()
    main(args)
