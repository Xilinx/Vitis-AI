#Copyright 2019 Xilinx Inc.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
from tqdm import tqdm
import toml
from dataset import AudioToTextDataLayer
from helpers import process_evaluation_batch, process_evaluation_epoch, add_blank_label, print_dict
from decoders import RNNTGreedyDecoder
from model_rnnt import RNNT
from preprocessing import AudioPreprocessing
import torch
import random
import numpy as np
import torchvision
import torch.nn as nn
import threading
import time
import sys
import os
sys.path.append(os.getcwd()+'/hwlib/libxvrnn')
from run_test import RNNT_infer_model
from parts.fbank import fbank

thread_Lock = threading.Lock()
Process_max_num = threading.Semaphore(4)
hard_time, lstm_run_time_t = [],[]
torch.set_default_dtype(torch.float32)

                
def parse_args():
    parser = argparse.ArgumentParser(description='Jasper')
    parser.add_argument("--batch_size", default=16,
                        type=int, help='data batch size')
    parser.add_argument("--steps", default=None,
                        help='if not specified do evaluation on full dataset. otherwise only evaluates the specified number of iterations for each worker', type=int)
    parser.add_argument("--model_toml", type=str,
                        help='relative model configuration path given dataset folder')
    parser.add_argument("--dataset_dir", type=str,
                        help='absolute path to dataset folder')
    parser.add_argument("--val_manifest", type=str,
                        help='relative path to evaluation dataset manifest file')
    parser.add_argument("--ckpt", default=None, type=str,
                        required=True, help='path to model checkpoint')
    parser.add_argument("--pad_to", default=None, type=int,
                        help="default is pad to value as specified in model configurations. if -1 pad to maximum duration. If > 0 pad batch to next multiple of value")
    parser.add_argument("--cudnn_benchmark",
                        action='store_true', help="enable cudnn benchmark")
    parser.add_argument("--save_prediction", type=str, default='prediction.txt',
                        help="if specified saves predictions in text form at this location")
    parser.add_argument("--logits_save_to", default=None,
                        type=str, help="if specified will save logits to path")
    parser.add_argument("--seed", default=42, type=int, help='seed')
    parser.add_argument("--mode", default = 1, type=int, help='run mode')
    parser.add_argument("--cuda",
                        action='store_true', help="use cuda", default=False)
    return parser.parse_args()


class MyProcess(threading.Thread):
        def __init__(self, data, audio_processor, _global_var_dict, labels ,func, rnnt_hw_model,gd):
                threading.Thread.__init__(self)
                self._global_var_dict = _global_var_dict
                self.labels = labels
                self.func = func
                self.data = data
                self.audio_processor = audio_processor
                self.model = rnnt_hw_model
                self.gd = gd
        def run(self):
            with Process_max_num:
                self.func(self.data, self.audio_processor, self._global_var_dict, self.labels, self.model,self.gd)

def ver_Process(data, audio_processor, _global_var_dict, labels, rnnt_hw_model, greedy_decoder):
        s = time.time()
        (t_audio_signal_e, t_a_sig_length_e,
            transcript_list, t_transcript_e,
            t_transcript_len_e) = audio_processor(data)
        h_rnns =(None,None)
        label=[]
        hidden = None
        mode = args.mode
        thread_Lock.acquire()
        if mode ==1:
                #dpu inference on dev clean
                s_t = time.time()
                (t_predictions_e, lstm_run_time,
                    convert_time, read_time,
                    max_length) = rnnt_hw_model.run_librispeech(t_audio_signal_e, t_a_sig_length_e)
                rnnt_hw_model.rnnt_reflash_ddr()
                e_t = time.time()
                hard_time.append(e_t-s_t)
                lstm_run_time_t.append(lstm_run_time)
        
        elif mode == 2:
            # test long wav
            wav_path = './my_work_dir/demo/demo4.wav'

            rnnt_hw_model.print_wav_info(t_transcript_e.data)
            cut_time = 6
            t_predictions_e = rnnt_hw_model.run_long_demo(wav_path, fbank, cut_time)

        else:
            print('mode not supported, please check it')
        
        values_dict = dict(
                predictions=[t_predictions_e],
                transcript=transcript_list,
                transcript_length=t_transcript_len_e,
        )
        process_evaluation_batch(values_dict, _global_var_dict, labels=labels)
        thread_Lock.release()

def eval(
        data_layer,
        audio_processor,
        greedy_decoder,
        labels,
        args):
        """performs inference / evaluation
        Args:
            data_layer: data layer object that holds data loader
            audio_processor: data processing module
            greedy_decoder: greedy decoder
            labels: list of labels as output vocabulary
            args: script input arguments
        """
        start_t = time.time()
        if args.mode==1 or args.mode==2:
                rnnt_hw_model = RNNT_infer_model()
        else:
                rnnt_hw_model = None
        
        logits_save_to = args.logits_save_to
        with torch.no_grad():
                _global_var_dict = {
                        'predictions': [],
                        'transcripts': [],
                        'logits': [],
                }

        Processnum = []
        for it, data in enumerate(data_layer.data_iterator):
            if args.mode == 3:
                (t_audio_signal_e, t_a_sig_length_e,
                    transcript_list, t_transcript_e,
                    t_transcript_len_e) = audio_processor(data)
                h_rnns =(None,None)
                label=[]
                hidden = None
                #greedy decode on cpu                
                t_transcript_e = torch.nn.utils.rnn.pad_packed_sequence(t_transcript_e, batch_first=True)[0]            
                t_predictions_e, h_pre_rnns, hidden_predict, decode_batch_length = greedy_decoder.decode(t_audio_signal_e, t_a_sig_length_e, h_rnns,label,hidden, None)
            else:
                Process_ver = MyProcess(data, audio_processor, _global_var_dict, labels, ver_Process,rnnt_hw_model,greedy_decoder)
                Process_ver.start()
                Processnum.append(it)
                
                if args.steps is not None and it + 1 >= args.steps:
                    break
        if args.mode !=3:
            for id in Processnum:
                Process_ver.join()
        else:
            values_dict = dict(
                    predictions=[t_predictions_e],
                    transcript=transcript_list,
                    transcript_length=t_transcript_len_e,
            )
            process_evaluation_batch(values_dict, _global_var_dict, labels=labels)

        wer = process_evaluation_epoch(_global_var_dict)
        print("=================>Evaluation WER: {0}\n".format(wer))
        if args.save_prediction is not None:
            with open(args.save_prediction, 'w') as fp:
                fp.write('\n'.join(_global_var_dict['predictions']))
        
        end_t =time.time()
        if args.mode == 1:
            print('dpu computation time (lstm_run time)', sum(lstm_run_time_t))
        print('e2e decode time:',end_t-start_t)

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    #torch.set_default_dtype(torch.double)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    #print("CUDNN BENCHMARK ", args.cudnn_benchmark)
    if args.cuda:
        assert(torch.cuda.is_available())

    model_definition = toml.load(args.model_toml)
    dataset_vocab = model_definition['labels']['labels']
    ctc_vocab = add_blank_label(dataset_vocab)

    val_manifest = args.val_manifest
    featurizer_config = model_definition['input_eval']

    if args.pad_to is not None:
        featurizer_config['pad_to'] = args.pad_to if args.pad_to >= 0 else "max"

    #print('model_config')
    #print_dict(model_definition)
    #print('feature_config')
    #print_dict(featurizer_config)
    data_layer = None
    data_layer = AudioToTextDataLayer(
        dataset_dir=args.dataset_dir,
        featurizer_config=featurizer_config,
        manifest_filepath=val_manifest,
        labels=dataset_vocab,
        batch_size=args.batch_size,
        pad_to_max=featurizer_config['pad_to'] == "max",
        shuffle=False,
        sampler='bucket' #sort by duration 
        )
    audio_preprocessor = AudioPreprocessing(**featurizer_config)

    model = RNNT(
        feature_config=featurizer_config,
        rnnt=model_definition['rnnt'],
        num_classes=len(ctc_vocab)
    )
  
    if args.ckpt is not None and args.mode in[3]:
        #print("loading model from ", args.ckpt)
        checkpoint = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    audio_preprocessor.featurizer.normalize = "per_feature"

    if args.cuda:
        audio_preprocessor.cuda()
    audio_preprocessor.eval()

    eval_transforms = []
    if args.cuda:
        eval_transforms.append(lambda xs: [xs[0].cuda(),xs[1].cuda(), *xs[2:]])
   
    eval_transforms.append(lambda xs: [*audio_preprocessor(xs[0:2]), *xs[2:]])
    # These are just some very confusing transposes, that's all.
    # BxFxT -> TxBxF
    eval_transforms.append(lambda xs: [xs[0].permute(2, 0, 1), *xs[1:]])
    eval_transforms = torchvision.transforms.Compose(eval_transforms)

    if args.cuda:
        model.cuda()
    # Ideally, I would jit this as well... But this is just the constructor...
    greedy_decoder = RNNTGreedyDecoder(len(ctc_vocab) - 1, model)

    eval(
        data_layer=data_layer,
        audio_processor=eval_transforms,
        greedy_decoder=greedy_decoder,
        labels=ctc_vocab,
        args=args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
