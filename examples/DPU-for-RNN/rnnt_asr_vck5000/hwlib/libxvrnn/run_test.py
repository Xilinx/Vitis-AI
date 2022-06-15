"""
Copyright 2019 Xilinx Inc.
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

import xrnn_py
import numpy as np
import soundfile as sf
import torch
import prettytable as pt
import time
import pdb
#from metrics import word_error_rate

class RNNT_infer_model():
        def __init__(self):
                #init model
                self.model = xrnn_py.xrnn("rnnt")
                #self.output_size = 0x200000
                self.output_size = 0x100000
                self.decoder_frame_in_ddr_bin = 0x7c40000
                self.split_length = 400

                self.output_num = np.zeros(self.output_size, dtype=np.int16)
                self.labels = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', "'", '<BLANK>']
                self.max_output_length = 0

        def test_op(self):
                '''
                for debug, copy from test.py
                '''
                frame_num = 400
                a = xrnn_py.xrnn("rnnt")
                input_num = np.fromfile("32input", dtype=np.int8)
                input_size = input_num.size # number of input data

                offset_in_ddr_bin = 0x7c40000
                decoder_frm_num = np.fromfile("32frm", dtype=np.int32)
                frm_num_size = decoder_frm_num.size
                a.rnnt_update_ddr(decoder_frm_num.flatten(), frm_num_size, offset_in_ddr_bin)
                r_out = np.zeros(0x400, dtype=np.int8)
                a.rnnt_download_ddr(r_out, 0x400, offset_in_ddr_bin)
                r_out.tofile("./r_out_bin")

                output_size = 0x200000  # max bytes number of output buffer is 0x100000
                output_num = np.zeros(output_size, dtype=np.int16)
                a.lstm_run(input_num.flatten(), input_size, output_num, output_size, frame_num)
                output_num.tofile("./rslt_bin")

        
        def convert(self, t_audio_signal_e,cut_frm=None):
                #new version with new data design
                """
                t_audio_signal_e: input from preprocessed torch tensor feature. shape,[frm,batch,feature]
                padding to (frm,32,320)
                auto_judge,if true, split input to several inputs list less than self.split_length(400) frms
                """ 
                self.batch = t_audio_signal_e.shape[1]
                self.input = (t_audio_signal_e*2**5).floor().numpy()

                #batch=32 最后一个batch不到32，导致输入是[frm,15,feature]]
                if self.batch ==32:
                        self.input = np.pad(self.input,((0,0),(0,0),(0,80)), 'constant', constant_values=((0,0),(0,0),(0,0)))
                elif self.batch==1:
                        self.input = np.pad(self.input,((0,0),(0,0),(0,80)), 'constant', constant_values=((0,0),(0,0),(0,0))).repeat(32,axis=1)
                elif self.batch>32:
                        print('Not support batch>32')
                else:
                        self.input = np.pad(self.input,((0,0),(0,32-self.batch),(0,80)), 'constant', constant_values=((0,0),(0,0),(0,0)))
                        
                if cut_frm:
                        self.input = self.input[:cut_frm,:,:] 

                self.frm_num = self.input.shape[0]

                if self.frm_num<=4:
                        pad_frm = 4-self.frm_num
                        self.input = np.pad(self.input,((0,pad_frm),(0,0),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
                        self.frm_num = self.input.shape[0] 
                
                #add transpose and pad frame
                self.input = np.pad(self.input,((0,400-self.frm_num),(0,0),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
                self.input = np.transpose(self.input,(1,0,2))
                self.input = self.input.astype(np.int8)
                self.input = self.input.flatten()
                self.input_size = self.input.size

        def read_np_output(self):
                """
                output are stored in numpy bin file in self.output_num, with ffffff as end, read it to character
                and print it
                内存里一行8个数，一行对应一个输出，内存新版本是batch占据固定长度memory，内部有所有的输出了,
                每个输出占用8，然后空出3行，也就是24，没有输出后拿ffffff作为结尾
                """
                memory = 16000
                batch_output_index= []
                for batch_idx in range(self.batch):
                        i = 0
                        self.output_char =''
                        output_index = []

                        if self.fake_real[batch_idx]:
                        #if True:
                                while self.output_num[4*8*i+batch_idx*memory] != -1:
                                        if self.output_num[batch_idx*memory+4*8*i] != 28:
                                                self.output_char += self.labels[self.output_num[batch_idx*memory+4*8*i]]
                                                output_index.append(int(self.output_num[batch_idx*memory+4*8*i]))
                                        i += 1
                        self.max_output_length = max(self.max_output_length,i)
                        #print(self.output_char, end='',flush = True)
                        batch_output_index.append(output_index)

                return batch_output_index

        def read_np_output_longdemo(self):
                """
                output are stored in numpy bin file in self.output_num, with ffffff as end, read it to character
                and print it
                内存里一行8个数，一行对应一个输出，内存新版本是batch占据固定长度memory，内部有所有的输出了,
                每个输出占用8，然后空出3行，也就是24，没有输出后拿ffffff作为结尾
                """
                memory = 16000
                batch_output_index= []
                for batch_idx in range(self.batch):
                        i = 0
                        self.output_char =''
                        output_index = []

                        if True:
                                while self.output_num[4*8*i+batch_idx*memory] != -1:
                                        if self.output_num[batch_idx*memory+4*8*i] != 28:
                                                self.output_char += self.labels[self.output_num[batch_idx*memory+4*8*i]]
                                                output_index.append(int(self.output_num[batch_idx*memory+4*8*i]))
                                        i += 1
                        self.max_output_length = max(self.max_output_length,i)
                        print(self.output_char, end='',flush = True)
                        batch_output_index.append(output_index)

                return batch_output_index

        def dump_output(self,output_path):
                self.output_num.tofile(output_path)

        def print_info(self):
                print('input frame number:',self.frm_num)

        def dump_input(self):
                self.input.tofile('input_bin')

        def convert_label(self, label_tensor):
                out = ""
                for i in label_tensor:
                        out += self.labels[i]

                return out

        def print_wav_info(self, label_tensor):
                #print demo4 long wav info
                tb = pt.PrettyTable()
                tb.field_names = [ "Channels", "Sample Rate", "Bitrate", "Duration(s)"]
                tb.add_row([1, 16000, 16,103])
                print()
                print('Test wav information summary:')
                print(tb)
                print()
                print("Test wav label:")
                print(self.convert_label(label_tensor))
                print()


        def compute_wer(self, prediction, label):
                pass
        
        def rnnt_reflash_ddr(self):
                self.model.rnnt_reflash_ddr()

        def print_summary(self):

                tb = pt.PrettyTable()
                tb.field_names = [ "Decoding Time(s)", "Real Time Factor (RTF)","Block size (samples)", "Cpu decoding time(s)", "Word Error Rate (WER)"]
                tb.add_row([self.decoding_time, self.decoding_time/103.0, self.block_size, 9.72, '7.02%'])
                print(tb)

        def preprocess_part(self,x):
                '''
                change sample to shape [1,L] tensor  for streaming load usage
                input x: samples read from soundfile
                return  tensor with shape [1,L] where L means sample number
                '''
                part = x.transpose()
                part = torch.tensor(part,dtype=torch.double)
                part = torch.unsqueeze(part,0)
                return part

        def concate_2list(self,a,b):
                '''
                a,b 2-d list, with batch_size element,each element has different length
                '''
                result = []
                for i in range(len(a)):
                        result.append(a[i]+b[i])
                return result

        def update_ddr(self):
                '''
                seq_len: torch tensor with shaape torch.Tensor([batch]), sequence length of each sample after cut the origin frame to less than 400 frames, and after time reduction
                fake real to check if the sample in batch is zeros, don't read them when finish computation
                
                '''
                self.seq_len = torch.zeros(self.batch)
                self.fake_real = torch.zeros(self.batch)
                for i in range(self.batch):
                        tmp_l = self.orig_input_batch_seq_len[i]
                        if tmp_l>self.split_length:
                                self.seq_len[i] = int(self.split_length/2)
                                self.fake_real[i] = 1 
                        elif tmp_l<1:
                                self.seq_len[i] = 2
                        else:
                                self.seq_len[i] = int((int(tmp_l)+1)/2)
                                self.fake_real[i] = 1
                self.orig_input_batch_seq_len -= self.split_length

                frame_info = self.seq_len.numpy().astype(np.int32)

                self.model.rnnt_update_ddr(frame_info.flatten(), frame_info.size, self.decoder_frame_in_ddr_bin)
                
        def test_run(self, input, cut_frm):
                #run from tensor feature
                self.convert(input,cut_frm)
                self.print_info()
                self.dump_input()
                self.model.lstm_run(self.input, self.input_size, self.output_num, self.output_size, self.frm_num)
        
        def run(self, input):
                """
                input: input tensor feature
                output are stored in numpy bin file in self.output_num
                """
                self.convert(input)
                self.print_info()
                
                self.model.lstm_run(self.input, self.input_size, self.output_num, self.output_size, self.frm_num)

        def run_librispeech(self, input, seq_len):

                """
                input: input tensor feature
                seq_len:torch.Size([batch]),input seq_length for each sample
                output are stored in numpy bin file in self.output_num
                cut the feature if frm num is larger than split_length(400)
                """
                frm_orig = input.shape[0]
                self.batch = input.shape[1]
                self.orig_input_batch_seq_len = seq_len
                result_index_list = [[] for i in range(self.batch)]
                lstm_run_time = 0
                convert_time = 0
                read_time = 0
                if frm_orig>self.split_length:
                        self.split_num = int(frm_orig/self.split_length)
                        for i in range(self.split_num):
                                c_s = time.time()
                                self.convert(input[self.split_length*i:self.split_length*(i+1),:,:])
                                convert_time += time.time()-c_s
                                self.update_ddr()
                                s = time.time()
                                self.model.lstm_run(self.input, self.input_size, self.output_num, self.output_size, self.frm_num)
                                lstm_run_time += time.time()-s
                                r_s = time.time()
                                out_index = self.read_np_output()
                                read_time += time.time()-r_s
                                result_index_list = self.concate_2list(result_index_list,out_index)
                        c_s = time.time()
                        self.convert(input[self.split_length*self.split_num:,:,:])
                        convert_time += time.time()-c_s
                        self.update_ddr()
                        s = time.time()
                        self.model.lstm_run(self.input, self.input_size, self.output_num, self.output_size, self.frm_num)
                        lstm_run_time += time.time()-s
                        r_s = time.time()
                        out_index = self.read_np_output()
                        read_time += time.time()-r_s
                        result_index_list = self.concate_2list(result_index_list,out_index)
                else:
                        c_s = time.time()
                        self.convert(input)
                        convert_time += time.time()-c_s
                        self.update_ddr()
                        s = time.time()
                        self.model.lstm_run(self.input, self.input_size, self.output_num, self.output_size, self.frm_num)
                        lstm_run_time += time.time()-s
                        r_s = time.time()
                        out_index = self.read_np_output()
                        read_time += time.time()-r_s
                        result_index_list = self.concate_2list(result_index_list,out_index)

                return result_index_list, lstm_run_time,convert_time,read_time, self.max_output_length
                        
        def _run_short(self, input):
                """
                input: input tensor feature,less than 400 frame
                output are stored in numpy bin file in self.output_num
                """
                convert_time,lstm_run_time,read_time = 0,0,0

                self.orig_input_batch_seq_len = torch.zeros(1,1)
                self.orig_input_batch_seq_len[0] = input.shape[0]
                tmp = time.time() 
                self.convert(input)
                self.update_ddr()
                convert_time += time.time()-tmp
                tmp = time.time()
                self.model.lstm_run(self.input, self.input_size, self.output_num, self.output_size, self.frm_num)
                lstm_run_time = time.time()-tmp
                tmp=time.time()
                out_index = self.read_np_output_longdemo()
                read_time = time.time()-tmp
                
                time_info = {'lstmrun':lstm_run_time,
                             'convert':convert_time,
                             'read':read_time}

                return out_index, time_info

        def run_from_np(self, input_path, frm_num):
                self.input = np.fromfile(input_path, dtype=np.int8)
                self.input_size = self.input.size
                self.frm_num = frm_num
                self.model.lstm_run(self.input.flatten(), self.input_size, self.output_num, self.output_size, self.frm_num)
                out_index = self.read_np_output()


        def run_long_demo(self, wav, feature_converter, cut_time):
                """
                wav: wav file path 
                feature_converter: to processs feature, class filterbank
                cut_time: cut long wac into equal length wav file, block size
                """

                print('Start Decoding, decoding in blocks, block size is approximately 6 seconds')
                self.long_wav = ''
                #self.block_size = cut_time*16320
                self.block_size = cut_time*15840
                wav = sf.blocks(wav, blocksize=self.block_size, overlap=0, dtype='float32')        
                result_index=[]
                result_index_list = [[]]
                preprocess_time = 0
                time_info = {'lstmrun':0,
                             'convert':0,
                             'read':0,
                             'preprocess_time':0}
                start = time.time()
                for part in wav:
                        part = self.preprocess_part(part)
                        part_length = torch.tensor([int(part.shape[1])])
                        p_s = time.time()
                        input = feature_converter([part, part_length])
                        preprocess_time += time.time()-p_s
                        index, time_info_dict = self._run_short(input)
                        result_index_list = self.concate_2list(result_index_list,index)
                        for k, v in time_info_dict.items():
                                time_info[k]+=v
                time_info['preprocess_time'] = preprocess_time
                end = time.time()
                self.decoding_time = end-start

                print()
                self.print_summary()
                print('dpu time:',time_info['lstmrun'])
                # Fix me , result_index is batch 1 ,need to support multi batch index
                return result_index_list
                
if __name__=='__main__':
        m = RNNT_infer_model()
        #input_num = np.fromfile("./golden_inout/vector_bin", dtype=np.uint16)
        #m.run_from_np("./golden_inout/vector_bin", 26)
        #m.dump_output("./rslt/rslt_bin")


