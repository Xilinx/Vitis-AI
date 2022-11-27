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

# Copyright (c) 2019, Myrtle Software Limited. All rights reserved.
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

import torch

import torch.nn.functional as F
from model_rnnt import label_collate
#from utils.torchquantizer import do_scan,do_quantize, set_value,get_mode, update_blobs, save_blobs
l = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"]
DICT ={i:l[i] for i in range(len(l))}
class TransducerDecoder:
    """Decoder base class.

    Args:
        alphabet: An Alphabet object.
        blank_symbol: The symbol in `alphabet` to use as the blank during CTC
            decoding.
        model: Model to use for prediction.
    """

    def __init__(self, blank_index, model):
        self._model = model
        self._SOS = -1   # start of sequence
        self._blank_id = blank_index

    def _pred_step(self, label, hidden, device):
        if label == self._SOS:
            return self._model.predict(None, hidden, add_sos=False)
        if label > self._blank_id:
            label -= 1
        label = label_collate([[label]]).to(device)
        return self._model.predict(label, hidden, add_sos=False)

    def _joint_step(self, enc, pred, log_normalize=False):
        logits = self._model.joint(enc, pred)[:, 0, 0, :]
        if not log_normalize:
            return logits

        probs = F.log_softmax(logits, dim=len(logits.shape) - 1)

        return probs
    def _get_last_symb(self, labels):
        return self._SOS if labels == [] else labels[-1]


class RNNTGreedyDecoder(TransducerDecoder):
    """A greedy transducer decoder.

    Args:
        blank_symbol: See `Decoder`.
        model: Model to use for prediction.
        max_symbols_per_step: The maximum number of symbols that can be added
            to a sequence in a single time step; if set to None then there is
            no limit.
        cutoff_prob: Skip to next step in search if current highest character
            probability is less than this.
    """

    def __init__(self, blank_index, model, max_symbols_per_step=30):
        super().__init__(blank_index, model)
        assert max_symbols_per_step is None or max_symbols_per_step > 0
        self.max_symbols = max_symbols_per_step

    def decode(self, x, out_lens, h_rnns,label,hidden, quantizer=None):
        """Returns a list of sentences given an input batch.

        Args:
            x: A tensor of size (seq_len, batch, in_features).
            out_lens: list of int representing the length of each sequence
                output sequence.
            h_rnns: primier hidden of encoder Pre & Post  (hidden_pre,hidden_post)
            label: total label
            hidden: primier hidden of prediction net
        Returns:
            list containing batch number of sentences (strings).
        """
        decode_count_batch = []
        with torch.no_grad():
            # Apply optional preprocessing
            #x_packed = torch.nn.utils.rnn.pack_padded_sequence(x, out_lens)
            x_packed = torch.nn.utils.rnn.pack_padded_sequence(x, out_lens)

            logits, out_lens, h_prim_rnns = self._model.encode(x_packed,h_rnns)

            output = []
            for batch_idx in range(logits.size(0)):
                #wrong with batch32, label not [] for the post 31 data,should be []
                label = []
                inseq = logits[batch_idx, :, :].unsqueeze(1)
                logitlen = out_lens[batch_idx]
                sentence, hidden_predict, decode_count= self._greedy_decode(inseq, logitlen,label,hidden,quantizer=quantizer)
                print()
                decode_count_batch.append(decode_count)
                output.append(sentence)

        return output,h_prim_rnns,hidden_predict,decode_count_batch

    def _greedy_decode(self, x, out_len, label,hidden, quantizer=None):
        training_state = self._model.training
        self._model.eval()

        device = x.device

        #hidden = None
        pred_step = 0
        #fake_label = []
        decode_count = 0
        for time_idx in range(out_len):
            f = x[time_idx, :, :].unsqueeze(0)

            not_blank = True
            symbols_added = 0

            while not_blank and (self.max_symbols is None or symbols_added < self.max_symbols):
                #**2 embedding

                g, hidden_prime = self._pred_step(
                    self._get_last_symb(label),
                    hidden,
                    device
                )
                decode_count += 1
                #**3 linear
                logp = self._joint_step(f, g, log_normalize=False)[0, :]

                # get index k, of max prob
                v, k = logp.max(0)
                k = k.item()

                if k == self._blank_id:
                    not_blank = False
                else:
                    label.append(k)
                    hidden = hidden_prime
                    pred_step += 1
                    print(DICT[k],end='')
                symbols_added += 1

        self._model.train(training_state)
        return label,hidden, decode_count
                                                                                                     

