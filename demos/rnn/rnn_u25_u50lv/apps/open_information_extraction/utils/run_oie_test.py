#Copyright 2021 Xilinx Inc.
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

#MIT License
#
#Copyright (c) 2016 Gabriel Stanovsky
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

""" Usage:
    <file-name> --in=INPUT_FILE --batch-size=BATCH-SIZE --out=OUTPUT_FILE --model-path=MODEL_PATH [--cuda-device=CUDA_DEVICE] [--debug]
"""
# External imports
import os
import logging
from pprint import pprint
from pprint import pformat
from docopt import docopt
import json
import pdb
from tqdm import tqdm
import torch
from allennlp.common.params import Params
from allennlp.models.model import Model
from allennlp.models.archival import Archive, CONFIG_NAME, _WEIGHTS_NAME
from allennlp.predictors import Predictor
from allennlp.nn import util
from allennlp.data import Instance, Vocabulary
from collections import defaultdict
from operator import itemgetter
import functools
import operator
import time

# Local imports
from format_oie import format_extractions, Mock_token
#=-----

def chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]

def create_instances(model, sent):
    """
    Convert a sentence into a list of instances.
    """
    sent_tokens = model._tokenizer.tokenize(sent)
    # Find all verbs in the input sentence
    pred_ids = [i for (i, t) in enumerate(sent_tokens)
                if t.pos_ == "VERB"]

    # Create instances
    instances = [{"sentence": sent_tokens,
                  "predicate_index": pred_id}
                 for pred_id in pred_ids]

    return instances

def get_confidence(model, tag_per_token, class_probs):
    """
    Get the confidence of a given model in a token list, using the class probabilities
    associated with this prediction.
    """
    #tag_per_token = list([tag for tag in tag_per_token if tag != 'O'])
    token_indexes = [model._model.vocab.get_token_index(tag, namespace = "labels") for tag in tag_per_token]
    # Get probability per tag
    #probs = [class_prob[token_index] for token_index, class_prob in zip(token_indexes, class_probs) if token_index != 0]
    probs = [class_prob[token_index] for token_index, class_prob in zip(token_indexes, class_probs)]

    # Combine (product)
    if len(probs) == 0:
        prod_prob = 0.0
        print(tag_per_token)
    else:
        prod_prob = functools.reduce(operator.mul, probs)

    return prod_prob

def remove_pretrained_embedding_params(params: Params):
    if isinstance(params, Params):  # The model could possible be a string, for example.
        keys = params.keys()
        if "pretrained_file" in keys:
            del params["pretrained_file"]
        for value in params.values():
            if isinstance(value, Params):
                remove_pretrained_embedding_params(value)

def openie_model(serialization_dir, weights_file=None, cuda_device=-1):
    """
    Instantiates an already-trained model, based on the experiment
    configuration and some optional overrides.
    """
    # Load config
    config = Params.from_file(os.path.join(serialization_dir, CONFIG_NAME), "")
    config.loading_from_archive = True
    config = config.duplicate()

    weights_file = weights_file or os.path.join(serialization_dir, _WEIGHTS_NAME)

    # Load vocabulary from file
    vocab_dir = os.path.join(serialization_dir, "vocabulary")
    # If the config specifies a vocabulary subclass, we need to use it.
    vocab_params = config.get("vocabulary", Params({}))
    vocab_choice = vocab_params.pop_choice("type", Vocabulary.list_available(), True)
    vocab = Vocabulary.by_name(vocab_choice).from_files(vocab_dir)

    model_params = config.get("model")

    # The experiment config tells us how to _train_ a model, including where to get pre-trained
    # embeddings from.  We're now _loading_ the model, so those embeddings will already be
    # stored in our weights.  We don't need any pretrained weight file anymore, and we don't
    # want the code to look for it, so we remove it from the parameters here.
    remove_pretrained_embedding_params(model_params)
    model = Model.from_params(vocab=vocab, params=model_params)

    # If vocab+embedding extension was done, the model initialized from from_params
    # and one defined by state dict in weights_file might not have same embedding shapes.
    # Eg. when model embedder module was transferred along with vocab extension, the
    # initialized embedding weight shape would be smaller than one in the state_dict.
    # So calling model embedding extension is required before load_state_dict.
    # If vocab and model embeddings are in sync, following would be just a no-op.
    model.extend_embedder_vocab()

    model_state = torch.load(weights_file, map_location=util.device_mapping(cuda_device))

    model.load_state_dict(model_state)

    # Force model to cpu or gpu, as appropriate, to make sure that the embeddings are
    # in sync with the weights
    if cuda_device >= 0:
        model.cuda(cuda_device)
    else:
        model.cpu()

    return model, config

if __name__ == "__main__":
    # Parse command line arguments
    args = docopt(__doc__)
    inp_fn = args["--in"]
    batch_size = int(args["--batch-size"])
    out_fn = args["--out"]
    model_path = args["--model-path"]
    cuda_device = int(args["--cuda-device"]) if (args["--cuda-device"] is not None) \
                  else -1
    debug = args["--debug"]
    if debug:
        logging.basicConfig(level = logging.DEBUG)
    else:
        logging.basicConfig(level = logging.INFO)

    # Init OIE
    raw_model, config = openie_model(model_path)
    # insert nndct code
    archive = Archive(model=raw_model, config=config)
    model = Predictor.from_archive(archive, "open-information-extraction")
    # model = open_information_extraction_stanovsky_2018()

    # Move model to gpu, if requested
    if cuda_device >= 0:
        model._model.cuda(cuda_device)

    lines = [line.strip()
             for line in open(inp_fn, encoding = "utf8")]

    # process sentences
    logging.info("Processing sentences")
    oie_lines = []
    loops = 0
    while (True):
        t1 = time.time()
        for chunk in tqdm(chunks(lines, batch_size)):
            oie_inputs = []
            for sent in chunk:
                oie_inputs.extend(create_instances(model, sent))
            if not oie_inputs:
                # No predicates in this sentence
                continue

            # Run oie on sents
            sent_preds = model.predict_batch_json(oie_inputs)

            # Collect outputs in batches
            predictions_by_sent = defaultdict(list)
            for outputs in sent_preds:
                sent_tokens = outputs["words"]
                tags = outputs["tags"]
                sent_str = " ".join(sent_tokens)
                assert(len(sent_tokens) == len(tags))
                predictions_by_sent[sent_str].append((outputs["tags"], outputs["class_probabilities"]))

            # Create extractions by sentence
            for sent_tokens, predictions_for_sent in predictions_by_sent.items():
                raw_tags = list(map(itemgetter(0), predictions_for_sent))
                class_probs = list(map(itemgetter(1), predictions_for_sent))

                # Compute confidence per extraction
                confs = [get_confidence(model, tag_per_token, class_prob)
                         for tag_per_token, class_prob in zip(raw_tags, class_probs)]

                extractions, tags = format_extractions([Mock_token(tok) for tok in sent_tokens.split(" ")], raw_tags)
                oie_lines.extend([extraction + f"\t{conf}" for extraction, conf in zip(extractions, confs)])
        loops+=1
        print("CURRENT LOOP: ", loops)
        t2 = time.time()
        print("E2E time: ", t2-t1)

    # Write to file
    logging.info(f"Writing output to {out_fn}")
    with open(out_fn, "w", encoding = "utf8") as fout:
        fout.write("\n".join(oie_lines))
    logging.info("DONE")
