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
#
#MIT License

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

''' 
Usage:
   benchmark --gold=GOLD_OIE --out=OUTPUT_FILE (--stanford=STANFORD_OIE | --ollie=OLLIE_OIE |--reverb=REVERB_OIE | --clausie=CLAUSIE_OIE | --openiefour=OPENIEFOUR_OIE | --props=PROPS_OIE | --tabbed=TABBED)

Options:
  --gold=GOLD_OIE              The gold reference Open IE file (by default, it should be under ./oie_corpus/all.oie).
  --out-OUTPUT_FILE            The output file, into which the precision recall curve will be written.
  --clausie=CLAUSIE_OIE        Read ClausIE format from file CLAUSIE_OIE.
  --ollie=OLLIE_OIE            Read OLLIE format from file OLLIE_OIE.
  --openiefour=OPENIEFOUR_OIE  Read Open IE 4 format from file OPENIEFOUR_OIE.
  --props=PROPS_OIE            Read PropS format from file PROPS_OIE
  --reverb=REVERB_OIE          Read ReVerb format from file REVERB_OIE
  --stanford=STANFORD_OIE      Read Stanford format from file STANFORD_OIE
  --tabbed=TABBED              Read tabbed format from file TABBED
'''
import docopt
import string
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc, f1_score, precision_recall_fscore_support
import re
import logging
logging.basicConfig(level = logging.INFO)

from oie_readers.stanfordReader import StanfordReader
from oie_readers.ollieReader import OllieReader
from oie_readers.reVerbReader import ReVerbReader
from oie_readers.clausieReader import ClausieReader
from oie_readers.openieFourReader import OpenieFourReader
from oie_readers.propsReader import PropSReader
from oie_readers.tabReader import TabReader

from oie_readers.goldReader import GoldReader
from matcher import Matcher

class Benchmark:
    ''' Compare the gold OIE dataset against a predicted equivalent '''
    def __init__(self, gold_fn):
        ''' Load gold Open IE, this will serve to compare against using the compare function '''
        gr = GoldReader() 
        gr.read(gold_fn)
        self.gold = gr.oie

    def compare(self, predicted, matchingFunc, output_fn):
        ''' Compare gold against predicted using a specified matching function. 
            Outputs PR curve to output_fn '''
        
        y_true = []
        y_scores = []
        
        correctTotal = 0
        unmatchedCount = 0        
        predicted = Benchmark.normalizeDict(predicted)
        gold = Benchmark.normalizeDict(self.gold)
                
        for sent, goldExtractions in list(gold.items()):
            if sent not in predicted:
                # The extractor didn't find any extractions for this sentence
                unmatchedCount += len(goldExtractions)
                correctTotal += len(goldExtractions)
                continue
                
            predictedExtractions = predicted[sent]
            
            for goldEx in goldExtractions:
                correctTotal += 1
                found = False
                
                for predictedEx in predictedExtractions:
                    if matchingFunc(goldEx, 
                                    predictedEx, 
                                    ignoreStopwords = True, 
                                    ignoreCase = True):
                        
                        y_true.append(1)
                        y_scores.append(predictedEx.confidence)
                        predictedEx.matched.append(output_fn)

                        # Also mark any other predictions with the
                        # same exact predicate as matched.
                        # This is to support packages that do conjunction
                        # splitting, and doesn't affect the results for
                        # packages that don't.
                        if predictedEx.splits_conjunctions:
                            for otherPredictedEx in predictedExtractions:
                                if otherPredictedEx.pred == predictedEx.pred:
                                    otherPredictedEx.matched.append(output_fn)

                        found = True
                        break
                    
                if not found:
                    unmatchedCount += 1
                    
            for predictedEx in [x for x in predictedExtractions if (output_fn not in x.matched)]:
                # Add false positives
                y_true.append(0)
                y_scores.append(predictedEx.confidence)
                
        y_true = y_true
        y_scores = y_scores
        # recall on y_true, y  (r')_scores computes |covered by extractor| / |True in what's covered by extractor|
        # to get to true recall we do r' * (|True in what's covered by extractor| / |True in gold|) = |true in what's covered| / |true in gold|
        p, r, thr = Benchmark.prCurve(np.array(y_true), np.array(y_scores),
                       recallMultiplier = ((correctTotal - unmatchedCount)/float(correctTotal)))
        #f1_scores = 2*(p*r)/(p+r)
        # write PR to file
        with open(output_fn, 'w') as fout:
            fout.write('{0}\t{1}\n'.format("Precision", "Recall"))
            for cur_p, cur_r in sorted(zip(p, r), key = lambda cur_p_cur_r: cur_p_cur_r[1]):
                fout.write('{0}\t{1}\n'.format(cur_p, cur_r))
        #print('fake thr: ', thr[np.nanargmax(f1_scores)])
        #print('fake f1 : ', np.nanmax(f1_scores))
        print('auc: ', auc(r, p))
        thr_gold = 1.204235e-12
        for i, score in enumerate(y_scores):
            if score > thr_gold:
                y_scores[i] = 1.0
            else:
                y_scores[i] = 0.0
        #p, r, thr = Benchmark.prCurve(np.array(y_true), np.array(y_scores),
        #               recallMultiplier = ((correctTotal - unmatchedCount)/float(correctTotal)))
        recallMultiplier = (correctTotal - unmatchedCount)/float(correctTotal)
        p, r, _, _ = precision_recall_fscore_support(y_true, np.array(y_scores), average="binary")
        r *= recallMultiplier
        #print(p, r)
        f1_scores = 2*(p*r)/(p+r)
        print('f1 : ', f1_scores)
    @staticmethod
    def prCurve(y_true, y_scores, recallMultiplier):
        # Recall multiplier - accounts for the percentage examples unreached by 
        precision, recall, thr = precision_recall_curve(y_true, y_scores)
        recall = recall * recallMultiplier
        return precision, recall, thr

    # Helper functions:
    @staticmethod
    def normalizeDict(d):
        return dict([(Benchmark.normalizeKey(k), v) for k, v in list(d.items())])
    
    @staticmethod
    def normalizeKey(k):
        return Benchmark.removePunct(str(Benchmark.PTB_unescape(k.replace(' ',''))))

    @staticmethod
    def PTB_escape(s):
        for u, e in Benchmark.PTB_ESCAPES:
            s = s.replace(u, e)
        return s
    
    @staticmethod
    def PTB_unescape(s):
        for u, e in Benchmark.PTB_ESCAPES:
            s = s.replace(e, u)
        return s
    
    @staticmethod
    def removePunct(s):
        return Benchmark.regex.sub('', s)
    
    # CONSTANTS
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    
    # Penn treebank bracket escapes 
    # Taken from: https://github.com/nlplab/brat/blob/master/server/src/gtbtokenize.py
    PTB_ESCAPES = [('(', '-LRB-'),
                   (')', '-RRB-'),
                   ('[', '-LSB-'),
                   (']', '-RSB-'),
                   ('{', '-LCB-'),
                   ('}', '-RCB-'),]


if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    logging.debug(args)
    if args['--stanford']:
        predicted = StanfordReader()
        predicted.read(args['--stanford'])
    
    if args['--props']:
        predicted = PropSReader()
        predicted.read(args['--props'])
       
    if args['--ollie']:
        predicted = OllieReader()
        predicted.read(args['--ollie'])
    
    if args['--reverb']:
        predicted = ReVerbReader()
        predicted.read(args['--reverb'])
    
    if args['--clausie']:
        predicted = ClausieReader()
        predicted.read(args['--clausie'])
        
    if args['--openiefour']:
        predicted = OpenieFourReader()
        predicted.read(args['--openiefour'])
    if args['--tabbed']:
        predicted = TabReader()
        predicted.read(args['--tabbed'])


    b = Benchmark(args['--gold'])
    out_filename = args['--out']

    logging.info("Writing PR curve of {} to {}".format(predicted.name, out_filename))
    b.compare(predicted = predicted.oie, 
               matchingFunc = Matcher.lexicalMatch,
               output_fn = out_filename)
    
        
        
