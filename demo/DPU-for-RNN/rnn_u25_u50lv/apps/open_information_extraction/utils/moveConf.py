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

"""Usage:
    <file-name> --in=INPUT_FILE --out=OUTPUT_FILE
"""
from docopt import docopt

def run(inf, out):
    tab_output = open(out, 'w')

    with open(inf, 'r') as f:
        for line in f:
            items = line.strip().split("\t")
            sent = items[0]
            conf = items[-1]
            args = []
            for arg in items[1:-1]:
                if arg.startswith("V"):
                    pred = arg[arg.index(':') + 1:]
                elif arg.startswith("ARG"):
                    args.append(arg[arg.index(':') + 1:])
            newline = "{}\t{}\t{}\t{}\n".format(sent, conf, pred, "\t".join(args))
            tab_output.write(newline)

    tab_output.close()

if __name__=="__main__":
    args = docopt(__doc__)
    inf = args['--in']
    out = args['--out']
    run(inf, out)
