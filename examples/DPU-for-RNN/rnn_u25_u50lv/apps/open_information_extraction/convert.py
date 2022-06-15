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

sents = set()
with open("./utils/test.oie", 'r') as f:
    for line in f:
        sent = line.strip().split("\t")[0]
        sents.add(sent)
with open("./test/test.oie.sent", 'w') as f:
    for sent in sents:
        f.write(sent + "\n")
with open("./test/test_in.txt", 'w') as f:
    f.write(sent)
