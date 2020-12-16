# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PART OF THIS FILE AT ALL TIMES.

import json
import os

def bdd2yolo5(categorys,jsonFile,writepath):
    strs=""
    f = open(jsonFile)
    info = json.load(f)
    #print(len(info))
    #print(info["name"])
    write = open(writepath + "%s.txt" % info["name"],'w')
    for obj in info["frames"]:
        #print(obj["objects"])
        for objects in obj["objects"]:
            #print(objects)
            if objects["category"] in categorys:
                #dw = 1.0 / 1280
                #dh = 1.0 / 720
                if objects["category"] in Car:
                    objects["category"]="Vehicle"
                elif objects["category"] in Person:
                    objects["category"]="Person"
                elif objects["category"] in Sign:
                    objects["category"]="Sign"
                strs += str(info["name"])
                strs += " "
                strs += str(objects["category"])
                strs += " "
                strs += str(objects["box2d"]["x1"])
                strs += " "
                strs += str(objects["box2d"]["y1"])
                strs += " "
                strs += str(objects["box2d"]["x2"])
                strs += " "
                strs += str(objects["box2d"]["y2"])
                strs += "\n"
        write.writelines(strs)
        write.close()
        #print("%s has been dealt!" % info["name"])

if __name__ == "__main__":
    ####################args#####################
    categorys = ["person","rider","car","bus","truck","bike","motor","traffic sign","traffic light"]
    Car=["car","bus","truck","motor","bike"]
    Person=["person","rider"]
    Sign=["traffic sign","traffic light"]
    readpath = "../../bdd100k/labels/100k/val/"
    writepath = "./"
    fileList = os.listdir(readpath)
    #print(fileList)
    for file in fileList:
        print(file)
        filepath = readpath + file
        bdd2yolo5(categorys,filepath,writepath)