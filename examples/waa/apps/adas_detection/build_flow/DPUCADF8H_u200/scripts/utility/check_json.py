#!/usr/bin/env python

#
# utility that reports mismatches between data encoded
# in both the description.json and the Makefile
#

import os
import json
import collections
from collections import OrderedDict
import re
import sys

okExitCode = os.EX_OK
failExitCode = 1
exitCode = os.EX_OK
# read data from the json file

jsonLibs = []
jsonKernels = {}

match_makefile = True

fail = "FAIL"
with open("description.json") as json_file:
    data = json.load(json_file, object_pairs_hook=OrderedDict)
    if data.get("match_makefile"):
        match_makefile = data.get("match_makefile")
        match_makefile = match_makefile.lower()
        if match_makefile not in [ "true", "1", "yes" ]:
            print "WARNING: allowing mismatches between description.json and Makefile"
            fail = "WARNING"
            failExitCode = okExitCode
            match_makefile = False
    if data.get("libs"):
        jsonLibs = data.get("libs")
    if data.get("containers"):
        containers = data.get("containers")
        for containerobj in containers:
            container = containerobj.get("name")
            if containerobj.get("accelerators"):
                accelerators = containerobj.get("accelerators")
                if accelerators:
                    for accelerator in accelerators:
                        name = container + "/" + accelerator.get("name")
                        src = accelerator.get("location")
                        if src.startswith("./"):
                            src = src[2:]
                        jsonKernels[name] = src
                        print "found accelerator %s, src = %s" % (name,jsonKernels[name])
    if data.get("accelerators"):
        accelerators = data.get("accelerators")
        if accelerators:
            for accelerator in accelerators:
                name = accelerator.get("name")
                container = accelerator.get("container")
                if container:
                    name = container + "/" + name
                src = accelerator.get("location")
                if src.startswith("./"):
                    src = src[2:]
                jsonKernels[name] = src
                print "found accelerator %s, src = %s" % (name,jsonKernels[name])

# read data from the makefile

makeLibs = []
makeKernels = {}

assignments = {}

libPat = re.compile("\s*include\s+\$[{\(]COMMON_REPO[}\)]/libs/([^/]+)/.*.mk.*")
assignPat = re.compile("\s*([^=:]+)\s*[:]*=\s*(.*)\s*")
for line in open("Makefile",'r'):
    m = libPat.match(line)
    if m:
        lib = m.group(1)
        makeLibs.append(lib)
    m = assignPat.match(line)
    if m:
        key = m.group(1)
        val = m.group(2)
        assignments[key] = val

x = None
if "XCLBINS" in assignments.keys():
    x = assignments["XCLBINS"]
    x = x.strip()
xclbins = []
if x:
    xclbins = re.split(r"\W+", x)
print "xclbins: %s" % xclbins
for xclbin in xclbins:
    name = xclbin
    src = ""
    if name + "_SRCS" in assignments.keys():
        src = assignments[name + "_SRCS"]
        src = src.strip()
    # multiple kernels get their own xclbins, matching kernel/container names
    if len(xclbins) > 1:
        name += "/" + name
    if src.startswith("./"):
        src = src[2:]
    makeKernels[name] = src
    print "xclbin=%s src=%s" % (name, src)

# opencl support already baked into gui projects
# no need to include in "libs"
if "opencl" in makeLibs:
    makeLibs.remove("opencl")

# ensure libs from json match the makefile
jsonOnlyLibs = list(set(jsonLibs) - set(makeLibs))
makeOnlyLibs = list(set(makeLibs) - set(jsonLibs))

print "json libs: %s" % jsonLibs
print "make libs: %s" % makeLibs

if len(jsonOnlyLibs) > 0:
    print "%s: libs in json, missing from Makefile: %s" % (fail, jsonOnlyLibs)
    exitCode = 1
if len(makeOnlyLibs) > 0:
    print "%s: libs in Makefile, missing from json: %s" % (fail, makeOnlyLibs)
    exitCode = 1

# ensure kernels and xclbins match the makefile
jsonOnlyKernels = list(set(jsonKernels.keys()) - set(makeKernels.keys()))
makeOnlyKernels = list(set(makeKernels.keys()) - set(jsonKernels.keys()))

commonKernels = list(set(makeKernels.keys()).intersection(set(jsonKernels.keys())))

print "json kernels: %s" % jsonKernels
print "make kernels: %s" % makeKernels

if len(jsonOnlyKernels) > 0:
    print "%s: kernels in json, missing from Makefile: %s" % (fail, jsonOnlyKernels)
    exitCode = 1
if len(makeOnlyKernels) > 0:
    print "%s: kernels in Makefile, missing from json: %s" % (fail, makeOnlyKernels)
    exitCode = 1

# confirm source paths match for each kernel
for kernel in commonKernels:
    jsonSrc = jsonKernels[kernel]
    makeSrc = makeKernels[kernel]
    if jsonSrc != makeSrc:
        print "%s: source paths differ: json=%s Makefile=%s" % (fail, jsonSrc, makeSrc)
        exitCode = 1


if exitCode == okExitCode:
    print "PASS: description.json and Makefile libs and kernels match"
else:
    print "%s: description.json and Makefile do not match." % fail
    if match_makefile:
        print "If unavoidable, add \"match_makefile\" : \"false\" to description.json"
    exitCode = failExitCode

sys.exit(exitCode)
