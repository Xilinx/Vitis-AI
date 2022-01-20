#!/bin/env python

import sys
import xml.etree.ElementTree

def vbnv(root):
	v = root.attrib['{http://www.xilinx.com/sdx}vendor']
	b = root.attrib['{http://www.xilinx.com/sdx}library']
	n = root.attrib['{http://www.xilinx.com/sdx}name']
	r = root.attrib['{http://www.xilinx.com/sdx}version']

	return ':'.join([v,b,n,r])

def hw(root):
	xpath = './{http://www.xilinx.com/sdx}hardwarePlatforms/{http://www.xilinx.com/sdx}hardwarePlatform'
	sdx = root.find(xpath).attrib

	dir = sdx['{http://www.xilinx.com/sdx}path']
	dsa = sdx['{http://www.xilinx.com/sdx}name']

	return dir + '/' + dsa

file = sys.argv[1]
mode = sys.argv[2]

tree = xml.etree.ElementTree.parse(file)

root = tree.getroot()

if mode == "dsa":
	print vbnv(root)

if mode == "hw":
	print hw(root)
