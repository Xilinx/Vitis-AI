#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
#
# Yuanjie Huang

import os
import sys
import json
import numbers
import jinja2 as j2

from argparse import ArgumentParser

impl2file = {
    'LoadDdrToStream': 'load_master_to_stream.cpp.j2',
	'SendRamToStream': 'send_ram_to_stream.cpp.j2',
	'SendRomToStream': 'send_rom_to_stream.cpp.j2',
	'StoreStreamToMaster': 'store_stream_to_master.cpp.j2',
	'ValidateStreamWithMaster': 'validate_stream_with_master.cpp.j2',
	'ValidateStreamWithRam': 'validate_stream_with_ram.cpp.j2',
	'ValidateStreamWithRom': 'validate_stream_with_rom.cpp.j2'
}

def canonical_name(s):
    o = []
    if not s[0].isalpha():
        o.append('_')
    for c in s:
        if c.isalpha() or c.isdigit() or c == '_':
            o.append(c)
    return ''.join(o)

def get_name(e):
    """get name for in or out specification"""
    if 'buffer' in e:
        return canonical_name(e['buffer'])
    elif 'stream' in e:
        if e['width'] != 16 and e['width'] != 32 and e['width'] != 64 and e['width'] != 128:
            sys.exit('[ERROR] Width of AXI stream have to be 16, 32, 64, or 128 bits');
        return canonical_name(e['stream'])
    else:
        sys.exit('[ERROR] File name not defined');

def main():
    """docstring for main"""
    parser = ArgumentParser()
    parser.add_argument('json', help='JSON file path, static data file will be searched in same directory as this file')
    parser.add_argument('--tool-dir', dest='tool_dir', help='path to directory of input data converter')
    parser.add_argument('-d', dest='dest_dir', help='folder to write output file(s), default to same directory as JSON file',
                        default="")
    parser.add_argument('-n', '--dry-run', action='store_true', help='print to STDOUT instead of writing file')
    args = parser.parse_args()

    with open(args.json) as file:
        d = json.load(file)
    print('[INFO] Loaded DataMover kernel info from %s.' % os.path.basename(args.json))

    in_dir = os.path.dirname(args.json)
    out_dir = args.dest_dir
    if not out_dir:
        out_dir = in_dir

    rom_width = 32
    for kn, k in d.items():
        k['name'] = kn
        for i in range(len(k['map'])):
            m = k['map'][i]
            if m.get('in_port'):
                m['in_port']['name'] = get_name(m['in_port'])
            if m.get('in_file'):
                if m['in_file']['type'] == 'int8_t':
                    data_width = 8
                elif m['in_file']['type'] == 'int16_t' or m['in_file']['type'] == 'half':
                    data_width = 16
                elif m['in_file']['type'] == 'int32_t' or m['in_file']['type'] == 'float':
                    data_width = 32
                elif m['in_file']['type'] == 'int64_t' or m['in_file']['type'] == 'double':
                    data_width = 64
                if m.get('in_port'):
                    rom_width = m['in_port']['width']
                elif 'stream' in m['out']:
                    rom_width = m['out']['width']
                cmd = args.tool_dir + '/dataConverter -i ' + in_dir + '/' + m['in_file']['name'] \
                        + ' -o ' + out_dir + '/' + m['in_file']['name'] + '.inc -w ' + str(rom_width) \
                        + ' -t ' + m['in_file']['type'] + ' -n ' + str(m['in_file']['num'])
                byteLeft = 0
                if (m['in_file']['num'] % (rom_width / data_width)) > 0:
                    byteLeft = 1
                m['in_file']['depth'] = int(m['in_file']['num'] / (rom_width / data_width) + byteLeft)
                os.system(cmd)
                print('[INFO] created ' + m['in_file']['name'] + '.inc')
            if m.get('in_ref'):
                m['in_ref']['name'] = get_name(m['in_ref'])
            m['out']['name'] = get_name(m['out'])

    #print(d)

    tmpl_loader = j2.FileSystemLoader(
        searchpath=os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'templates', 'datamover'))
    tmpl_env = j2.Environment(loader=tmpl_loader)
    templates = {}
    for impl in impl2file.keys():
        templates[impl] = tmpl_env.get_template(impl2file[impl])

    for k in d.values():
        fname = os.path.join(out_dir, k['name'] + '.cpp')
        if args.dry_run:
            print('[DEBUG] the content of file %s would be:' % os.path.basename(fname))
            print(templates[k['impl']].render(k = k))
            print('[DEBUG] end of file.')
        else:
            with open(fname, 'w') as file:
                file.write(templates[k['impl']].render(k = k))
            print('[INFO] created %s.' % os.path.basename(fname))

if __name__ == '__main__':
    main()
