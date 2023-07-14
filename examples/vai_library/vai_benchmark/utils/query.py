# Copyright 2022-2023 Advanced Micro Devices Inc.
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

import os
import sys
import json
import re
import logging
from utils import utilities


class TargetQuery:
    def __init__(self, args):
        self.args = args
        self.target_info = None
        self.target_info_all = ''

    @staticmethod
    def xclbin_exists(vart_config='/etc/vart.conf'):
        vart_path = utilities.read_file(vart_config)
        xclbin_firmware_config = vart_path.split(' ')
        if len(xclbin_firmware_config) != 2 or xclbin_firmware_config[0].strip(': ') != 'firmware':
            logging.error('Please check your /etc/vart.conf, the config format is not correct!')
            sys.exit(1)

        xclbin_path = xclbin_firmware_config[1].strip()
        BOOT_bin_path = os.path.join(os.path.split(xclbin_path)[0], 'BOOT.bin')
        if utilities.file_exists(xclbin_path) and utilities.file_exists(BOOT_bin_path):
            return

        logging.error('The xclbin in /etc/vart.conf is not existed!')
        possible_xclbin_path = ['/run/media/mmcblk0p1', '/boot', '/media/sd-mmcblk0p1']
        xclbin_files = [os.path.join(xclbin_path, x) for xclbin_path in possible_xclbin_path if
                        os.path.exists(xclbin_path) for x in os.listdir(xclbin_path) if x.endswith('.xclbin')]
        if xclbin_files:
            logging.info('The possible xclbin may be %s' % ' '.join(xclbin_files))
        sys.exit(1)

    def query_platform(self):
        xclbin_path = utilities.read_file('/etc/vart.conf').split(' ')[1].strip()
        logging.debug("Reading xclbin file into memory.  File: {}".format(xclbin_path))
        pl_query_cmd = 'xclbinutil --info --input {}'.format(xclbin_path)
        rc = utilities.execute_cmd(pl_query_cmd)
        if rc.returncode != 0:
            return

        output = rc.stdout.decode("utf-8")
        self.target_info_all += output
        lines = [x for x in output.split('\n')]
        pl_sptags = [];
        argument = None
        pairs = []
        clk_pairs = []
        for line in lines:
            if 'Created' in line:
                pl_create = lines[lines.index(line)+1].strip()
            if 'Board Part' in line:
                pl_name = line[line.index(':') + 1:].strip()
            if 'Kernels' in line:
                pl_kernel = line[line.index(':') + 1:].strip()
            if 'Version' in line:
                pl_ver = line[line.index(':') + 1:].strip()
            if 'Instance:' in line:
                instance = line.split(':')[1].strip()
                pairs.append({'Instance': instance})
            elif 'Base Address:' in line:
                address = line.split(':')[1].strip()
                if pairs:
                    pairs[-1]['Base Address'] = address
            if 'Argument:' in line:
                argument = line.split(':')[1].strip()
                pairs.append({'Argument': argument})
            elif 'Register Offset:' in line and argument is not None:
                reg_offset = line.split(':')[1].strip()
                if pairs:
                    pairs[-1]['Offset'] = reg_offset
            elif 'Port:' in line and argument is not None:
                port = line.split(':')[1].strip()
                if pairs:
                    pairs[-1]['Port'] = port
            elif 'Memory:' in line and argument is not None:
                memory = line.split(':')[1].strip()
                if pairs:
                    pairs[-1]['Memory'] = memory
            if len(line) <=256 and '--connectivity.sp' in line:
                pl_sptag = line.split('--connectivity.sp')[1].strip().split('.')[1]
                pl_sptags.append(pl_sptag)
            if len(line) <= 256 and '--clock.freqHz' in line:
                frequency, clock_info = line.split(":")
                instance_name, clock_name = clock_info.split(".")
                clock_data = {
                    "Instance": instance_name.strip(),
                    "Clock Name": clock_name.strip(),
                    "Frequency": int(frequency.strip().split()[-1])
                }
                clk_pairs.append(clock_data)

        pl_info_datas = {}
        if pl_name:
            pl_info_datas['Platform Board Part'] = pl_name
        if pl_create:
            pl_info_datas['Platform Created'] = pl_create
        if pl_kernel:
            pl_info_datas['XCLBIN Kernels'] = pl_kernel
        if pl_ver:
            pl_info_datas['XCLBIN Version'] = pl_ver

        self.target_info.update(pl_info_datas)

        # Filter out pairs with Memory value as '<not applicable>'
        pairs = [pair for pair in pairs if pair.get('Memory', '') != '<not applicable>']
        # Filter out pairs with Argument value as 'dpu_prof_addr'
        pairs = [pair for pair in pairs if pair.get('Argument', '') != 'dpu_prof_addr']
        # update the connectivity if specified the sptag
        if pl_sptags:
            sptag_pairs = [line.split(':') for line in pl_sptags]
            for pair in pairs:
                port = pair.get('Port', '')
                for pair2 in sptag_pairs:
                    if port == pair2[0]:
                        pair['Memory'] = pair2[1]
                        break

        widths = [20, 15, 20, 10, 15, 20]
        # Print the headers
        xclbin_str_list = ['', ]
        headers = ['Instance', 'Base Address', 'Argument', 'Offset', 'Port', 'Memory']
        xclbin_str_list.append('=' * sum(widths))
        xclbin_str_list.append(' | '.join(header.ljust(width) for header, width in zip(headers, widths)))
        xclbin_str_list.append('-' * sum(widths))        # print(' | '.join(header.ljust(width) for header, width in zip(headers, widths)))        # print('-' * sum(widths))
        for pair in pairs:
            row = [pair.get(header, '').ljust(width) for header, width in zip(headers, widths)]            # print(' | '.join(row))
            xclbin_str_list.append(' | '.join(row))
        xclbin_str_list.append('=' * sum(widths))
        self.target_info['XCLBIN Information'] = '\n'.join(xclbin_str_list)

        if clk_pairs:
            clk_paris_str_list = ['', ]
            widths = [20, 15, 20]
            headers = ['Instance', 'Clock Name', 'Frequency']
            clk_paris_str_list.append('=' * sum(widths))
            clk_paris_str_list.append((' | '.join(header.ljust(width) for header, width in zip(headers, widths))))
            clk_paris_str_list.append('-' * sum(widths))
            for pair in clk_pairs:
                row = [pair.get('Instance', '').ljust(widths[0]), pair.get('Clock Name', '').ljust(widths[1]), str(pair.get('Frequency', '')).ljust(widths[2])]
                clk_paris_str_list.append(' | '.join(row))
            clk_paris_str_list.append('=' * sum(widths))
            self.target_info['CLock Frequency'] = '\n'.join(clk_paris_str_list)
 
    def query_dpu(self):
        aie_freq = ''
        if os.path.exists('/sys/kernel/debug/clk/aie_core_ref_clk_0/clk_rate'):
            aie_freq_cmd = 'cat /sys/kernel/debug/clk/aie_core_ref_clk_0/clk_rate'
            rc_c = utilities.execute_cmd(aie_freq_cmd)
            if rc_c.returncode == 0:
                aie_freq = rc_c.stdout.decode("utf-8").strip()

        aie_actual_freq = ''
        if os.path.exists('/sys/kernel/debug/clk/hsm0_ref_mux/clk_rate'):
            aie_actual_freq_cmd = 'cat /sys/kernel/debug/clk/hsm0_ref_mux/clk_rate'
            rc_c = utilities.execute_cmd(aie_actual_freq_cmd)
            if rc_c.returncode == 0:
                aie_actual_freq = rc_c.stdout.decode("utf-8").strip()

        run_cmd = 'xdputil query'
        rc = utilities.execute_cmd(run_cmd)
        if rc.returncode != 0:
            return

        output = rc.stdout.decode("utf-8")
        self.target_info_all += output

        lines = [x for x in output.split('\n') if all([keyword not in x for keyword in ('XAIEFAL', 'hsm0_ref_value')])]

        xdputil_query = json.loads(''.join(lines))
        kernals = xdputil_query.get("kernels", "")
        if not kernals:
            logging.error('Get xdputil query failed!')
            return

        dpu_info_datas = {}
        cu_num = len(kernals)
        dpu_info_datas['DPU Compute Unit Number'] = cu_num

        for kernal in kernals:
            if kernal.get('name', '') != 'DPU Core 0':
                continue
            batch_num = kernal.get("DPU Batch Number", 0)
            dpu_arch = kernal.get("DPU Arch", '')
            dpu_freg = kernal.get("DPU Frequency (MHz)", '')
            fingerprint = kernal.get("fingerprint", '')

            if dpu_arch:
                dpu_info_datas['DPU Arch'] = dpu_arch
            if batch_num:
                dpu_info_datas['DPU Batch Number'] = batch_num
            if dpu_freg:
                dpu_info_datas['DPU Frequency'] = str(dpu_freg) + ' (MHz)'
            if fingerprint:
                dpu_info_datas['DPU Fingerprint'] = str(fingerprint)

        if aie_freq:
            aie_freq_str = str(int(aie_freq) / 1000000)
            dpu_info_datas['AIE Frequency'] = aie_freq_str + ' (MHz)'
        if aie_actual_freq:
            aie_actual_freq_str = str(int(aie_actual_freq) / 1000000)
            dpu_info_datas['AIE Actual Frequency'] = aie_actual_freq_str + ' (MHz)'

        self.target_info.update(dpu_info_datas)

    def query_os(self):
        cpu_freq_cmd = 'cat /sys/devices/system/cpu/cpufreq/policy0/cpuinfo_cur_freq'
        rc_c = utilities.execute_cmd(cpu_freq_cmd)
        cpu_freq = ''
        if rc_c.returncode == 0:
            cpu_freq = rc_c.stdout.decode("utf-8").strip()

        os_query_cmd = 'xbutil examine'
        rc = utilities.execute_cmd(os_query_cmd)
        if rc.returncode != 0:
            return

        output = rc.stdout.decode("utf-8")
        self.target_info_all += output
        lines = [x for x in output.split('\n')]
        sys_distribution = sys_memory = cpu_cores = ''
        for line in lines:
            if 'Memory' in line:
                sys_memory = line[line.index(':')+1:].strip()
            if 'Distribution' in line:
                sys_distribution = line[line.index(':')+1:].strip()
            if 'CPU Cores' in line:
                cpu_cores = line[line.index(':') + 1:].strip()

        os_info_datas = {}
        if sys_distribution:
            os_info_datas['System Distribution'] = sys_distribution
        if cpu_cores:
            os_info_datas['System CPU Cores'] = cpu_cores
        if cpu_freq:
            cpu_freq_str = str(int(cpu_freq) / 1000)
            os_info_datas['System CPU Frequency'] = cpu_freq_str + ' (MHz)'
        if sys_memory:
            os_info_datas['System Memory'] = sys_memory
        self.target_info.update(os_info_datas)

    def query(self):
        self.target_info = {}
        self.xclbin_exists()
        self.query_os()
        self.query_dpu()
        self.query_platform()

    def show(self, scope=''):
        format_row = "{:<25}:  {:<30}"
        if scope == 'all':
            logging.info(self.target_info_all)
            return
        for k, v in self.target_info.items():
            logging.info(format_row.format(k, v))
