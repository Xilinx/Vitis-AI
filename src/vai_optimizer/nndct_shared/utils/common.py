# Copyright 2021 Xilinx Inc.
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

from enum import Enum

def readable_num(number):
  s = ''
  if number < 0:
    s += '-'
    number = -number

  if number < 1000:
    s += '%d' % number
  elif number > 1e15:
    s += '%0.3G' % number
  else:
    units = 'KMGT'
    unit_index = 0
    while number > 1000000:
      number /= 1000
      unit_index += 1
    s += '%.2f%s' % (number / 1000.0, units[unit_index])
  return s

def print_table(header, rows):
  if any(len(row) != len(header) for row in rows):
    raise ValueError('Column length must be equal to headers')

  column_widths = [len(field) for field in header]
  for row in rows:
    for i, field in enumerate(row):
      column_widths[i] = max(len(str(field)), column_widths[i])

  spaces_between_columns = 1
  current_pos = 0
  column_positions = []
  for i in range(len(column_widths)):
    column_positions.append(current_pos + column_widths[i] +
                            spaces_between_columns)
    current_pos = column_positions[-1]
  line_length = column_positions[-1]

  def print_row(fields, positions):
    line = ''
    for i in range(len(fields)):
      if i > 0:
        line = line[:-1] + ' '
      line += str(fields[i])
      line = line[:positions[i]]
      line += ' ' * (positions[i] - len(line))
    print(line)

  print('=' * line_length)
  print_row(header, column_positions)
  print('=' * line_length)
  for row in rows:
    print_row(row, column_positions)
    print('-' * line_length)

class AutoName(Enum):
  def _generate_next_value_(name, start, count, last_values):
    return name.lower()
