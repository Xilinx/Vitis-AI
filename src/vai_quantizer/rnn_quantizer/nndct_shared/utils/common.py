
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
