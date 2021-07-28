#!/usr/bin/python3
import sys,os

logs = open(sys.argv[1], "rt").readlines()

logs = [l for l in logs if l.startswith('EVENT_HOST_')]

record_num = len(logs)
first = logs[0]
last = logs[-1]

#EVENT_HOST_START 1330 4 HELLO_sta -1 4411277.674832291
first_t = float(first.split()[-1])
last_t = float(last.split()[-1])
dur_t = last_t -first_t

average_t = dur_t / record_num 
average_t_us = average_t * 1000 * 1000
print("items: {}, dur: {:.3f}\npre_record_time: {:.3f} us".format(record_num, dur_t, average_t_us))
