#!/usr/bin/env python

'''
./convert_to_csv.py sensor-reading.json sensor-reading-acceleration.csv

'''

import os, sys, json

data = []

out_fh = open(sys.argv[2], 'w')

tmp = ''
bra = 0
for l in open(sys.argv[1]).readlines():
    if l.startswith(']') or l.startswith('['):
        continue
    elif '{' in l:
        bra+= 1
    elif '}' in l:
        bra-= 1
    tmp += l
        
    if bra == 0:
        tmp = tmp.rstrip()
        if tmp[-1] == ',':
            tmp = tmp[0:-1]

        d = json.loads(tmp.rstrip())
        
        if d['type'] == 'Accelerometer':
            out_fh.write(','.join(map(str, [ d['date'], d['x'], d['y'], d['z'], ])) + '\n')
     
        tmp = ''
