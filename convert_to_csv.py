#!/usr/bin/env python

'''
./convert_to_csv.py sensor-reading.json sensor-reading # the second argument is only a prefix

'''

import os, sys, json

data = []

out_fh_accel = open(sys.argv[2] + '-accel.csv', 'w')
out_fh_accel.write(','.join([ 'date', 'x', 'y', 'z' ]))
out_fh_accel.write('\n')

out_fh_baro = open(sys.argv[2] + '-baro.csv', 'w')
out_fh_baro.write(','.join([ 'date', 'pres', 'alt' ]))
out_fh_baro.write('\n')

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
            out_fh_accel.write(','.join(map(str, [ d['date'], d['x'], d['y'], d['z'], ])) + '\n')
            
        elif d['type'] == 'Barometer':
            out_fh_baro.write(','.join(map(str, [ d['date'], d['pressure'], d['relativeAltitude'] ])) + '\n')
     
        tmp = ''
