#!/usr/bin/env python

from collections import OrderedDict
execfile('helpers.py')

OUT_DIR = '.'

DATA = OrderedDict()
DATA['sensor-reading'] = {'detect_step_width': 10}

for DATASET in DATA:

    print '### %s ###' % DATASET
    
    execfile('convert_to_csv.py')
    convert(DATASET + '.json', DATASET)

    execfile('analyze_load.py')
    
    execfile('analyze_baro.py')
    
    execfile('foo.py')
    