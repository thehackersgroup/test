#!/usr/bin/env python

from collections import OrderedDict
execfile('helpers.py')

CHALLENGE_NUMBERS = [1]

DATA = OrderedDict()
DATA[1] = {'detect_step_width': 30, 'start': timestampToDateTime('2016-09-17 09:05:47.849000'), 'end': timestampToDateTime('2016-09-17 09:35:47.849000'), 'plot_gt': True}

OUT_DIR = '.'

for DATASET in DATA:

    print '### %s ###' % DATASET
    
    execfile('data_to_df.py')
    
    execfile('analyze_baro.py')
    
    #execfile('foo.py')
