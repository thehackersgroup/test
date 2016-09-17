#!/usr/bin/env python

CHALLENGE_NUMBERS = [1]
OUT_DIR = '.'

for CHALLENGE_NUMBER in CHALLENGE_NUMBERS:

    print '### %s ###' % CHALLENGE_NUMBER
    
    execfile('data_to_df.py')
    
    execfile('analyze_baro.py')
    
    execfile('foo.py')
