#!/usr/bin/env python

FILES_PREFIXES = ['sensor-reading']
OUT_DIR = '.'

for FILES_PREFIX in FILES_PREFIXES:

    print '### %s ###' % FILES_PREFIX
    
    execfile('convert_to_csv.py')
    convert(FILES_PREFIX + '.json', FILES_PREFIX)

    execfile('analyze_load.py')
    
    execfile('foo.py')
    