import json
import matplotlib.pyplot as plt
from datetime import datetime
import glob, os
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

import numpy as np
import scipy.signal

#os.chdir("//Users/maria/Desktop/hack/sensor-reading.json")
#for file in glob.glob("*.Accelerometer"):
#2016-09-17T05:24:22.701+02:00,96.7838668823,1.70025634766
xs=[] #pres
ys=[] # alt
t=[]

file='sensor-reading-baro.csv'
re= open(file, 'r')
flag=False
for l in  re.readlines():
	if not flag:
		flag=True
		continue

	ll=l.split(',')

	
	#date_object = datetime.strptime(ll[0], '%Y-%m-%dT%H:%M:%S.%f+02:00')
	#t.append((date_object-datetime(1970,1,1)).total_seconds())
	xs.append(float(ll[1]))
	ys.append(float(ll[2]))
ts=range(len(xs))


d=(max(xs)+min(xs))/2
print d, max(xs) , min(xs)
xxs=[]
for x in xs:
	xxs.append(x-d)

print xs
print xxs
xs=xxs
window_size, poly_order = 25, 3

itp = interp1d(ts,xs, kind='linear')
x_sg = savgol_filter(itp(ts), window_size, poly_order)

abs_val_sg = savgol_filter(itp(ts), window_size, poly_order)

#indexes = scipy.signal.find_peaks_cwt(xs, np.arange(1, 4))
#indexes = np.array(indexes) - 1
#print('Peaks are: %s' % (indexes))


indexes=0.5 * (np.sign(abs_val_sg) + 1)

print 'len(indexes)',len(indexes), indexes



p=[]
'''
for i in indexes:
	p.append(xs[i])
plt.figure(1)
plt.subplot(211)
plt.plot(range( len(xs)), xs, 'r', indexes,p,'bs'  )
'''

plt.figure(1)
plt.subplot(211)
plt.plot(range( len(xs)), xs,'bs',range(len(xs)) ,x_sg,'r' )
plt.subplot(212)
plt.plot( ys)

plt.step(range( len(xs)), xs, 'r', )
#plt.plot(t, xs, 'r--', t, ys, 'bs', t, zs, 'g^')
plt.show()


