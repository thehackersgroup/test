import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

def regularise_time(ts, xs):
    t1 = np.arange(ts[0], ts[len(ts) - 1], 0.1)
    x1 = np.interp(t1, ts, xs)

    return(x1, t1)

t0 = df_accel.index - df_accel.index[0]
t0 = [_t.seconds+float(_t.microseconds)/1000000 for _t in t0]
x0 = df_accel['x']
y0 = df_accel['y']
z0 = df_accel['z']

(x, t) = regularise_time(t0, x0)
(y, t) = regularise_time(t0, y0)
(z, t) = regularise_time(t0, z0)

abs_val = []
for i in range(0, len(t)):
    abs_val.append(x[i]*x[i] + y[i]*y[i] + z[i]*z[i])

# some sample data
ts = pd.Series(abs_val, index=t)

#plot the time series
plt.subplot(3,1,1)
ts.plot(style='k--')

# calculate a 60 day rolling mean and plot
plt.subplot(3,1,2)
ts.rolling(30).mean().plot(style='k')

# add the 20 day rolling variance:
plt.subplot(3,1,3)
ts.rolling(30).std().plot(style='b')

plt.show()