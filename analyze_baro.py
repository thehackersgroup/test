#!/usr/bin/env python

from scipy.interpolate import interp1d
from pylab import *

def detect_step(xs, threshold=1.0, width=10, smoothing = 0.5):
    from scipy.signal import find_peaks_cwt
    xs_mean = mean(xs)
    detected_steps = zeros(len(xs))
    detected_steps_binary = zeros(len(xs))
    for center in arange(width, len(xs) - width):
        sum_before = 0
        sum_from = 0
        for i in arange(center - width, center):
            sum_before *= smoothing
            sum_before += xs[i]
        # for i in arange(center, center + width): <- in reverse
        for i in arange(center + width -1, center -1, -1):
            sum_from *= smoothing
            sum_from += xs[i]
        detected_steps[center] = float(sum_from - sum_before) / xs_mean

    current_max = 0
    current_max_index = 0
    for i, x in enumerate(detected_steps):
        if (abs(x) > threshold):
            if (abs(x) > abs(current_max)):
                current_max_index = i
                current_max = x
        else:
            # end of over threshold series if current_max != 0
            if current_max > 0:
                detected_steps_binary[current_max_index] = 1
            elif current_max < 0:
                detected_steps_binary[current_max_index] = -1
            current_max = 0
            current_max_index = 0

    return detected_steps_binary

# Smoothing data
#t = df_baro.index - df_baro.index[0]
#t = [_t.seconds for _t in t]
#df_baro['t'] = t

#window_size, poly_order = 25, 3
#itp = interp1d(t, df_baro['alt'], kind='linear')
#df_baro['alt_fil'] = savgol_filter(itp(t), window_size, poly_order)


df_baro['alt_steps'] = detect_step(df_baro['alt'])
df_baro[['alt_steps', 'alt']].plot(figsize=[20,5])
legend()
