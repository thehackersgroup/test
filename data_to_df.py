import pandas as pd
import glob, os
os.chdir("/Users/maria/Desktop/Sample Data/Liftfahrt/test/1")
df={}
i=0
for file in glob.glob("*.Barometer"):
	print file
	df[i] = pd.read_json(file)
	i=i+1



vertical_stack= df[0]

for i in range(1, len(df)):
  vertical_stack = pd.concat([vertical_stack,df[i]], axis=0)

vertical_stack.columns = [u'date', u'pressure', u'alt', u'type']

df_accel=vertical_stack.copy()

df={}
i=0
for file in glob.glob("*.Accelerometer"):
	print file
	df[i] = pd.read_json(file)
	i=i+1



vertical_stack= df[0]

for i in range(1, len(df)):
  vertical_stack = pd.concat([vertical_stack,df[i]], axis=0)

df_baro=vertical_stack.copy()
