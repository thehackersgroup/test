import pandas as pd
import glob, os

_dfs=[]
i=0
for file in glob.glob("%s/*.Barometer" % CHALLENGE_NUMBER):
	print file
	_dfs.append(pd.read_json(file))

df_baro = pd.concat(_dfs, axis=0)
df_baro.columns = [u'date', u'pressure', u'alt', u'type']
df_baro.index = df_baro['date']

_dfs=[]
i=0
for file in glob.glob("%s/*.Accelerometer" % CHALLENGE_NUMBER):
	print file
	_dfs.append(pd.read_json(file))

df_accel = pd.concat(_dfs, axis=0)
df_accel.index = df_accel['date']



#df={}
#i=0
#for file in glob.glob("%s/*.Accelerometer" % CHALLENGE_NUMBER):
	#print file
	#df[i] = pd.read_json(file)
	#i=i+1

#vertical_stack= df[0]

#for i in range(1, len(df)):
  #vertical_stack = pd.concat([vertical_stack,df[i]], axis=0)

#df_accel=vertical_stack.copy()
