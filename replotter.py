import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

path='C:\Users\Peter Duggins\Documents\GitHub\SYDE556-750\data'
filename='\FearConditioningCombinedV4pt521807'
params=pd.read_json(path+filename+'_params.json')
dataframe=pd.read_pickle(path+filename+'_data.pkl')

experiment=params['experiment'][0]
# print experiment
# print dataframe

print 'Plotting...'
sns.set(context='paper')
if experiment=='muller-tone' or experiment=='muller-context':
	figure, (ax1, ax2) = plt.subplots(2, 1)
	sns.barplot(x="drug",y="freeze",data=dataframe.query("phase=='test'"),ax=ax1)
	sns.tsplot(time="time", value="freeze", data=dataframe.query("phase=='test'"),
					unit="trial", condition="drug",ax=ax2)
	ax1.set(xlabel='',ylabel='freezing (%)', ylim=(0.0,1.0), title=experiment)
	ax2.set(xlabel='time (s)', ylabel='freezing (%)', ylim=(0.0,1.0))
elif experiment=='viviani':
	figure, (ax1, ax2) = plt.subplots(2, 1)
	sns.barplot(x="phase",y="freeze",hue='drug',data=dataframe,ax=ax1)
	sns.tsplot(time="time", value="freeze",data=dataframe,
					unit="trial", condition="drug",ax=ax2)
	ax1.set(xlabel='',ylabel='freezing (%)', ylim=(0.0,1.0), title=experiment)
	ax2.set(xlabel='time (s)', ylabel='freezing (%)', ylim=(0.0,1.0))
elif experiment=='validate-tone' or experiment=='validate-context':
	current_palette = sns.color_palette()
	sns.set_palette('deep', color_codes=True)
	figure, (ax1,ax2,ax3,ax4) = plt.subplots(4, 1, sharex=False)
	#plot stimuli
	a1=sns.tsplot(time="time",value="c",color='b',data=dataframe,unit="trial",ax=ax1)
	a2=sns.tsplot(time="time",value="u",color='g',data=dataframe,unit="trial",ax=ax1)
	a3=sns.tsplot(time="time",value="context",color='r',data=dataframe,unit="trial",ax=ax1)
	# legend1=plt.legend(['CS', 'US', 'context'],loc='upper right')
	ax1.set(xlabel='',ylabel='stimulus',title=experiment)
	#plot LA, BA_fear, BA_extinct
	a4=sns.tsplot(time="time",value="la",color='b',data=dataframe,unit="trial",ax=ax2)
	a5=sns.tsplot(time="time",value="ba_fear",color='g',data=dataframe,unit="trial",ax=ax2)
	a6=sns.tsplot(time="time",value="ba_extinct",color='r',data=dataframe,unit="trial",ax=ax2)
	# legend2=plt.legend(['LA', 'BA_fear', 'BA_extinct'],loc='upper right')
	ax2.set(xlabel='',ylabel='response',)
	# #plot error_cond, error_context, error_extinct
	a7=sns.tsplot(time="time",value="error_cond",color='b',data=dataframe,unit="trial",ax=ax3)
	a8=sns.tsplot(time="time",value="error_context",color='g',data=dataframe,unit="trial",ax=ax3)
	a9=sns.tsplot(time="time",value="error_extinct",color='r',data=dataframe,unit="trial",ax=ax3)
	ax4.set(xlabel='',ylabel='error',)
	# legend3=plt.legend(['error_cond', 'error_context', 'error_extinct'],loc='upper right')
	# #plot freeze
	a10=sns.tsplot(time="time", value="freeze",data=dataframe,unit="trial",ax=ax4)
	ax4.set(xlabel='time',ylabel='freezing (%)',)
	# legend4=plt.legend(['freeze'],loc='upper right')
plt.show()