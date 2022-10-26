# %% READ DATAFRAME
from logging import warning
import pandas as pd
import scipy as sp
import numpy as np
from datetime import datetime
from maad import sound, features
from maad.util import (date_parser, plot_correlation_map,
                       plot_features_map, plot_features, false_Color_Spectro)

import matplotlib.pyplot as plt
from mytoolbox import plot_daytime_map


#%% =================== Load file===========================
df_indices=pd.read_csv("df_indices.csv")
df_indices.drop('file', inplace=True, axis=1) # drop path column

# Retrieve datetime index
fileformat='%Y%m%d_%H%M%S.WAV'
df_indices['Date']=df_indices["filename"].apply(lambda x: datetime.strptime(x, fileformat))
df_indices.set_index('Date', inplace=True)
df_indices=df_indices.sort_values(['period','recpos','Date'])

# Calculate missing H indice
df_indices["H"]=df_indices["Ht"]*df_indices["Hf"]
# Drop EPS which is not calculated
df_indices.drop('EPS',inplace=True, axis=1)

# %% ==================Get the Subset of the data frame ==========
''' ===================USER INPUT=================='''
# ================= user input ===============
usefull_dates=["2021-03-29","2021-03-30"] # P1=["2021-03-26","2021-04-27"] P2=["2021-07-28","2021-08-26"]
sel_recopos=["CH11"]#["CH9"] # ["CH9","CH10","CH11","CH12"]
time_sample_per_hour=6 # Integer (there is maximum 6 samples per hour)
indexName='Ht' # Name of the index to plot
outdir=''#'./out_plot/' # your output directory name
# ===============end user input ==============
''' ===================END USER INPUT=================='''

# Compute the subset
df=df_indices
df=df_indices[df_indices['recpos'].isin(sel_recopos)]
df=df.iloc[::int(6/time_sample_per_hour), :]
df=df[((df.index >= usefull_dates[0]) & (df.index <= usefull_dates[1]))]

#%% =============Plot feature day time map ==========
fig, ax, df_daytime=plot_daytime_map(df,indexName)
if not outdir == "":
    fig.savefig(outdir+indexName+'_'+"daytimeMap"+'_'+'_'.join(sel_recopos)+'_'+'_'.join(usefull_dates)+'.png')

#%% =================Box plot =============
df_timeday=df_daytime.transpose()
df_timeday=df_timeday.iloc[:,::6] # select only round hours
ax = df_timeday.boxplot()
plt.setp(ax.get_yticklabels(), rotation=0, ha="right", fontsize=6)
plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=6)

if not outdir == "":
    fig2=plt.gcf()
    fig2.savefig(outdir+indexName+'_'+"boxplot"+'_'+'_'.join(sel_recopos)+'_'+'_'.join(usefull_dates)+'.png')

#%% ========== plot median or mean of the indicator ==========
methodagg='median'
df_daytime['mean'] = df_daytime.mean(axis=1)
df_daytime['median'] = df_daytime.median(axis=1)
ax=df_daytime[methodagg].plot()
plt.setp(ax.get_yticklabels(), rotation=0, ha="right", fontsize=6)
plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=6)
if not outdir == "":
    fig3=plt.gcf()
    fig3.savefig(outdir+indexName+'_'+methodagg+'_'+'_'.join(sel_recopos)+'_'+'_'.join(usefull_dates)+'.png')

# %% Subset the dataset 

sel_recopos=["CH9","CH10","CH11","CH12"]#["CH9"]#["CH9","CH10","CH11","CH12"]#["CH9"] # ["CH9","CH10","CH11","CH12"]

# Select only
df=df_indices
df=df.iloc[::, :] # drop interhour data subset by hour
df=df[df['recpos'].isin(sel_recopos)]
#mask=(df.index == "2021-03-29")
#df=df[mask]
df1=df[(df['period']=="P1")]
df2=df[(df['period']=="P2")]

#%% """# %% PLOT FEATURE MAP

'''
SPECTRAL_FEATURES=['MEANf','VARf','SKEWf','KURTf','NBPEAKS','LEQf',
'ENRf','BGNf','SNRf','Hf', 'EAS','ECU','ECV','EPS_KURT','EPS_SKEW','ACI',
'NDSI','rBA','AnthroEnergy','BioEnergy','BI','ROU','ADI','AEI','LFC','MFC','HFC',
'ACTspFract','ACTspCount','ACTspMean', 'EVNspFract','EVNspMean','EVNspCount',
'TFSD','H_Havrda','H_Renyi','H_pairedShannon', 'H_gamma', 'H_GiniSimpson','RAOQ',
'AGI','ROItotal','ROIcover']

TEMPORAL_FEATURES=['ZCR','MEANt', 'VARt', 'SKEWt', 'KURTt',
               'LEQt','BGNt', 'SNRt','MED', 'Ht','ACTtFraction', 'ACTtCount',
               'ACTtMean','EVNtFraction', 'EVNtMean', 'EVNtCount']

'''
FEATURES_SHORT=['LEQf','SNRf','BioEnergy','AnthroEnergy','ROItotal','NDSI','H','Hf','Ht','ACI']

plot_features_map(df1[FEATURES_SHORT],mode='24h')
plot_features_map(df2[FEATURES_SHORT],mode='24h')

plot_correlation_map(df_indices[FEATURES_SHORT], R_threshold=0)

# %%
