# This script reads annotations
#%% Imports

import pandas as pd
from maad.util import format_features,  overlay_rois, power2dB
from maad import sound
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime

#%%
dirpathannot='Annotations/'
dirpathwave='data/'

ol=0.5
Nfft=1024
flag_plt_spectro=1

dfa = pd.read_csv(dirpathannot+'Filelist.csv')
dfa = dfa[0:160]

fileformat='%Y%m%d_%H%M%S.WAV'
dfa['Date']=dfa["filename"].apply(lambda x: datetime.strptime(x, fileformat))
dfa['Time'] = dfa['Date'].dt.strftime('%H:%M')
dfa.set_index('Time', inplace=True)


dfa = dfa[dfa['Done'] == True]
dfa = dfa[dfa['period'] == 'P1']
dfa = dfa.reset_index()

NumA = []
NumB = []
Acover = []
Bcover = []

include_bgnd=True

for i in range(0,dfa.shape[0]):
    print(i)
    annotfname = dirpathannot+dfa['recpos'][i] + \
        '/'+dfa['filename'][i][0:-4]+'.txt'
    fullfilename = dirpathwave + \
        dfa['recpos'][i]+'_'+dfa['period'][i] + \
        '_2021/Son/'+dfa['filename'][i][0:-4]+'.WAV'

    print(annotfname)
    print(fullfilename)

    a = pd.read_csv(annotfname, delimiter='\t')
    a = a.drop(columns=['Selection', 'View', 'Channel'])
    a = a.rename(columns={"Begin Time (s)": "min_t", "End Time (s)": "max_t",
                 "Low Freq (Hz)": "min_f", "High Freq (Hz)": "max_f"})

    wave, fs = sound.load(filename=fullfilename,
                          channel='left', detrend=True, verbose=False)

    Sxx_power, tn, fn, ext = sound.spectrogram(wave, fs, window='hann',
                                               nperseg=Nfft, noverlap=Nfft*ol,
                                               verbose=False, display=False,
                                               savefig=None)
    #Sxx_db = power2dB(Sxx_power)+96
    #plt.figure()
    #ax, fig = overlay_rois(Sxx_db, a, savefig=None, **
    #                      {'vmin': 0, 'vmax': 96, 'extent': ext, 'cmap': 'viridis'})
    #ax.set(ylim=(10, 24000))

    a = format_features(a, tn, fn)

    if include_bgnd==False:
        # Delete every annotation that last more than 80% of the duration of the recording
        a = a.loc[a["Delta Time (s)"] <= tn[-1]*0.8]

    a['area'] = (a.max_y - a.min_y) * (a.max_x - a.min_x)

    spectro_bandwidth_y = (fn[-1]-fn[0]) / ((fn[1]-fn[0]))
    spectro_duration_x = (tn[-1]-fn[0]) / (tn[1]-tn[0])
    spectro_area_xy = spectro_duration_x*spectro_bandwidth_y

    NumAtmp = a[(a['Annotation'] != 'B')].shape[0]
    NumBtmp = a[a['Annotation'] == 'B'].shape[0]
    Acovertmp = a[a['Annotation'] != 'B'].area.sum()/spectro_area_xy*100
    Bcovertmp = a[a['Annotation'] == 'B'].area.sum()/spectro_area_xy*100

    NumA.append(NumAtmp)
    NumB.append(NumBtmp)
    Acover.append(Acovertmp)
    Bcover.append(Bcovertmp)

        #fn1=np.linspace(0,fs/2,int(Nfft*ol)+1)[:-1]
        #tn1=np.linespace(0,1,np.ceil(wave.shape[0])*2/Nfft)

dfa['NumA']=NumA
dfa['NumB']=NumB 
dfa['Acover']=Acover
dfa['Bcover']=Bcover
dfa['AcoverProp']=dfa['Acover']/(dfa['Acover']+dfa['Bcover'])
dfa['BcoverProp']=dfa['Bcover']/(dfa['Acover']+dfa['Bcover'])

#%% Merge with existing dataframe
df_indices=pd.read_csv("df_indices_weather.csv")
fileformat='%Y%m%d_%H%M%S.WAV'
df_indices['Date']=df_indices["filename"].apply(lambda x: datetime.strptime(x, fileformat))
df_indices['Time'] = df_indices['Date'].dt.strftime('%H:%M')
df_indices.set_index('Time', inplace=True)

df_indices.set_index(['recpos','Date'])
dfa.set_index(['recpos','Date'])

df_indices_small=df_indices.merge(dfa,how='inner')
df_indices_full=df_indices.merge(dfa,how='outer')

#df_indices_small.to_csv('df_indices_GTsmall.csv')
#df_indices_full.to_csv('df_indices_full.csv')

#%%
# PLOT
dfa=dfa.sort_index()
ax=dfa[dfa['recpos']=='CH9'].plot.bar(x='Time',y=['Bcover','Acover'],stacked=True,color=['lightgreen','orange'],xlabel='Time',ylabel='Cover',figsize=(10,1),fontsize=6,ylim=[0,100])
ax=dfa[dfa['recpos']=='CH10'].plot.bar(x='Time',y=['Bcover','Acover'],stacked=True,color=['lightgreen','orange'],xlabel='Time',ylabel='Cover',figsize=(10,1),fontsize=6,ylim=[0,100])
ax=dfa[dfa['recpos']=='CH11'].plot.bar(x='Time',y=['Bcover','Acover'],stacked=True,color=['lightgreen','orange'],xlabel='Time',ylabel='Cover',figsize=(10,1),fontsize=6,ylim=[0,100])
ax=dfa[dfa['recpos']=='CH12'].plot.bar(x='Time',y=['Bcover','Acover'],stacked=True,color=['lightgreen','orange'],xlabel='Time',ylabel='Cover',figsize=(10,1),fontsize=6,ylim=[0,100])

#%%
ax=dfa[dfa['recpos']=='CH9'].plot.bar(x='Time',y=['BcoverProp','AcoverProp'],stacked=True,color=['lightgreen','orange'],xlabel='Time',ylabel='Cover',figsize=(10,2),fontsize=6)
#%%
ax=dfa[dfa['recpos']=='CH9'].plot.bar(x='Time',y=['NumB','NumA'],stacked=True,color=['lightgreen','orange'],xlabel='Time',ylabel='N events',figsize=(10,2),fontsize=6)
#%%
dfa.boxplot(column='NumA',by='recpos')
#sns.violinplot(dfa,x='recpos',y='NumA')

# %% Stats paired tests 

# STATISTICS
import scipy.stats as stats
from itertools import combinations

indexName='Acover'

sel_recopos=['CH9','CH10','CH11','CH12']
combi=list(combinations(sel_recopos,2))
print('\nComputing %s combinations',len(combi))
df_test=dfa[['recpos',indexName]]
print('\nMann Whitney U test (Wilcoxon rank sum test)')
print(indexName)
for c in combi:
    print('\nComparing '+c[0]+' and '+c[1])
    dfx1=dfa[df_test['recpos']==c[0]]
    dfy1=df_test[df_test['recpos']==c[1]]
    # perform two-sided test. You can use 'greater' or 'less' for one-sided test
    Statvalue,p=stats.mannwhitneyu(x=dfx1[indexName], y=dfy1[indexName], alternative = 'two-sided')
    print(f"\nStat = {Statvalue:.1f} \t p = {p}")
    # output

# %%

Features=['ZCR', 'MEANt', 'VARt', 'SKEWt', 'KURTt', 'LEQt',
       'BGNt', 'SNRt', 'MED', 'Ht', 'ACTtFraction', 'ACTtCount', 'ACTtMean',
       'EVNtFraction', 'EVNtMean', 'EVNtCount', 'MEANf', 'VARf', 'SKEWf',
       'KURTf', 'NBPEAKS', 'LEQf', 'ENRf', 'BGNf', 'SNRf', 'Hf', 'EAS', 'ECU',
       'ECV', 'EPS', 'ACI', 'NDSI', 'rBA',
       'AnthroEnergy', 'BioEnergy', 'BI', 'ROU', 'ADI', 'AEI', 'LFC', 'MFC',
       'HFC', 'ACTspFract', 'ACTspCount', 'ACTspMean', 'EVNspFract',
       'EVNspMean', 'EVNspCount', 'TFSD', 'H_Havrda', 'H_Renyi',
       'H_pairedShannon', 'H_gamma', 'H_GiniSimpson', 'RAOQ', 'AGI',
       'ROItotal', 'ROIcover', 'H', 'NumA',
       'NumB', 'Acover', 'Bcover', 'AcoverProp', 'BcoverProp']

Features_short=['NumA','NumB', 'Acover', 'Bcover','LEQt', 'Ht','Hf','H', 'ACI', 'NDSI', 
       'AnthroEnergy', 'BioEnergy','ADI','BI',
       'ROItotal', 'ROIcover']
matcorr=df_indices_small[Features_short].corr()
matcorr=matcorr[['NumA','Acover','NumB','Bcover']].transpose()

heatmap = sns.heatmap(matcorr, annot=True, cmap="Blues",square=True,annot_kws={"fontsize":6})

#%%
matcorr=df_indices_small[Features_short].corr()
heatmap = sns.heatmap(matcorr, annot=False, cmap="Blues",square=True,annot_kws={"fontsize":6})

#%%
sns.scatterplot(df_indices_small,x='NumB',y='NDSI')

# %%
