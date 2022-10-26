# %% import libraries
import os
import glob
import pandas as pd
import scipy as sp
import numpy as np
from datetime import datetime
from maad import sound, features
from maad.util import (date_parser, plot_correlation_map,
                       plot_features_map, plot_features, false_Color_Spectro)

# %% Prepare configurations to study
import itertools as it

usefull_dates_P1=["2021-03-26","2021-04-26"]
usefull_dates_P2=["2021-07-28","2021-08-25"]
my_dict={'recPos':['CH9','CH10','CH11','CH12'],'period':['P1','P2'],'year':['2021']}
allNames = sorted(my_dict)
configList = list(it.product(*(my_dict[Name] for Name in allNames)))
print((configList))

# %% List files and build dataframe
df=pd.DataFrame()
path = os.getcwd()
fileformat='%Y%m%d_%H%M%S.WAV'

# loop over folders to agregate filenames in a big dataframe
for config in configList:
    dftmp=pd.DataFrame()
    recPos=config[1]
    period=config[0]
    year=config[2]
    subfolder=recPos+"_"+period+"_"+year
    print("Processing "+subfolder+"...")

    # Add all filename in the directory in a dataframe
    dftmp=pd.DataFrame({"file":glob.glob(os.path.join(path, 'data/'+subfolder+'/Son/*.WAV'))})

    dftmp['period']=period
    dftmp['recpos']=recPos
    df=pd.concat([df,dftmp])

# Extract filename from fullpath
df['filename']=df["file"].apply(lambda x: os.path.basename(x))
# Extract Date and time from filename
df['Date']=df["filename"].apply(lambda x: datetime.strptime(x, fileformat))

# Set date as index
# sort dataframe by date
#df = df.sort_index(axis=0)
df.set_index('Date', inplace=True)
df=df.sort_values(['period','recpos','Date'])

# Remove not usable dates from P1  & Remove not usable dates from P2
mask1=((df.index > usefull_dates_P1[0]) & (df.index < usefull_dates_P1[1]))
mask2=((df.index > usefull_dates_P2[0]) & (df.index < usefull_dates_P2[1]))
mask=np.logical_or(mask1, mask2)
df=df[mask]

# Add columns with year, month, and weekday 
df['Year'] = df.index.year
df['Month'] = df.index.month
df['Weekday Name'] = df.index.weekday

#select only 1 hour
#df=df.loc['2021-03-26' : '2021-04-01']
#df=df[df['recpos']=="CH9"]

#df=df.iloc[::100, :]
print(df.head())
print(df.tail())
# %

# %% BATCH COMPUTING OF ACOUSTIC INDICES

df_indices = pd.DataFrame()
df_indices_per_bin = pd.DataFrame()
N=df.shape[0]
id=0

for index, row in df.iterrows() :
    id=id+1
    # get the full filename of the corresponding row
    fullfilename = row['file']
    # Save file basename
    path, filename = os.path.split(fullfilename)
    print ('\n**************************************************************')
    print(str(id)+"/"+str(N))
    print (filename)

    #### Load the original sound (16bits) and get the sampling frequency fs
    try :
        wave,fs = sound.load(filename=fullfilename, channel='left', detrend=True, verbose=False)

    except:
        # Delete the row if the file does not exist or raise a value error (i.e. no EOF)
        df.drop(index, inplace=True)
        continue

    """ =======================================================================
                     Computation in the time domain
    ========================================================================"""

    # Parameters of the audio recorder. This is not a mandatory but it allows
    # to compute the sound pressure level of the audio file (dB SPL) as a
    # sonometer would do.
    S = -18         # Sensbility microphone-35dBV (SM4) / -18dBV (Audiomoth)
    G = 15       # Amplification gain (26dB (SM4 preamplifier))

    # compute all the audio indices and store them into a DataFrame
    # dB_threshold and rejectDuration are used to select audio events.
    df_audio_ind = features.all_temporal_alpha_indices(wave, fs,
                                          gain = G, sensibility = S,
                                          dB_threshold = 3, rejectDuration = 0.01,
                                          verbose = False, display = False)

    """ =======================================================================
                     Computation in the frequency domain
    ========================================================================"""

    # Compute the Power Spectrogram Density (PSD) : Sxx_power
    Sxx_power,tn,fn,ext = sound.spectrogram (wave, fs, window='hann',
                                             nperseg = 1024, noverlap=1024//2,
                                             verbose = False, display = False,
                                             savefig = None)

    # compute all the spectral indices and store them into a DataFrame
    # flim_low, flim_mid, flim_hi corresponds to the frequency limits in Hz
    # that are required to compute somes indices (i.e. NDSI)
    # if R_compatible is set to 'soundecology', then the output are similar to
    # soundecology R package.
    # mask_param1 and mask_param2 are two parameters to find the regions of
    # interest (ROIs). These parameters need to be adapted to the dataset in
    # order to select ROIs
    df_spec_ind, df_spec_ind_per_bin = features.all_spectral_alpha_indices(Sxx_power,
                                                            tn,fn,
                                                            flim_low = [0,1500],
                                                            flim_mid = [1500,8000],
                                                            flim_hi  = [8000,20000],
                                                            gain = G, sensitivity = S,
                                                            verbose = False,
                                                            R_compatible = 'soundecology',
                                                            mask_param1 = 6,
                                                            mask_param2=0.5,
                                                            display = False)

    """ =======================================================================
                     Create a dataframe
    ========================================================================"""
    # First, we create a dataframe from row that contains the date and the
    # full filename. This is done by creating a DataFrame from row (ie. TimeSeries)
    # then transposing the DataFrame.
    df_row = pd.DataFrame(row)
    df_row =df_row.T
    df_row.index.name = 'Date'
    df_row = df_row.reset_index()

    # add scalar indices into the df_indices dataframe
    df_indices = df_indices.append(pd.concat([df_row,
                                              df_audio_ind,
                                              df_spec_ind], axis=1))
    # add vector indices into the df_indices_per_bin dataframe
    #df_indices_per_bin = df_indices_per_bin.append(pd.concat([df_row,
    #                                                          df_spec_ind_per_bin], axis=1))
# Set back Date as index
df_indices = df_indices.set_index('Date')
#df_indices_per_bin = df_indices_per_bin.set_index('Date')

df_indices.to_csv('df_indices.csv')
#df_indices_per_bin.to_csv('df_indices_per_bin.csv')

# %% PLOT FEATURE MAP


SPECTRAL_FEATURES=['MEANf','VARf','SKEWf','KURTf','NBPEAKS','LEQf',
'ENRf','BGNf','SNRf','Hf', 'EAS','ECU','ECV','EPS_KURT','EPS_SKEW','ACI',
'NDSI','rBA','AnthroEnergy','BioEnergy','BI','ROU','ADI','AEI','LFC','MFC','HFC',
'ACTspFract','ACTspCount','ACTspMean', 'EVNspFract','EVNspMean','EVNspCount',
'TFSD','H_Havrda','H_Renyi','H_pairedShannon', 'H_gamma', 'H_GiniSimpson','RAOQ',
'AGI','ROItotal','ROIcover']

TEMPORAL_FEATURES=['ZCR','MEANt', 'VARt', 'SKEWt', 'KURTt',
               'LEQt','BGNt', 'SNRt','MED', 'Ht','ACTtFraction', 'ACTtCount',
               'ACTtMean','EVNtFraction', 'EVNtMean', 'EVNtCount']

plot_features_map(df_indices[SPECTRAL_FEATURES], mode='24h')
plot_features_map(df_indices[TEMPORAL_FEATURES], mode='24h')

#plot_features_map(df_indices[SPECTRAL_FEATURES],mode='none')
#plot_features_map(df_indices[TEMPORAL_FEATURES],mode='none')

# %%
