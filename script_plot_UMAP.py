# SCRIPT PLOT UMAP
# This scripts perform UMAP calculation over the different features of the sounds
# A DBSCAN in performed on the UMAP to identify clusters
#%% IMPORT packages
import pandas as pd
import scipy as sp
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
from datetime import datetime
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import umap
import umap.plot  # pip install "umap-learn[plot]"
from babyplots import Babyplot
from sklearn.preprocessing import QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


#%% ------------------------------------------------
# =================== Load file===========================
df_indices=pd.read_csv("df_indices_weather.csv")

# Retrieve datetime index
fileformat='%Y%m%d_%H%M%S.WAV'
df_indices['Date']=df_indices["filename"].apply(lambda x: datetime.strptime(x, fileformat))
df_indices.set_index('Date', inplace=True)
df_indices=df_indices.sort_values(['period','recpos','Date'])

# Calculate missing H indice
df_indices["H"]=df_indices["Ht"]*df_indices["Hf"]
# Drop EPS which is not calculated
df_indices.drop('EPS',inplace=True, axis=1)

df_indices["date"]=pd.to_datetime(df_indices['date'])
df_indices['Time'] = df_indices['date'].dt.strftime('%H:%M')
df_indices['hour']=df_indices['date'].dt.hour
df_indices['isloud']=df_indices['LEQf']>60


ALL_FEATURES=['MEANf','VARf','SKEWf','KURTf','NBPEAKS','LEQf',
'ENRf','BGNf','SNRf','Hf', 'EAS','ECU','ECV','EPS_KURT','EPS_SKEW','ACI',
'NDSI','rBA','AnthroEnergy','BioEnergy','BI','ROU','ADI','AEI','LFC','MFC','HFC',
'ACTspFract','ACTspCount','ACTspMean', 'EVNspFract','EVNspMean','EVNspCount',
'TFSD','H_Havrda','H_Renyi','H_pairedShannon', 'H_gamma', 'H_GiniSimpson','RAOQ',
'AGI','ROItotal','ROIcover','ZCR','MEANt', 'VARt', 'SKEWt', 'KURTt',
'LEQt','BGNt', 'SNRt','MED', 'Ht','H','ACTtFraction', 'ACTtCount','EVNtFraction', 'EVNtMean', 'EVNtCount']

SHORT_FEATURES=['LEQf','SNRf','BioEnergy','AnthroEnergy','ROItotal','NDSI','H','Hf','Ht','ACI']
#SHORT_FEATURES=['AnthroEnergy','ROItotal','NDSI','LEQf']

 
# %% ------------------------------------------------
# ================= UMAP input ===================
#----------- USER INPUT------------
usefull_dates=["2021-03-26","2021-04-27"] # P1=["2021-03-26","2021-04-27"] P2=["2021-07-28","2021-08-26"]
sel_recopos=["CH9","CH10","CH11","CH12"]#["CH9"] # ["CH9","CH10","CH11","CH12"]

Ndim_umap=3 # Number of dimensions for the UMAP
n_neighbors_umap=15# Number of neighbors for UMAP
random_seed=42 # Random seed for UMAP
FEATURES=ALL_FEATURES # All features or a subset ALL_FEATURES or SHORT_FEATURES
#----------------------------------

# Compute the subset of dates and recpos
df=df_indices[df_indices['recpos'].isin(sel_recopos)]
df=df[((df.index >= usefull_dates[0]) & (df.index <= usefull_dates[1]))]
#df=df[(df.index.hour>=3) & (df.index.hour=<6)]

X=df[ALL_FEATURES].to_numpy()

# ==================== PROCESSING UMAP=======================
# Preprocess with a quantile transformer
pipe = make_pipeline(SimpleImputer(strategy="mean"), QuantileTransformer())
X = pipe.fit_transform(X.copy())

# Fit UMAP to processed data
manifold = umap.UMAP(random_state=random_seed,n_neighbors=n_neighbors_umap,n_components=Ndim_umap).fit(X)
X_reduced = manifold.transform(X)

# %% ------------------------------------------------
# ==============Cluster Analysis with DBSCAN ==============
from sklearn.cluster import DBSCAN

db=DBSCAN(eps=0.6, min_samples=20)
db.fit(X_reduced)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labelsdb=  db.labels_
df['labelsdbscan']=labelsdb

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labelsdb)) - (1 if -1 in labelsdb else 0)
n_noise_ = list(labelsdb).count(-1)


print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)


#%% ------------------------------------------------
# ================== PLOT UMAP ====================
umap_label_by='labelsdbscan' # 'recpos' 'israining' 'hour' 'labelsdbscan' 'isloud'
mycolorscale="Spectral"
if Ndim_umap==2:  # Use UMAP.plot
    umap.plot.points(manifold, labels=df[umap_label_by],color_key_cmap=mycolorscale)

elif Ndim_umap==3: # Use babyplot
    outname='UMAP_3D.html'
    labelbb=df[umap_label_by].to_numpy().tolist() # labels
    bp=Babyplot()
    bp.add_plot(X_reduced[:,:].tolist(), "pointCloud", "categories",labelbb, {"shape": "sphere",
                                                                 "colorScale": mycolorscale,
                                                                 "showAxes": [True, True, True],
                                                                 "axisLabels": ["UMAP 1", "UMAP 2", "UMAP 3"],
                                                                 "showLegend": True,
                                                                 "fontSize": 12,
                                                                 "labelSize":20})
    
    #bp
    bp.save_as_html(outname)

