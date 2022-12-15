# Libaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================================
def plot_daytime_map(df,indexName, **kwargs):
    """
    Plot features values on a in heatmap with 'daytime' on the Y axis and 'date' on the X axis.
    
    If two values of indicators are presented at the same time and date it does the mean.
    
    Parameters
    ----------
    df : Panda DataFrame
        DataFrame with features (ie. indices).
        /!\must have datetime index

    indexName: String
        The name of the index to plot
            
    **kwargs
            
        Specific to matplotlib:
            
        - figsize : tuple of integers, optional, default: (4,10) width, height in inches.  
        
        - title : string, optional, default : 'Spectrogram' title of the figure
        
        - xlabel : string, optional, default : 'Time [s]' label of the horizontal axis

        - ylabel : string, optional, default : 'Amplitude [AU]' label of the vertical axis
            
        - xticks : tuple of ndarrays, optional, default : none
            * ticks : array_like => A list of positions at which ticks should be placed. You can pass an empty list to disable yticks.
            * labels : array_like, optional =>  A list of explicit labels to place at the given locs.
            
        - yticks : tuple of ndarrays, optional, default : none
            * ticks : array_like => A list of positions at which ticks should be placed. You can pass an empty list to disable yticks.
            * labels : array_like, optional =>  A list of explicit labels to place at the given locs.
        
        - cmap : string or Colormap object, optional, default is 'gray'
            See https://matplotlib.org/examples/color/colormaps_reference.html
            in order to get all the  existing colormaps
            examples: 'hsv', 'hot', 'bone', 'tab20c', 'jet', 'seismic', 
            'viridis'...
        
        - vmin, vmax : scalar, optional, default: None
            `vmin` and `vmax` are used in conjunction with norm to normalize
            luminance data.  Note if you pass a `norm` instance, your
            settings for `vmin` and `vmax` will be ignored.
        
        - extent : list of scalars [left, right, bottom, top], optional, default: None
            The location, in data-coordinates, of the lower-left and
            upper-right corners. If `None`, the image is positioned such that
            the pixel centers fall on zero-based (row, column) indices.
        
        - now : boolean, optional, default : True
            if True, display now. Cannot display multiple images. 
            To display mutliple images, set now=False until the last call for 
            the last image      
            
        ... and more, see matplotlib
        
    Returns
    -------
    fig : Figure
        The Figure instance 
    ax : Axis
        The Axis instance   
        """
    if isinstance(df, pd.DataFrame) == False:
        raise TypeError("df must be a Pandas Dataframe")
    elif isinstance(df.index, pd.DatetimeIndex) == False:
        raise TypeError("df must have an index of type DateTimeIndex")

    # kwargs
    cmap = kwargs.pop("cmap", "RdBu_r")
    figsize = kwargs.pop("figsize", None)


    #============ Compute New Dataframe =================
    df_daily=pd.DataFrame() # create empty Dataframe
    alldays=pd.date_range(start=np.min(df.index),end=np.max(df.index),freq='D')
    alldays=alldays.strftime('%Y-%m-%d')
    for day in alldays:
        dftmp=df.loc[day,[indexName]]
        ind_per_day=df.loc[day,[indexName]].reset_index(drop=True)
        if ind_per_day.empty:
            print('Empty day in dataframe ignored')
        else:
            ind_per_day['DayTime']=dftmp.index.strftime('%H:%M') # set new index
            # If more than one value in the timestamp do the mean
            ind_per_day=pd.DataFrame([{'DayTime': k,
                            day: v[indexName].mean()}
                        for k,v in ind_per_day.groupby(['DayTime'])],
                        columns=['DayTime', day])

            ind_per_day.set_index('DayTime', inplace=True)
            ind_per_day.rename(columns={indexName:day},inplace=True)
            df_daily=pd.concat([df_daily,ind_per_day],axis=1)

    #====================================================

    # plot
    if figsize is None :
        fig = plt.figure(figsize=(len(df_daily)*0.33, len(list(df_daily))*0.27))
    ax = fig.add_subplot(111)
    caxes = ax.matshow(df_daily.transpose(), cmap=cmap, aspect="auto", **kwargs)
    fig.colorbar(caxes, shrink=0.75, label="Value")    
    # Set ticks on both sides of axes on
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    # We want to show all ticks...
    ax.set_yticks(np.arange(len(df_daily.columns)))
    ax.set_xticks(np.arange(len(df_daily.index)))
    ax.set_title(indexName)
    # ... and label them with the respective list entries
    ax.set_yticklabels(df_daily.columns)
    ax.set_xticklabels(df_daily.index)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=10)
    fig.tight_layout()
    plt.show()

    return fig, ax , df_daily



def calc_daytime_map(df,indexName):
    """
    Calculate daytime map on a specific index from datetime index dataframe 
    
    Parameters
    ----------
    df : Panda DataFrame
        DataFrame with features (ie. indices).
        /!\must have datetime index

    indexName: String
        The name of the index to plot
        
        
    Returns
    -------
    df_daily : dataframe of daily and hourly 
    """
    if isinstance(df, pd.DataFrame) == False:
        raise TypeError("df must be a Pandas Dataframe")
    elif isinstance(df.index, pd.DatetimeIndex) == False:
        raise TypeError("df must have an index of type DateTimeIndex")


    #============ Compute New Dataframe =================
    df_daily=pd.DataFrame() # create empty Dataframe
    alldays=pd.date_range(start=np.min(df.index),end=np.max(df.index),freq='D')
    alldays=alldays.strftime('%Y-%m-%d')
    for day in alldays:
        dftmp=df.loc[day,[indexName]]
        ind_per_day=df.loc[day,[indexName]].reset_index(drop=True)
        if ind_per_day.empty:
            print('Empty day in dataframe ignored')
        else:
            ind_per_day['DayTime']=dftmp.index.strftime('%H:%M') # set new index
            # If more than one value in the timestamp do the mean
            ind_per_day=pd.DataFrame([{'DayTime': k,
                            day: v[indexName].mean()}
                        for k,v in ind_per_day.groupby(['DayTime'])],
                        columns=['DayTime', day])

            ind_per_day.set_index('DayTime', inplace=True)
            ind_per_day.rename(columns={indexName:day},inplace=True)
            df_daily=pd.concat([df_daily,ind_per_day],axis=1)

    #====================================================
    return df_daily