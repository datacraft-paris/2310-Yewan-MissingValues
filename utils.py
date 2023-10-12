import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.coordinates import get_sun, AltAz, EarthLocation
import astropy.coordinates as coord
from astropy.time import Time
import astropy.units as u


dic_location = {
                'tenerife': {'lat': 28.3005372, 'lon': -16.513448, 'height': 731, 'timezone': 'Atlantic/Canary'}
                }


def get_astro_sunAlt(loc, given_time, utc=True):
    earth_loc = EarthLocation(lat=loc['lat'] * u.deg, lon=loc['lon'] * u.deg, height=loc['height'] * u.m)
    if utc:
        utc_time = Time(given_time)
    else:
        timezone = pytz.timezone(loc['timezone'])
        utc_time = Time(given_time.astimezone(timezone))

    alt_az = coord.AltAz(location=earth_loc, obstime=utc_time)
    alt = get_sun(utc_time).transform_to(alt_az).alt
    # az = get_sun(utc_time).transform_to(alt_az).az

    return alt.degree

def add_features_from_raw_data(df_original, loc):
    '''
    Extract meaningful info to expand features of original data.
    Extract timing from time object: hour of day, month, season
    Extract current sun altitude
    :param df_original:
    :param loc:
    :return:
    '''
    df = df_original.copy()
    df['hour_of_day'] = df['time'].dt.hour
    df['month'] = df['time'].dt.month
    df['season'] = (df['month'] % 12 + 3) // 3  # 1: Winter, 2: Spring, 3: Summer, 4: Fall

    # Add sun altitude
    df['sun_alt'] = df.apply(lambda row: get_astro_sunAlt(loc, row['time']), axis=1)
    
    return df

def missingDF(df) : 
    
    n = len(df)
    #df.isna().sum().sum() = 3 153 573 
    missing = df.isna().sum()
    index = list(missing.index)
    for i in index :
        missing[i] = float("{0:.2f}".format((missing[i]/n)*100))
        
    missing = pd.DataFrame(missing)
    return missing


def replace_nan_by_mean(df,var):
    
    indexNan = list(df[df[var].isna() == True].index)

    for index in indexNan :
        var_0,var_1 = df[var][index-1],df[var][index+1]
        if np.isfinite(var_0) and np.isnan(var_1):
            df[var][index] = var_0
        elif np.isfinite(var_1) and np.isnan(var_0):
            df[var][index] = var_1
        else :
            df[var][index] = (var_0 + var_1)/2
            
    return df


def complete_dataset(df,missing_rate):
    
    missing_df = missingDF(df)
    list_var = list(missing_df.index)
    list_var_complete = ['time']

    for i in range(1,len(list_var)-1) :
        var = list_var[i]
        if missing_df[0][var] <= missing_rate :
            list_var_complete.append(var)
      
    dfCompleteData = df[list_var_complete]

    df_to_droped = dfCompleteData[dfCompleteData['d_ext_temp'].isna() == True]

    dfCompleteData = dfCompleteData.drop(list(df_to_droped.index)).reset_index().drop('index',axis = 1)
    dfCompleteData = replace_nan_by_mean(dfCompleteData,'irrad')
    dfCompleteData = replace_nan_by_mean(dfCompleteData,'d_humid')
    dfCompleteData = replace_nan_by_mean(dfCompleteData,'humid')
    dfCompleteData = replace_nan_by_mean(dfCompleteData,'press')
    
    return dfCompleteData

def missing_dataset(df,missing_rate):
    
    missing_df = missingDF(df)
    list_var = list(missing_df.index)
    list_var_missing = ['time']

    for i in range(1,len(list_var)-1) :
        var = list_var[i]
        if missing_df[0][var] >= missing_rate :
            list_var_missing.append(var)

    dfMissingData = df[list_var_missing]        

    return dfMissingData



#Fonction de plotting

#courbe pour un paramètre
def plot_one_param(data, param):
    fig = plt.figure(figsize=(10, 6))
    plt.plot(data['time'], data[param])
    plt.xlabel('Time')
    plt.ylabel(param)
    plt.title('Evolution de la variable '+param+' en fonction du temps')

#courbe pour un paramètre en fonction de deux variables : le temps et une "catégorie"   
def plot_one_params_based_categories(data, param, catgr):
    plot = sns.relplot(data = data, x = data.time, y = data[param], hue = data[catgr], height=6, aspect=8/6)
    plt.title('Evolution de la variable '+param+ ' en fonction du temps et la catégorie '+catgr)

#courbe pour deux paramètres
def plot_two_params(data, param1, param2):
    fig, ax = plt.subplots(figsize=(10, 6))
    plot1 = ax.plot(data['time'], data[param1], 'r', label = param1)
    ax1 = ax.twinx()
    plot2 = ax1.plot(data['time'], data[param2], 'g', label = param2)
    ax.legend(loc=1)
    ax1.legend(loc=2)
    ax.set_ylabel(param1)
    ax1.set_ylabel(param2)
    ax.set_xlabel('Time')
    plt.title('Evolution des variables '+param1+' et '+param2+' en fonction du temps')
    
#tableau des corrélations
def plot_corr(df) :
    corr = df.corr()
    plt.figure(figsize=(11,8))
    plt.title("Corrélation")
    sns.heatmap(corr, cmap="Greens",annot=True)
    plt.show()
    
def plot_corr_saison_variable(x,y,saison) :

    print(f"Le coefficient de correlation entre {x} et {y} pour la saison {saison[1]} est de", "{0:.3f}".format(saison[0][x].corr(saison[0][y])*100),'% \n')
    saison[0].plot.scatter(x, y)
    plt.show()


def print_results_table(table, header = None):
    '''
    Print table with aligened format, add header if needed.
    '''
    if isinstance(header, list) and len(header) >=1:
        table.insert(0, header)
    
    longest_cols = [
        (max([len(str(row[i])) for row in table]) + 3)
        for i in range(len(table[0]))
    ]
    row_format = "".join(["{:<" + str(longest_col) + "}" for longest_col in longest_cols])
    for row in table:
        print(row_format.format(*row))

        

def correlation_table(df_to_compute, coef = 0.7, params = None):
    '''
    Compute pearson correlation between parameters(params) in given dataframe(df_to_compute), 
    find all the pairs with correlation coefficient higher than coef.
    Inputs:
    - df_to_compute: the dataframe with multiple continuous variables
    - coef: correlation coefficient threadshold
    - params: given features, the name of the variables to compute
    Outputs:
    Parameter pairs table with the correlation results (without header), .eg:
    [[a,b,corr1], [a,c,corr2], [b,d,corr3]]    
    '''
    if params == None:
        params = list(df_to_compute.columns)
        
    corr = df_to_compute[params].corr(method = 'spearman')
    corr = df_to_compute.corr(method = 'spearman')
    corr_table = []
    #Add elements, when pearson correlation between two params are greater than 0.7
    for i in range(len(params)-1):
        for j in range(i+1, len(params)):
            param1 = params[i]
            param2 = params[j]
            if (abs(corr[param1][param2]) >= coef):
                corr_table.append([param1, param2, corr[param1][param2]])
    corr_table.sort(key = lambda x: abs(x[2]), reverse=True) # Sort by correlation value, descending order
    return corr_table

