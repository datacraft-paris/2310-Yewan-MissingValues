'''
All that is needed for run the workshop.
'''
from IPython.display import Markdown
from IPython.display import display
import inspect
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


def showsrc(fcn):
    display(Markdown(f'''```python\n{inspect.getsource(fcn)}```'''))

    
def load_raw_data(path_data_raw: str) -> pd.DataFrame:
    """
    Load raw data from a CSV file into a DataFrame.
    
    Parameters:
    - path_data_raw (str): The path to the raw data CSV file.
    
    Returns:
    - pd.DataFrame: A DataFrame containing the loaded data with specified optimizations.
    
    Notes:
    - The function skips unnecessary columns while reading the CSV.
    - It removes duplicate rows by checking columns 'time'
    - It replaces zeros and infinite values with NaN.
    - The 'time' column is converted to datetime format and set as the index.
    """
    
    # Removed 'isoplan' and 'd_rain_rate' from the list as they will not be used.   
    cols_names = [
        'time', 'status', 'd_ext_temp', 'd_humid', 'd_wind', 'd_wind_dir', 'day_r0', 
        'day_see', 'day_see_stddev', 'down_ir', 'humid', 'irrad', 'night_r0', 'night_see', 
        'press','pyr_temp', 'scint', 'sky_temp', 'transp', 'wat_col_hei'
    ]
    column_types = [
        'string', 'string', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 
        'float32', 'float32','float32', 'float32', 'float32', 'float32', 'float32', 'float32', 
        'float32', 'float32', 'float32', 'float32'
    ]
    dtype = dict(zip(cols_names, column_types))

    df = pd.read_csv(path_data_raw, usecols=cols_names, dtype=dtype)
    
    # Replace zero and infinity values with NaN
    df.replace([0, np.inf, -np.inf], np.nan, inplace=True)
    
    # Convert time to datetime format and set as index
    df['time'] = pd.to_datetime(df['time'], unit='ns')
    # Remove duplicated values
    df = df[~df.time.duplicated()]
    df.set_index('time', inplace=True)
    
    return df


def get_astro_sunAlt(loc, given_time, utc=True):
    """
    Get the real-time altitude of the sun based on GPS location(latitude et longitude) and the utc time
    :param: loc: site name, related to GPS and time zone in global dictionary dic-location
    :param: given time: time in format
    """
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

def add_skystates_from_model(df_original: pd.DataFrame, path_label: str) -> pd.DataFrame:
    '''
    Load sky states from classification model, merge the states with original dataframe.
    :param df_original: raw dataframe
    :param path_label: csv file, result from classification model
    :return: new dataframe with extended sky states: 
    c0 night
    c1 sunny
    c2 cloud
    c3 fog
    c4 rain
    c5 foreign
    c6 freeze
    '''
    df = df_original.copy()
    df_states = pd.read_csv(path_label, usecols=['utc', 'c0', 'c1', 'c2','c3', 'c4', 'c5', 'c6'])
    df_states['utc'] = pd.to_datetime(df_states['utc'], unit='ns')
    #Truncate `time` column to minute precision 
    #(ref: https://stackoverflow.com/questions/28773342/truncate-timestamp-column-to-hour-precision-in-pandas-dataframe)
    df['hourly'] = df['time'].values.astype('<M8[m]') 
    df = df.merge(df_states, left_on=df.hourly, right_on=df_states.utc, how='left')
    df.drop(columns=['key_0', 'utc', 'hourly'], inplace=True)
    return df

def add_features_from_raw_data(df_original: pd.DataFrame, path_label: str, loc: dict):
    '''
    Extract meaningful info to expand features of original data.
    Extract timing from time object: hour of day, month, season
    Extract current sun altitude
    :param df_original: raw dataframe
    :param loc: site name, related to GPS and time zone in global dictionary dic-location
    :return: new dataframe with extended features, hour, month, season, numeric status, sun altitude
    '''
    df = df_original.copy()

    # Add timing
    df['hour_of_day'] = df['time'].dt.hour
    df['month'] = df['time'].dt.month
    df['season'] = (df['month'] % 12 + 3) // 3  # 1: Winter, 2: Spring, 3: Summer, 4: Fall

    # Encode DIMM status to numeric values
    numeric_values, unique_statuses = pd.factorize(df['status'], use_na_sentinel=-1)
    df['status_numeric'] = numeric_values +1

    # Add sky states from a classification model
    df = add_skystates_from_model(df, path_label)
    
    # Add sun altitude
    def get_sunalt(x):
        return get_astro_sunAlt(loc, x)

    df['sun_alt'] = df['time'].parallel_apply(get_sunalt)

    return df


def missingDF(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Return a dataframe with the percentage of missing values.
    '''
    n = len(df)
    missing = df.isna().sum()
    index = list(missing.index)
    for i in index :
        missing[i] = float("{0:.2f}".format((missing[i]/n)*100))
    return pd.DataFrame(missing, columns=["%"]).sort_values("%", ascending=False)


def replace_nan_by_mean(df,var):
    
    indexNan = list(df[df[var].isna() == True].index)

    for index in indexNan :
        var_0, var_1 = df[var][index-1],df[var][index+1]
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

def missing_dataset(df, missing_rate):
    '''
    A fonction that print the missing value in a dataframe that are supperior or equal to a given value.
    '''
    missing_df = missingDF(df)
    list_var = list(missing_df.index)
    list_var_missing = ['time']

    for i in range(1,len(list_var)-1) :
        var = list_var[i]
        if missing_df[0][var] >= missing_rate :
            list_var_missing.append(var)

    dfMissingData = df[list_var_missing].sort_values(ascending=False)

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
    plot = sns.relplot(data = data, x = data.index, y = data[param], hue = data[catgr], height=6, aspect=8/6)
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
        
    corr = df_to_compute[params].corr(method = 'spearman', numeric_only=True)
    corr = df_to_compute.corr(method = 'spearman', numeric_only=True)
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

