import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
    