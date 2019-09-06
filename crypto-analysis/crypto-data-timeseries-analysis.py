####################################
#  Author : Bastien Girardet
#  Goal : Fetch historical data for each cryptocurrencies in our dataset
#         To assess the performance of each.
#  Date : 05-09-2019
####################################

# We import the required libraries
import os
import re
import pandas as pd 
from datetime import datetime
from Cryptocurrency import Cryptocurrency


# We define our main functions
def plot_crypto(name,colname,data,log=False, standardize = False, bollinger=False, SMA = False):
    """Plot the colname depending on their 
    
    Args:
        name (string) : Name of the crypto used for visualization
        colname (string) : Column of interest
        data (pandas.DataFrame) : Dataframe containing the cryptocurrency data.
        log (boolean) : If logaritmic scale
        standardize (boolean) : If standardized
        bollinger (boolean) : If bollinger
        SMA (boolean) : If SMA
    
    Return: 
        plot the desired graph
    """
    # Import the libraries
    import matplotlib.pyplot as plt
    import numpy as np
        
    # Define the fig as so
    fig, ax = plt.subplots(figsize=(30,10))
       
    # If logarithmic scale   
    if log:    
        x = np.log(data[[colname]])
    else:
        x = data[[colname]]
    
    # If standardization
    if standardize: 
        mean = data[[colname]].mean()
        std = data[[colname]].std()
        x = data[[colname]]
        x = (x - mean)/std
    else:
        x = data[[colname]]
    
    # We plot
    ax.plot(data.index,x)
    
    # If bollinger bands
    if bollinger:
        ax.plot(data.index,data['boll_bands_lower_band'])
        ax.plot(data.index,data['boll_bands_upper_band'])
    
    label = colname
    
    # Label if log
    if log:
        label += "log scale"
        
    # Label if log
    if standardize:
        label += "standardized"

    ax.set(xlabel='(t) Date', ylabel=label,
           title=f'{name} Analysis of {colname} - (2019-01 / 2019-09)')
    
    ax.legend(name)
    ax.grid()
    
    if log:
        fig.savefig(f"./figs/{name}-{colname}-log-analysis.png")
    else:
        fig.savefig(f"./figs/{name}-{colname}-analysis.png")
        
    plt.show()



def compare_crypto(colname,crypto_dict,log=False, standardize = False):
    """Compare all the crypto contained in the crypto_dict in the specific column
    
    Args:
        colname (string) : Column of interest
        crypto_dict (dictionnary) : Dictionnary containing Cryptocurrency objects
        log (boolean) : If logaritmic scale
    
    Return: 
        plot containing the comparison
    """
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    label_list = []
    fig, ax = plt.subplots(figsize=(30,10))

    for each in crypto_dict.items():
        try :     
            
            x = each[1].data[[colname]]
            
            if log:    
                x = np.log(each[1].data[[colname]])             
                
            if standardize: 
                mean = each[1].data[[colname]].mean()
                std = each[1].data[[colname]].std()
                x = each[1].data[[colname]]
                x = (x - mean)/std
                
            ax.plot(each[1].data.index,x)
            label_list.append(each[0])
        except:
            print(f"An error occured at {each[0]}")
            print(f"{colname} unkown for {each[0]}")
            
    label = colname

    # Label if log
    if log:
        label += "log scale"

    # Label if log
    if standardize:
        label += "standardized"
                
    ax.set(xlabel='(t) Date', ylabel=label,
           title=f"Comparison of {colname} - (2019-01 / 2019-09)")
    ax.legend(label_list)
    ax.grid()
    if log:
        fig.savefig(f"./figs/{colname}-log-comparison.png")
    else:
        fig.savefig(f"./figs/{colname}-comparison.png")
        
    plt.show()


# If script run is run
if __name__ == "__main__":

    # We get the list of cryptocurrencies
    df_crypto_meta = pd.read_csv("./data/crypto.csv")
    crypto_dict = {}


    # We gather the data for each crypto in our crypto dataset
    for idx, crypto in df_crypto_meta.iterrows():
        print("Retrieve crypto_data...")
        data = pd.read_csv(f"./data/crypto-ohlc/{crypto['filename']}")
        data = data.iloc[:,1:len(data['Date'])]
        crypto_dict[crypto['symbol']] = Cryptocurrency(crypto['name'], data= data )
        print("Done...")  

    print(f"Available columns are :{list(crypto_dict['CHSB'].data.columns)}")


    plot_crypto('CHSB','close',crypto_dict['CHSB'].data, standardize=False, bollinger=True)
    compare_crypto('daily_volatility',crypto_dict, log=False, standardize=True)
    compare_crypto('close',crypto_dict, log=False, standardize=True)