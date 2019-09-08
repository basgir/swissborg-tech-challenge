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

def plot_RSI(crypto, start=0, end=700):
    import numpy as np
    import matplotlib.pyplot as plt
    
    # If the end index is bigger than the index itself
    if end >= len(crypto.data['close']):
        end = len(crypto.data['close'])-1
        
    # Starting index vs ending index
    start_date = crypto.data.index[end].strftime("%d-%b-%Y")
    end_date = crypto.data.index[start].strftime("%d-%b-%Y")
    
    # We plot the series 
    t = crypto.data.iloc[start:end].index
    s1 = crypto.data['close'].iloc[start:end]
    s2 = crypto.data['boll_bands_upper_band'].iloc[start:end]
    s3 = crypto.data['boll_bands_lower_band'].iloc[start:end]
    s4 = crypto.data['RSI'].iloc[start:end]

    fig, axs = plt.subplots(2, 1, figsize=(30,20))
    axs[0].set_title('{0} close {1} to {2}'.format(crypto.name, start_date, end_date))
    axs[0].plot(t, s1, label='close')
    axs[0].plot(t, s2, label='BB up')
    axs[0].plot(t, s3, label='BB down')
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('close')
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(t, s4)
    # Overbought line
    axs[1].axhline(y=80, c='r')
    axs[1].fill_between(t, 80, s4, where=s4>80, color='r')

    # Oversold line
    axs[1].axhline(y=20, c='g')
    axs[1].fill_between(t, 20, s4, where=s4<20,  color='g')
    axs[1].set_ylabel('RSI')
    axs[1].grid(True)


    fig.tight_layout()
    plt.show()


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
        legend = [name, 'bollinger bands upper','bollinger bands lower' ]
        ax.legend(legend)
        ax.plot(data.index,data['boll_bands_upper_band'])
        ax.plot(data.index,data['boll_bands_lower_band'])
    
    label = colname
    
    # Label if log
    if log:
        label += "log scale"
        
    # Label if log
    if standardize:
        label += "standardized"

    ax.set(xlabel='(t) Date', ylabel=label,
           title=f'{name} Analysis of {colname} - (2019-01 / 2019-09)')
    
    ax.grid()
    
    if log:
        fig.savefig(f"./figs/{name}-{colname}-log-analysis.png", bbox_inches='tight', pad_inches=0)
    else:
        fig.savefig(f"./figs/{name}-{colname}-analysis.png", bbox_inches='tight', pad_inches=0)



def compare_crypto(colname,crypto_dict,log=False, standardize = False, avoid=[]):
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
        try:
            x = each[1].data[colname]
            x_1 = x[0]

            if log:    
                x = np.log(each[1].data[colname])             
                
            # If we want to standardize the serie
            # We also check if the items in the pandas.Series are numeric
            if (standardize):

                mean = each[1].data[colname].mean()
                std = each[1].data[colname].std()

                print(f"Crypto {each[0]}\tmean : {mean}\tstd :{std}")
                x = pd.to_numeric(x)
                x = (x - mean)/std

            if each[0] not in avoid:

                ax.plot(each[1].data.index,x)
                label_list.append(each[0])

        except:
            print(f"Error while processing {each[0]}")
            
    label = colname.replace("_"," ")
    fixed_legend = colname.replace("_"," ")

    # Label if log
    if log:
        label += " log scale"

    # Label if log
    if standardize:
        label += " standardized"
                
    ax.set(xlabel='(t) Date', ylabel=label,
           title=f"{fixed_legend} comparison - Year To Date")
    ax.legend(label_list)
    ax.grid()

    filename = colname.replace("_", "-")

    if log:
        fig.savefig(f"./figs/ALL-{filename}-log-comparison.png", bbox_inches='tight', pad_inches=0)
    else:
        fig.savefig(f"./figs/ALL-{filename}-comparison.png", bbox_inches='tight', pad_inches=0)




def compute_annualized_volatility(crypto_dict, trading_days=365):
    """Compute the returns of each cryptocurrencies
    
    Args:
        crypto_dict (dictionnary) : Dictionnary containing Cryptocurrency objects
    
    Return: 
        dict_crypto_volatility (dict) : dict of crypto returns
    """
    
    import matplotlib.pyplot as plt
    import numpy as np

    dict_crypto_volatility = {}

    for each in crypto_dict.items():
            
        num = len(each[1].data['returns'])
        r_n = each[1].data['returns']

        ann_vol = np.sqrt(trading_days) * np.std(r_n)

        dict_crypto_volatility[each[0]] = ann_vol
        
    label_list = []

    fig, ax = plt.subplots(figsize=(20,10))
    
    ax.bar(dict_crypto_volatility.keys(), dict_crypto_volatility.values())

    ax.set(xlabel='Cryptocurrencies', ylabel="% returns",
           title=f"Annualized Volatility on {trading_days} trading days - (2019-01 / 2019-09)")

    ax.grid()

    fig.savefig(f"./figs/ALL-annualized-volatility-2019-01-to-2019-09.png", bbox_inches='tight', pad_inches=0)

    # plt.show()

    return dict_crypto_volatility


def compute_performance(crypto_dict):
    """Compute the returns of each cryptocurrencies
    
    Args:
        crypto_dict (dictionnary) : Dictionnary containing Cryptocurrency objects
    
    Return: 
        dict_crypto_returns (dict) : dict of crypto returns
    """
    
    import matplotlib.pyplot as plt
    import numpy as np

    dict_crypto_returns = {}

    for each in crypto_dict.items():
        # try :     
            
        P_0 = each[1].data['close'].tail(1)[0]
        P_n = each[1].data['close'].head(1)[0]
        
        r = (P_n - P_0) / P_0 * 100

        dict_crypto_returns[each[0]] = r
        
    label_list = []
    fig, ax = plt.subplots(figsize=(20,10))
    
    ax.bar(dict_crypto_returns.keys(), dict_crypto_returns.values())

    ax.set(xlabel='Cryptocurrencies', ylabel="% returns",
           title=f"Performance comparison - (2019-01 / 2019-09)")

    ax.grid()

    fig.savefig(f"./figs/performance-comparison-2019-01-to-2019-09.png", bbox_inches='tight', pad_inches=0)

    # plt.show()

    return dict_crypto_returns


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




    plot_crypto('MCO','close',crypto_dict['MCO'].data, standardize=False, bollinger=True)
    plot_crypto('MCO','daily_volatility',crypto_dict['MCO'].data, standardize=False, bollinger=False)
    plot_crypto('MCO','volume',crypto_dict['MCO'].data, standardize=False, bollinger=False)

    plot_crypto('EDO','close',crypto_dict['EDO'].data, standardize=False, bollinger=True)
    plot_crypto('EDO','daily_volatility',crypto_dict['EDO'].data, standardize=False, bollinger=False)
    plot_crypto('EDO','volume',crypto_dict['EDO'].data, standardize=False, bollinger=False)

    plot_crypto('CRPT','close',crypto_dict['CRPT'].data, standardize=False, bollinger=True)
    plot_crypto('CRPT','daily_volatility',crypto_dict['CRPT'].data, standardize=False, bollinger=False)
    plot_crypto('CRPT','volume',crypto_dict['CRPT'].data, standardize=False, bollinger=False)

    plot_crypto('NEXO','close',crypto_dict['NEXO'].data, standardize=False, bollinger=True)
    plot_crypto('NEXO','daily_volatility',crypto_dict['NEXO'].data, standardize=False, bollinger=False)
    plot_crypto('NEXO','volume',crypto_dict['NEXO'].data, standardize=False, bollinger=False)

    plot_crypto('SXP','close',crypto_dict['SXP'].data, standardize=False, bollinger=True)
    plot_crypto('SXP','daily_volatility',crypto_dict['SXP'].data, standardize=False, bollinger=False)
    plot_crypto('SXP','volume',crypto_dict['SXP'].data, standardize=False, bollinger=False)

    plot_crypto('DROP','close',crypto_dict['DROP'].data, standardize=False, bollinger=True)
    plot_crypto('DROP','daily_volatility',crypto_dict['DROP'].data, standardize=False, bollinger=False)
    plot_crypto('DROP','volume',crypto_dict['DROP'].data, standardize=False, bollinger=False)
    
    plot_crypto('CHSB','close',crypto_dict['CHSB'].data, standardize=False, bollinger=True)
    plot_crypto('CHSB','daily_volatility',crypto_dict['CHSB'].data, standardize=False, bollinger=False)
    plot_crypto('CHSB','volume',crypto_dict['CHSB'].data, standardize=False, bollinger=False)

    plot_crypto('ETH','close',crypto_dict['ETH'].data, standardize=False, bollinger=True)
    plot_crypto('ETH','daily_volatility',crypto_dict['ETH'].data, standardize=False, bollinger=False)
    plot_crypto('ETH','volume',crypto_dict['ETH'].data, standardize=False, bollinger=False)


    compare_crypto('daily_volatility',crypto_dict, log=False, standardize=True)
    compare_crypto('market_cap',crypto_dict, log=False, standardize=False, avoid=["ETH"])
    compare_crypto('market_cap',crypto_dict, log=False, standardize=True, avoid=["ETH"])

    compare_crypto('close',crypto_dict, log=False, standardize=True, avoid=["ETH","MCO"])
    compare_crypto('volume',crypto_dict, log=False, standardize=True, avoid=["ETH"])


    print(compute_performance(crypto_dict))
    print(compute_annualized_volatility(crypto_dict))