####################################
#  Author : Bastien Girardet
#  Goal : Graphs the evolution of ETHEREUM addresses
#  Date : 06-09-2019
####################################

# We import the required libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# Data processing
df_addresses = pd.read_csv("./data/export-AddressCount.csv")
df_addresses.columns= ["date","timestamp","addr_count"]
df_addresses.index = pd.to_datetime(df_addresses["date"])
df_addresses = df_addresses[['addr_count']]

# We compute the growth rate
r_t = df_addresses['addr_count'].pct_change()

# We create a subplot
fig, ax = plt.subplots(2,1,figsize=(20,10))

# Plot the evolution of ethereum wallet addresses evolution
ax[0].plot(df_addresses.index, df_addresses["addr_count"])
ax[0].set(xlabel='(t) Date', ylabel="Amount",
        title=f"Number of Ethereum wallet addresses")
ax[0].grid()

# Plot the evolution of ethereum wallet addresses evolution
ax[1].plot(df_addresses.index,r_t)
ax[1].set(xlabel='(t) Date', ylabel="%age growth",
        title=f"Growth rate")
ax[1].grid()

# We save the fig into png
fig.savefig(f"./figs/ETH-addresses-evolution.png", bbox_inches='tight', pad_inches=0)