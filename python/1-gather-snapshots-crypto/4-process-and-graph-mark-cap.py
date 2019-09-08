####################################
#  Author : Bastien Girardet
#  Email : bastien.girardet@gmail.com
#  Website : bastien.girardet.me
#  Goal : Market cap analysis by snapshots
#  Date : 03-09-2019
####################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import re
from datetime import datetime

# Folder containing all the snapshots
mypath = "./data/snapshots/"

# We create a list of all the filenames
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
df_files = pd.DataFrame({"filename" : onlyfiles})

# We save few important information such as the 
# Number of Crypto per snapshots
num_crypto_per_snapshot= []
# The time of the snapshot
timestamps = []
# If swissborg appeared on the market
is_swissborg_in = []

# We retrieve the data from each csv file
for idx, row in df_files.iterrows():

    # We read each csv file containing the snapshot data
    data = pd.read_csv(join(mypath, row['filename']))

    # We check if swissborg appears
    if len(data[data['Symbol'] == 'CHSB']) > 0:
        is_swissborg_in.append(True)
    else:
        is_swissborg_in.append(False)
    
    # We add the time of the snapshot and the number of crypto.
    date_time = re.findall("[0-9]{8}",row['filename'])[0]
    num_crypto_per_snapshot.append(len(data)) 
    timestamps.append(date_time)


# We process the data in order to display it.
df_files.index = pd.to_datetime(timestamps, format="%Y%m%d")
df_files['num_crypto'] = pd.to_numeric(num_crypto_per_snapshot)
df_files['is_swissborg_in'] = is_swissborg_in
df_files.sort_index(inplace=True)

# We add swissborg introduction as a vertical timestamp line
CHSB introduction = df_files[df_files['is_swissborg_in'] == True].head(1).index.values[0]


# We plot the Cryptocurrencies number evolution
fig, ax = plt.subplots(figsize=(30,10))
ax.plot(df_files.index, df_files['num_crypto'])

ax.set(xlabel='Date', ylabel='Amount',title='Cryptocurrencies number evolution (2013-2019)')
ax.grid()
ax.vlines(CHSB introduction, ymin = 0, ymax = max(df_files['num_crypto']),color="red",label="CHSB introduction")
ax.legend(["Cryptocurrencies number evolution", "CHSB introduction"])

# We save the fig
fig.savefig("./figs/cryptocurrencies-number-evolution-2013-2019.png")


# We read the snapshots meta data
snapshots_data = pd.read_csv("./data/list.csv")
snapshots_data.index = pd.to_datetime(snapshots_data['snapshots_date'])
snapshots_data.drop(labels=['Unnamed: 0','Unnamed: 0.1', 'snapshots_date'],axis=1, inplace=True)

subperiod = snapshots_data[snapshots_data.index >= "2017-01-01"]

fig, ax = plt.subplots(figsize=(30,10))
ax.plot(subperiod.index, subperiod['total_cap'])

ax.set(xlabel='Date', ylabel='Market cap', title='Cryptocurrencies market capitalisation (2013-2019)')
ax.grid()
ax.vlines(CHSB introduction, ymin = 0, ymax = max(subperiod['total_cap']),color="red",label="CHSB introduction")

ax.legend(["Market Cap Evolution", "CHSB introduction"])

fig.savefig("./figs/cryptocurrencies-market-cap-2013-2019.png")

print("Done !")