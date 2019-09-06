####################################
#  Author : Bastien Girardet
#  Goal : Fetch historical data for each cryptocurrencies in our dataset
#         To assess the performance of each.
#  Date : 05-09-2019
####################################

# We import the required libraries
import pandas as pd 
import re
import time
from datetime import datetime

# We get the list of cryptocurrencies
df_crypto = pd.read_csv("crypto.csv")

# List to build the metadata
ohlc_data_list = []
crypto_filenames_list = []
start_list = []
end_list = []
format = "%Y%m%d"

# We gather the data for each crypto in our crypto dataset
for idx, crypto in df_crypto.iterrows():
    try:
        print(f"Fetching {crypto['symbol']} data...")
        df = pd.read_html(crypto['url'])[0]
        print("Fetched!")

        print("Processing Data...")
        start = re.search("start=([0-9]{8})",crypto['url'])[1]
        end = re.search("end=([0-9]{8})",crypto['url'])[1]

        start_date_format = datetime.strptime(start, format)
        end_date_format = datetime.strptime(end, format)

        start_list.append(start_date_format)
        end_list.append(end_date_format)
        print("Data processed!")

        print("Saving dataset...")
        filename = f"{crypto['symbol']}-{start}-to-{end}.csv"
        crypto_filenames_list.append(filename)

        df.to_csv(f"./crypto-ohlc/{filename}")
        print(f"Saved under {filename}")
        print("="*50)
    except:
        print("Something went wrong...")
        
# We save our metadata
df_crypto['filename'] = crypto_filenames_list
df_crypto['start_date'] = start_list
df_crypto['end_date'] = end_list

# we save the relevant metadata by specifying the columns in order to avoid a double index.
df_crypto[['symbol','name','url','identifier','filename','start_date','end_date']].to_csv("crypto.csv")