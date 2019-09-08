####################################
#  Author : Bastien Girardet
#  Email : bastien.girardet@gmail.com
#  Website : bastien.girardet.me
#  Goal : Get all the snapshots data from coin market cap. save it as csv etc...
#  Date : 08-09-2019
####################################

# We import the required libraries
import requests
import pandas as pd 
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import re
import time

# We read the csv file
df = pd.read_csv("./snapshots/list.csv")

def non_stop_gather(snapshot_url_list):
    """Provided the list of urls, the method gathers the snapshot data of coinmarket cap
    
    Args: 
        snapshot_url_list (list) : List of all urls in the form of ["https://coinmarketcap.com/historical/20190901/"]
    Output:
        list_of_cryptocurrencies (list) : contains the list of pandas.Dataframe which contain the respective snapshot data.
    """

    idx = 0
    filename_list = []
    total_cap_list = []

    # We go through all urls, extract the timestamp and gather their respective data from coinmarketcap.com
    # We use a while here in order to keep track of the index when a request is not performed correctly.
    # When the request fails, we wait and try again after 30 seconds.
    while idx < len(snapshot_url_list):
        try :

            # Log message
            print(f"Currently fetching : {snapshot_url_list[idx]} \tindex : {idx}/{len(snapshot_url_list)}")


            # We try to fetch the data from coin market cap
            df = pd.read_html(snapshot_url_list[idx])

            # We only are concerned by the first table
            df = df[0] 

            # Log message
            print("Done fetching !")

            # we filer the time stamp out of the urls
            timestamp = re.search('[0-9]{8}',snapshot_url_list[idx])[0]

            # Log messages
            print(f"snapshot url : {snapshot_url_list[idx]}")
            print(f"timestamp : {timestamp}")
            
            # We define the filename based on timestamp
            filename = f"./snapshots/snapshot-{timestamp}.csv"

            # Log message
            print(f"Saving...\t{filename}")
            
            # We save the file to the folder
            df.to_csv(filename)
            
            idx += 1
        except:
            # In case of there is a problem fetching the data
            # Usually it is because the website's DDOS policy.
            # We wait for 30 seconds and retry the same url.
            print("Something went wrong...")
            print(f"Need to resume at {idx}")
            print("Waiting 5seconds before re-trying...")
            time.sleep(30)
            continue

    return filename_list

non_stop_gather(df['snapshots_url'])
