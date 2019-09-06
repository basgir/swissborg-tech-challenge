####################################
#  Author : Bastien Girardet
#  Goal : Gather all snapshots from coin market cap
#         in order to create a graph on the evolution of crpypto.
#  Date : 03-09-2019
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

def get_total_cap(df):
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
    while idx < len(df):
        try :
            url = df['snapshots_url'].iloc[idx]

            # Log message
            print(f"Currently fetching : {url} \tindex : {idx}/{len(df['snapshots_url'])}")

            # Retrieve the total cap at snapshot 

            # Retrieve the html 
            html_doc = requests.get(url).text
            
            
            # Create the Soup from the html document
            soup = BeautifulSoup(html_doc, 'html.parser')

            # We retrieve the total cap
            total_cap = soup.find("span", {"id": "total-marketcap"}).attrs['data-usd']


            # We add the total cap value of the snapshot

            total_cap_list.append(total_cap)



            # Log message
            print("Done fetching !")

            print(f"Current state of toal_cap {total_cap_list}")
            
            # We go to the next iteration
            idx += 1
        except:
            # In case of there is a problem fetching the data
            # Usually it is because the website's DDOS policy.
            # We wait for 30 seconds and retry the same url.
            print("Something went wrong...")
            print(f"Need to resume at {idx}")
            print("Waiting 30 seconds before re-trying...")
            time.sleep(30)
            continue
        
    df['total_cap'] = total_cap_list
    return df

df = get_total_cap(df)

df.to_csv("./snapshots/list.csv")
