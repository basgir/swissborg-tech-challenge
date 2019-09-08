####################################
#  Author : Bastien Girardet
#  Email : bastien.girardet@gmail.com
#  Website : bastien.girardet.me
#  Goal : Fetch all the urls and snapshots into a csv file
#         in order to create a graph on the evolution of crpypto.
#  Date : 05-09-2019
####################################

# We import the required libraries
import re
import requests
import pandas as pd 
import numpy as np
from bs4 import BeautifulSoup
from datetime import datetime


# We first need to gather all the urls
url_where_urls_are = "https://coinmarketcap.com/historical/"
snapshot_url_html_page = requests.get(url_where_urls_are).text
soup = BeautifulSoup(snapshot_url_html_page, 'html.parser')
all_urls = soup.find_all('a')

snapshots_url_list = []
snapshots = []
snapshots_dates = []

# We go through each urls and process them.
for url in all_urls:

    # The length of the url is equals 3 (attributes)
    if len(url.text) == 3:

        # We search for the value of the snapshot (e.g. 20130428)
        # Then create the url in the form of https://coinmarketcap.com/historical/20130428
        if re.search("\/historical\/[0-9]{8}",url.attrs['href']):
            snapshots_url_list.append(f"https://coinmarketcap.com{url.attrs['href']}")
        else:
            snapshots_url_list.append(np.nan)

        # If we sucessfully fetched the href we retrieve the current snapshot value (e.g. 20130428)
        # As well as the current snapshot date (e.g. 28-04-2013)
        if re.search("[0-9]{8}",url.attrs['href']):
            current_snapshot = re.search("[0-9]{8}",url.attrs['href'])[0]
            snapshots.append(current_snapshot)

            # If snapshot exists
            current_snapshot_date = datetime.strptime(current_snapshot, '%Y%m%d')
            snapshots_dates.append(current_snapshot_date)
        else:
            snapshots.append(np.nan)
            snapshots_dates.append(np.nan)

# We put the retrieved data into a pandas.DataFrame
data = pd.DataFrame({'snapshots_url': snapshots_url_list, 'snapshots': snapshots, 'snapshots_date': snapshots_dates})

# We delete useless rows 
data = data.iloc[4:len(data)-1,:]

# We export the data into a csv file
data.to_csv("./data/list.csv")