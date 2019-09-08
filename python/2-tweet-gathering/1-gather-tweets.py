####################################
#  Author : Bastien Girardet
#  Email : bastien.girardet@gmail.com
#  Website : bastien.girardet.me
#  Goal : Gather tweets from a list of crypto
#  Date : 08-09-2019
####################################

# We import the required libraries
import pandas as pd
import numpy as np
import tweepy
import time
import sys


# We import the csv(s) and select the relevant columns
# crypto list
df_crypto = pd.read_csv("./data/crypto.csv")
df_crypto = df_crypto[['symbol','symbol']]
df_crypto.columns = ['Symbol','Name']


# We authenticate on twitter
consumer_key = "z7gp32ZtNDlQOj4W92v6wJqm5" 
consumer_secret = "vbDEgRXlqALekExDz2wqhTiU6CwpBsLeEVNOF1al9hapqywsl6" 

access_token = "901719311143903233-V2t305dvpgFtMwonEIXoof8FOAKZxiH"
access_token_secret = "z9knLifKkDRtJwgTQqxZxpA0Gy05HWVqsVj9K5M8fX0up"


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)



# Main functions
# If there is an error creating the api instance
if (not api):
    print ("Can't Authenticate")
    sys.exit(-1)

# define parsing functions
def parse_tweet(tweet):
    """
    Takes result object from tweepy and parses it
    """

    # initialize dict
    parsed_tweet = {}

    # extract relevant info
    parsed_tweet['Author Name'] = tweet.author.name
    parsed_tweet['Text'] = tweet.text
    parsed_tweet['Message ID'] = tweet.id
    parsed_tweet['Published At'] = tweet.created_at
    parsed_tweet['Retweet Count'] = tweet.retweet_count
    parsed_tweet['Favorite Count'] = tweet.favorite_count

    return parsed_tweet

def format_response(response, crypto):
    """
    Takes list of result objects from tweepy and formats it
    """
    try:
        parsed_tweets = pd.DataFrame([parse_tweet(tweet) for tweet in response], columns=[ 'Crypto', 'Author Name', 'Text', 'Message ID', 'Published At', 'Retweet Count', 'Favorite Count'])
        parsed_tweets['Crypto'] = str(crypto)
    except TypeError as e:
        print(e)
        parsed_tweets = pd.DataFrame(columns=[ 'Crypto', 'Author Name', 'Text', 'Message ID', 'Published At', 'Retweet Count', 'Favorite Count'])

    return parsed_tweets



# # main program
if __name__ == '__main__':
    
    # We create a timestamp
    ts = int(time.time())

    # if twitter data exists
    try:
        data = pd.read_csv('./data/update.csv')
    except:
        data = False

    # loop over cryptos
    for idx,crypto in enumerate(df_crypto['Name']):

        print('Processing {0} {1} of {2}'.format(crypto, str(idx), str(len(df_crypto['Name']))))

        # define params dict
        params={
            'q' : crypto+" cryptocurrency"
        }

        # add max_id if prior data file exists
        if data is not False:

            # if this company exists in our dataset
            if crypto in data['Crypto']:

                # add the max_id param so we dont collect redundant tweets
                params['since_id'] = data[data['Crypto']==crypto]['Message ID'].max()

        try:
            # make the call to twitter
            response = api.search(**params)

        # handle error
        except tweepy.error.TweepError as e:

            print(e)

            # Will run up to the point where it reaches the Rate limit per 15 min.
            response = api.search(**params)

        # format response
        formatted_response = format_response(response, crypto)

        # write out the result
        if data is not False:
            formatted_response.to_csv(f'./data/1-raw-tweets/twitter_data_{ts}.csv', mode='a', header=False, encoding='utf-8')
        elif idx == 0:
            formatted_response.to_csv(f'./data/1-raw-tweets/twitter_data_{ts}.csv', encoding='utf-8')
        else:
            formatted_response.to_csv(f'./data/1-raw-tweets/twitter_data_{ts}.csv', mode='a', header=False, encoding='utf-8')
            
    print("Tweet gathered and saved under 1-raw-tweets/twitter_data_{ts}.csv")
    print("Should wait at least half a day until re-run")