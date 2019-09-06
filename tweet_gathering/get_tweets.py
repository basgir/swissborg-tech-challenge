import tweepy

consumer_key = "z7gp32ZtNDlQOj4W92v6wJqm5" 
consumer_secret = "vbDEgRXlqALekExDz2wqhTiU6CwpBsLeEVNOF1al9hapqywsl6" 

access_token = "901719311143903233-V2t305dvpgFtMwonEIXoof8FOAKZxiH"
access_token_secret = "z9knLifKkDRtJwgTQqxZxpA0Gy05HWVqsVj9K5M8fX0up"


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

public_tweets = api.home_timeline()
for tweet in public_tweets:
    print(tweet.text)
