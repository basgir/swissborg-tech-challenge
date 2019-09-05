import requests


url = "https://medium.com/swissborg"
data = requests.get(url).text

print(data)



