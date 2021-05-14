import urllib.request
import pandas as pd

songlist = pd.read_csv('csv/songlist.csv')

for index, row in songlist.iterrows():
    # if the url exists
    if(not pd.isna(row['songURL'])):
        try:
            url = row['songURL']
            name = '../mp3/'+str(index)+'.mp3'
            urllib.request.urlretrieve(url, name)
            print(str(int(index/len(songlist)*100)) + "% - ", row['songURL'])
        except:
            print(str(int(index/len(songlist)*100)) + "% - Error - " + row['songURL'])
print("100%")