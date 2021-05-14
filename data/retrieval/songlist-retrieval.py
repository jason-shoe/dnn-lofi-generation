import spotipy
#To access authorised Spotify data
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import re
import string
import config

'''
    Function Name: get_playlist_tracks
    Description:
        Retrieves the list of songs in a playlist
    Input:
        username (string) - username of track creator
        playlist_id (string) - URI of playlist
    Output:
        List of tracks (dictionaries)
        https://developer.spotify.com/documentation/web-api/reference/#category-tracks
'''
def getPlaylistTracks(username,playlist_id):
    results = sp.user_playlist_tracks(username,playlist_id)
    tracks = results['items']
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
    return tracks

def parseTrack(track):
        name = track['track']['name']
        url = track['track']['preview_url']
        uri = track['track']['uri']
        return [name, url, uri]


if __name__ == "__main__":
    regex = re.compile(".*?\((.*?)\)")

    # client_id and client_secret are defined in hidden config.py
    credentials_manager = SpotifyClientCredentials(client_id=config.client_id,
                                             client_secret=config.client_secret)

    # spotify object to access API
    sp = spotipy.Spotify(client_credentials_manager=credentials_manager)
    
    # get the requested playlists
    playlists = pd.read_csv('csv/playlists.csv')

    # generate the song information
    songdf = pd.DataFrame(columns=['songName','songURL', 'songURI'])
    for index, row in playlists.iterrows():
        # get the songs in the playlist
        tracks = getPlaylistTracks(row['Creator'],row['URI'])
        print(int(index/len(playlists)*10000)/100, row['Playlist Name'])
        for track in tracks:
            trackInfo = parseTrack(track)
            if (trackInfo):
                songdf.loc[len(songdf)] = trackInfo
    
    print("Number of Songs Before Removing Duplicates: ", len(songdf))
    songdf.drop_duplicates(subset=['songURI'], keep='first', inplace = True)
    print("Number of Songs After Removing Duplicates:  ", len(songdf))
    songdf.to_csv('csv/songlist.csv', index=None)
