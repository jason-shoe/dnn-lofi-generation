from scipy.io import wavfile
import pandas as pd
import numpy as np
import os

'''
chunking with pandas

data is split into csvs with sectionSizes songs
each song has 2 rows, 10-5 second splits between x-y
'''
sectionNumber = 11
sectionSizes = 10

'''
x data is 10 seconds long
y data is 5 seconds long
'''
factor = 10
samples30 = 1323000 // factor
directory = '../seperation'
counter = 0
nums = [int(x) for x in os.listdir(directory) if x.isdigit() ]
all_files = np.sort(nums)
print(all_files)

# creating the dataframes
total = pd.DataFrame(columns = [x for x in range(samples30)])
other = pd.DataFrame(columns = [x for x in range(samples30)])
piano = pd.DataFrame(columns = [x for x in range(samples30)])
bass = pd.DataFrame(columns = [x for x in range(samples30)])
drums = pd.DataFrame(columns = [x for x in range(samples30)])
vocals = pd.DataFrame(columns = [x for x in range(samples30)])


for filename in all_files[sectionNumber * sectionSizes:
                                        sectionSizes * (sectionNumber + 1)]:
    filename = str(filename)
    samplerate, totalsong = wavfile.read('../wav/' + filename + '.wav')
    samplerate, basssong = wavfile.read('../seperation/' + filename + '/bass.wav')
    samplerate, drumssong = wavfile.read('../seperation/' + filename + '/drums.wav')
    samplerate, othersong = wavfile.read('../seperation/' + filename + '/other.wav')
    samplerate, pianosong = wavfile.read('../seperation/' + filename + '/piano.wav')
    samplerate, vocalssong = wavfile.read('../seperation/' + filename + '/vocals.wav')
    
    # progress
    print(counter + 1, filename)
    print("\tSample Rate: ", samplerate)
    print("\tData Length: ", len(totalsong[:, 0]))

    if(len(totalsong[:, 0]) >= 1323000):
        total.loc[filename] = totalsong[::factor,0][:samples30]
        bass.loc[filename] = basssong[::factor, 0][:samples30]
        drums.loc[filename] = drumssong[::factor, 0][:samples30]
        other.loc[filename] = othersong[::factor, 0][:samples30]
        piano.loc[filename] = pianosong[::factor, 0][:samples30]
        vocals.loc[filename] = vocalssong[::factor, 0][:samples30]
    else:
        print("DIDNT WORK", len(totalsong[:, 0]))

    counter += 1

total.to_csv('../wav_csvs/seperation/total' + str(sectionNumber) + '.csv')
bass.to_csv('../wav_csvs/seperation/bass' + str(sectionNumber) + '.csv')
other.to_csv('../wav_csvs/seperation/other' + str(sectionNumber) + '.csv')
piano.to_csv('../wav_csvs/seperation/piano' + str(sectionNumber) + '.csv')
drums.to_csv('../wav_csvs/seperation/drums' + str(sectionNumber) + '.csv')
vocals.to_csv('../wav_csvs/seperation/vocals' + str(sectionNumber) + '.csv')