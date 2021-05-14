from scipy.io import wavfile
import pandas as pd
import numpy as np
import os

'''
chunking with pandas

data is split into csvs with sectionSizes songs
each song has 2 rows, 10-5 second splits between x-y
'''
sectionNumber = 3
sectionSizes = 50

'''
x data is 10 seconds long
y data is 5 seconds long
'''
samples30 = 1323000
xSamples = samples30 // 3
ySamples = samples30 // 6

# creating the dataframes
xData = pd.DataFrame(columns = [x for x in range(xSamples)])
yData = pd.DataFrame(columns = [x for x in range(ySamples)])

directory = '../wav'
counter = 0

for filename in os.listdir(directory)[sectionNumber * sectionSizes:
                                        sectionSizes * (sectionNumber + 1)]:
    samplerate, data = wavfile.read('../wav/' + filename)
    data = np.array(data)[:,0]

    # progress
    print(counter + 1, filename)
    print("\tSample Rate: ", samplerate)
    print("\tData Length: ", len(data))

    if(len(data) >= samples30):
        xData.loc[filename+'-1'] = data[:xSamples]
        yData.loc[filename+'-1'] = data[xSamples: xSamples + ySamples]
        xData.loc[filename+'-2'] = data[xSamples + ySamples:
                                        xSamples * 2 + ySamples]
        yData.loc[filename+'-2'] = data[xSamples * 2 + ySamples:
                                        2 * (xSamples + ySamples)]
    else:
        print("DIDNT WORK", len(data))

    counter += 1

xData.to_csv('../wav_csvs/xData' + str(sectionNumber) + '-'
                                 + str(sectionSizes) + '.csv')
yData.to_csv('../wav_csvs/yData' + str(sectionNumber) + '-'
                                 + str(sectionSizes) + '.csv')