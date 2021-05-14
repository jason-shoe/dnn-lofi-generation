# Deep Neural Networks (COMP137) Final Project - Ashwin Swar and Jason Xu
Description:

### Data:

### Enviornment:
```
conda create -n dnnlofienv python=3.7
conda activate dnnlofienv
pip install tensorflow==2.3
conda install -c conda-forge notebook
conda install -c anaconda matplotlib
conda install scikit-learn
pip install keras-tcn
conda install pandas
```

### How to Train:
    training occured using Kaggle as well as the Tufts HPC.
    Running TCN/training.ipynb as well as STFT/training.ipynb should be sufficient to train the models.

### How to Use Model:
    Running TCN/results.ipynb and STFT/results.ipynb will allow one to see the models in use!

### File Structure
```
data/
└---mp3/                        - mp3 files of songs used for testing data
└---retrieval/
│   └---csv/
│   │   │   playlists.csv
│   │   │   songlist.csv
│   │
│   │   mp3-retrieval.py        - Gets mp3 files from spotify
│   │   songlist-retrieval.py   - Gets sample URLs and creates songlist.csv
│   │   vectorize-seperation.py - Converts songs in seperation/ to csv files in wav_csvs/seperation/
│   │   vectorize-wav-factor.py - Creates downsampled wav_csvs/downsample from wav/ songs
│   │   vectorize-wav.py        - Discontinued
│   |   wav-converter.py        - Converts mp3/ songs into wav/ songs
|
└---seperation/                 - folders for each of the songs split up into channels
└---wav/                        - wav files of songs used for testing data
└---wav_csvs/
    └---downsample/             - TCN/ model data
    └---seperation/             - STFT/ model data
TCN/
|   results.ipynb               -
│   training.ipynb              -
└---model/                      -
STFT/                           -
│   helper.py                   - Helper functions
│   results.ipynb               - Reults/Exploration Notebook
│   training.ipynb              - Training Notebook
│
└---model/                      - Trained models for STFT Encoder Decoder
└---samples/                    - 
practice_models/                - Random practice files
requirements.txt                -
```