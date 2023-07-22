'''
extract_features.py
-------------------
This script extracts features from the audio files and stores them in a CSV file.
Run this script for AP1, AP2, and AP3 separately (change APDIR accordingly) to generate audio features file for each AP.
Usage:
    python extract_features.py
Authors:
    - Atharv Naik
    - Deepak Bajaj
'''

import os
from audio_features_extractor import AudioFeaturesExtractor
import pandas as pd
import numpy as np
from tqdm import tqdm
from warnings import filterwarnings

filterwarnings('ignore')

AP = 'AP1'  # Activity Phase; change this for AP2 and AP3

BASEDIR = '/mnt/sirlshare/SAD study Data/Audio'  # path/to/base/directory
APDIR = os.path.join(BASEDIR, f'Extracted_{AP}')  # path/to/AP/directory
FEATURESDIR = os.path.join(BASEDIR, f'{AP}Features') # path/to/features/directory
OUTPUTCSV = f'{AP}Features.csv'  # output csv file name
# FEATURESDIR = 'features'
os.makedirs(FEATURESDIR, exist_ok=True)


if __name__ == '__main__':
    # load each audio file and extract features, store their mean in a dataframe like this:
    # pid, mean_feature1, mean_feature2, mean_feature3, ...

    # get list of audio files
    audioFiles = os.listdir(APDIR)

    # create dataframe to store features
    columns = ['pid', 'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6', 'mfcc_7', 'mfcc_8', 'mfcc_9', 'mfcc_10', 'mfcc_11', 'mfcc_12', 'mfcc_13', 'f0', 'chroma_1', 'chroma_2', 'chroma_3', 'chroma_4', 'chroma_5', 'chroma_6', 'chroma_7', 'chroma_8', 'chroma_9', 'chroma_10', 'chroma_11', 'chroma_12', 'energy', 'zcr',
               'spectral_centroid', 'spectral_bandwidth', 'spectral_contrast_1', 'spectral_contrast_2', 'spectral_contrast_3', 'spectral_contrast_4', 'spectral_contrast_5', 'spectral_contrast_6', 'spectral_contrast_7', 'spectral_rolloff', 'spectral_flux', 'spectral_flatness', 'f2', 'jitter', 'shimmer', 'band_energy_ratio', 'pause_rate']

    df = pd.DataFrame(columns=columns)
    lst = []

    # iterate through each audio file
    for audioFile in tqdm(audioFiles):
        audio = AudioFeaturesExtractor(os.path.join(APDIR, audioFile))

        # extract features; these will be stored as attributes of the AudioFeaturesExtractor instance
        audio.mfcc()
        audio.f0()
        audio.chroma()
        audio.energy()
        audio.zcr()
        audio.spectral_centroid()
        audio.spectral_bandwidth()
        audio.spectral_contrast()
        audio.spectral_rolloff()
        audio.spectral_flux()
        audio.spectral_flatness()
        audio.jitter()
        audio.shimmer()
        audio.band_energy_ratio()
        audio.f2()
        audio.pause_rate()

        # get features array
        featuresArray = audio.getMeanFeaturesArray(features=['mfcc', 'f0', 'chroma', 'energy', 'zcr', 'spectral_centroid',
                                                   'spectral_bandwidth', 'spectral_contrast', 'spectral_rolloff', 'spectral_flux', 'spectral_flatness'])

        # append pid to feature array
        featuresArray = np.insert(featuresArray, 0, audioFile.split('_')[0])

        # append mean features to feature array
        meanFeaturesArray = np.array(
            [audio.f2, audio.jitter, audio.shimmer, audio.band_energy_ratio, audio.pause_rate])
        featureArray = np.concatenate((featuresArray, meanFeaturesArray))

        lst.append(featureArray)

    df = pd.DataFrame(lst, columns=columns)

    # save dataframe to csv
    df.to_csv(os.path.join(FEATURESDIR, OUTPUTCSV), index=False)
