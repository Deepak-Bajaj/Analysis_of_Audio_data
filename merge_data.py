'''
make_final_csv_data.py
----------------------
This script merges the features extracted from the audio files with the self-assessment scores and the responses of the candidates.
Usage:
    python make_final_csv_data.py
Authors:
    - Atharv Naik
    - Deepak Bajaj
'''

import numpy as np
import pandas as pd

# Load the CSV file into a pandas DataFrame
df1 = pd.read_csv('/mnt/sirlshare/SAD study Data/Audio/AP1Features/AP1Features.csv')
df2 = pd.read_csv('/mnt/sirlshare/SAD study Data/Audio/candidate_self_assessments/spin_scores.csv')
df3 = pd.read_csv('/mnt/sirlshare/SAD study Data/Audio/candidate_self_assessments/candidate_responses.csv')

df3['AS1_2'] = df3['AS1_2'].apply(lambda x: 6 - x)

# for df3 only consider 

# Rename the first column of df2 and df3 to 'pid'
df2.rename(columns={df2.columns[0]: 'pid'}, inplace=True)
df3.rename(columns={df3.columns[0]: 'pid'}, inplace=True)

# Merge the two DataFrames based on the 'pid' column for common pid values in both DataFrames

df = pd.merge(df1, df2, on='pid', how='inner')
df = pd.merge(df, df3, on='pid', how='inner')

final_columns = list(df1.columns) + list(df2.columns.drop('pid')) + list(df3.columns.drop('pid'))

df = df[final_columns]

print(df.head())

# df.to_csv('/mnt/sirlshare/SAD study Data/Audio/featuresForML/final_features_AP1.csv', index=False)
df.to_csv('final/final_features_AP1.csv', index=False)
