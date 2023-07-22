'''
split_audio.py
--------------
This script extracts the audio segments corresponding to the activity phases from the audio files and stores them in the Extracted_AP directory.
Usage:
    python split_audio.py
Authors:
    - Atharv Naik
    - Deepak Bajaj
'''

import os
import pandas as pd
from pydub import AudioSegment
from datetime import datetime


def main():
    APs = ['AP1', 'AP2', 'AP3']  # Activity Phases
    APFILE = f'/mnt/sirlshare/SAD study Data/Audio/timestamps.xlsx'

    for AP in APs:

        AUDIODIR = '/mnt/sirlshare/SAD study Data/Audio/Audio_mp3'
        OUTPUTDIR = f'/mnt/sirlshare/SAD study Data/Audio/Extracted_{AP}'
        # OUTPUTDIR = 'final/test'

        print(f'Extracting {AP}...\n')

        # Load the excel file into a pandas DataFrame
        df = pd.read_excel(APFILE)

        # Iterate over the rows of the DataFrame
        for index, row in df.iterrows():
            pid = str(row['PID'])
            start_time = str(row[f'{AP}_S'])
            end_time = str(row[f'{AP}_E'])
            print(pid, start_time, end_time)

            # if start_time or end_time is NaN, or blank, skip ahead
            skip_values = ('nan', 'NaT', '', ' ')
            if pd.isna(start_time) or pd.isna(end_time) or start_time in skip_values or end_time in skip_values:
                print(f'No timestamps found for pid: {pid}. Skipping ahead...')
                continue

            # check if pid is not already extracted in output directory
            if os.path.exists(os.path.join(OUTPUTDIR, pid + f'_{AP}.mp3')):
                print(f'Audio file for pid: {pid} already exists. Skipping ahead...')
                continue

            print(f'Extracting {AP} for pid: {pid}...', end='', flush=True)

            # infer the format of start and end times as hh:mm:ss or mm:ss
            if len(start_time.split(':')) == 2:
                # convert to hh:mm:ss
                start_time = '00:' + start_time
                end_time = '00:' + end_time

            file_name = ''

            for file in os.listdir(AUDIODIR):
                if file.startswith(pid):
                    file_name = file
                    break
            else:  # if no break
                print(f'No audio file found for pid: {pid}. Skipping ahead...')
                continue

            # Construct the input and output file paths
            input_file = os.path.join(AUDIODIR, file_name)
            output_file = os.path.join(OUTPUTDIR, pid + f'_{AP}.mp3')

            # Load the audio file
            audio = AudioSegment.from_file(input_file, format='mp3')

            # Convert start and end times to seconds

            start_seconds = (datetime.strptime(
                start_time, '%H:%M:%S') - datetime(1900, 1, 1)).total_seconds()
            end_seconds = (datetime.strptime(end_time, '%H:%M:%S') -
                           datetime(1900, 1, 1)).total_seconds()

            # Extract the specified segment
            segment = audio[start_seconds * 1000:end_seconds * 1000]

            # Save the extracted segment as a new audio file
            segment.export(output_file, format='mp3')

            print('Done')
        
    print('Done')


if __name__ == '__main__':
    main()

