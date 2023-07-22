'''
conv2audio.py
-------------
Converts all the videos in the VIDEOSDIR directory to audio files and stores them in the AUDIOSDIR directory as mp3 files.
Usage:
    python conv2audio.py
Authors:
    - Atharv Naik
    - Deepak Bajaj
'''

import os
import tqdm
import moviepy.editor

# # Set the base directory
BASEDIR = '/mnt/sirlshare/SAD study Data'
VIDEOSDIR = os.path.join(BASEDIR, 'Video')
AUDIOSDIR = os.path.join(BASEDIR, 'Audio', 'Audio_mp3_4')

# # Read the videos directory and get all the videos in it into a list
videos = os.listdir(VIDEOSDIR)

if __name__ == '__main__':
    for video in tqdm.tqdm(videos):
        try:
            videoFile = moviepy.editor.VideoFileClip(
                os.path.join(VIDEOSDIR, video))
            audio = videoFile.audio
            audio.write_audiofile(os.path.join(
                AUDIOSDIR, video.split('.')[0] + '.mp3'))
        except Exception as e:
            print(e)
