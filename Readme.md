# Audio Data Analysis Project

## Introduction
This project is a part of the summer internship program at IISER Bhopal in 2023. The objective of this project is to analyze audio files from 111 participants and identify correlations between audio features and the emotions expressed during their speech activity. The audio files are in WAV format, and the emotions are classified into four categories: Happy, Sad, Anxious, and Neutral. Each audio is divided into three activity phases: AP1, AP2, and AP3. Audio features are extracted for each activity phase and used to train a machine learning model on AP1, which is then used to predict the emotions of the participants based on their audio features.

In addition to audio analysis, the transcriptions of each activity phase are also performed for future application of natural language processing (NLP) techniques to identify correlations between participants' emotions and the words they use during their speech activity.

## Dataset
The dataset consists of video recordings from approximately 111 participants. Participants were asked to speak on a given topic for around 2 minutes in AP1. The video files were recorded using a mobile phone and saved in MP4 format. The audio from each video file was then extracted and saved in WAV format. The final dataset for ML training was prepared by extracting the features from AP1 and the questionnire responses and SPIN scores of each participant and included the following columns:

| Column Name | Description |
| ----------- | ----------- |
| PID | Participant ID |
| feature1 | Mean of the first feature |
| feature2 | Mean of the second feature |
...
| feature45 | Mean of the forty-fifth feature |
| AS1_1 | Answer to the first question of the first questionnaire |
...
| AS1_5 | Answer to the fifth question of the first questionnaire |
| SPIN | SPIN score of the participant |

## Feature Extraction Pipeline
The audio is extracted from the video files using the `moviepy` library. The extracted audio files are saved in the "audio" folder. The audio files are then split into three activity phases (AP1, AP2, and AP3) based on provided timestamps using the `split_audio.py` script.

For each activity phase, audio features are extracted using the `extract_features.py` script. The script utilizes the `AudioFeatureExtractor` class defined in the `audio_feature_extractor.py` file. A total of 45 features are extracted for each activity phase.

The extracted features are saved in separate CSV files for each activity phase. The feature extraction pipeline involves the following steps:

1. Run `python conv2audio.py` to extract audio from video files.
2. Run `python split_audio.py` to split the audio files into activity phases.
3. Run `python extract_features.py` to extract audio features for each activity phase.

The extracted features CSV files are then merged with the participants' SPIN scores and questionnaire responses using the `merge_data.py` script to create a final dataset for ML classification.


## Machine Learning Classification
The final dataset is utilized to train a machine learning model for classifying participants' emotions. The dataset is split into training and testing sets using the `train_test_split` function from the `sklearn.model_selection` module. The training set is used to train the model, and the testing set is used to evaluate the model's accuracy.

The machine learning model is trained using the `RandomForestClassifier` class from the `sklearn.ensemble` module. After training, the model is saved using the `joblib.dump` function from the `sklearn.externals` module.


## Transcription Pipeline
Transcription of each activity phase is performed for future NLP tasks. The transcription process involves using Buzz AI (with the openai-whisper large model) for AP1 and the otter.ai platform for AP2 and AP3. The transcriptions are manually corrected and saved in separate TXT files for each participant audio in each activity phase.