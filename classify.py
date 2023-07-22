'''
classify.py
-----------
This script trains a Random Forest Classifier on the features extracted from the audio files and predicts participants experiencing negative/anxious feelings based on their self-assessment scores.
Usage:
    python classify.py
Authors:
    - Atharv Naik
    - Deepak Bajaj
'''



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
import warnings

warnings.filterwarnings('ignore')


MLFeaturesCSV = '/mnt/sirlshare/SAD study Data/Audio/featuresForML/final_features_AP1.csv'

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(MLFeaturesCSV)


# add a new column that sums values in AS1_1,AS1_2,AS1_3,AS1_4,AS1_5; but value in AS1_2 is taken as 5 - value in AS1_2
df['AS1_2'] = df['AS1_2'].apply(lambda x: 6 - x)
# add values in AS1_1,AS1_2,AS1_3,AS1_4,AS1_5 to a new column
df['AS1'] = df['AS1_1'] + df['AS1_2'] + df['AS1_3'] + df['AS1_4'] + df['AS1_5']

# add a column to df that is the mean of mfccfeatures
df['mfcc'] = df[['mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6', 'mfcc_7', 'mfcc_8', 'mfcc_9', 'mfcc_10', 'mfcc_11', 'mfcc_12', 'mfcc_13']].mean(axis=1)

# Split the DataFrame into X and y
X = df.drop(['pid', 'SPIN', 'AS1_1','AS1_2','AS1_3','AS1_4','AS1_5','RS1_1', 'RS1_2', 'RS1_3', 'RS1_4', 'RS1_5', 'ATS2_1', 'ATS2_2', 'ATS2_3', 'ATS2_4', 'ATS2_5', 'AS2_1', 'AS2_2',
             'AS2_3', 'AS2_4', 'AS2_5', 'RS2_1', 'RS2_2', 'RS2_3', 'RS2_4', 'RS2_5', 'ATS3_1', 'ATS3_2', 'ATS3_3', 'ATS3_4',
             'ATS3_5', 'AS3_1', 'AS3_2', 'AS3_3', 'AS3_4', 'AS3_5', 'RS3_1', 'RS3_2', 'RS3_3', 'RS3_4', 'RS3_5', 'PS', 'Age',
             'Gender', 'Location', 'FOPS', 'FONE', 'FOUSC', 'SPIN (Q1+..+Q17)', 'SPIN=FOPS+FONE+FOUSC', 'SPIN Score',
             'PBS', 'ATS1_1', 'ATS1_2', 'ATS1_3', 'ATS1_4', 'ATS1_5', 'mfcc'], axis=1)
# select ['mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6', 'mfcc_7', 'mfcc_8', 'mfcc_9', 'mfcc_10', 'mfcc_11', 'mfcc_12', 'mfcc_13', 'f0', 'chroma_1', 'chroma_2', 'chroma_3', 'chroma_4', 'chroma_5', 'chroma_6', 'chroma_7', 'chroma_8', 'chroma_9', 'chroma_10', 'chroma_11', 'chroma_12', 'energy', 'zcr'] columns from the DataFrame
# X = df[['mfcc']]
# print(X.head())

# y is 1 if SPIN is greater than 30, else 0
y = df['AS1'].apply(lambda x: 1 if x >= 15 else 0)
# y = df['SPIN'].apply(lambda x: 1 if x >= 30 else 0)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10, random_state=21)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


param_grid = {
    'n_estimators': [1, 10, 25, 50, 100],  # Number of trees in the forest
    'max_depth': [None, 5, 10],  # Maximum depth of each tree
    'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10]  # Minimum number of samples required to split an internal node
}

# using random forest classifier

# Train the model
model = RandomForestClassifier(random_state=21)
# model.set_params(class_weight=class_weights)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters found
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Retrain the model with the best hyperparameters
best_rf = RandomForestClassifier(random_state=21, **best_params)
best_rf.fit(X_train, y_train)


# Make predictions on the test set
y_pred = best_rf.predict(X_test)

# Calculate accuracy on the test set
accuracy = mean_absolute_error(y_test, y_pred)
print("Test Set MAE:", accuracy)
f1score = f1_score(y_test, y_pred)
print("Test Set F1 Score:", f1score)
precision = precision_score(y_test, y_pred)
print("Test Set Precision:", precision)
recall = recall_score(y_test, y_pred)
print("Test Set Recall:", recall)
