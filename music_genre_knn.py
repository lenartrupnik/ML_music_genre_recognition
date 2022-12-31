import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils.helper_functions import *

directory = "Data/genres_original"
dataset = []

def run_KNN_classification(custom_path:str=""):
    # Read all files 
    features = []
    for idx, folder in tqdm(enumerate(os.listdir(directory)), "Reading files", total=len(os.listdir(directory))):
        for file in os.listdir(directory+"/"+folder):
            try:
                (rate, sig) = wav.read(directory+"/"+folder+"/"+file)
                mfcc_feat = mfcc(sig, rate, winlen = 0.020, numcep= 40, appendEnergy=False)
                covariance = np.cov(np.matrix.transpose(mfcc_feat))
                mean_matrix = mfcc_feat.mean(0)
                feature = (mean_matrix, covariance, idx)
                features.append(feature)
            
            except Exception as e:
                print("Got an exception: ", e, 'in folder: ', folder, ' filename: ', file)

    x_train, x_test = train_test_split(features, test_size = 0.25, random_state=10)
    #load_dataset('mydataset.dat', 0.75, trainingSet, test_set)

    # Make the prediction using KNN(K nearest Neighbors)
    length = len(x_test)
    predictions = []
       
    for x in range(length):
        predictions.append(nearest_class(get_neighbors(x_train, x_test[x], 5)))

    y_test = []
    for x in range(len(x_test)):
        y_test.append(x_test[x][-1])
    accuracy = get_accuracy(x_test, predictions)
    print(f"KNN accuracy is {accuracy}")
    
    plot_confusion_matrix(predictions, y_test)
    return predictions, y_test

run_KNN_classification()

