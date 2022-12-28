import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import os
import pickle
import random
from tqdm import tqdm
from utils import *

directory = "Data/genres_original"
f = open("mydataset.dat", "wb")
dataset = []

# Read all files 
for idx, folder in tqdm(enumerate(os.listdir(directory)), "Reading files", total=len(os.listdir(directory))):
    for file in os.listdir(directory+"/"+folder):
        try:
            (rate, sig) = wav.read(directory+"/"+folder+"/"+file)
            mfcc_feat = mfcc(sig, rate, winlen = 0.020, appendEnergy=False)
            covariance = np.cov(np.matrix.transpose(mfcc_feat))
            mean_matrix = mfcc_feat.mean(0)
            feature = (mean_matrix, covariance, idx)
            pickle.dump(feature, f)
        except Exception as e:
            print("Got an exception: ", e, 'in folder: ', folder, ' filename: ', file)
f.close()


def load_dataset(filename, split, tr_set, te_set):
    with open(filename,'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break
    for x in range(len(dataset)):
        if random.random() < split:
            tr_set.append(dataset[x])
        else:
            te_set.append(dataset[x])

trainingSet = []
test_set = []
load_dataset('mydataset.dat', 0.7, trainingSet, test_set)

# Make the prediction using KNN(K nearest Neighbors)
length = len(test_set)
predictions = []
for x in range(length):
    predictions.append(nearest_class(get_neighbors(trainingSet, test_set[x], 5)))

accuracy1 = get_accuracy(test_set, predictions)
print(accuracy1)