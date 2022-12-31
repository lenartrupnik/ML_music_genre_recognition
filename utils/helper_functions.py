import math
import os
import numpy as np
import operator
import numpy as np
from sklearn import metrics
from tqdm import tqdm
import librosa
import matplotlib.pyplot as plt

#GTZAN library path
directory = "Data/genres_original"

#Define a function to get distance between feature vectors and find neighbors
def get_neighbors(training_set, instance, k):
    distances = []
    for x in range(len(training_set)):
        dist = distance(training_set[x], instance, k) + distance(instance,training_set[x],k)
        distances.append((training_set[x][2], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
        
    return neighbors
    
    
#Function to identify the nearest neighbors
def nearest_class(neighbors):
    classVote = {}
    
    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVote:
            classVote[response] += 1
        else:
            classVote[response] = 1
            
    sorter = sorted(classVote.items(), key=operator.itemgetter(1), reverse=True)
    return sorter[0][0]

#Function that return accuracy of KNN model
def get_accuracy(test_set, prediction):
    correct = 0
    for x in range(len(test_set)):
        if test_set[x][-1] == prediction[x]:
            correct += 1
    return 1.0 * correct / len(test_set)
    
#Function to calculate distance for KNN model
def distance(instance1, instance2, k):
    distance = 0
    mm1 = instance1[0]
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]
    distance = np.trace(np.dot(np.linalg.inv(cm2), cm1))
    distance += (np.dot(np.dot((mm2-mm1).transpose(), np.linalg.inv(cm2)), mm2-mm1))
    distance += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance -= k
    return distance

# Make feature extracti on based on GTZAN library
def extract_feature_GTZAN(path, num_mfcc=40, n_fft=2048, hop_length=512, num_segment=10):
    
    #Prepare empty dictionary for extracted features
    extracted_features = {"labels": [], "mfcc":[]}
    
    #Empirical number suited only for GTZAN dictionary
    sr = 22050
    samples_per_segment = int(sr*30/num_segment)
    
    #Go through every audio file in data and extract mfcc features
    for label_idx, (dirpath, dirnames, filenames) in tqdm(enumerate(os.walk(path)), total=11):
        if dirpath == path:
            continue
        
        for f in sorted(filenames):
            if not f.endswith('.wav'):
                continue
            
            file_path = os.path.join(dirpath, f)
            
            #Load and read .wav file into 2D audio time series and sampling rate
            try:
                y, sr = librosa.load(file_path, sr=None)
                
            except Exception as e:
                print("Got an exception: ", e, 'in folder: ', file_path)
                
            #Restructure into n number of segments
            for n in range (num_segment):
                mfcc_feature = librosa.feature.mfcc(y = y[samples_per_segment*n:samples_per_segment*(n+1)],
                                            sr=sr, 
                                            n_mfcc=num_mfcc, 
                                            n_fft = n_fft,
                                            hop_length=hop_length)
                mfcc_feature = mfcc_feature.T
                
                if len(mfcc_feature) == math.ceil(samples_per_segment / hop_length):
                    extracted_features["mfcc"].append(mfcc_feature.tolist())
                    extracted_features["labels"].append(label_idx-1)
        
    return extracted_features


def plot_history(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    
def custom_track_features(path=str):
    y, sr = librosa.load(path)
    num_mfcc=40
    n_fft=2048
    hop_length=512
    sr = 22050

    mfcc_feature = librosa.feature.mfcc(y = y,
                            sr=sr, 
                            n_mfcc=num_mfcc, 
                            n_fft = n_fft,
                            hop_length=hop_length)
    mfcc_feature = mfcc_feature.T
    
    num_sectors = mfcc_feature.shape[0] // 130
    reshaped_mfcc = []
    for i in range(num_sectors):
        reshaped_mfcc.append(mfcc_feature[130*i:130*(i+1)])
    reshaped_mfcc = np.stack(tuple(reshaped_mfcc))
    return reshaped_mfcc
    
    
def map_genre(index:int):
    genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "raggae", "rock"]
    return genres[index]
    
def plot_confusion_matrix(y_pred, y_test):
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    cm_display = metrics.ConfusionMatrixDisplay(
        confusion_matrix = confusion_matrix, 
        display_labels = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "raggae", "rock"])
    cm_display.plot()
    plt.show()