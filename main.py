import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from music_genre_lstm import create_LSTM_model
from music_genre_CNN import create_CNN_model
from helper_function import *
import statistics
import matplotlib.pyplot as plt

def test_custom_music():
    return


def main():
    #Load features
    custom_features = features_custom_track("Dr.Dre_Still_D.R.E..wav")
    
    #data = extract_feature_GTZAN("data/genres_original")
    #x = np.array(data["mfcc"])
    #y = np.array(data["labels"])

    
    #Prepare LSTM model
    #create_LSTM_model()
    
    #Run LSTM model
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    #x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2)
    #model_lstm = tf.keras.models.load_model('GTZAN_LSTM_01.h5')
    #genre_predict_lstm = model_lstm.predict(custom_features)
    #genre_predict_lstm = np.argmax(genre_predict_lstm, axis=1)
    #print(genre_predict_lstm)
    #modus = statistics.mode(genre_predict_lstm.tolist())
    #mapped_genre_lstm = map_genre(modus)
    #print(f"Classified genre with LSTM is = {mapped_genre_lstm}")
    
    #Prepare CNN model
    #create_CNN_model()
    model_CNN = tf.keras.models.load_model('GTZAN_CNN_01.h5')
    genre_predict_CNN = model_CNN.predict(custom_features)
    genre_predict_CNN = np.argmax(genre_predict_CNN, axis=1)
    print(genre_predict_CNN)
    genres = [map_genre(genre) for genre in genre_predict_CNN.tolist()]

    df = pd.DataFrame(genres)
    count = df.value_counts()
    count_index = count.index.to_list()
    count_index = [item[0] for item in count_index]
    print(count_index)
    print(count.to_list())
    

    plt.bar(count_index, count.to_list(), color="orange")
    plt.title("Dr. Dre - Still D.R.E song classification by genres")
    plt.show()
    mapped_genre_CNN = map_genre(statistics.mode(genre_predict_CNN.tolist()))
    print(f"Classified genre with CNN for custom track is = {mapped_genre_CNN}.")


if __name__ == "__main__":
    main()