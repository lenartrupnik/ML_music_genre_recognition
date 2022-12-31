import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from music_genre_lstm import create_LSTM_model
from music_genre_CNN import create_CNN_model
from utils.helper_functions import *
import statistics
import matplotlib.pyplot as plt

# Custom track path
# Track must be in .wav format
TRACK_PATH = "Dr.Dre_Still_D.R.E..wav"
GENRES = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "raggae", "rock"]


def LSTM_model(extracted_features, create_model:bool = False) -> dict:
    """"Returns classified genres based on LSTM model
    """
    
    if create_model:
        create_LSTM_model()
        
    model_lstm = tf.keras.models.load_model('GTZAN_LSTM.h5')
    genre_predict_lstm = model_lstm.predict(extracted_features)
    classified_genres = np.argmax(genre_predict_lstm, axis=1)
    modus = statistics.mode(classified_genres.tolist())
    genre_lstm = map_genre(modus)
    
    lstm_genres = [map_genre(genre) for genre in classified_genres.tolist()]
    lstm_dict = {genre:0 for genre in GENRES}
    for genre in lstm_genres:
        lstm_dict[genre] += 1
    
    
    print(f"Accuracy for classified genre with LSTM module for {TRACK_PATH}is = {genre_lstm}")
    
    return lstm_dict


def CNN_model(extracted_features, create_model: bool = False) -> dict:
    """"Returns classified genre based on CNN model
    """
    
    if create_model:
        create_CNN_model()

    model_CNN = tf.keras.models.load_model('GTZAN_CNN.h5')
    genre_predict_CNN = model_CNN.predict(extracted_features)
    classified_genres = np.argmax(genre_predict_CNN, axis=1)
    mapped_genre_CNN = map_genre(statistics.mode(classified_genres.tolist()))
    
    cnn_genres = [map_genre(genre) for genre in classified_genres.tolist()]
    cnn_dict = {genre:0 for genre in GENRES}
    for genre in cnn_genres:
        cnn_dict[genre] += 1
    print(f"Accuracy for classified genre with LSTM module for {TRACK_PATH}is =  {mapped_genre_CNN}.")
    
    return cnn_dict


def main():
    #Load custom features
    extracted_features = custom_track_features(TRACK_PATH)
    
    #Run lstm model
    lstm_predictions = LSTM_model(extracted_features)
    
    #Run cnn model
    cnn_predictions = CNN_model(extracted_features)

    for name in GENRES:
        if lstm_predictions[name] == 0 and cnn_predictions[name] == 0:
            del lstm_predictions[name]
            del cnn_predictions[name]
            
    df = pd.DataFrame({"LSTM model":lstm_predictions.values(),
                  "CNN model":cnn_predictions.values()}, lstm_predictions.keys())

    df.plot(kind="bar")
    plt.title("Dr. Dre - Still D.R.E song classification by genres")
    plt.xlabel("Genres")
    plt.ylabel("Number of classifications")
    plt.tight_layout()
    plt.show()
    
    
if __name__ == "__main__":
    main()