import pandas as pd
from utils.helper_functions import *
import tensorflow as tf
from sklearn.model_selection import train_test_split


def create_LSTM_model():
    #Extract features from GTZAN library
    data = extract_feature_GTZAN("data/genres_original", num_mfcc=40)
    x = np.array(data["mfcc"])
    y = np.array(data["labels"])
    
    # Preprocess data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=10)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, random_state=10)

    input_shape = (x_train.shape[1], x_train.shape[2])
    
    #Build LSTM model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Normalization())
    model.add(tf.keras.layers.LSTM(units=130, input_shape=input_shape, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.15))
    model.add(tf.keras.layers.LSTM(units=300))
    model.add(tf.keras.layers.Dropout(0.15))
    model.add(tf.keras.layers.Dense(units=300, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                loss="sparse_categorical_crossentropy",
                metrics=['accuracy'])

    history = model.fit(x_train, 
                        y_train, 
                        validation_data=(x_val, y_val), 
                        batch_size=32, 
                        epochs=35, 
                        verbose=2)
    plot_history(history)
    model.save("GTZAN_LSTM.h5")
    #pd.DataFrame.from_dict(history.history).to_csv('history_LSTM', index=False)
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    
    #Plot confusion matrix
    plot_confusion_matrix(y_pred, y_test)
    
    #Print accuracy
    print(f'Accuracy for lstm model = {np.sum(y_pred==y_test)/len(y_pred)}')
    