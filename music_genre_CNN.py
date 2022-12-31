from sklearn import metrics
from utils.helper_functions import *
import tensorflow as tf
from keras import models, layers
from sklearn.model_selection import train_test_split
import pandas as pd


def create_CNN_model():
    data = extract_feature_GTZAN("data/genres_original", num_mfcc = 13)

    x = np.array(data["mfcc"])
    y = np.array(data["labels"])

    x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
    y = tf.keras.utils.to_categorical(y, num_classes=10)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state=10)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=10)

    y_train[y_train==10] = 9
    y_val[y_val==10] = 9
    y_test[y_test==10] = 9

    input_shape = x_train.shape[1:]

    cnn_model = models.Sequential([
        layers.Conv2D(64, (3,3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D(2, padding='same'),
        
        layers.Conv2D(256, (3,3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D(2, pvfgadding='same'),
        layers.Dropout(0.3),
        
        layers.Conv2D(256, (3,3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D(2, padding='same'),
        layers.Dropout(0.3),
        
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    cnn_model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics='accuracy')
    cnn_model.summary()

    history = cnn_model.fit(x_train, y_train,
                            validation_data=(x_val, y_val),
                            epochs=40,
                            verbose=2,
                            batch_size=32)

    cnn_model.save("GTZAN_CNN.h5")
    y_pred = cnn_model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)
    #pd.DataFrame.from_dict(history.history).to_csv('history_CNN', index=False)
    plot_history(history)
    
    #Plot confusion matrix
    plot_confusion_matrix(y_pred, y_test)

    print(f'Accuracy for CNN model = {np.sum(y_pred==y_test)/len(y_pred)}')
    