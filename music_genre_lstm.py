from utils import *
import tensorflow as tf
from tensorflow import keras
import keras_tuner

from sklearn.model_selection import train_test_split

data = extract_feature_GTZAN("data/genres_original")

x = np.array(data["mfcc"])
y = np.array(data["labels"])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2)

input_shape = (x_train.shape[1], x_train.shape[2])

model = tf.keras.Sequential()

model.add(tf.keras.layers.LSTM(units=299, input_shape=input_shape, return_sequences=True))
model.add(tf.keras.layers.LSTM(units=299))

model.add(tf.keras.layers.Dense(units=299, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation="softmax"))

hp_learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)
model.compile(optimizer=optimizer,
                loss="sparse_categorical_crossentropy",
                metrics=['accuracy'])

model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=32, epochs=28, verbose=2)
model.save("GTZAN_LSTM.h5")

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
print(np.sum(y_pred==y_test)/len(y_pred))


""""
def build_model(hp):
    model = tf.keras.Sequential()
    
    model.add(tf.keras.layers.LSTM(units=hp.Int('units', min_value=32, max_value=320), input_shape=input_shape, return_sequences=True))
    model.add(tf.keras.layers.LSTM(units=hp.Int('units', min_value=32, max_value=320, step=32)))

    model.add(tf.keras.layers.Dense(units=hp.Int('units', min_value=32, max_value=320, step=32), activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
    optimizer = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])
    
    return model

tuner = keras_tuner.Hyperband(build_model,
                              objective="val_accuracy",
                              max_epochs=30,
                              factor=3,
                              hyperband_iterations=10,
                              directory="kt_dir",
                              project_name="kt_hyperband")

tuner.search_space_summary()
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search(x_train, 
             y_train, 
             epochs=20, 
             validation_data=(x_val, y_val), 
             callbacks=[stop_early],
             verbose = 2)

best_hps = tuner.get_best_hyperparameters()[0]
print(best_hps)


"""