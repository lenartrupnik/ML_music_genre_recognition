import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
gtzan_data = pd.read_csv('Data/features_3_sec.csv')
gtzan_data.head()

x = gtzan_data.iloc[:,1:-1]
y = gtzan_data.label

x_np = x.to_numpy()
print(x_np.shape)
x_ts = []
y_ts = []
window_size = 50

for i in range(x_np.shape[0] - window_size):
    x_ts.append(x_np[i:window_size + i, :].tolist())
    y_ts.append(y[i])
    
x_ts_np = np.array(x_ts, dtype=object).astype('float32')
y_ts_np = np.array(y_ts, dtype=object)

print(x_ts_np.shape)
print(y_ts_np.shape)

label_encoder = LabelEncoder()
y_ts_np = label_encoder.fit_transform(y_ts_np)

x_train, x_test, y_train, y_test = train_test_split(x_ts_np, y_ts_np, test_size=0.2,
                                                    stratify=y_ts_np, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2,
                                                  stratify=y_train, random_state=42)

input_shape = x_train.shape[1:]

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(64, input_shape = input_shape, return_sequences=True))
model.add(tf.keras.layers.LSTM(64))
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))
optimiser = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimiser,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())

model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=32, epochs=60, verbose=2)
model.save("GTZAN_full_LSTM.h5")

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)

print(np.sum(y_pred==y_test)/len(y_pred))