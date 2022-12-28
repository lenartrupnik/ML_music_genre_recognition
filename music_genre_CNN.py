from utils import *
import tensorflow as tf
from keras import models, layers
from sklearn.model_selection import train_test_split

data = extract_feature_GTZAN("data/genres_original")

x = np.array(data["mfcc"])
y = np.array(data["labels"])

x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
y = tf.keras.utils.to_categorical(y, num_classes=10)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

y_train[y_train==10] = 9
y_val[y_val==10] = 9
y_test[y_test==10] = 9

input_shape = x_train.shape[1:]
print(input_shape)

cnn_model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape),
    layers.MaxPooling2D(2, padding='same'),
    
    layers.Conv2D(128, (3,3), activation='relu', padding='same', input_shape=input_shape),
    layers.MaxPooling2D(2, padding='same'),
    layers.Dropout(0.3),
    
    layers.Conv2D(128, (3,3), activation='relu', padding='same', input_shape=input_shape),
    layers.MaxPooling2D(2, padding='same'),
    layers.Dropout(0.3),
    
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dense(10, activation='softmax')
])

cnn_model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics='acc')
cnn_model.summary()

history = cnn_model.fit(x_train, y_train,
                        validation_data=(x_val, y_val),
                        epochs=40,
                        verbose=2,
                        batch_size=32)

y_pred = cnn_model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

print(np.sum(y_pred==y_test)/len(y_pred))